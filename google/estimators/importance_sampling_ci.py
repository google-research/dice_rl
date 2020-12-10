# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats as stats
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tf_agents.specs import tensor_spec
from tf_agents.policies import tf_policy
from tf_agents.utils import common as tfagents_common
from typing import Any, Callable, Iterable, Optional, Sequence, Text, Tuple, Union

import dice_rl.data.dataset as dataset_lib
import dice_rl.utils.common as common_lib


class ImportanceSamplingCI(object):
  """Approximate average reward of policy using importance sampling."""

  def __init__(self,
               dataset_spec,
               policy_optimizer,
               policy_network,
               mode,
               ci_method,
               delta_tail,
               gamma: Union[float, tf.Tensor],
               reward_fn: Callable = None,
               clipping: Optional[float] = 2000.,
               policy_regularizer: float = 0.,
               q_network=None,
               q_optimizer=None,
               target_update_tau: Union[float, tf.Tensor] = 0.01,
               target_update_period: int = 1,
               num_samples: Optional[int] = None):
    """Initializes the importance sampling estimator.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      policy_optimizer: The optimizer to use for learning policy.
      policy_network: The policy NN network.
      mode: Importance sampling estimator (e.g., "weighted-step-wise").
      ci_method: Method for constructing confidence intervals (e.g., "CH" for
        Chernoff-Hoeffding).
      delta_tail: Total probability quantile threshold (will be halved in code
        for 2-tail)
      gamma: The discount factor to use.
      reward_fn: A function that takes in an EnvStep and returns the reward for
        that step. If not specified, defaults to just EnvStep.reward.
      clipping: Threshold for clipping IS factor.
      policy_regularizer: float on policy regularizer.
      q_network: A function that returns the values for each observation and
        action. If specified, the Q-values are learned and used for
        doubly-robust estimation.
      q_optimizer: TF optimizer for q_network.
      target_update_tau: Rate at which to set target network parameters.
      target_update_period: Rate at which to set target network parameters.
      num_samples: Number of samples to take from policy to estimate average
        next state value. If actions are discrete, this defaults to computing
        average explicitly. If actions are not discrete, this defaults to using
        a single sample.
    """
    self._dataset_spec = dataset_spec
    self._policy_optimizer = policy_optimizer
    self._policy_network = policy_network
    if self._policy_network is not None:
      self._policy_network.create_variables()
    self._mode = mode
    self._ci_method = ci_method
    self._delta_tail = delta_tail
    self._gamma = gamma
    if reward_fn is None:
      reward_fn = lambda env_step: env_step.reward
    self._reward_fn = reward_fn
    self._clipping = clipping
    self._policy_regularizer = policy_regularizer

    self._q_network = q_network
    if self._q_network is not None:
      self._q_network.create_variables()
      self._target_network = self._q_network.copy(name='TargetQNetwork')
      self._target_network.create_variables()
      self._target_update_tau = target_update_tau
      self._target_update_period = target_update_period
      self._update_targets = self._get_target_updater(
          tau=self._target_update_tau, period=self._target_update_period)
      self._q_optimizer = q_optimizer
      self._initialize()

    self._num_samples = num_samples
    self._categorical_action = common_lib.is_categorical_spec(self._dataset_spec.action)
    if not self._categorical_action and self._num_samples is None:
      self._num_samples = 1

  def _get_target_updater(self, tau=1.0, period=1):

    def update():
      return tfagents_common.soft_variables_update(
          self._q_network.variables,
          self._target_network.variables,
          tau,
          tau_non_trainable=1.0)

    return tfagents_common.Periodically(update, period, 'update_targets')

  def _initialize(self):
    tfagents_common.soft_variables_update(
        self._q_network.variables, self._target_network.variables, tau=1.0)

  def _orthogonal_regularization(self, network):
    reg = 0
    for layer in network.layers:
      if isinstance(layer, tf.keras.layers.Dense):
        prod = tf.matmul(tf.transpose(layer.kernel), layer.kernel)
        reg += tf.reduce_sum(tf.math.square(prod * (1 - tf.eye(prod.shape[0]))))
    return reg

  def _get_q_value(self, env_step):
    if self._q_network is None:
      return tf.zeros_like(env_step.reward)
    return self._q_network((env_step.observation, env_step.action))[0]

  def _get_v_value(self, env_step, policy):
    return self._get_average_value(self._q_network, env_step, policy)

  def _get_target_value(self, env_step, policy):
    return self._get_average_value(self._target_network, env_step, policy)

  def _get_average_value(self, network, env_step, policy):
    if self._q_network is None:
      return tf.zeros_like(env_step.reward)

    tfagents_step = dataset_lib.convert_to_tfagents_timestep(env_step)
    if self._categorical_action and self._num_samples is None:
      action_weights = policy.distribution(
          tfagents_step).action.probs_parameter()
      action_dtype = self._dataset_spec.action.dtype
      batch_size = tf.shape(action_weights)[0]
      num_actions = tf.shape(action_weights)[-1]
      actions = (  # Broadcast actions
          tf.ones([batch_size, 1], dtype=action_dtype) *
          tf.range(num_actions, dtype=action_dtype)[None, :])
    else:
      batch_size = tf.shape(env_step.observation)[0]
      num_actions = self._num_samples
      action_weights = tf.ones([batch_size, num_actions]) / num_actions
      actions = tf.stack(
          [policy.action(tfagents_step).action for _ in range(num_actions)],
          axis=1)

    flat_actions = tf.reshape(
        actions,
        tf.concat([[batch_size * num_actions], tf.shape(actions)[2:]], axis=0))
    flat_observations = tf.reshape(
        tf.tile(env_step.observation[:, None, ...],
                [1, num_actions] + [1] * len(env_step.observation.shape[1:])),
        tf.concat([[batch_size * num_actions], tf.shape(env_step.observation)[1:]], axis=0))
    flat_values, _ = network((flat_observations, flat_actions))

    values = tf.reshape(
        flat_values,
        tf.concat([[batch_size, num_actions], tf.shape(flat_values)[1:]], axis=0))
    return tf.reduce_sum(values * action_weights, axis=1)


  def _get_log_prob(self, policy_network, env_step):
    # TODO(ofirnachum): env_step.action is shaped [B] but network's action_spec
    # is BoundedTensorSpec(shape=[1], ...); which leads network to use a
    # MVNDiag distribution here with event_shape=[1].  MVNDiag expects inputs of
    # shape [B, 1].
    return policy_network(env_step.observation)[0].log_prob(
        env_step.action[..., tf.newaxis])

  def clip_is_factor(self, is_factor):
    return tf.minimum(self._clipping, tf.maximum(-self._clipping, is_factor))

  def clip_log_factor(self, log_factor):
    return tf.minimum(tf.math.log(self._clipping),
                      tf.maximum(-tf.math.log(self._clipping), log_factor))

  def get_is_weighted_reward_samples(self,
                                     dataset: dataset_lib.OffpolicyDataset,
                                     target_policy: tf_policy.TFPolicy,
                                     episode_limit: Optional[int] = None,
                                     eps: Optional[float] = 1e-8):
    """Get the IS weighted reweard samples."""
    episodes, valid_steps = dataset.get_all_episodes(limit=episode_limit)
    total_num_steps_per_episode = tf.shape(valid_steps)[1] - 1
    num_episodes = tf.shape(valid_steps)[0]
    num_samples = num_episodes * total_num_steps_per_episode

    init_env_step = tf.nest.map_structure(
        lambda t: t[:, 0, ...], episodes)
    env_step = tf.nest.map_structure(
        lambda t: tf.squeeze(
            tf.reshape(t[:, 0:total_num_steps_per_episode, ...],
                       [num_samples, -1])), episodes)
    next_env_step = tf.nest.map_structure(
        lambda t: tf.squeeze(
            tf.reshape(t[:, 1:1 + total_num_steps_per_episode, ...],
                       [num_samples, -1])), episodes)
    tfagents_env_step = dataset_lib.convert_to_tfagents_timestep(env_step)

    gamma_weights = tf.reshape(
        tf.pow(self._gamma, tf.cast(env_step.step_num, tf.float32)),
        [num_episodes, total_num_steps_per_episode])

    rewards = (-self._get_q_value(env_step) +
               self._reward_fn(env_step) +
               self._gamma * next_env_step.discount *
               self._get_v_value(next_env_step, target_policy))
    rewards = tf.reshape(rewards, [num_episodes, total_num_steps_per_episode])

    init_values = self._get_v_value(init_env_step, target_policy)
    init_offset = (1 - self._gamma) * init_values

    target_log_probabilities = target_policy.distribution(
        tfagents_env_step).action.log_prob(env_step.action)
    if tf.rank(target_log_probabilities) > 1:
      target_log_probabilities = tf.reduce_sum(target_log_probabilities, -1)
    if self._policy_network is not None:
      baseline_policy_log_probability = self._get_log_prob(
          self._policy_network, env_step)
      if tf.rank(baseline_policy_log_probability) > 1:
        baseline_policy_log_probability = tf.reduce_sum(
            baseline_policy_log_probability, -1)
      policy_log_ratios = tf.reshape(
          tf.maximum(-1.0 / eps, target_log_probabilities -
                     baseline_policy_log_probability),
          [num_episodes, total_num_steps_per_episode])
    else:
      policy_log_ratios = tf.reshape(
          tf.maximum(-1.0 / eps,
                     target_log_probabilities - env_step.get_log_probability()),
          [num_episodes, total_num_steps_per_episode])
    valid_steps_in = valid_steps[:, 0:total_num_steps_per_episode]
    mask = tf.cast(
        tf.logical_and(valid_steps_in, episodes.discount[:, :-1] > 0.),
        tf.float32)

    masked_rewards = tf.where(mask > 0, rewards, tf.zeros_like(rewards))
    clipped_policy_log_ratios = mask * self.clip_log_factor(policy_log_ratios)

    if self._mode in ['trajectory-wise', 'weighted-trajectory-wise']:
      trajectory_avg_rewards = tf.reduce_sum(
          masked_rewards * gamma_weights, axis=1) / tf.reduce_sum(
              gamma_weights, axis=1)
      trajectory_log_ratios = tf.reduce_sum(clipped_policy_log_ratios, axis=1)
      if self._mode == 'trajectory-wise':
        trajectory_avg_rewards *= tf.exp(trajectory_log_ratios)
        return init_offset + trajectory_avg_rewards
      else:
        offset = tf.reduce_max(trajectory_log_ratios)
        normalized_clipped_ratios = tf.exp(trajectory_log_ratios - offset)
        normalized_clipped_ratios /= tf.maximum(
            eps, tf.reduce_mean(normalized_clipped_ratios))
        trajectory_avg_rewards *= normalized_clipped_ratios
        return init_offset + trajectory_avg_rewards

    elif self._mode in ['step-wise', 'weighted-step-wise']:
      trajectory_log_ratios = mask * tf.cumsum(policy_log_ratios, axis=1)
      if self._mode == 'step-wise':
        trajectory_avg_rewards = tf.reduce_sum(
            masked_rewards * gamma_weights * tf.exp(trajectory_log_ratios),
            axis=1) / tf.reduce_sum(
                gamma_weights, axis=1)
        return init_offset + trajectory_avg_rewards
      else:
        # Average over data, for each time step.
        offset = tf.reduce_max(trajectory_log_ratios, axis=0)  # TODO: Handle mask.
        normalized_imp_weights = tf.exp(trajectory_log_ratios - offset)
        normalized_imp_weights /= tf.maximum(
            eps,
            tf.reduce_sum(mask * normalized_imp_weights, axis=0) /
            tf.maximum(eps, tf.reduce_sum(mask, axis=0)))[None, :]

        trajectory_avg_rewards = tf.reduce_sum(
            masked_rewards * gamma_weights * normalized_imp_weights,
            axis=1) / tf.reduce_sum(
                gamma_weights, axis=1)
        return init_offset + trajectory_avg_rewards
    else:
      ValueError('Estimator is not implemented!')

  def estimate_average_reward(self,
                              dataset: dataset_lib.OffpolicyDataset,
                              target_policy: tf_policy.TFPolicy,
                              episode_limit: Optional[int] = None):
    is_weighted_reward_samples = self.get_is_weighted_reward_samples(
        dataset, target_policy, episode_limit)
    return tf.reduce_mean(is_weighted_reward_samples)

  def estimate_reward_ci(self,
                         dataset: dataset_lib.OffpolicyDataset,
                         target_policy: tf_policy.TFPolicy,
                         episode_limit: Optional[int] = None,
                         num_grid: Optional[int] = 100,
                         eps: Optional[float] = 1e-6,
                         num_bootstraps: Optional[int] = 10000,
                         num_bootstrap_samples: Optional[int] = 10000):
    """Estimate the confidence interval of reward."""
    is_weighted_reward_samples = self.get_is_weighted_reward_samples(
        dataset, target_policy, episode_limit)
    episodes, valid_steps = dataset.get_all_episodes(limit=episode_limit)
    num_episodes = tf.shape(valid_steps)[0]
    max_abs_reward = tf.reduce_max(
        tf.where(valid_steps, tf.abs(self._reward_fn(episodes)), 0.))

    # mean estimate
    center = self.estimate_average_reward(dataset, target_policy)
    delta_tail_half = self._delta_tail / 2.0
    num_episodes_float = tf.cast(num_episodes, tf.float32)

    if self._ci_method == 'CH':  # Chernoff-Hoeffding
      width = max_abs_reward * tf.math.sqrt(
          tf.math.log(1.0 / delta_tail_half) / num_episodes_float)
      lb = center - width
      ub = center + width
    elif self._ci_method == 'BE':  # Empirical Bernstein
      constant_term = 7 * max_abs_reward * tf.math.log(
          2.0 / delta_tail_half) / (3 * (num_episodes_float - 1))
      variance_term = tf.reduce_sum(
          tf.square(is_weighted_reward_samples - center))

      variance_term *= tf.math.log(2.0 / delta_tail_half) / (
          num_episodes_float - 1)
      width = constant_term + tf.math.sqrt(variance_term) / num_episodes_float
      lb = center - width
      ub = center + width
    elif self._ci_method == 'C-BE':  # Clipped empirical Bernstein
      # need to learn c
      def compute_center_width(c_const):
        """Compute the center and width of CI."""
        c_vec = c_const * tf.ones_like(is_weighted_reward_samples)
        c_is_weighted_reward_samples = tf.minimum(is_weighted_reward_samples,
                                                  c_vec) / c_vec
        constant_term = 7 * num_episodes_float * tf.math.log(
            2.0 / delta_tail_half) / (3 * (num_episodes_float - 1))

        center = tf.reduce_sum(c_is_weighted_reward_samples) / tf.reduce_sum(
            1.0 / c_vec)
        variance_term = tf.reduce_sum(
            tf.square(c_is_weighted_reward_samples - center))
        variance_term *= tf.math.log(2.0 / delta_tail_half) / (
            num_episodes_float - 1)

        width = (constant_term + tf.math.sqrt(variance_term)) / tf.reduce_sum(
            1.0 / c_vec)
        return center, width

      def compute_bdd(c_const):
        center, width = compute_center_width(c_const)
        return center - width, center + width

      def compute_obj(c_const, obj='width'):
        center, width = compute_center_width(c_const)
        if obj == 'lb':
          return center - width
        elif obj == 'ub':  # minimize ub
          return -(center + width)
        elif obj == 'width':
          return width
        elif obj == 'lb_ub':
          return -2 * width
        else:
          ValueError('Objective is not implemented')

      c_grid = tf.linspace(eps, max_abs_reward, num_grid)
      objs = tf.map_fn(compute_obj, c_grid, dtype=tf.float32)

      star_index = tf.argmax(objs)
      c_star = tf.gather(c_grid, star_index)

      lb, ub = compute_bdd(c_star)

    elif self._ci_method == 'TT':  # Student-t test
      # Two-tailed confidence intervals
      t_statistic_quantile = stats.t.ppf(1 - delta_tail_half,
                                         num_episodes_float - 1)
      std_term = tf.math.sqrt(
          tf.reduce_sum(tf.square(is_weighted_reward_samples - center)) /
          (num_episodes_float - 1))
      width = t_statistic_quantile * std_term / tf.math.sqrt(num_episodes_float)
      lb = center - width
      ub = center + width
    elif self._ci_method == 'BCa':  # Bootstrap
      # see references
      # https://faculty.washington.edu/heagerty/Courses/b572/public/GregImholte-1.pdf
      # http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf
      gaussian_rv = tfp.distributions.Normal(loc=0, scale=1)

      def _compute_bootstrap_lb_ub(reward_samples):
        """Compute Efron's bootstrap lb."""
        sample_mean = tf.reduce_mean(reward_samples)
        # Step 1, sample with replacement and compute subsampled mean
        uniform_log_prob = tf.tile(
            tf.expand_dims(tf.zeros(num_episodes), 0), [num_bootstraps, 1])
        ind = tf.random.categorical(uniform_log_prob, num_bootstrap_samples)
        bootstrap_subsamples = tf.gather(reward_samples, ind)
        subsample_means = tf.reduce_mean(bootstrap_subsamples, axis=1)

        # Step 2, sort subsample means, compute y, z_0, and a
        sorted_subsample_means = tf.sort(
            subsample_means, axis=0, direction='ASCENDING')

        # bias factor
        z_0 = gaussian_rv.quantile(
            tf.reduce_sum(
                tf.cast(
                    tf.greater(sample_mean, sorted_subsample_means),
                    tf.float32)) / float(num_bootstraps))
        # y is the leave-one-out, jackknife sample mean
        mask_matrix = tf.ones([num_episodes, num_episodes
                              ]) - tf.eye(num_episodes)
        leave_one_out_subsample_sums = tf.einsum('j,jk->k', reward_samples,
                                                 mask_matrix)
        ys = leave_one_out_subsample_sums / (num_episodes_float - 1)

        # average of jackknife estimate
        y_bar = tf.reduce_mean(ys)

        # acceleration factor
        d_ys = y_bar - ys
        a = tf.reduce_sum(tf.pow(d_ys, 3.0)) / tf.maximum(
            eps, 6.0 * tf.pow(tf.reduce_sum(tf.pow(d_ys, 2.0)), 1.5))

        # Step 3, compute z_scores for lb and ub
        z_score_delta_tail = gaussian_rv.quantile(delta_tail_half)
        z_score_1_delta_tail = gaussian_rv.quantile(1.0 - delta_tail_half)

        z_lb = z_0 + (z_score_delta_tail + z_0) / tf.maximum(
            eps, 1 - a * (z_score_delta_tail + z_0))
        z_ub = z_0 + (z_score_1_delta_tail + z_0) / tf.maximum(
            eps, 1 - a * (z_score_1_delta_tail + z_0))

        # Step 4, compute corresponding quantiles and get bootstrap intervals
        lb_index = tf.cast(
            tf.maximum(
                tf.minimum(
                    tf.floor(num_bootstraps * gaussian_rv.cdf(z_lb)),
                    num_bootstraps - 1), 1), tf.int64)
        ub_index = tf.cast(
            tf.maximum(
                tf.minimum(
                    tf.floor(num_bootstraps * gaussian_rv.cdf(z_ub)),
                    num_bootstraps - 1), 1), tf.int64)

        lb = tf.gather(sorted_subsample_means, lb_index)
        ub = tf.gather(sorted_subsample_means, ub_index)

        return lb, ub

      lb, ub = _compute_bootstrap_lb_ub(is_weighted_reward_samples)
    else:
      ValueError('Confidence interval is not implemented!')
    return [lb, ub]

  @tf.function
  def train_step(self, experience: dataset_lib.EnvStep,
                 target_policy: tf_policy.TFPolicy):
    """Performs a single training step based on batch and MLE.

    Args:
      experience: A batch of transitions. Elements must have shape [batch_size,
        2, ...].
      target_policy: The policy whose value we want to estimate.

    Returns:
      The losses and the train op.
    """
    env_step = tf.nest.map_structure(lambda t: t[:, 0, ...], experience)
    next_env_step = tf.nest.map_structure(lambda t: t[:, 1, ...], experience)

    if self._policy_network is not None:
      assert self._policy_optimizer is not None
      with tf.GradientTape(
          watch_accessed_variables=False, persistent=True) as tape:
        tape.watch(self._policy_network.variables)
        policy_loss = self.compute_policy_loss(env_step)
        policy_loss += self._policy_regularizer * self._orthogonal_regularization(
            self._policy_network)

      policy_grads = tape.gradient(policy_loss, self._policy_network.variables)
      policy_grad_op = self._policy_optimizer.apply_gradients(
          zip(policy_grads, self._policy_network.variables))
    else:
      policy_loss = 0.0

    if self._q_network is not None:
      assert self._q_optimizer is not None
      with tf.GradientTape(
          watch_accessed_variables=False, persistent=True) as tape:
        tape.watch(self._q_network.variables)
        q_loss = self.compute_q_loss(env_step, next_env_step, target_policy)

      q_grads = tape.gradient(q_loss, self._q_network.variables)
      q_grad_op = self._q_optimizer.apply_gradients(
          zip(q_grads, self._q_network.variables))
      update_op = self._update_targets()
    else:
      q_loss = 0.0

    return (tf.reduce_mean(policy_loss), tf.reduce_mean(q_loss))

  def compute_policy_loss(self, env_step):
    policy_loss = -tf.reduce_mean(
        self._get_log_prob(self._policy_network, env_step))

    return policy_loss

  def compute_q_loss(self, env_step, next_env_step, target_policy):
    q_value = self._get_q_value(env_step)
    target_value = tf.stop_gradient(
        self._get_target_value(next_env_step, target_policy))
    reward = self._reward_fn(env_step)
    td_error = (-q_value + reward +
                self._gamma * next_env_step.discount * target_value)

    return tf.math.square(td_error)
