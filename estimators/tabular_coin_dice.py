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
import tensorflow.compat.v2 as tf
from tf_agents.specs import tensor_spec
from tf_agents.policies import tf_policy
from typing import Any, Callable, Iterable, Optional, Sequence, Text, Tuple, Union

import dice_rl.data.dataset as dataset_lib
import dice_rl.utils.common as common_lib
import dice_rl.estimators.estimator as estimator_lib


class TabularCoinDice(object):
  """Robust policy evaluation."""

  def __init__(self,
               dataset_spec,
               gamma: Union[float, tf.Tensor],
               reward_fn: Optional[Callable] = None,
               solve_for_state_action_ratio: bool = True,
               divergence_limit: Union[float, np.ndarray, tf.Tensor] = 0.0,
               divergence_type: Text = 'rkl',
               nu_learning_rate: Union[float, tf.Tensor] = 0.1,
               zeta_learning_rate: Union[float, tf.Tensor] = 0.1,
               algae_alpha: Union[float, tf.Tensor] = 1.0,
               limit_episodes: Optional[int] = None):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      gamma: The discount factor to use.
      reward_fn: A function that takes in an EnvStep and returns the reward for
        that step. If not specified, defaults to just EnvStep.reward.
      solve_for_state_action_ratio: Whether to solve for state-action density
        ratio. Defaults to True. When solving an environment with a large
        state/action space (taxi), better to set this to False to avoid OOM
        issues.
      divergence_limit: The limit on the f-divergence between the weights and
        the empirical distribution.
      divergence_type: The type of f-divergence to use, e.g., 'kl'. Defaults to
        'rkl', reverse KL.
      nu_learning_rate: Learning rate for nu.
      zeta_learning_rate: Learning rate for zeta.
      algae_alpha: Regularizer coefficient on Df(dpi || dD).
      limit_episodes: How many episodes to take from the dataset. Defaults to
        None (take whole dataset).
    """
    self._dataset_spec = dataset_spec
    self._gamma = gamma
    if reward_fn is None:
      reward_fn = lambda env_step: env_step.reward
    self._reward_fn = reward_fn

    self._solve_for_state_action_ratio = solve_for_state_action_ratio
    if (not self._solve_for_state_action_ratio and
        not self._dataset_spec.has_log_probability()):
      raise ValueError('Dataset must contain log-probability when '
                       'solve_for_state_action_ratio is False.')

    # Get number of states/actions.
    observation_spec = self._dataset_spec.observation
    action_spec = self._dataset_spec.action
    if not common_lib.is_categorical_spec(observation_spec):
      raise ValueError('Observation spec must be discrete and bounded.')
    self._num_states = observation_spec.maximum + 1

    if not common_lib.is_categorical_spec(action_spec):
      raise ValueError('Action spec must be discrete and bounded.')
    self._num_actions = action_spec.maximum + 1
    self._dimension = 1 + (
        self._num_states * self._num_actions
        if self._solve_for_state_action_ratio else self._num_states)

    # For learning data weight
    self._divergence_limit = tf.convert_to_tensor(
        divergence_limit, dtype=tf.float32)
    if tf.rank(self._divergence_limit) < 1:
      self._divergence_limit = tf.expand_dims(self._divergence_limit, -1)
    self._two_sided_limit = tf.concat(
        [self._divergence_limit, self._divergence_limit], -1)
    self._num_limits = int(self._two_sided_limit.shape[0])
    # The lagrange multiplier w.r.t. data weight constraint
    self._alpha = tf.Variable(
        np.zeros(self._two_sided_limit.shape), dtype=tf.float32)

    self._algae_alpha = tf.convert_to_tensor(algae_alpha, dtype=tf.float32)
    if tf.rank(self._algae_alpha) < 1:
      self._algae_alpha = tf.expand_dims(self._algae_alpha, -1)
    if self._algae_alpha.shape[-1] != self._two_sided_limit.shape[-1]:
      self._algae_alpha *= tf.ones_like(self._two_sided_limit)
    self._algae_alpha_sign = 2 * (
        tf.cast(self._algae_alpha >= 0, tf.float32) - 0.5)

    self._divergence_type = divergence_type
    if self._divergence_type not in ['kl', 'rkl', 'chi2']:
      raise ValueError('Unsupported divergence type %s.' %
                       self._divergence_type)

    self._nu_learning_rate = nu_learning_rate
    self._zeta_learning_rate = zeta_learning_rate

    # We have two variables to counteract the bias introduced by algae_alpha.
    self._nu = tf.zeros([self._dimension, self._num_limits])
    self._nu2 = tf.zeros([self._dimension, self._num_limits])
    self._zeta = tf.zeros([self._dimension, self._num_limits])
    self._zeta2 = tf.zeros([self._dimension, self._num_limits])
    self._limit_episodes = limit_episodes

  def _get_index(self, state, action):
    if self._solve_for_state_action_ratio:
      return state * self._num_actions + action
    else:
      return state

  def _get_state_action_counts(self, env_step):
    if self._solve_for_state_action_ratio:
      index = env_step.observation * self._num_actions + env_step.action
      dim = self._num_states * self._num_actions
    else:
      # only count states
      index = env_step.observation
      dim = self._num_states
    y, _, count = tf.unique_with_counts(index)
    sparse_count = tf.sparse.SparseTensor(
        tf.stack([tf.zeros_like(y), y], axis=1), count, [1, dim])
    count = tf.sparse.to_dense(tf.sparse.reorder(sparse_count), default_value=0)

    count_samples = tf.gather_nd(
        count, tf.stack([tf.zeros_like(index), index], axis=1))

    return count_samples

  def _get_weights(self, nu_loss):
    """Get weights for both upper and lower confidence intervals."""
    if self._divergence_type in ['kl', 'rkl']:
      weight_loss_multiplier = self._algae_alpha_sign * tf.concat([
          tf.ones_like(self._divergence_limit),
          -tf.ones_like(self._divergence_limit)
      ],
                                                                  axis=-1)
      log_weights = tf.reshape(
          -weight_loss_multiplier * nu_loss / tf.exp(self._alpha),
          [-1, self._num_limits])
    elif self._divergence_type == 'chi2':
      weight_loss_multiplier = self._algae_alpha_sign * tf.concat([
          tf.ones_like(self._divergence_limit),
          -tf.ones_like(self._divergence_limit)
      ],
                                                                  axis=-1)
      logits = -weight_loss_multiplier * nu_loss / tf.exp(self._alpha)
      weights = tf.transpose(_compute_2d_sparsemax(tf.transpose(logits)))
      batch_size = tf.cast(tf.shape(weights)[0], tf.float32)
      return (batch_size * weights,
              tf.math.log(batch_size) + tf.math.log(1e-6 + weights))
    else:
      raise ValueError('Divergence is not implemented.')
    batch_size = tf.cast(tf.shape(log_weights)[0], tf.float32)
    return (batch_size * tf.nn.softmax(log_weights, axis=0),
            tf.math.log(batch_size) + tf.nn.log_softmax(log_weights, 0))

  def _compute_divergence(self, weights, log_weights):
    if self._divergence_type == 'kl':
      return tf.reduce_mean(2 * weights * log_weights - 2 * weights + 2, axis=0)
    elif self._divergence_type == 'rkl':
      return tf.reduce_mean(2 * -log_weights + 2 * weights - 2, axis=0)
    elif self._divergence_type == 'chi2':
      return tf.reduce_mean((weights - 1)**2, axis=0)
    else:
      raise ValueError('Divergence is not implemented.')

  def prepare_dataset(self, dataset: dataset_lib.OffpolicyDataset,
                      target_policy: tf_policy.TFPolicy):
    """Performs pre-computations on dataset to make solving easier."""
    episodes, valid_steps = dataset.get_all_episodes(limit=self._limit_episodes)
    total_num_steps_per_episode = tf.shape(valid_steps)[1] - 1
    num_episodes = tf.shape(valid_steps)[0]
    num_samples = num_episodes * total_num_steps_per_episode
    valid_and_not_last = tf.logical_and(valid_steps, episodes.discount > 0)
    valid_indices = tf.squeeze(
        tf.where(tf.reshape(valid_and_not_last[:, :-1], [-1])))

    # Flatten all tensors so that each data sample is a tuple of
    # (initial_env_step, env_step, next_env_step).
    initial_env_step = tf.nest.map_structure(
        lambda t: tf.squeeze(
            tf.reshape(
                tf.repeat(
                    t[:, 0:1, ...], axis=1, repeats=total_num_steps_per_episode
                ), [num_samples, -1])), episodes)
    initial_env_step = tf.nest.map_structure(
        lambda t: tf.gather(t, valid_indices), initial_env_step)
    tfagents_initial_env_step = dataset_lib.convert_to_tfagents_timestep(
        initial_env_step)

    env_step = tf.nest.map_structure(
        lambda t: tf.squeeze(
            tf.reshape(t[:, 0:total_num_steps_per_episode, ...],
                       [num_samples, -1])), episodes)
    env_step = tf.nest.map_structure(lambda t: tf.gather(t, valid_indices),
                                     env_step)
    tfagents_env_step = dataset_lib.convert_to_tfagents_timestep(env_step)

    next_env_step = tf.nest.map_structure(
        lambda t: tf.squeeze(
            tf.reshape(t[:, 1:total_num_steps_per_episode + 1, ...],
                       [num_samples, -1])), episodes)
    next_env_step = tf.nest.map_structure(lambda t: tf.gather(t, valid_indices),
                                          next_env_step)
    tfagents_next_env_step = dataset_lib.convert_to_tfagents_timestep(
        next_env_step)

    # Get target probabilities for initial and next steps.
    initial_target_probs = target_policy.distribution(
        tfagents_initial_env_step).action.probs_parameter()
    next_target_probs = target_policy.distribution(
        tfagents_next_env_step).action.probs_parameter()

    # Map states and actions to indices into tabular representation.
    initial_states = tf.tile(
        tf.reshape(initial_env_step.observation, [-1, 1]),
        [1, self._num_actions])
    initial_actions = tf.tile(
        tf.reshape(tf.range(self._num_actions), [1, -1]),
        [initial_env_step.observation.shape[0], 1])
    initial_nu_indices = self._get_index(initial_states, initial_actions)

    next_states = tf.tile(
        tf.reshape(next_env_step.observation, [-1, 1]), [1, self._num_actions])
    next_actions = tf.tile(
        tf.reshape(tf.range(self._num_actions), [1, -1]),
        [next_env_step.observation.shape[0], 1])
    next_nu_indices = self._get_index(next_states, next_actions)
    next_nu_indices = tf.where(
        tf.expand_dims(next_env_step.is_absorbing(), -1),
        -1 * tf.ones_like(next_nu_indices), next_nu_indices)

    nu_indices = self._get_index(env_step.observation, env_step.action)

    target_log_probabilities = target_policy.distribution(
        tfagents_env_step).action.log_prob(env_step.action)
    if not self._solve_for_state_action_ratio:
      policy_ratio = tf.exp(target_log_probabilities -
                            env_step.get_log_probability())
    else:
      policy_ratio = tf.ones([
          target_log_probabilities.shape[0],
      ])
    policy_ratios = tf.tile(
        tf.reshape(policy_ratio, [-1, 1]), [1, self._num_actions])

    # Bellman residual matrix of size [n_data, n_dim].
    a_vec = tf.one_hot(nu_indices, self._dimension) - tf.reduce_sum(
        self._gamma *
        tf.expand_dims(next_target_probs * policy_ratios, axis=-1) *
        tf.one_hot(next_nu_indices, self._dimension),
        axis=1)

    state_action_count = self._get_state_action_counts(env_step)
    # Bellman residual matrix of size [n_dim, n_dim].
    td_mat = tf.einsum('ai, a, aj -> ij', tf.one_hot(nu_indices,
                                                     self._dimension),
                       1.0 / tf.cast(state_action_count, tf.float32), a_vec)

    # Reward vector of size [n_data].
    weighted_rewards = policy_ratio * self._reward_fn(env_step)

    # Reward vector of size [n_dim].
    bias = tf.reduce_sum(
        tf.one_hot(nu_indices, self._dimension) *
        tf.reshape(weighted_rewards, [-1, 1]) * 1.0 /
        tf.cast(state_action_count, tf.float32)[:, None],
        axis=0)

    # Initialize.
    self._nu = np.ones_like(self._nu) * bias[:, None]
    self._nu2 = np.ones_like(self._nu2) * bias[:, None]

    self._a_vec = a_vec
    self._td_mat = td_mat
    self._bias = bias
    self._weighted_rewards = weighted_rewards
    self._state_action_count = state_action_count
    self._nu_indices = nu_indices
    self._initial_nu_indices = initial_nu_indices
    self._initial_target_probs = initial_target_probs

  def train_step(self,
                 dataset: dataset_lib.OffpolicyDataset,
                 target_policy: tf_policy.TFPolicy,
                 regularizer: float = 1e-6):
    """Performs single iteration of CoinDICE.

    Args:
      dataset: The dataset to sample experience from.
      target_policy: The policy whose value we want to estimate.
      regularizer: A small constant to add to matrices before inverting them or
        to floats before taking square root.

    Returns:
      Estimated average per-step reward of the target policy.
    """
    # First compute Lagrangian loss.
    saddle_bellman_residuals = (
        tf.matmul(self._a_vec, self._nu) - self._weighted_rewards[:, None])
    saddle_bellman_residuals *= -1 * self._algae_alpha_sign
    saddle_zetas = tf.gather(self._zeta, self._nu_indices)
    saddle_initial_nu_values = tf.reduce_sum(  # Average over actions.
        self._initial_target_probs[:, :, None] *
        tf.gather(self._nu, self._initial_nu_indices),
        axis=1)
    saddle_init_nu_loss = ((1 - self._gamma) * saddle_initial_nu_values *
                           self._algae_alpha_sign)

    # This second optimization switches the sign of algae_alpha.
    # We add these two together to get the final loss, and thus counteract
    # the bias introduced by algae_alpha.
    saddle_bellman_residuals2 = (
        tf.matmul(self._a_vec, self._nu2) - self._weighted_rewards[:, None])
    saddle_bellman_residuals2 *= 1 * self._algae_alpha_sign
    saddle_zetas2 = tf.gather(self._zeta2, self._nu_indices)
    saddle_initial_nu_values2 = tf.reduce_sum(  # Average over actions.
        self._initial_target_probs[:, :, None] *
        tf.gather(self._nu2, self._initial_nu_indices),
        axis=1)
    saddle_init_nu_loss2 = ((1 - self._gamma) * saddle_initial_nu_values2 * -1 *
                            self._algae_alpha_sign)

    saddle_loss = 0.5 * (
        saddle_init_nu_loss + saddle_bellman_residuals * saddle_zetas +
        -tf.math.abs(self._algae_alpha) * 0.5 * tf.square(saddle_zetas) +
        -saddle_init_nu_loss2 + -saddle_bellman_residuals2 * saddle_zetas2 +
        tf.math.abs(self._algae_alpha) * 0.5 * tf.square(saddle_zetas2))

    # Find optimal weights by doing binary search on alpha (lambda in the
    # paper).
    left = tf.constant([-8., -8.])
    right = tf.constant([32., 32.])
    for _ in range(16):
      mid = 0.5 * (left + right)
      self._alpha.assign(mid)
      weights, log_weights = self._get_weights(saddle_loss)

      divergence = self._compute_divergence(weights, log_weights)
      divergence_violation = divergence - self._two_sided_limit
      left = tf.where(divergence_violation > 0., mid, left)
      right = tf.where(divergence_violation > 0., right, mid)
    self._alpha.assign(0.5 * (left + right))
    weights, log_weights = self._get_weights(saddle_loss)

    # Now that we have weights, we reconstruct the Bellman residual matrices.
    data_weights = tf.stop_gradient(weights)
    avg_saddle_loss = (
        tf.reduce_sum(data_weights * saddle_loss, axis=0) /
        tf.reduce_sum(data_weights, axis=0))

    weighted_state_action_count = tf.reduce_sum(
        tf.one_hot(self._nu_indices, self._dimension)[:, :, None] *
        weights[:, None, :],
        axis=0)
    weighted_state_action_count = tf.gather(weighted_state_action_count,
                                            self._nu_indices)
    my_td_mat = tf.einsum('ai, ab, ab, aj -> bij',
                          tf.one_hot(self._nu_indices, self._dimension),
                          1.0 / weighted_state_action_count, weights,
                          self._a_vec)
    my_bias = tf.reduce_sum(
        tf.transpose(weights)[:, :, None] *
        tf.one_hot(self._nu_indices, self._dimension)[None, :, :] *
        tf.reshape(self._weighted_rewards, [1, -1, 1]) * 1.0 /
        tf.transpose(weighted_state_action_count)[:, :, None],
        axis=1)

    # Solve for nu using primal form; i.e., E[(nu - B nu)^2] - (1-g) * E[nu0].
    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      tape.watch([self._nu, self._nu2, self._alpha])
      bellman_residuals = tf.matmul(
          my_td_mat,
          tf.transpose(self._nu)[:, :, None]) - my_bias[:, :, None]
      bellman_residuals = tf.transpose(tf.squeeze(bellman_residuals, -1))
      bellman_residuals = tf.gather(bellman_residuals, self._nu_indices)
      initial_nu_values = tf.reduce_sum(  # Average over actions.
          self._initial_target_probs[:, :, None] *
          tf.gather(self._nu, self._initial_nu_indices),
          axis=1)

      bellman_residuals *= self._algae_alpha_sign

      init_nu_loss = ((1 - self._gamma) * initial_nu_values *
                      self._algae_alpha_sign)

      nu_loss = (
          tf.math.square(bellman_residuals) / 2.0 +
          tf.math.abs(self._algae_alpha) * init_nu_loss)

      loss = (
          data_weights * nu_loss /
          tf.reduce_sum(data_weights, axis=0, keepdims=True))

      bellman_residuals2 = tf.matmul(
          my_td_mat,
          tf.transpose(self._nu2)[:, :, None]) - my_bias[:, :, None]
      bellman_residuals2 = tf.transpose(tf.squeeze(bellman_residuals2, -1))
      bellman_residuals2 = tf.gather(bellman_residuals2, self._nu_indices)
      initial_nu_values2 = tf.reduce_sum(  # Average over actions.
          self._initial_target_probs[:, :, None] *
          tf.gather(self._nu2, self._initial_nu_indices),
          axis=1)

      bellman_residuals2 *= -1 * self._algae_alpha_sign

      init_nu_loss2 = ((1 - self._gamma) * initial_nu_values2 * -1 *
                       self._algae_alpha_sign)

      nu_loss2 = (
          tf.math.square(bellman_residuals2) / 2.0 +
          tf.math.abs(self._algae_alpha) * init_nu_loss2)

      loss2 = (
          data_weights * nu_loss2 /
          tf.reduce_sum(data_weights, axis=0, keepdims=True))

      divergence = self._compute_divergence(weights, log_weights)
      divergence_violation = divergence - self._two_sided_limit

      # Extra loss if for the 'terminal' state (index = -1).
      extra_loss = tf.reduce_sum(tf.math.square(self._nu[-1, :]))
      extra_loss2 = tf.reduce_sum(tf.math.square(self._nu2[-1, :]))

      nu_grad = tape.gradient(loss + extra_loss, [self._nu])[0]
      nu_grad2 = tape.gradient(loss2 + extra_loss2, [self._nu2])[0]

    avg_loss = tf.reduce_sum(
        0.5 * (loss - loss2) / tf.math.abs(self._algae_alpha), axis=0)
    nu_jacob = tape.jacobian(nu_grad, [self._nu])[0]
    nu_hess = tf.stack([nu_jacob[:, i, :, i] for i in range(self._num_limits)],
                       axis=0)

    nu_jacob2 = tape.jacobian(nu_grad2, [self._nu2])[0]
    nu_hess2 = tf.stack(
        [nu_jacob2[:, i, :, i] for i in range(self._num_limits)], axis=0)

    for idx, div in enumerate(divergence):
      tf.summary.scalar('divergence%d' % idx, div)

    # Perform Newton step on nu.
    nu_transformed = tf.transpose(
        tf.squeeze(
            tf.linalg.solve(nu_hess + regularizer * tf.eye(self._dimension),
                            tf.expand_dims(-tf.transpose(nu_grad), axis=-1))))
    self._nu = self._nu + self._nu_learning_rate * nu_transformed
    nu_transformed2 = tf.transpose(
        tf.squeeze(
            tf.linalg.solve(nu_hess2 + regularizer * tf.eye(self._dimension),
                            tf.expand_dims(-tf.transpose(nu_grad2), axis=-1))))
    self._nu2 = self._nu2 + self._nu_learning_rate * nu_transformed2

    # Perform step on zeta based on fact that zeta* = (nu* - bellman nu*)/a.
    zetas = tf.matmul(my_td_mat,
                      tf.transpose(self._nu)[:, :, None]) - my_bias[:, :, None]
    zetas = tf.transpose(tf.squeeze(zetas, -1))
    zetas *= -self._algae_alpha_sign
    zetas /= tf.math.abs(self._algae_alpha)
    self._zeta = self._zeta + self._zeta_learning_rate * (zetas - self._zeta)

    zetas2 = tf.matmul(my_td_mat,
                       tf.transpose(self._nu2)[:, :, None]) - my_bias[:, :,
                                                                      None]
    zetas2 = tf.transpose(tf.squeeze(zetas2, -1))
    zetas2 *= 1 * self._algae_alpha_sign
    zetas2 /= tf.math.abs(self._algae_alpha)
    self._zeta2 = (
        self._zeta2 + self._zeta_learning_rate * (zetas2 - self._zeta2))

    return [
        avg_saddle_loss * self._algae_alpha_sign,
        avg_loss * self._algae_alpha_sign, divergence
    ]


def _compute_2d_sparsemax(logits):
  """Performs the sparsemax operation when axis=-1."""
  shape_op = tf.shape(logits)
  obs = tf.math.reduce_prod(shape_op[:-1])
  dims = shape_op[-1]

  # In the paper, they call the logits z.
  # The mean(logits) can be subtracted from logits to make the algorithm
  # more numerically stable. the instability in this algorithm comes mostly
  # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
  # to zero. However, in practise the numerical instability issues are very
  # minor and substacting the mean causes extra issues with inf and nan
  # input.
  # Reshape to [obs, dims] as it is almost free and means the remanining
  # code doesn't need to worry about the rank.
  z = tf.reshape(logits, [obs, dims])

  # sort z
  z_sorted, _ = tf.nn.top_k(z, k=dims)

  # calculate k(z)
  z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
  k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
  z_check = 1 + k * z_sorted > z_cumsum
  # because the z_check vector is always [1,1,...1,0,0,...0] finding the
  # (index + 1) of the last `1` is the same as just summing the number of 1.
  k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)

  # calculate tau(z)
  # If there are inf values or all values are -inf, the k_z will be zero,
  # this is mathematically invalid and will also cause the gather_nd to fail.
  # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
  # fixed later (see p_safe) by returning p = nan. This results in the same
  # behavior as softmax.
  k_z_safe = tf.math.maximum(k_z, 1)
  indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
  tau_sum = tf.gather_nd(z_cumsum, indices)
  tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

  # calculate p
  p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
  # If k_z = 0 or if z = nan, then the input is invalid
  p_safe = tf.where(
      tf.expand_dims(
          tf.math.logical_or(
              tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
          axis=-1), tf.fill([obs, dims], tf.cast(float('nan'), logits.dtype)),
      p)

  # Reshape back to original size
  p_safe = tf.reshape(p_safe, shape_op)
  return p_safe
