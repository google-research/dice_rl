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
from tf_agents.utils import common as tfagents_common
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import dice_rl.data.dataset as dataset_lib
import dice_rl.estimators.estimator as estimator_lib


def _is_categorical_spec(spec):
  return (tensor_spec.is_discrete(spec) and tensor_spec.is_bounded(spec) and
          spec.shape == [] and spec.minimum == 0)


class NeuralQLearning(object):
  """Policy evaluation with Q-learning."""

  def __init__(self,
               dataset_spec,
               value_network,
               optimizer,
               gamma: Union[float, tf.Tensor],
               reward_fn: Optional[Callable] = None,
               solve_for_state_action_value: bool = True,
               num_qvalues: Optional[int] = None,
               target_update_tau: Union[float, tf.Tensor] = 0.02,
               target_update_period: int = 1,
               num_samples: Optional[int] = None):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      value_network: A function that returns the values for each observation and
        action. If num_qvalues is not None, should return a vector of values.
      optimizer: TF optimizer to use.
      gamma: The discount factor to use.
      reward_fn: A function that takes in an EnvStep and returns the reward for
        that step. If not specified, defaults to just EnvStep.reward.
      solve_for_state_action_value: Whether to solve for Q-values (default) or
        V-values, i.e., state-values.
      num_qvalues: If specified, maintains an ensemble of Q-values for
        confidence bound estimation.
      num_samples: Number of samples to take from policy to estimate average
        next state value. If actions are discrete, this defaults to computing
        average explicitly. If actions are not discrete, this defaults to using
        a single sample.
    """
    self._dataset_spec = dataset_spec
    self._value_network = value_network
    self._value_network.create_variables()
    self._target_network = self._value_network.copy(name='TargetValueNetwork')
    self._target_network.create_variables()
    self._target_update_tau = target_update_tau
    self._target_update_period = target_update_period
    self._update_targets = self._get_target_updater(
        tau=self._target_update_tau, period=self._target_update_period)

    self._optimizer = optimizer

    self._gamma = gamma
    if reward_fn is None:
      reward_fn = lambda env_step: env_step.reward
    self._reward_fn = reward_fn
    self._num_qvalues = num_qvalues
    self._num_samples = num_samples

    self._solve_for_state_action_value = solve_for_state_action_value
    if (not self._solve_for_state_action_value and
        not self._dataset_spec.has_log_probability()):
      raise ValueError('Dataset must contain log-probability when '
                       'solve_for_state_action_value is False.')

    self._categorical_action = _is_categorical_spec(self._dataset_spec.action)
    if not self._categorical_action and self._num_samples is None:
      self._num_samples = 1
    self._initialize()

  def _initialize(self):
    tfagents_common.soft_variables_update(
        self._value_network.variables, self._target_network.variables, tau=1.0)

  def _get_target_updater(self, tau=1.0, period=1):

    def update():
      return tfagents_common.soft_variables_update(
          self._value_network.variables,
          self._target_network.variables,
          tau,
          tau_non_trainable=1.0)

    return tfagents_common.Periodically(update, period, 'update_targets')

  def _get_value(self, env_step):
    if self._solve_for_state_action_value:
      return self._value_network((env_step.observation, env_step.action))[0]
    else:
      return self._value_network(env_step.observation)[0]

  def _get_average_value(self, network, env_step, policy):
    if self._solve_for_state_action_value:
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

      flat_actions = tf.reshape(actions,
                                [batch_size * num_actions] + actions.shape[2:])
      flat_observations = tf.reshape(
          tf.tile(env_step.observation[:, None, ...],
                  [1, num_actions] + [1] * len(env_step.observation.shape[1:])),
          [batch_size * num_actions] + env_step.observation.shape[1:])
      flat_values, _ = network((flat_observations, flat_actions))

      values = tf.reshape(flat_values,
                          [batch_size, num_actions] + flat_values.shape[1:])
      if self._num_qvalues is not None:
        action_weights = action_weights[..., None]
      return tf.reduce_sum(values * action_weights, axis=1)
    else:
      return network(env_step.observation)[0]

  def _get_target_value(self, env_step, policy):
    return self._get_average_value(self._target_network, env_step, policy)

  def train_loss(self, env_step, rewards, next_env_step, policy, gamma):
    values = self._get_value(env_step)
    discounts = gamma * next_env_step.discount
    target_values = self._get_target_value(next_env_step, policy)
    #target_values = tf.reduce_min(target_values, axis=-1, keepdims=True)

    if self._num_qvalues is not None and tf.rank(discounts) == 1:
      discounts = discounts[:, None]
    td_targets = rewards + discounts * tf.stop_gradient(target_values)

    policy_ratio = 1.0
    if not self._solve_for_state_action_value:
      tfagents_step = dataset_lib.convert_to_tfagents_timestep(env_step)
      policy_log_probabilities = policy.distribution(
          tfagents_step).action.log_prob(env_step.action)
      policy_ratio = tf.exp(policy_log_probabilities -
                            env_step.get_log_probability())

    td_errors = policy_ratio * td_targets - values
    return tf.square(td_errors)

  def train_step(self, experience: dataset_lib.EnvStep,
                 target_policy: tf_policy.TFPolicy):
    """Performs a single training step based on experience batch.

    Args:
      experience: A batch of experience. Members should have shape [batch_size,
        time_length, ...].
      target_policy: The policy whose value we want to estimate.

    Returns:
      A train op.
    """
    first_env_step = tf.nest.map_structure(lambda t: t[:, 0, ...], experience)

    is_last = tf.cast(experience.is_last(), tf.float32)
    batch_size = tf.shape(is_last)[0]
    time_length = tf.shape(is_last)[1]
    batch_range = tf.range(batch_size, dtype=tf.int64)
    last_indices = tf.where(
        tf.equal(tf.reduce_max(is_last, axis=-1), 0.),
        tf.cast(time_length - 1, tf.int64) *
        tf.ones([batch_size], dtype=tf.int64), tf.argmax(is_last, axis=-1))
    last_env_step = tf.nest.map_structure(
        lambda t: tf.gather_nd(t, tf.stack([batch_range, last_indices], -1)),
        experience)

    rewards = self._reward_fn(experience)[:, :-1]
    if self._num_qvalues is not None and tf.rank(rewards) == 2:
      rewards = rewards[:, :, None]

    # Mask out rewards after episode end.
    mask = (
        tf.range(time_length - 1, dtype=tf.int64)[None, :] < last_indices[:,
                                                                          None])
    if self._num_qvalues is not None:
      mask = mask[:, :, None]
    rewards *= tf.cast(mask, tf.float32)

    # Sum up trajectory rewards.
    discounts = tf.pow(self._gamma, tf.range(time_length - 1, dtype=tf.float32))
    if self._num_qvalues is None:
      discounts = discounts[None, :]
    else:
      discounts = discounts[None, :, None]
    sum_discounted_rewards = tf.reduce_sum(rewards * discounts, 1)

    # Discount to be applied on last env step.
    last_discounts = tf.pow(self._gamma, tf.cast(time_length - 1, tf.float32))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self._value_network.variables)
      loss = self.train_loss(first_env_step, sum_discounted_rewards,
                             last_env_step, target_policy, last_discounts)

    grads = tape.gradient(loss, self._value_network.variables)
    grad_op = self._optimizer.apply_gradients(
        zip(grads, self._value_network.variables))
    update_op = self._update_targets()
    return tf.reduce_mean(loss), tf.group(grad_op, update_op)

  def estimate_average_reward(self, dataset: dataset_lib.OffpolicyDataset,
                              target_policy: tf_policy.TFPolicy):
    """Estimates value (average per-step reward) of policy.

    Args:
      dataset: The dataset to sample experience from.
      target_policy: The policy whose value we want to estimate.

    Returns:
      Estimated average per-step reward of the target policy.
    """

    def reward_fn(env_step, valid_steps, value_index=None):
      """Computes average initial Q-values of episodes."""
      # env_step is an episode, and we just want the first step.
      if tf.rank(valid_steps) == 1:
        first_step = tf.nest.map_structure(lambda t: t[0, ...], env_step)
      else:
        first_step = tf.nest.map_structure(lambda t: t[:, 0, ...], env_step)

      value = self._get_average_value(self._value_network, first_step,
                                      target_policy)
      if value_index is None:
        return value
      return value[..., value_index]

    def weight_fn(env_step, valid_steps):
      return tf.ones([tf.shape(valid_steps)[0]], dtype=tf.float32)

    if self._num_qvalues is None:
      return (1 - self._gamma) * estimator_lib.get_fullbatch_average(
          dataset,
          limit=None,
          by_steps=False,
          truncate_episode_at=1,
          reward_fn=reward_fn,
          weight_fn=weight_fn)
    else:
      estimates = []
      for i in range(self._num_qvalues):
        estimates.append(
            (1 - self._gamma) * estimator_lib.get_fullbatch_average(
                dataset,
                limit=None,
                by_steps=False,
                truncate_episode_at=1,
                reward_fn=lambda *args: reward_fn(*args, value_index=i),
                weight_fn=weight_fn))
      return np.array(estimates)
