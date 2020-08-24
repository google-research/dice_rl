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
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import dice_rl.data.dataset as dataset_lib
import dice_rl.estimators.estimator as estimator_lib


def _is_categorical_spec(spec):
  return (tensor_spec.is_discrete(spec) and tensor_spec.is_bounded(spec) and
          spec.shape == [] and spec.minimum == 0)


class TabularQLearning(object):
  """Approximate the density ratio using exact matrix solves."""

  def __init__(self,
               dataset_spec,
               gamma: Union[float, tf.Tensor],
               reward_fn: Callable = None,
               solve_for_state_action_value: bool = True,
               num_qvalues: Optional[int] = None,
               bootstrap: bool = True,
               perturbation_scale: Union[float, tf.Tensor] = 1.0,
               default_reward_value: Union[float, tf.Tensor] = 0.0,
               limit_episodes: Optional[int] = None):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      gamma: The discount factor to use.
      reward_fn: A function that takes in an EnvStep and returns the reward for
        that step. If not specified, defaults to just EnvStep.reward.
      solve_for_state_action_value: Whether to solve for Q-values (default) or
        V-values, i.e., state-values.
      num_qvalues: If specified, maintains an ensemble of Q-values for
        confidence bound estimation.
      bootstrap: Whether to bootstrap the dataset.
      perturbation_scale: Scale of reward perturbation.
      default_reward_value: Value to use for reward of unseen state-actions.
      limit_episodes: How many episodes to take from the dataset. Defaults to
        None (take all episodes).
    """
    self._dataset_spec = dataset_spec
    self._gamma = gamma
    if reward_fn is None:
      reward_fn = lambda env_step: env_step.reward
    self._reward_fn = reward_fn
    self._num_qvalues = num_qvalues
    self._bootstrap = bootstrap
    self._perturbation_scale = np.array(perturbation_scale)
    if len(np.shape(self._perturbation_scale)) < 1:
      self._perturbation_scale = np.reshape(self._perturbation_scale, [-1])
    self._num_perturbations = len(self._perturbation_scale)
    self._default_reward_value = default_reward_value
    self._limit_episodes = limit_episodes

    self._solve_for_state_action_value = solve_for_state_action_value
    if (not self._solve_for_state_action_value and
        not self._dataset_spec.has_log_probability()):
      raise ValueError('Dataset must contain log-probability when '
                       'solve_for_state_action_value is False.')

    # Get number of states/actions.
    observation_spec = self._dataset_spec.observation
    action_spec = self._dataset_spec.action
    if not _is_categorical_spec(observation_spec):
      raise ValueError('Observation spec must be discrete and bounded.')
    self._num_states = observation_spec.maximum + 1

    if not _is_categorical_spec(action_spec):
      raise ValueError('Action spec must be discrete and bounded.')
    self._num_actions = action_spec.maximum + 1
    self._dimension = (
        self._num_states * self._num_actions
        if self._solve_for_state_action_value else self._num_states)
    self._dimension += 1  # Add 1 for terminal absorbing state.
    self._point_qvalues = np.zeros([self._dimension])
    if self._num_qvalues is not None:
      self._ensemble_qvalues = np.zeros([self._num_qvalues, self._dimension])

  def _get_index(self, state, action):
    if self._solve_for_state_action_value:
      return state * self._num_actions + action
    else:
      return state

  def solve(self,
            dataset: dataset_lib.OffpolicyDataset,
            target_policy: tf_policy.TFPolicy,
            regularizer: float = 1e-8):
    """Solves for Q-values and then approximates target policy value.

    Args:
      dataset: The dataset to sample experience from.
      target_policy: The policy whose value we want to estimate.
      regularizer: A small constant to add before dividing.

    Returns:
      Estimated average per-step reward of the target policy.
    """
    num_estimates = 1 + int(self._num_qvalues)
    transition_matrix = np.zeros(
        [self._dimension, self._dimension, num_estimates])
    reward_vector = np.zeros(
        [self._dimension, num_estimates, self._num_perturbations])
    total_weights = np.zeros([self._dimension, num_estimates])

    episodes, valid_steps = dataset.get_all_episodes(limit=self._limit_episodes)
    #all_rewards = self._reward_fn(episodes)
    #reward_std = np.ma.MaskedArray(all_rewards, valid_steps).std()
    tfagents_episodes = dataset_lib.convert_to_tfagents_timestep(episodes)

    sample_weights = np.array(valid_steps, dtype=np.int64)
    if not self._bootstrap or self._num_qvalues is None:
      sample_weights = (
          sample_weights[:, :, None] * np.ones([1, 1, num_estimates]))
    else:
      probs = np.reshape(sample_weights, [-1]) / np.sum(sample_weights)
      weights = np.random.multinomial(
          np.sum(sample_weights), probs,
          size=self._num_qvalues).astype(np.float32)
      weights = np.reshape(
          np.transpose(weights),
          list(np.shape(sample_weights)) + [self._num_qvalues])
      sample_weights = np.concatenate([sample_weights[:, :, None], weights],
                                      axis=-1)

    for episode_num in range(tf.shape(valid_steps)[0]):
      # Precompute probabilites for this episode.
      this_episode = tf.nest.map_structure(lambda t: t[episode_num], episodes)
      this_tfagents_episode = dataset_lib.convert_to_tfagents_timestep(
          this_episode)
      episode_target_log_probabilities = target_policy.distribution(
          this_tfagents_episode).action.log_prob(this_episode.action)
      episode_target_probs = target_policy.distribution(
          this_tfagents_episode).action.probs_parameter()

      for step_num in range(tf.shape(valid_steps)[1] - 1):
        this_step = tf.nest.map_structure(lambda t: t[episode_num, step_num],
                                          episodes)
        next_step = tf.nest.map_structure(
            lambda t: t[episode_num, step_num + 1], episodes)
        this_tfagents_step = dataset_lib.convert_to_tfagents_timestep(this_step)
        next_tfagents_step = dataset_lib.convert_to_tfagents_timestep(next_step)
        this_weights = sample_weights[episode_num, step_num, :]
        if this_step.is_last() or not valid_steps[episode_num, step_num]:
          continue

        weight = this_weights
        this_index = self._get_index(this_step.observation, this_step.action)

        reward_vector[this_index, :, :] += np.expand_dims(
            self._reward_fn(this_step) * weight, -1)
        if self._num_qvalues is not None:
          random_noise = np.random.binomial(this_weights[1:].astype('int64'),
                                            0.5)
          reward_vector[this_index, 1:, :] += (
              self._perturbation_scale[None, :] *
              (2 * random_noise - this_weights[1:])[:, None])

        total_weights[this_index] += weight

        policy_ratio = 1.0
        if not self._solve_for_state_action_value:
          policy_ratio = tf.exp(episode_target_log_probabilities[step_num] -
                                this_step.get_log_probability())

        # Need to weight next nu by importance weight.
        next_weight = (
            weight if self._solve_for_state_action_value else policy_ratio *
            weight)
        if next_step.is_absorbing():
          next_index = -1  # Absorbing state.
          transition_matrix[this_index, next_index] += next_weight
        else:
          next_probs = episode_target_probs[step_num + 1]
          for next_action, next_prob in enumerate(next_probs):
            next_index = self._get_index(next_step.observation, next_action)
            transition_matrix[this_index, next_index] += next_prob * next_weight
    print('Done processing data.')

    transition_matrix /= (regularizer + total_weights)[:, None, :]
    reward_vector /= (regularizer + total_weights)[:, :, None]
    reward_vector[np.where(np.equal(total_weights,
                                    0.0))] = self._default_reward_value
    reward_vector[-1, :, :] = 0.0  # Terminal absorbing state has 0 reward.

    self._point_qvalues = np.linalg.solve(
        np.eye(self._dimension) - self._gamma * transition_matrix[:, :, 0],
        reward_vector[:, 0])
    if self._num_qvalues is not None:
      self._ensemble_qvalues = np.linalg.solve(
          (np.eye(self._dimension) -
           self._gamma * np.transpose(transition_matrix, [2, 0, 1])),
          np.transpose(reward_vector, [1, 0, 2]))

    return self.estimate_average_reward(dataset, target_policy)

  def estimate_average_reward(self, dataset: dataset_lib.OffpolicyDataset,
                              target_policy: tf_policy.TFPolicy):
    """Estimates value (average per-step reward) of policy.

    Args:
      dataset: The dataset to sample experience from.
      target_policy: The policy whose value we want to estimate.

    Returns:
      Estimated average per-step reward of the target policy.
    """

    def reward_fn(env_step, valid_steps, qvalues=self._point_qvalues):
      """Computes average initial Q-values of episodes."""
      # env_step is an episode, and we just want the first step.
      if tf.rank(valid_steps) == 1:
        first_step = tf.nest.map_structure(lambda t: t[0, ...], env_step)
      else:
        first_step = tf.nest.map_structure(lambda t: t[:, 0, ...], env_step)

      if self._solve_for_state_action_value:
        indices = self._get_index(first_step.observation[:, None],
                                  np.arange(self._num_actions)[None, :])
        initial_qvalues = tf.cast(tf.gather(qvalues, indices), tf.float32)

        tfagents_first_step = dataset_lib.convert_to_tfagents_timestep(
            first_step)
        initial_target_probs = target_policy.distribution(
            tfagents_first_step).action.probs_parameter()
        value = tf.reduce_sum(initial_qvalues * initial_target_probs, axis=-1)
      else:
        indices = self._get_index(first_step.observation, first_step.action)
        value = tf.cast(tf.gather(qvalues, indices), tf.float32)

      return value

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
        estimates.append([])
        for j in range(self._num_perturbations):
          estimates[-1].append(
              (1 - self._gamma) * estimator_lib.get_fullbatch_average(
                  dataset,
                  limit=None,
                  by_steps=False,
                  truncate_episode_at=1,
                  reward_fn=lambda *args: reward_fn(
                      *args, qvalues=self._ensemble_qvalues[i, :, j]),
                  weight_fn=weight_fn))
      return np.array(estimates)
