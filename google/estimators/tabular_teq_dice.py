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
import scipy.linalg
import tensorflow.compat.v2 as tf
from tf_agents.specs import tensor_spec
from tf_agents.policies import tf_policy
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import dice_rl.data.dataset as dataset_lib
import dice_rl.estimators.estimator as estimator_lib


def _is_categorical_spec(spec):
  return (tensor_spec.is_discrete(spec) and tensor_spec.is_bounded(spec) and
          spec.shape == [] and spec.minimum == 0)


class TabularTeQDice(object):
  """Approximate the density ratio using exact matrix solves."""

  def __init__(self,
               dataset_spec,
               gamma: Union[float, tf.Tensor],
               max_trajectory_length: int,
               step_encoding: str = None,
               reward_fn: Callable = None,
               solve_for_state_action_ratio: bool = True):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      gamma: The discount factor to use.
      step_encoding: A string that describes the type of step encoding.
      max_trajectory_length: Evaluate up to this trajectory length.
      reward_fn: A function that takes in an EnvStep and returns the reward for
        that step. If not specified, defaults to just EnvStep.reward.
      solve_for_state_action_ratio: Whether to solve for state-action density
        ratio. Defaults to False, which instead solves for state density ratio.
        Although the estimated policy value should be the same, approximating
        using the state density ratio is much faster (especially in large
        environments) and more accurate (especially in low-data regimes).
    """
    self._dataset_spec = dataset_spec
    self._gamma = gamma
    self._step_encoding = step_encoding
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
    step_num_spec = self._dataset_spec.step_num
    if not _is_categorical_spec(observation_spec):
      raise ValueError('Observation spec must be discrete and bounded.')
    self._num_states = observation_spec.maximum + 1

    if not _is_categorical_spec(action_spec):
      raise ValueError('Action spec must be discrete and bounded.')
    self._num_actions = action_spec.maximum + 1

    if not _is_categorical_spec(step_num_spec):
      raise ValueError('Step number spec must be discrete and bounded.')
    self._num_steps = step_num_spec.maximum
    self._max_trajectory_length = max_trajectory_length

    self._dimension = (
        self._num_states * self._num_actions
        if self._solve_for_state_action_ratio else self._num_states)

    if self._step_encoding is not None:
      emb_blocks = np.repeat(self._create_step_emb(), self._dimension, axis=0)
      self._step_emb = scipy.linalg.block_diag(*emb_blocks)
      self._dimension *= self._num_steps

    self._nu = np.zeros([self._dimension])
    self._zeta = np.zeros([self._dimension])

  def _get_index(self, state, action, step_num):
    if self._solve_for_state_action_ratio:
      index = state * self._num_actions + action
    else:
      index = state
    if self._step_encoding is not None:
      index = index * self._num_steps + step_num
    return index

  def _create_step_emb(self):
    if self._step_encoding == 'one_hot':
      return np.eye(self._num_steps)[None, ...]

    d_step_emb = 5

    def get_angles(pos, i):
      angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_step_emb))
      return pos * angle_rates

    angle_rads = get_angles(0 + np.arange(self._num_steps)[:, np.newaxis],
                            0 + np.arange(d_step_emb)[np.newaxis, :])
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads[np.newaxis, ...]

  def solve(self,
            dataset: dataset_lib.OffpolicyDataset,
            target_policy: tf_policy.TFPolicy,
            regularizer: float = 1e-8):
    """Solves for density ratios and then approximates target policy value.

    Args:
      dataset: The dataset to sample experience from.
      target_policy: The policy whose value we want to estimate.
      regularizer: A small constant to add to matrices before inverting them or
        to floats before taking square root.

    Returns:
      Estimated average per-step reward of the target policy.
    """
    td_residuals = np.zeros([self._dimension, self._dimension])
    total_weights = np.zeros([self._dimension])
    initial_weights = np.zeros([self._dimension])

    episodes, valid_steps = dataset.get_all_episodes(limit=None)
    tfagents_episodes = dataset_lib.convert_to_tfagents_timestep(episodes)

    for episode_num in range(tf.shape(valid_steps)[0]):
      # Precompute probabilites for this episode.
      this_episode = tf.nest.map_structure(lambda t: t[episode_num], episodes)
      first_step = tf.nest.map_structure(lambda t: t[0], this_episode)
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
        if this_step.is_last() or not valid_steps[episode_num, step_num]:
          continue

        weight = 1.0
        step_num_range = np.arange(self._num_steps - 1)
        nu_index = self._get_index(this_step.observation, this_step.action,
                                   step_num_range).numpy()
        td_residuals[nu_index, nu_index] += weight
        np.add.at(total_weights, nu_index, weight)

        policy_ratio = 1.0
        if not self._solve_for_state_action_ratio:
          policy_ratio = tf.exp(episode_target_log_probabilities[step_num] -
                                this_step.get_log_probability())

        # Need to weight next nu by importance weight.
        next_weight = (
            weight if self._solve_for_state_action_ratio else policy_ratio *
            weight)
        next_probs = episode_target_probs[step_num + 1]
        for next_action, next_prob in enumerate(next_probs):
          next_nu_index = self._get_index(next_step.observation, next_action,
                                          step_num_range + 1).numpy()
          td_residuals[next_nu_index,
                       nu_index] += (-next_prob * self._gamma * next_weight)

        initial_probs = episode_target_probs[0]
        for initial_action, initial_prob in enumerate(initial_probs):
          initial_nu_index = self._get_index(first_step.observation,
                                             initial_action, 0)
          initial_weights[initial_nu_index] += weight * initial_prob * len(
              step_num_range)

    td_residuals /= np.sqrt(regularizer + total_weights)[None, :]
    if self._step_encoding is not None:
      td_residuals = np.dot(self._step_emb.T, td_residuals)
      initial_weights = np.dot(self._step_emb.T, initial_weights)
    td_errors = np.dot(td_residuals, td_residuals.T)
    self._nu = np.linalg.solve(
        td_errors + regularizer * np.eye(td_errors.shape[0]),
        (1 - self._gamma) * initial_weights)
    self._zeta = np.dot(self._nu,
                        td_residuals) / np.sqrt(regularizer + total_weights)
    return self.estimate_average_reward(dataset, target_policy)

  def estimate_average_reward(self, dataset: dataset_lib.OffpolicyDataset,
                              target_policy: tf_policy.TFPolicy):
    """Estimates value (average per-step reward) of policy.

    The estimation is based on solved values of zeta, so one should call
    solve() before calling this function.

    Args:
      dataset: The dataset to sample experience from.
      target_policy: The policy whose value we want to estimate.

    Returns:
      Estimated average per-step reward of the target policy.
    """

    def weight_fn(env_step):
      if self._step_encoding is not None:
        zeta = 0.
        for step_num in range(self._max_trajectory_length):
          index = self._get_index(env_step.observation, env_step.action,
                                  step_num)
          zeta += self._gamma**step_num * self._zeta[index]
        zeta *= (1 - self._gamma) / (1 - self._gamma**(self._num_steps - 1))
      else:
        index = self._get_index(env_step.observation, env_step.action,
                                env_step.step_num)
        zeta = self._zeta[index]
        zeta = tf.where(env_step.step_num >= self._max_trajectory_length,
                        tf.zeros_like(zeta), zeta)

      policy_ratio = 1.0
      if not self._solve_for_state_action_ratio:
        tfagents_timestep = dataset_lib.convert_to_tfagents_timestep(env_step)
        target_log_probabilities = target_policy.distribution(
            tfagents_timestep).action.log_prob(env_step.action)
        policy_ratio = tf.exp(target_log_probabilities -
                              env_step.get_log_probability())

      return tf.cast(
          zeta, dtype=tf.float32) * tf.cast(
              policy_ratio, dtype=tf.float32)

    return estimator_lib.get_fullbatch_average(
        dataset,
        limit=None,
        by_steps=True,
        reward_fn=self._reward_fn,
        weight_fn=weight_fn)
