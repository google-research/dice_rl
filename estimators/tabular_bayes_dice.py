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


class TabularBayesDice(object):
  """Robust policy evaluation."""

  def __init__(self,
               dataset_spec,
               gamma: Union[float, tf.Tensor],
               reward_fn: Callable = None,
               solve_for_state_action_ratio: bool = True,
               nu_learning_rate: Union[float, tf.Tensor] = 0.1,
               zeta_learning_rate: Union[float, tf.Tensor] = 0.1,
               kl_regularizer: Union[float, tf.Tensor] = 1.,
               eps_std: Union[float, tf.Tensor] = 1):
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
      nu_learning_rate: Learning rate for nu.
      zeta_learning_rate: Learning rate for zeta.
      kl_regularizer: Regularization constant for D_kl(q || p).
      eps_std: epsilon standard deviation for sampling from the posterior.
    """
    self._dataset_spec = dataset_spec
    self._gamma = gamma
    if reward_fn is None:
      reward_fn = lambda env_step: env_step.reward
    self._reward_fn = reward_fn
    self._kl_regularizer = kl_regularizer
    self._eps_std = eps_std

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
    self._dimension = (
        self._num_states * self._num_actions
        if self._solve_for_state_action_ratio else self._num_states)

    self._td_residuals = np.zeros([self._dimension, self._dimension])
    self._total_weights = np.zeros([self._dimension])
    self._initial_weights = np.zeros([self._dimension])

    self._nu_optimizer = tf.keras.optimizers.Adam(nu_learning_rate)
    self._zeta_optimizer = tf.keras.optimizers.Adam(zeta_learning_rate)

    # Initialize variational Bayes parameters
    self._nu_mu = tf.Variable(tf.zeros([self._dimension]))
    self._nu_log_sigma = tf.Variable(tf.zeros([self._dimension]))
    self._prior_mu = tf.Variable(tf.zeros([self._dimension]), trainable=True)
    self._prior_log_sigma = tf.Variable(
        tf.zeros([self._dimension]), trainable=False)

  def _get_index(self, state, action):
    if self._solve_for_state_action_ratio:
      return state * self._num_actions + action
    else:
      return state

  def prepare_dataset(self, dataset: dataset_lib.OffpolicyDataset,
                      target_policy: tf_policy.TFPolicy):
    episodes, valid_steps = dataset.get_all_episodes()
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
        if this_step.is_last() or not valid_steps[episode_num, step_num]:
          continue

        weight = 1.0
        nu_index = self._get_index(this_step.observation, this_step.action)
        self._td_residuals[nu_index, nu_index] += -weight
        self._total_weights[nu_index] += weight

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
          next_nu_index = self._get_index(next_step.observation, next_action)
          self._td_residuals[next_nu_index, nu_index] += (
              next_prob * self._gamma * next_weight)

        initial_probs = episode_target_probs[0]
        for initial_action, initial_prob in enumerate(initial_probs):
          initial_nu_index = self._get_index(first_step.observation,
                                             initial_action)
          self._initial_weights[initial_nu_index] += weight * initial_prob

    self._initial_weights = tf.cast(self._initial_weights, tf.float32)
    self._total_weights = tf.cast(self._total_weights, tf.float32)
    self._td_residuals = self._td_residuals / np.sqrt(
        1e-8 + self._total_weights)[None, :]
    self._td_errors = tf.cast(
        np.dot(self._td_residuals, self._td_residuals.T), tf.float32)
    self._td_residuals = tf.cast(self._td_residuals, tf.float32)

  @tf.function
  def train_step(self, regularizer: float = 1e-6):
    # Solve primal form min (1-g) * E[nu0] + E[(B nu - nu)^2].
    with tf.GradientTape() as tape:
      nu_sigma = tf.sqrt(tf.exp(self._nu_log_sigma))
      eps = tf.random.normal(tf.shape(nu_sigma), 0, self._eps_std)
      nu = self._nu_mu + nu_sigma * eps
      init_nu_loss = tf.einsum('m,m', (1 - self._gamma) * self._initial_weights,
                               nu)
      residuals = tf.einsum('n,nm->m', nu, self._td_residuals)
      bellman_loss = 0.5 * tf.einsum('m,m', residuals, residuals)

      prior_sigma = tf.sqrt(tf.exp(self._prior_log_sigma))
      prior_var = tf.square(prior_sigma)
      prior_var = 1.
      neg_kl = (0.5 * (1. - 2. * tf.math.log(prior_sigma / nu_sigma + 1e-8) -
                       (self._nu_mu - self._prior_mu)**2 / prior_var -
                       nu_sigma**2 / prior_var))
      loss = init_nu_loss + bellman_loss - self._kl_regularizer * neg_kl

    grads = tape.gradient(loss, [
        self._nu_mu, self._nu_log_sigma, self._prior_mu, self._prior_log_sigma
    ])
    self._nu_optimizer.apply_gradients(
        zip(grads, [
            self._nu_mu, self._nu_log_sigma, self._prior_mu,
            self._prior_log_sigma
        ]))
    return loss

  def estimate_average_reward(self,
                              dataset: dataset_lib.OffpolicyDataset,
                              target_policy: tf_policy.TFPolicy,
                              num_samples=100):
    """Estimates value (average per-step reward) of policy.

    The estimation is based on solved values of zeta, so one should call
    solve() before calling this function.

    Args:
      dataset: The dataset to sample experience from.
      target_policy: The policy whose value we want to estimate.
      num_samples: number of posterior samples.

    Returns:
      A tensor with num_samples samples of estimated average per-step reward
      of the target policy.
    """
    nu_sigma = tf.sqrt(tf.exp(self._nu_log_sigma))
    eps = tf.random.normal(
        tf.concat([[num_samples], tf.shape(nu_sigma)], axis=-1), 0,
        self._eps_std)
    nu = self._nu_mu + nu_sigma * eps
    self._zeta = (
        tf.einsum('bn,nm->bm', nu, self._td_residuals) /
        tf.math.sqrt(1e-8 + self._total_weights))

    def weight_fn(env_step):
      index = self._get_index(env_step.observation, env_step.action)
      zeta = tf.gather(
          self._zeta, tf.tile(index[None, :], [num_samples, 1]), batch_dims=1)
      policy_ratio = 1.0
      if not self._solve_for_state_action_ratio:
        tfagents_timestep = dataset_lib.convert_to_tfagents_timestep(env_step)
        target_log_probabilities = target_policy.distribution(
            tfagents_timestep).action.log_prob(env_step.action)
        policy_ratio = tf.exp(target_log_probabilities -
                              env_step.get_log_probability())

      return tf.cast(zeta * policy_ratio, tf.float32)

    return estimator_lib.get_fullbatch_average(
        dataset,
        limit=None,
        by_steps=True,
        reward_fn=self._reward_fn,
        weight_fn=weight_fn)
