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
import dice_rl.utils.common as common_lib
import dice_rl.estimators.estimator as estimator_lib
from dice_rl.estimators.neural_dice import NeuralDice


class NeuralBayesDice(NeuralDice):

  def __init__(self,
               dataset_spec,
               nu_network,
               zeta_network,
               nu_optimizer,
               zeta_optimizer,
               lam_optimizer,
               gamma: Union[float, tf.Tensor],
               kl_regularizer: Union[float, tf.Tensor] = 1.,
               eps_std: Union[float, tf.Tensor] = 1.,
               **kwargs):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      nu_network: The nu-value network.
      zeta_network: The zeta-value network.
      nu_optimizer: The optimizer to use for nu.
      zeta_optimizer: The optimizer to use for zeta.
      lam_optimizer: The optimizer to use for lambda.
      gamma: The discount factor to use.
      kl_regularizer: Regularization constant for D_kl(q || p).
      eps_std: epsilon standard deviation for sampling from the posterior.
    """
    self._kl_regularizer = kl_regularizer
    self._eps_std = eps_std
    super(NeuralBayesDice,
          self).__init__(dataset_spec, nu_network, zeta_network, nu_optimizer,
                         zeta_optimizer, lam_optimizer, gamma, **kwargs)

  def train_loss(self, initial_env_step, env_step, next_env_step, policy):
    nu_values, _, eps = self._sample_value(self._nu_network, env_step)
    initial_nu_values, _, _ = self._sample_average_value(
        self._nu_network, initial_env_step, policy)
    next_nu_values, _, _ = self._sample_average_value(self._nu_network,
                                                      next_env_step, policy)

    zeta_values, zeta_neg_kl, _ = self._sample_value(self._zeta_network,
                                                     env_step, eps)

    discounts = self._gamma * env_step.discount
    policy_ratio = 1.0
    if not self._solve_for_state_action_ratio:
      tfagents_step = dataset_lib.convert_to_tfagents_timestep(env_step)
      policy_log_probabilities = policy.distribution(
          tfagents_step).action.log_prob(env_step.action)
      policy_ratio = tf.exp(policy_log_probabilities -
                            env_step.get_log_probability())

    bellman_residuals = (
        common_lib.reverse_broadcast(discounts * policy_ratio, nu_values) *
        next_nu_values - nu_values - self._norm_regularizer * self._lam)
    if not self._zero_reward:
      bellman_residuals += policy_ratio * self._reward_fn(env_step)

    zeta_loss = -zeta_values * bellman_residuals
    nu_loss = (1 - self._gamma) * initial_nu_values
    lam_loss = self._norm_regularizer * self._lam
    if self._primal_form:
      nu_loss += self._fstar_fn(bellman_residuals)
      lam_loss = lam_loss + self._fstar_fn(bellman_residuals)
    else:
      nu_loss += zeta_values * bellman_residuals
      lam_loss = lam_loss - self._norm_regularizer * zeta_values * self._lam

    nu_loss += self._primal_regularizer * self._f_fn(nu_values)
    zeta_loss += self._dual_regularizer * self._f_fn(zeta_values)
    zeta_loss -= self._kl_regularizer * tf.reduce_mean(zeta_neg_kl)

    if self._weight_by_gamma:
      weights = self._gamma**tf.cast(env_step.step_num, tf.float32)[:, None]
      weights /= 1e-6 + tf.reduce_mean(weights)
      nu_loss *= weights
      zeta_loss *= weights

    return nu_loss, zeta_loss, lam_loss

  def _sample_helper(self, value, eps=None):
    mu, sigma = tf.split(value, num_or_size_splits=2, axis=-1)
    sigma = tf.sigmoid(sigma)
    if eps is None:
      eps = tf.random.normal(
          shape=tf.shape(sigma),
          mean=0.,
          stddev=self._eps_std,
          dtype=tf.float32)
    value = mu + sigma * eps
    neg_kl = 0.5 + tf.math.log(sigma + 1e-8) - 0.5 * (sigma**2 + mu**2)
    return tf.squeeze(value, axis=-1), tf.squeeze(neg_kl, axis=-1), eps

  def _sample_value(self, network, env_step, eps=None):
    value = self._get_value(network, env_step)
    return self._sample_helper(value, eps)

  def _sample_average_value(self, network, env_step, policy):
    action_weights = 1.
    if self._solve_for_state_action_ratio:
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

      flat_actions = tf.reshape(actions, [batch_size * num_actions] +
                                actions.shape[2:].as_list())
      flat_observations = tf.reshape(
          tf.tile(env_step.observation[:, None, ...],
                  [1, num_actions] + [1] * len(env_step.observation.shape[1:])),
          [batch_size * num_actions] + env_step.observation.shape[1:].as_list())
      flat_values, _ = network((flat_observations, flat_actions))

      params = tf.reshape(flat_values, [batch_size, num_actions] +
                          flat_values.shape[1:].as_list())
    else:
      params = network(env_step.observation)[0]

    values, neg_kl, _ = self._sample_helper(params)
    policy_weights = common_lib.reverse_broadcast(action_weights, values)
    return (tf.reduce_sum(values * policy_weights, axis=1),
            tf.reduce_sum(neg_kl * policy_weights, axis=1), None)

  def estimate_average_reward(self,
                              dataset: dataset_lib.OffpolicyDataset,
                              target_policy: tf_policy.TFPolicy,
                              write_summary: bool = False):
    """Estimates value (average per-step reward) of policy.

    Args:
      dataset: The dataset to sample experience from.
      target_policy: The policy whose value we want to estimate.

    Returns:
      Estimated average per-step reward of the target policy.
    """

    def weight_fn(env_step):
      zeta, _, _ = self._sample_value(self._zeta_network, env_step)
      policy_ratio = 1.0
      if not self._solve_for_state_action_ratio:
        tfagents_timestep = dataset_lib.convert_to_tfagents_timestep(env_step)
        target_log_probabilities = target_policy.distribution(
            tfagents_timestep).action.log_prob(env_step.action)
        policy_ratio = tf.exp(target_log_probabilities -
                              env_step.get_log_probability())
      return zeta * common_lib.reverse_broadcast(policy_ratio, zeta)

    def init_nu_fn(env_step, valid_steps):
      """Computes average initial nu values of episodes."""
      # env_step is an episode, and we just want the first step.
      if tf.rank(valid_steps) == 1:
        first_step = tf.nest.map_structure(lambda t: t[0, ...], env_step)
      else:
        first_step = tf.nest.map_structure(lambda t: t[:, 0, ...], env_step)
      value, _, _ = self._sample_average_value(self._nu_network, first_step,
                                               target_policy)
      return value

    dual_step = estimator_lib.get_fullbatch_average(
        dataset,
        limit=None,
        by_steps=True,
        reward_fn=self._reward_fn,
        weight_fn=weight_fn)

    nu_zero = (1 - self._gamma) * estimator_lib.get_fullbatch_average(
        dataset,
        limit=None,
        by_steps=False,
        truncate_episode_at=1,
        reward_fn=init_nu_fn)

    if not write_summary:
      return dual_step

    tf.summary.scalar('eval/dual_step', dual_step)
    tf.summary.scalar('eval/nu_zero', nu_zero)
    tf.print('step', tf.summary.experimental.get_step(), 'dual_step =',
             dual_step, 'nu_zero =', nu_zero)

    return dual_step

  def _eval_constraint_and_regs(self, dataset: dataset_lib.OffpolicyDataset,
                                target_policy: tf_policy.TFPolicy):
    """Get the residual term and the primal and dual regularizers during eval.

    Args:
      dataset: The dataset to sample experience from.
      target_policy: The policy whose value we want to estimate.

    Returns:
      The residual term (weighted by zeta), primal, and dual reg values.
    """

    experience = dataset.get_all_steps(num_steps=2)
    env_step = tf.nest.map_structure(lambda t: t[:, 0, ...], experience)
    next_env_step = tf.nest.map_structure(lambda t: t[:, 1, ...], experience)
    nu_values, _, _ = self._sample_value(self._nu_network, env_step)
    next_nu_values, _, _ = self._sample_average_value(self._nu_network,
                                                      next_env_step,
                                                      target_policy)
    zeta_values, neg_kl, _ = self._sample_value(self._zeta_network, env_step)
    discounts = self._gamma * env_step.discount
    bellman_residuals = (
        common_lib.reverse_broadcast(discounts, nu_values) * next_nu_values -
        nu_values - self._norm_regularizer * self._lam)

    # Always include reward during eval
    bellman_residuals += self._reward_fn(env_step)
    constraint = tf.reduce_mean(zeta_values * bellman_residuals)

    f_nu = tf.reduce_mean(self._f_fn(nu_values))
    f_zeta = tf.reduce_mean(self._f_fn(zeta_values))

    return constraint, f_nu, f_zeta, tf.reduce_mean(neg_kl)
