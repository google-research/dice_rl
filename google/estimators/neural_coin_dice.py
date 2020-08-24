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
from typing import Any, Callable, Iterable, Optional, Sequence, Text, Tuple, Union

import dice_rl.data.dataset as dataset_lib
import dice_rl.utils.common as common_lib
import dice_rl.estimators.estimator as estimator_lib
from dice_rl.estimators.neural_dual_dice import NeuralDualDice


class NeuralCoinDice(NeuralDualDice):
  """Coinfidence interval policy evaluation."""

  def __init__(self,
               dataset_spec,
               nu_network,
               zeta_network,
               weight_network,
               nu_optimizer,
               zeta_optimizer,
               weight_optimizer,
               gamma: Union[float, tf.Tensor],
               divergence_limit: Union[float, np.ndarray, tf.Tensor],
               divergence_type: Text = 'rkl',
               algae_alpha: Union[float, tf.Tensor] = 1.0,
               unbias_algae_alpha: bool = False,
               closed_form_weights: bool = True,
               weight_by_gamma: bool = False,
               **kwargs):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      nu_network: The nu-value network.
      zeta_network: The zeta-value network.
      weight_network: The weights network.
      nu_optimizer: The optimizer to use for nu.
      zeta_optimizer: The optimizer to use for zeta.
      weight_optimizer: The optimizer to use for the weights.
      gamma: The discount factor to use.
      divergence_limit: The limit on the f-divergence between the weights and
        the empirical distribution. This should contain half as many elements as
        outputted by the nu, zeta, and weight networks.
      divergence_type: The type of f-divergence to use, e.g., 'kl'.
      algae_alpha: Regularizer coefficient on Df(dpi || dD).
      unbias_algae_alpha: Whether to learn two objectives, one with algae_alpha
        and the other with -algae_alpha, to counteract bias induced by
        algae_alpha. Defaults to False, which is more stable for optimization.
      closed_form_weights: Whether to use closed-form weights. If true,
        weight_network and weight_optimizer are ignored.
      weight_by_gamma: Weight nu and zeta losses by gamma ** step_num.
    """
    self._weight_network = weight_network
    if self._weight_network is not None:
      self._weight_network.create_variables()
    self._weight_optimizer = weight_optimizer

    self._divergence_limit = tf.convert_to_tensor(
        divergence_limit, dtype=tf.float32)
    if tf.rank(self._divergence_limit) < 1:
      self._divergence_limit = tf.expand_dims(self._divergence_limit, -1)
    self._two_sided_limit = tf.concat(
        [self._divergence_limit, self._divergence_limit], -1)
    self._num_limits = int(self._two_sided_limit.shape[0])
    self._alpha = tf.Variable(
        np.zeros(self._two_sided_limit.shape), dtype=tf.float32)

    self._divergence_type = divergence_type
    if self._divergence_type not in ['kl', 'rkl']:
      raise ValueError('Unsupported divergence type %s.' %
                       self._divergence_type)

    self._algae_alpha = tf.convert_to_tensor(algae_alpha, dtype=tf.float32)
    if tf.rank(self._algae_alpha) < 1:
      self._algae_alpha = tf.expand_dims(self._algae_alpha, -1)
    if self._algae_alpha.shape[-1] != self._two_sided_limit.shape[-1]:
      self._algae_alpha * tf.ones_like(self._two_sided_limit)
    self._algae_alpha_sign = 2 * (
        tf.cast(self._algae_alpha >= 0, tf.float32) - 0.5)
    self._algae_alpha_sign = tf.reshape(
        tf.stack([self._algae_alpha_sign, -self._algae_alpha_sign], axis=-1),
        [2 * self._num_limits])
    self._algae_alpha_abs = tf.concat(
        2 * [tf.math.abs(self._algae_alpha)], axis=-1)
    self._unbias_algae_alpha = unbias_algae_alpha

    self._closed_form_weights = closed_form_weights
    self._weight_by_gamma = weight_by_gamma

    super(NeuralCoinDice,
          self).__init__(dataset_spec, nu_network, zeta_network, nu_optimizer,
                         zeta_optimizer, gamma, **kwargs)

  def _get_weights(self,
                   initial_env_step,
                   env_step,
                   next_env_step,
                   nu_loss,
                   alpha=None):
    if alpha is None:
      alpha = self._alpha
    if not self._closed_form_weights:
      network_output = self._weight_network(
          (initial_env_step.observation, env_step.observation, env_step.action,
           next_env_step.observation))[0]
      log_weights = network_output
    else:
      if self._divergence_type in ['kl', 'rkl']:
        weight_loss_multiplier = self._algae_alpha_sign * tf.concat(
            2 * [tf.ones_like(self._divergence_limit)] +
            2 * [-tf.ones_like(self._divergence_limit)],
            axis=-1)
        multiplied_loss = weight_loss_multiplier * nu_loss
        combined_loss = tf.reduce_mean(
            tf.reshape(multiplied_loss, [-1, self._num_limits, 2]), axis=-1)
        log_weights = -combined_loss / tf.exp(alpha)
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
    else:
      raise ValueError('Divergence is not implemented.')

  def train_loss(self, initial_env_step, env_step, next_env_step, policy):
    nu_values = self._get_value(self._nu_network, env_step)
    initial_nu_values = self._get_average_value(self._nu_network,
                                                initial_env_step, policy)
    next_nu_values = self._get_average_value(self._nu_network, next_env_step,
                                             policy)
    zeta_values = self._get_value(self._zeta_network, env_step)
    rewards = self._reward_fn(env_step)

    discounts = self._gamma * env_step.discount
    policy_ratio = 1.0
    if not self._solve_for_state_action_ratio:
      tfagents_step = dataset_lib.convert_to_tfagents_timestep(env_step)
      policy_log_probabilities = policy.distribution(
          tfagents_step).action.log_prob(env_step.action)
      policy_ratio = tf.exp(policy_log_probabilities -
                            env_step.get_log_probability())

    bellman_residuals = (
        -nu_values + common_lib.reverse_broadcast(rewards, nu_values) +
        common_lib.reverse_broadcast(discounts * policy_ratio, nu_values) *
        next_nu_values)
    bellman_residuals *= self._algae_alpha_sign

    zeta_loss = (
        self._algae_alpha_abs * self._fstar_fn(zeta_values) -
        bellman_residuals * zeta_values)

    init_nu_loss = ((1 - self._gamma) * initial_nu_values *
                    self._algae_alpha_sign)
    if self._primal_form:
      nu_loss = (
          self._algae_alpha_abs *
          self._f_fn(bellman_residuals / self._algae_alpha_abs) + init_nu_loss)
    else:
      nu_loss = -zeta_loss + init_nu_loss

    if self._weight_by_gamma:
      weights = self._gamma**tf.cast(env_step.step_num, tf.float32)[:, None]
      weights /= 1e-6 + tf.reduce_mean(weights)
      nu_loss *= weights
      zeta_loss *= weights
    return nu_loss, zeta_loss

  def train_step(self, initial_env_step: dataset_lib.EnvStep,
                 experience: dataset_lib.EnvStep,
                 target_policy: tf_policy.TFPolicy):
    """Performs a single training step based on batch.

    Args:
      initial_env_step: A batch of initial steps.
      experience: A batch of transitions. Elements must have shape [batch_size,
        2, ...].
      target_policy: The policy whose value we want to estimate.

    Returns:
      The losses and the train op.
    """
    env_step = tf.nest.map_structure(lambda t: t[:, 0, ...], experience)
    next_env_step = tf.nest.map_structure(lambda t: t[:, 1, ...], experience)

    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      tape.watch(self._nu_network.variables)
      tape.watch(self._zeta_network.variables)
      tape.watch(self._weight_network.variables)
      tape.watch([self._alpha])
      nu_loss, zeta_loss = self.train_loss(initial_env_step, env_step,
                                           next_env_step, target_policy)
      if not self._unbias_algae_alpha:
        # Zero-out components of loss made for -algae_alpha.
        nu_loss *= tf.constant([1., 0., 1., 0.])
        zeta_loss *= tf.constant([1., 0., 1., 0.])

      nu_reg = self._nu_regularizer * self._orthogonal_regularization(
          self._nu_network)
      zeta_reg = self._zeta_regularizer * self._orthogonal_regularization(
          self._zeta_network)

      # Binary search to find best alpha. Limited to just be around the
      # neighborhood of the current alpha to save on compute.
      left = self._alpha - 1 * tf.ones_like(self._two_sided_limit)
      right = self._alpha + 1 * tf.ones_like(self._two_sided_limit)
      for _ in range(4):
        mid = 0.5 * (left + right)
        weights, log_weights = self._get_weights(
            initial_env_step, env_step, next_env_step, nu_loss, alpha=mid)
        divergence = self._compute_divergence(weights, log_weights)
        divergence_violation = divergence - self._two_sided_limit
        left = tf.where(divergence_violation > 0., mid, left)
        right = tf.where(divergence_violation > 0., right, mid)
      best_alpha = 0.5 * (left + right)
      self._alpha.assign(0.05 * best_alpha + 0.95 * self._alpha)

      weights, log_weights = self._get_weights(initial_env_step, env_step,
                                               next_env_step, nu_loss)
      divergence = self._compute_divergence(weights, log_weights)
      divergence_violation = divergence - self._two_sided_limit

      weighted_nu_loss = (
          tf.reshape(nu_loss, [-1, self._num_limits, 2]) * weights[:, :, None])
      weighted_zeta_loss = (
          tf.reshape(zeta_loss, [-1, self._num_limits, 2]) *
          weights[:, :, None])

      # Multiplier to make all weight optimizations minimizations.
      # Takes into account that sign of algae_alpha determines whether nu_loss
      # has switched signs (since all of nu_loss is used for minimization).
      weight_loss_multiplier = self._algae_alpha_sign * tf.concat(
          2 * [tf.ones_like(self._divergence_limit)] +
          2 * [-tf.ones_like(self._divergence_limit)],
          axis=-1)
      weight_loss = tf.reduce_mean(
          tf.reshape(weight_loss_multiplier * nu_loss,
                     [-1, self._num_limits, 2]), 1)
      weight_loss += tf.exp(self._alpha) * divergence_violation

      reg_weighted_nu_loss = weighted_nu_loss + nu_reg
      reg_weighted_zeta_loss = weighted_zeta_loss + nu_reg

    nu_grads = tape.gradient(reg_weighted_nu_loss, self._nu_network.variables)
    nu_grad_op = self._nu_optimizer.apply_gradients(
        zip(nu_grads, self._nu_network.variables))

    zeta_grads = tape.gradient(reg_weighted_zeta_loss,
                               self._zeta_network.variables)
    zeta_grad_op = self._zeta_optimizer.apply_gradients(
        zip(zeta_grads, self._zeta_network.variables))

    if not self._closed_form_weights:
      weight_grads = tape.gradient(weight_loss, self._weight_network.variables)
      weight_grad_op = self._weight_optimizer.apply_gradients(
          zip(weight_grads, self._weight_network.variables))
    else:
      weight_grad_op = tf.group()

    for idx in range(self._num_limits):
      tf.summary.scalar('divergence%d' % idx, divergence[idx])
      tf.summary.scalar('nu_loss%d' % idx, tf.reduce_mean(nu_loss, 0)[idx])
      tf.summary.scalar('zeta_loss%d' % idx, tf.reduce_mean(zeta_loss, 0)[idx])
      tf.summary.scalar('exp_alpha%d' % idx, tf.exp(self._alpha[idx]))
      tf.summary.histogram('weights%d' % idx, weights[:, idx])

    estimate = tf.reduce_mean(
        weighted_nu_loss *
        tf.reshape(self._algae_alpha_sign, [self._num_limits, 2]),
        axis=[0, -1])
    if not self._unbias_algae_alpha:
      estimate = 2 * estimate  # Counteract reduce_mean above.
    return ((estimate, tf.reshape(tf.reduce_mean(weighted_nu_loss, [0]), [-1]),
             tf.reshape(tf.reduce_mean(weighted_zeta_loss, [0]),
                        [-1]), tf.reduce_mean(weight_loss, 0), divergence),
            tf.group(nu_grad_op, zeta_grad_op, weight_grad_op))
