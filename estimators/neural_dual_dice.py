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


class NeuralDualDice(object):
  """Policy evaluation with DualDICE."""

  def __init__(self, dataset_spec,
               nu_network,
               zeta_network,
               nu_optimizer,
               zeta_optimizer,
               gamma: Union[float, tf.Tensor],
               reward_fn: Callable = None,
               solve_for_state_action_ratio: bool = True,
               f_exponent: float = 1.5,
               primal_form: bool = False,
               num_samples: Optional[int] = None,
               nu_regularizer: float = 0.,
               zeta_regularizer: float = 0.):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      nu_network: The nu-value network.
      zeta_network: The zeta-value network.
      nu_optimizer: The optimizer to use for nu.
      zeta_optimizer: The optimizer to use for zeta.
      gamma: The discount factor to use.
      reward_fn: A function that takes in an EnvStep and returns the reward
        for that step. If not specified, defaults to just EnvStep.reward.
      solve_for_state_action_ratio: Whether to solve for state-action density
        ratio. Defaults to True.
      f_exponent: Exponent p to use for f(x) = |x|^p / p.
      primal_form: Whether to use primal form of DualDICE, which optimizes for
        nu independent of zeta. This form is biased in stochastic environments.
        Defaults to False, which uses the saddle-point formulation of DualDICE.
      num_samples: Number of samples to take from policy to estimate average
        next nu value. If actions are discrete, this defaults to computing
        average explicitly. If actions are not discrete, this defaults to using
        a single sample.
      nu_regularizer: Regularization coefficient on nu network.
      zeta_regularizer: Regularization coefficient on zeta network.
    """
    self._dataset_spec = dataset_spec
    self._nu_network = nu_network
    self._nu_network.create_variables()
    self._zeta_network = zeta_network
    self._zeta_network.create_variables()

    self._nu_optimizer = nu_optimizer
    self._zeta_optimizer = zeta_optimizer
    self._nu_regularizer = nu_regularizer
    self._zeta_regularizer = zeta_regularizer

    self._gamma = gamma
    if reward_fn is None:
      reward_fn = lambda env_step: env_step.reward
    self._reward_fn = reward_fn
    self._num_samples = num_samples

    self._solve_for_state_action_ratio = solve_for_state_action_ratio
    if (not self._solve_for_state_action_ratio and
        not self._dataset_spec.has_log_probability()):
      raise ValueError('Dataset must contain log-probability when '
                       'solve_for_state_action_ratio is False.')

    if f_exponent <= 1:
      raise ValueError('Exponent for f must be greater than 1.')
    fstar_exponent = f_exponent / (f_exponent - 1)
    self._f_fn = lambda x: tf.abs(x) ** f_exponent / f_exponent
    self._fstar_fn = lambda x: tf.abs(x) ** fstar_exponent / fstar_exponent

    self._categorical_action = common_lib.is_categorical_spec(
        self._dataset_spec.action)
    if not self._categorical_action and self._num_samples is None:
      self._num_samples = 1

    self._primal_form = primal_form
    self._initialize()

  def _initialize(self):
    pass

  def _get_value(self, network, env_step):
    if self._solve_for_state_action_ratio:
      return network((env_step.observation, env_step.action))[0]
    else:
      return network(env_step.observation)[0]

  def _get_average_value(self, network, env_step, policy):
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
            [policy.action(tfagents_step).action
             for _ in range(num_actions)],
            axis=1)

      flat_actions = tf.reshape(actions, [batch_size * num_actions] +
                                actions.shape[2:].as_list())
      flat_observations = tf.reshape(
          tf.tile(env_step.observation[:, None, ...],
                  [1, num_actions] + [1] * len(env_step.observation.shape[1:])),
          [batch_size * num_actions] + env_step.observation.shape[1:].as_list())
      flat_values, _ = network((flat_observations, flat_actions))

      values = tf.reshape(flat_values, [batch_size, num_actions] +
                          flat_values.shape[1:].as_list())
      return tf.reduce_sum(
          values * common_lib.reverse_broadcast(action_weights, values),
          axis=1)
    else:
      return network(env_step.observation)[0]

  def _orthogonal_regularization(self, network):
    reg = 0
    for layer in network.layers:
      if isinstance(layer, tf.keras.layers.Dense):
        prod = tf.matmul(tf.transpose(layer.kernel), layer.kernel)
        reg += tf.reduce_sum(
            tf.math.square(prod * (1 - tf.eye(prod.shape[0]))))
    return reg

  def train_loss(self, initial_env_step, env_step, next_env_step, policy):
    nu_values = self._get_value(self._nu_network, env_step)
    initial_nu_values = self._get_average_value(
        self._nu_network, initial_env_step, policy)
    next_nu_values = self._get_average_value(
        self._nu_network, next_env_step, policy)
    zeta_values = self._get_value(self._zeta_network, env_step)

    discounts = self._gamma * env_step.discount
    policy_ratio = 1.0
    if not self._solve_for_state_action_ratio:
      tfagents_step = dataset_lib.convert_to_tfagents_timestep(env_step)
      policy_log_probabilities = policy.distribution(
          tfagents_step).action.log_prob(env_step.action)
      policy_ratio = tf.exp(
          policy_log_probabilities - env_step.get_log_probability())

    bellman_residuals = (
        nu_values - common_lib.reverse_broadcast(
            discounts * policy_ratio, nu_values) * next_nu_values)

    zeta_loss = self._fstar_fn(zeta_values) - bellman_residuals * zeta_values
    if self._primal_form:
      nu_loss = (self._f_fn(bellman_residuals)
                 - (1 - self._gamma) * initial_nu_values)
    else:
      nu_loss = -zeta_loss - (1 - self._gamma) * initial_nu_values

    return nu_loss, zeta_loss

  @tf.function
  def train_step(self,
                 initial_env_step: dataset_lib.EnvStep,
                 experience: dataset_lib.EnvStep,
                 target_policy: tf_policy.TFPolicy):
    """Performs a single training step based on batch.

    Args:
      initial_env_step: A batch of initial steps.
      experience: A batch of transitions. Elements must have shape
        [batch_size, 2, ...].
      target_policy: The policy whose value we want to estimate.

    Returns:
      The losses and the train op.
    """
    env_step = tf.nest.map_structure(lambda t: t[:, 0, ...], experience)
    next_env_step = tf.nest.map_structure(lambda t: t[:, 1, ...], experience)

    with tf.GradientTape(watch_accessed_variables=False,
                         persistent=True) as tape:
      tape.watch(self._nu_network.variables)
      tape.watch(self._zeta_network.variables)
      nu_loss, zeta_loss = self.train_loss(
          initial_env_step, env_step, next_env_step, target_policy)
      nu_loss += self._nu_regularizer * self._orthogonal_regularization(
          self._nu_network)
      zeta_loss += self._zeta_regularizer * self._orthogonal_regularization(
          self._zeta_network)

    nu_grads = tape.gradient(nu_loss, self._nu_network.variables)
    nu_grad_op = self._nu_optimizer.apply_gradients(
        zip(nu_grads, self._nu_network.variables))

    zeta_grads = tape.gradient(zeta_loss, self._zeta_network.variables)
    zeta_grad_op = self._zeta_optimizer.apply_gradients(
        zip(zeta_grads, self._zeta_network.variables))

    return (tf.reduce_mean(nu_loss), tf.reduce_mean(zeta_loss))

  def estimate_average_reward(self,
                              dataset: dataset_lib.OffpolicyDataset,
                              target_policy: tf_policy.TFPolicy):
    """Estimates value (average per-step reward) of policy.

    Args:
      dataset: The dataset to sample experience from.
      target_policy: The policy whose value we want to estimate.

    Returns:
      Estimated average per-step reward of the target policy.
    """
    def weight_fn(env_step):
      zeta = self._get_value(self._zeta_network, env_step)

      policy_ratio = 1.0
      if not self._solve_for_state_action_ratio:
        tfagents_timestep = dataset_lib.convert_to_tfagents_timestep(env_step)
        target_log_probabilities = target_policy.distribution(
            tfagents_timestep).action.log_prob(env_step.action)
        policy_ratio = tf.exp(
            target_log_probabilities -
            env_step.get_log_probability())

      return zeta * common_lib.reverse_broadcast(policy_ratio, zeta)

    return estimator_lib.get_fullbatch_average(
        dataset, limit=None, by_steps=True,
        reward_fn=self._reward_fn, weight_fn=weight_fn)
