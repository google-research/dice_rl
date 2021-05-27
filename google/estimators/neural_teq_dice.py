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

import tensorflow.compat.v2 as tf
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import dice_rl.data.dataset as dataset_lib
import dice_rl.utils.common as common_lib
from dice_rl.estimators.neural_dual_dice import NeuralDualDice


class NeuralTeQDice(NeuralDualDice):
  """Policy evaluation with TeQDICE."""

  def __init__(self,
               dataset_spec,
               nu_network,
               zeta_network,
               nu_optimizer,
               zeta_optimizer,
               gamma: Union[float, tf.Tensor],
               reward_fn: Optional[Callable] = None,
               **kwargs):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      nu_network: The nu-value network.
      zeta_network: The zeta-value network.
      nu_optimizer: The optimizer to use for nu.
      zeta_optimizer: The optimizer to use for zeta.
      gamma: The discount factor to use.
      reward_fn: A function that takes in an EnvStep and returns the reward for
        that step. If not specified, defaults to just EnvStep.reward.
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
    super(NeuralTeQDice,
          self).__init__(dataset_spec, nu_network, zeta_network, nu_optimizer,
                         zeta_optimizer, gamma, **kwargs)

  def _get_value(self, network, env_step):
    if self._solve_for_state_action_ratio:
      return network(
          (env_step.observation, env_step.action, env_step.step_num))[0]
    else:
      return network(env_step.observation, env_step.step_num)[0]

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
            [policy.action(tfagents_step).action for _ in range(num_actions)],
            axis=1)

      flat_actions = tf.reshape(actions, [batch_size * num_actions] +
                                actions.shape[2:].as_list())
      flat_observations = tf.reshape(
          tf.tile(env_step.observation[:, None, ...],
                  [1, num_actions] + [1] * len(env_step.observation.shape[1:])),
          [batch_size * num_actions] + env_step.observation.shape[1:].as_list())
      flat_step_nums = tf.reshape(
          tf.tile(env_step.step_num[:, None, ...],
                  [1, num_actions] + [1] * len(env_step.step_num.shape[1:])),
          [batch_size * num_actions] + env_step.step_num.shape[1:].as_list())
      flat_values, _ = network(
          (flat_observations, flat_actions, flat_step_nums))

      values = tf.reshape(flat_values, [batch_size, num_actions] +
                          flat_values.shape[1:].as_list())
      return tf.reduce_sum(
          values * common_lib.reverse_broadcast(action_weights, values), axis=1)
    else:
      return network(env_step.observation, env_step.step_num)[0]
