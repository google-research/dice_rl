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
import dice_rl.utils.common as common_lib


class TabularSaddlePoint(object):
  """Tabular approximatation of Mirror Prox for saddle-point optimization."""

  def __init__(self,
               dataset_spec,
               policy_optimizer,
               gamma: Union[float, tf.Tensor],
               z_learning_rate=0.5,
               v_learning_rate=0.5,
               entropy_reg=0.1,
               reward_fn: Callable = None):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      policy_optimizer: TF optimizer for distilling policy from z.
      gamma: The discount factor to use.
      z_learning_rate: Learning rate for z.
      v_learning_rate: Learning rate for v. entropy_reg; Coefficient on entropy
        regularization.
      reward_fn: A function that takes in an EnvStep and returns the reward for
        that step. If not specified, defaults to just EnvStep.reward.
    """
    self._dataset_spec = dataset_spec
    self._policy_optimizer = policy_optimizer
    self._z_learning_rate = z_learning_rate
    self._v_learning_rate = v_learning_rate
    self._entropy_reg = entropy_reg

    self._gamma = gamma
    if reward_fn is None:
      reward_fn = lambda env_step: env_step.reward
    self._reward_fn = reward_fn

    # Get number of states/actions.
    observation_spec = self._dataset_spec.observation
    action_spec = self._dataset_spec.action
    if not common_lib.is_categorical_spec(observation_spec):
      raise ValueError('Observation spec must be discrete and bounded.')
    self._num_states = observation_spec.maximum + 1

    if not common_lib.is_categorical_spec(action_spec):
      raise ValueError('Action spec must be discrete and bounded.')
    self._num_actions = action_spec.maximum + 1

    self._zetas = np.zeros([self._num_states * self._num_actions])
    self._values = np.zeros([self._num_states])
    self._policy = tf.Variable(np.zeros([self._num_states, self._num_actions]))

  def _get_z_index(self, env_step):
    return env_step.observation * self._num_actions + env_step.action

  def _get_v_index(self, env_step):
    return env_step.observation

  def _get_objective_and_grads(self, init_values, this_values, next_values,
                               zetas, rewards, discounts):
    batch_size = tf.cast(tf.shape(zetas)[0], tf.float32)
    normalized_zetas = batch_size * tf.nn.softmax(zetas)
    log_normalized_zetas = tf.math.log(batch_size) + tf.nn.log_softmax(zetas)

    residuals = rewards + self._gamma * discounts * next_values - this_values
    objective = ((1 - self._gamma) * tf.reduce_mean(init_values) +
                 tf.reduce_mean(normalized_zetas * residuals))
    entropy = -tf.reduce_mean(normalized_zetas * log_normalized_zetas)

    init_v_grads = (1 - self._gamma) / batch_size
    this_v_grads = -1 * normalized_zetas / batch_size
    next_v_grads = self._gamma * discounts * normalized_zetas / batch_size

    z_grads = (residuals -
               self._entropy_reg * log_normalized_zetas) / batch_size

    return ((objective, entropy), init_v_grads, this_v_grads, next_v_grads,
            z_grads)

  def _mirror_prox_step(self, init_steps, this_steps, next_steps):
    init_v_indices = self._get_v_index(init_steps)
    this_v_indices = self._get_v_index(this_steps)
    next_v_indices = self._get_v_index(next_steps)
    this_z_indices = self._get_z_index(this_steps)

    # Mirror prox step.
    init_v = tf.cast(tf.gather(self._values, init_v_indices), tf.float32)
    this_v = tf.cast(tf.gather(self._values, this_v_indices), tf.float32)
    next_v = tf.cast(tf.gather(self._values, next_v_indices), tf.float32)
    this_z = tf.cast(tf.gather(self._zetas, this_z_indices), tf.float32)

    rewards = self._reward_fn(this_steps)
    discounts = next_steps.discount

    (m_objective, m_init_v_grads, m_this_v_grads, m_next_v_grads,
     m_z_grads) = self._get_objective_and_grads(init_v, this_v, next_v, this_z,
                                                rewards, discounts)
    np.add.at(self._values, init_v_indices,
              -self._v_learning_rate * m_init_v_grads)
    np.add.at(self._values, this_v_indices,
              -self._v_learning_rate * m_this_v_grads)
    np.add.at(self._values, next_v_indices,
              -self._v_learning_rate * m_next_v_grads)
    np.add.at(self._zetas, this_z_indices, self._z_learning_rate * m_z_grads)

    # Mirror descent step.
    init_v = tf.cast(tf.gather(self._values, init_v_indices), tf.float32)
    this_v = tf.cast(tf.gather(self._values, this_v_indices), tf.float32)
    next_v = tf.cast(tf.gather(self._values, next_v_indices), tf.float32)
    this_z = tf.cast(tf.gather(self._zetas, this_z_indices), tf.float32)

    (objective, init_v_grads, this_v_grads, next_v_grads,
     z_grads) = self._get_objective_and_grads(init_v, this_v, next_v, this_z,
                                              rewards, discounts)
    np.add.at(self._values, init_v_indices,
              -self._v_learning_rate * (init_v_grads - m_init_v_grads))
    np.add.at(self._values, this_v_indices,
              -self._v_learning_rate * (this_v_grads - m_this_v_grads))
    np.add.at(self._values, next_v_indices,
              -self._v_learning_rate * (next_v_grads - m_next_v_grads))
    np.add.at(self._zetas, this_z_indices,
              self._z_learning_rate * (z_grads - m_z_grads))

    return objective

  def _policy_step(self, steps):
    z_indices = self._get_z_index(steps)
    z = tf.gather(self._zetas, z_indices)
    actions = steps.action
    batch_size = tf.cast(tf.shape(z)[0], actions.dtype)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch([self._policy])
      policy_logits = tf.gather(self._policy, steps.observation)
      #print(steps.observation[:3], tf.nn.softmax(policy_logits[:3], axis=-1),
      #      tf.nn.softmax(z)[:3], actions[:3])
      log_probs = tf.nn.log_softmax(policy_logits, axis=-1)
      selected_log_probs = tf.gather_nd(
          log_probs, tf.stack([tf.range(batch_size), actions], -1))
      loss = -tf.reduce_mean(tf.nn.softmax(z) * selected_log_probs)

    grads = tape.gradient(loss, [self._policy])
    grad_op = self._policy_optimizer.apply_gradients([(grads[0], self._policy)])

    return loss, grad_op

  def train_step(self, initial_steps: dataset_lib.EnvStep,
                 transitions: dataset_lib.EnvStep):
    """Performs training step on z, v, and policy based on batch of transitions.

    Args:
      initial_steps: A batch of initial steps.
      transitions: A batch of transitions. Members should have shape
        [batch_size, 2, ...].

    Returns:
      A train op.
    """
    this_steps = tf.nest.map_structure(lambda t: t[:, 0, ...], transitions)
    next_steps = tf.nest.map_structure(lambda t: t[:, 1, ...], transitions)

    loss = self._mirror_prox_step(initial_steps, this_steps, next_steps)
    policy_loss, _ = self._policy_step(this_steps)

    return loss, policy_loss

  def get_policy(self):
    """Returns learned policy."""
    return common_lib.create_tf_policy_from_table(
        tf.nn.softmax(self._policy, axis=-1).numpy(),
        obs_to_index_fn=lambda obs: obs,
        return_distribution=True)
