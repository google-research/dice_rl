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
from tf_agents import specs
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
import tensorflow_probability as tfp


def create_py_policy_from_table(probability_table, obs_to_index_fn):
  """Creates a callable policy function given a table of state to distribution.

  Args:
    probability_table: A NumPy array determining the action distribution.
    obs_to_index_fn: A function mapping environment observation to index in table.

  Returns:
    policy_fn: A function mapping observations to sampled actions and policy
      info.
    policy_info_spec: A spec that determines the type of objects returned by
      policy info.
  """
  def policy_fn(observation, probability_table=probability_table,
                obs_to_index_fn=obs_to_index_fn,
                dtype=np.int32):
    state = obs_to_index_fn(observation)
    distributions = probability_table[state]
    batched = np.ndim(distributions) > 1
    if not batched:
      distributions = distributions[None, :]

    cum_probs = distributions.cumsum(axis=-1)
    uniform_samples = np.random.rand(len(cum_probs), 1)
    actions = (uniform_samples < cum_probs).argmax(axis=1)
    probs = distributions[np.arange(len(actions)), actions]

    if not batched:
      action = actions[0]
      log_prob = np.log(1e-8 + probs[0])
    else:
      action = actions
      log_prob = np.log(1e-8 + probs)

    policy_info = {'log_probability': log_prob, 'distribution': distributions}
    return action.astype(dtype), policy_info

  policy_info_spec = {'log_probability': specs.ArraySpec([], np.float),
                      'distribution': specs.BoundedArraySpec(
                          [np.shape(probability_table)[-1]], np.float,
                          minimum=0.0, maximum=1.0)}
  return policy_fn, policy_info_spec


def create_tf_policy_from_table(probability_table, obs_to_index_fn,
                                return_distribution=False):
  """Creates a callable policy function given a table of state to distribution.

  Args:
    probability_table: A Tensor-like object determining the action distribution.
    obs_to_index_fn: A function mapping environment observation to index in
      table.
    return_distribution: Whether policy_fn should return a distribution. If not,
      returns sampled actions.

  Returns:
    policy_fn: A function mapping observations to action distribution or sampled
      actions and policy info.
    policy_info_spec: A spec that determines the type of objects returned by
      policy info.
  """
  probability_table = tf.convert_to_tensor(probability_table, dtype=tf.float32)
  n_actions = tf.shape(probability_table)[-1]

  def policy_fn(observation, probability_table=probability_table,
                obs_to_index_fn=obs_to_index_fn,
                return_distribution=return_distribution,
                dtype=tf.int32):
    state = obs_to_index_fn(observation)
    distribution = tf.gather(probability_table, state)
    batched = tf.rank(distribution) > 1
    if not batched:
      distributions = distribution[None, :]
    else:
      distributions = distribution

    batch_size = tf.shape(distributions)[0]

    actions = tf.random.categorical(tf.math.log(1e-8 + distributions), 1,
                                    dtype=dtype)
    actions = tf.squeeze(actions, -1)
    probs = tf.gather_nd(distributions,
                         tf.stack([tf.range(batch_size, dtype=dtype),
                                   actions], -1))

    if not batched:
      action = actions[0]
      log_prob = tf.math.log(1e-8 + probs[0])
    else:
      action = actions
      log_prob = tf.math.log(1e-8 + probs)

    if return_distribution:
      policy_info = {'distribution': distribution}
      return (tfp.distributions.Categorical(probs=distribution, dtype=dtype),
              policy_info)
    else:
      policy_info = {'log_probability': log_prob, 'distribution': distribution}
      return action, policy_info

  policy_info_spec = {'log_probability': specs.TensorSpec([], tf.float32),
                      'distribution': specs.BoundedTensorSpec(
                          [n_actions], tf.float32,
                          minimum=0.0, maximum=1.0)}
  return policy_fn, policy_info_spec


class TFAgentsWrappedPolicy(tf_policy.TFPolicy):
  """Wraps a policy function in a TF-Agents tf_policy.TFPolicy."""

  def __init__(self, time_step_spec, action_spec,
               policy_distribution_fn, policy_info_spec,
               emit_log_probability=True):
    """Wraps the policy function.

    Args:
      time_step_spec: Spec of time steps given by environment.
      action_spec: Intended spec of actions returned by policy_fn.
      policy_distribution_fn: A TF function mapping observation to action
        distribution and policy info.
      policy_info_spec: Spec determining policy info returned by policy_fn.
      emit_log_probability: Whether to emit log probabilities of sampled
        actions.
    """
    self._policy_distribution_fn = policy_distribution_fn
    self._action_dtype = tf.nest.map_structure(
        lambda spec: spec.dtype, action_spec)
    super(TFAgentsWrappedPolicy, self).__init__(
        time_step_spec, action_spec,
        policy_state_spec=(),
        info_spec=policy_info_spec,
        clip=False,
        emit_log_probability=emit_log_probability)

  def _distribution(self, time_step, policy_state):
    distribution, info = self._policy_distribution_fn(time_step.observation,
                                                      dtype=self._action_dtype)
    distribution = tf.nest.pack_sequence_as(
        self.action_spec,
        tf.nest.flatten(distribution))
    return policy_step.PolicyStep(
        distribution, policy_state, info)


def is_categorical_spec(spec):
  """Checks if spec is of a categorical value."""
  return (specs.tensor_spec.is_discrete(spec) and
          specs.tensor_spec.is_bounded(spec) and
          spec.shape == [] and
          spec.minimum == 0)


def reverse_broadcast(input_tensor, target_tensor):
  input_tensor = tf.convert_to_tensor(input_tensor)
  target_tensor = tf.convert_to_tensor(target_tensor)
  input_rank = len(input_tensor.shape.as_list())
  target_rank = len(target_tensor.shape.as_list())
  additional_rank = max(0, target_rank - input_rank)
  return tf.reshape(input_tensor, input_tensor.shape.as_list() +
                    [1] * additional_rank)
