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

import numpy as np
import tensorflow.compat.v2 as tf

from tf_agents.networks import categorical_projection_network
from tf_agents.networks import network
from tf_agents.networks import normal_projection_network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils

import dice_rl.utils.common as common_lib


def _categorical_projection_net(action_spec, logits_init_output_factor=0.1):
  return categorical_projection_network.CategoricalProjectionNetwork(
      action_spec, logits_init_output_factor=logits_init_output_factor)


def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1):
  std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      init_means_output_factor=init_means_output_factor,
      std_bias_initializer_value=std_bias_initializer_value,
      scale_distribution=False)


class PolicyNetwork(network.DistributionNetwork):
  """Creates a policy network."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               fc_layer_params=(200, 100),
               activation_fn=tf.nn.relu,
               output_activation_fn=None,
               kernel_initializer=None,
               last_kernel_initializer=None,
               discrete_projection_net=_categorical_projection_net,
               continuous_projection_net=_normal_projection_net,
               name='PolicyNetwork'):
    """Creates an instance of `ValueNetwork`.

    Args:
      input_tensor_spec: A possibly nested container of
        `tensor_spec.TensorSpec` representing the inputs.
      output_tensor_spec: A possibly nested container of
        `tensor_spec.TensorSpec` representing the outputs.
      fc_layer_params: Optional list of fully connected parameters after
        merging all inputs, where each item is the number of units
        in the layer.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      output_activation_fn: Activation function for the last layer. This can be
        used to restrict the range of the output. For example, one can pass
        tf.keras.activations.sigmoid here to restrict the output to be bounded
        between 0 and 1.
      kernel_initializer: kernel initializer for all layers except for the value
        regression layer. If None, a VarianceScaling initializer will be used.
      last_kernel_initializer: kernel initializer for the value regression
         layer. If None, a RandomUniform initializer will be used.
      discrete_projection_net: projection layer for discrete actions.
      continuous_projection_net: projection layer for continuous actions.
      name: A string representing name of the network.
    """

    def map_proj(spec):
      if tensor_spec.is_discrete(spec):
        return discrete_projection_net(spec)
      else:
        return continuous_projection_net(spec)

    projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
    output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                        projection_networks)
    if tensor_spec.is_discrete(output_tensor_spec):
      action_dim = np.unique(output_tensor_spec.maximum -
                             output_tensor_spec.minimum + 1)
    else:
      action_dim = output_tensor_spec.shape.num_elements()
    super(PolicyNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    self._flat_specs = tf.nest.flatten(input_tensor_spec)

    if kernel_initializer is None:
      kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(
          scale=1. / 3., mode='fan_in', distribution='uniform')
    if last_kernel_initializer is None:
      last_kernel_initializer = tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003)

    self._fc_layers = utils.mlp_layers(
        None,
        fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        name='mlp')
    self._fc_layers.append(
        tf.keras.layers.Dense(
            action_dim,
            activation=output_activation_fn,
            kernel_initializer=last_kernel_initializer,
            name='value'))

    self._projection_networks = projection_networks
    self._output_tensor_spec = output_tensor_spec

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self, inputs, step_type=(), network_state=(), training=False, mask=None):

    flat_inputs = tf.nest.flatten(inputs)
    del step_type  # unused.

    processed_inputs = []
    for single_input, input_spec in zip(flat_inputs, self._flat_specs):
      if common_lib.is_categorical_spec(input_spec):
        processed_input = tf.one_hot(single_input, input_spec.maximum + 1)
      else:
        if len(input_spec.shape) != 1:  # Only allow vector inputs.
          raise ValueError('Invalid input spec shape %s.' % input_spec.shape)
        processed_input = single_input
      processed_inputs.append(processed_input)

    joint = tf.concat(processed_inputs, -1)
    for layer in self._fc_layers:
      joint = layer(joint, training=training)

    outer_rank = nest_utils.get_outer_rank(inputs, self.input_tensor_spec)

    def call_projection_net(proj_net):
      distribution, _ = proj_net(
          joint, outer_rank, training=training, mask=mask)
      return distribution

    output_actions = tf.nest.map_structure(call_projection_net,
                                           self._projection_networks)
    return output_actions, network_state
