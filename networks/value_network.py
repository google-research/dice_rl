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

import tensorflow.compat.v2 as tf

from tf_agents.networks import network
from tf_agents.networks import utils

import dice_rl.utils.common as common_lib


class ValueNetwork(network.Network):
  """Creates a critic network."""

  def __init__(self,
               input_tensor_spec,
               fc_layer_params=None,
               activation_fn=tf.nn.relu,
               output_activation_fn=None,
               kernel_initializer=None,
               last_kernel_initializer=None,
               output_dim=None,
               name='ValueNetwork'):
    """Creates an instance of `ValueNetwork`.

    Args:
      input_tensor_spec: A possibly nested container of
        `tensor_spec.TensorSpec` representing the inputs.
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
      output_dim: If specified, the desired dimension of the output.
      name: A string representing name of the network.
    """
    super(ValueNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
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

    self._output_dim = output_dim
    self._fc_layers.append(
        tf.keras.layers.Dense(
            output_dim or 1,
            activation=output_activation_fn,
            kernel_initializer=last_kernel_initializer,
            name='value'))

  def call(self, inputs, step_type=(), network_state=(), training=False):
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

    if self._output_dim is None:
      joint = tf.reshape(joint, [-1])

    return joint, network_state
