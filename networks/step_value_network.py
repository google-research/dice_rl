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

import dice_rl.utils.common as common_lib
from dice_rl.networks.value_network import ValueNetwork


class StepValueNetwork(ValueNetwork):
  """Creates a step-aware critic network."""

  def __init__(self,
               input_tensor_spec,
               max_trajectory_length_train=None,
               step_encoding='one_hot',
               d_step_emb=64,
               name='StepValueNetwork',
               **kwargs):
    """Creates an instance of `StepValueNetwork`.

    Args:
      input_tensor_spec: A possibly nested container of `tensor_spec.TensorSpec`
        representing the inputs.
      max_trajectory_length_train: If specified, use max_trajectory_length_train
        as the step dimension.
      step_encoding: The step encoding to use. One of ["one_hot", "sinusoid",
        "learned"] or None. If None, step numbers are ignored and
        StepValueNetwork behaves the same as ValueNetwork.
      d_step_emb: Dimension of the step embedding.
      name: A string representing name of the network.
    """
    super(StepValueNetwork, self).__init__(
        input_tensor_spec, name=name, **kwargs)

    self._max_trajectory_length_train = max_trajectory_length_train
    self._step_encoding = step_encoding
    self._d_step_emb = d_step_emb

    if self._step_encoding == 'learned':
      self._step_embedding_layer = tf.keras.layers.Dense(
          self._d_step_emb, name='step_embedding')

  def _process_step_num(self, single_input, max_step):
    if self._step_encoding == 'one_hot':
      return tf.one_hot(single_input, max_step + 1)
    if self._step_encoding == 'sinusoid':
      i = tf.range(self._d_step_emb, dtype=tf.float32)[tf.newaxis, :]
      step_num = tf.cast(single_input, tf.float32)[:, tf.newaxis]
      rads = step_num / tf.math.pow(
          1.0e4, 2 * (i // 2) / tf.cast(self._d_step_emb, tf.float32))
      return tf.concat([tf.sin(rads[:, 0::2]), tf.cos(rads[:, 1::2])], axis=-1)
    if self._step_encoding == 'learned':
      return self._step_embedding_layer(tf.one_hot(single_input, max_step + 1))
    raise ValueError(
        'Step encoding must be one of ["one_hot, "sinusoid", "learned"].')

  def call(self, inputs, step_type=(), network_state=(), training=False):
    flat_inputs = tf.nest.flatten(inputs)
    del step_type  # unused.

    processed_inputs = []
    for single_input, input_spec in zip(flat_inputs, self._flat_specs):
      if common_lib.is_categorical_spec(input_spec):
        if input_spec.name == 'step_num':
          if self._step_encoding is None:
            continue

          if self._max_trajectory_length_train is not None:
            max_step = self._max_trajectory_length_train
          else:
            max_step = input_spec.maximum
          processed_input = self._process_step_num(single_input, max_step)
        else:
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
