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

import collections
import numpy as np
import tensorflow.compat.v2 as tf

from typing import Any, Callable, List, Optional, Text, Tuple, Union

from dice_rl.data.dataset import EnvStep, OffpolicyDataset, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset

EpisodeInfo = collections.namedtuple(
    'EpisodeInfo', ['episode_start_id', 'episode_end_id',
                    'episode_start_type', 'episode_end_type'])


class PerturbedDataset(OffpolicyDataset):
  """An off-policy dataset with perturbed rewards."""

  def __init__(self,
               dataset: TFOffpolicyDataset,
               num_perturbations: Optional[int] = None,
               perturbation_scale: float = 1.):
    """Creates a PerturbedDataset.

    Args:
      dataset: The off-policy dataset to perturb.
      num_perturbations: How many perturbations to apply. Defaults to None
        (a single perturbation).
      perturbation_scale: Scale of random perturbations.
    """
    self._dataset = dataset
    if num_perturbations is not None and num_perturbations >= 64:
      raise ValueError('Number of permutations %d is too high.' % num_perturbations)
    self._num_perturbations = num_perturbations
    self._perturbation_scale = perturbation_scale

    with tf.device(self._dataset.device):
      self._random_numbers = tf.random.uniform([self._dataset.capacity])

  @property
  def spec(self):
    return self._dataset.spec

  @property
  def num_steps(self) -> Union[int, tf.Tensor]:
    return self._dataset.num_steps

  @property
  def num_total_steps(self) -> Union[int, tf.Tensor]:
    return self._dataset.num_total_steps

  @property
  def num_episodes(self) -> Union[int, tf.Tensor]:
    return self._dataset.num_episodes

  @property
  def num_total_episodes(self) -> Union[int, tf.Tensor]:
    return self._dataset.num_total_episodes

  @property
  def constructor_args_and_kwargs(self):
    args = [self._dataset]
    kwargs = {'num_perturbations': self._num_perturbation,
              'perturbation_scale': self._perturbation_scale}
    return args, kwargs

  def add_step(self, env_step: EnvStep):
    self._dataset.add_step(env_step)

  def _add_perturbations(self, env_step: EnvStep, last_rows_read: tf.Tensor):
    """Add history perturbations to rewards."""
    randoms = tf.gather(self._random_numbers, last_rows_read)
    num_perturbations = self._num_perturbations or 1
    perturbations = tf.cast(
        randoms[..., None] *
        tf.pow(2., 1 + tf.range(num_perturbations, dtype=tf.float32)),
        tf.int64)
    perturbations = tf.cast(tf.math.mod(perturbations, 2),
                            env_step.reward.dtype) - 0.5

    new_reward = (env_step.reward[..., None] +
                  self._perturbation_scale * perturbations)
    if self._num_perturbations is None:
      new_reward = tf.squeeze(new_reward, -1)
      new_discount = env_step.discount
    else:
      new_discount = env_step.discount[..., None]
    return env_step.write(reward=new_reward, discount=new_discount)

  def get_step(self, batch_size: Optional[int] = None,
               num_steps: Optional[int] = None) -> EnvStep:
    env_steps = self._dataset.get_step(batch_size, num_steps)
    return self._add_perturbations(env_steps, self._dataset.last_rows_read)

  def get_episode(self, batch_size: Optional[int] = None,
                  truncate_episode_at: Optional[int] = None) -> Tuple[
                      EnvStep, Union[np.ndarray, tf.Tensor]]:
    env_steps, valid_steps = self._dataset.get_episode(
        batch_size, truncate_episode_at)
    perturbed_steps = self._add_perturbations(
        env_steps, self._dataset.last_rows_read)
    return perturbed_steps, valid_steps

  def get_all_steps(self, num_steps: Optional[int] = None,
                    limit: Optional[int] = None) -> EnvStep:
    env_steps = self._dataset.get_all_steps(num_steps, limit)
    perturbed_steps = self._add_perturbations(
        env_steps, self._dataset.last_rows_read)
    return perturbed_steps

  def get_all_episodes(self, truncate_episode_at: Optional[int] = None,
                       limit: Optional[int] = None) -> Tuple[
                           EnvStep, Union[np.ndarray, tf.Tensor]]:
    env_steps, valid_steps = self._dataset.get_all_episodes(
        truncate_episode_at, limit)
    perturbed_steps = self._add_perturbations(
        env_steps, self._dataset.last_rows_read)
    return perturbed_steps, valid_steps
