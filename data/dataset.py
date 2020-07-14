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

import abc
import collections
import numpy as np
import os
import pickle
import tensorflow.compat.v2 as tf

import six
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from tf_agents.trajectories import time_step
from tf_agents.environments import gym_wrapper
from tf_agents.utils import nest_utils

CONSTRUCTOR_PREFIX = 'dataset-ctr.pkl'
CHECKPOINT_PREFIX = 'dataset-ckpt'


class StepType(object):
  """Defines the type of step (first/mid/last) with some basic utilities."""
  FIRST = time_step.StepType.FIRST
  MID = time_step.StepType.MID
  LAST = time_step.StepType.LAST

  def __init__(self, step_type):
    self._step_type = step_type

  @classmethod
  def is_first(cls, val):
    if tf.is_tensor(val):
      return tf.equal(val, StepType.FIRST)
    return np.equal(val, StepType.FIRST)

  @classmethod
  def is_mid(cls, val):
    if tf.is_tensor(val):
      return tf.equal(val, StepType.MID)
    return np.equal(val, StepType.MID)

  @classmethod
  def is_last(cls, val):
    if tf.is_tensor(val):
      return tf.equal(val, StepType.LAST)
    return np.equal(val, StepType.LAST)


_EnvStep = collections.namedtuple(
    '_EnvStep', ['step_type', 'step_num', 'observation', 'action', 'reward',
                 'discount', 'policy_info', 'env_info', 'other_info'])


class EnvStep(_EnvStep):
  """A tuple containing the relevant information for a single environment step."""

  def is_first(self):
    return StepType.is_first(self.step_type)

  def is_mid(self):
    return StepType.is_mid(self.step_type)

  def is_last(self):
    return StepType.is_last(self.step_type)

  def is_absorbing(self):
    """Checks if step is an absorbing (terminal) step of an episode."""
    if tf.is_tensor(self.discount):
      return tf.logical_and(
          tf.equal(self.discount, tf.constant(0, self.discount.dtype)),
          self.is_last())
    return np.logical_and(
        np.equal(self.discount, 0),
        self.is_last())

  def has_log_probability(self):
    return 'log_probability' in self.policy_info

  def get_log_probability(self):
    if isinstance(self.policy_info, dict):
      if not self.has_log_probability():
        raise ValueError('No log probability in this EnvStep.')
      return self.policy_info['log_probability']
    else:
      if not hasattr(self.policy_info, 'log_probability'):
        raise ValueError('No log probability in this EnvStep.')
      return self.policy_info.log_probability

  def write(self, **kwargs):
    """Creates a new EnvStep with appropriate fields over-written."""
    new_fields = {}
    for field_name in self._fields:
      if field_name in kwargs:
        new_fields[field_name] = kwargs[field_name]
      else:
        new_fields[field_name] = getattr(self, field_name)

    return EnvStep(**new_fields)


def convert_to_tfagents_timestep(env_step: EnvStep):
  """Converts an EnvStep to a tf_agents.TimeStep.

  Args:
    env_step: An instance of EnvStep.

  Returns:
    A representation of env_step as a tf_agents.TimeStep.
  """
  #TODO(ofirnachum): Handle batched env_steps appropriately.
  return time_step.TimeStep(
      env_step.step_type,
      env_step.reward,
      env_step.discount,
      env_step.observation)


@six.add_metaclass(abc.ABCMeta)
class Dataset(object):
  """Abstract class for on or off-policy dataset."""

  @property
  @abc.abstractmethod
  def spec(self) -> EnvStep:
    """Returns the spec (expected shape and type) of steps."""

  @property
  @abc.abstractmethod
  def num_steps(self) -> Union[int, tf.Tensor]:
    """Returns the number of steps in the dataset.

    Following standard convention, this number excludes terminal steps in the
    episodes. The last step in an episode is typically the last observation and
    no action is taken or reward received.
    """
  @property
  @abc.abstractmethod
  def num_total_steps(self) -> Union[int, tf.Tensor]:
    """Returns the total, unfiltered number of steps in the dataset."""

  @property
  @abc.abstractmethod
  def num_episodes(self) -> Union[int, tf.Tensor]:
    """Returns the number of completed episodes in the dataset.

    Returns the number of completed episodes, meaning contiguous sequence of
    steps starting with StepType.FIRST and ending with StepType.LAST.
    """

  @property
  @abc.abstractmethod
  def num_total_episodes(self) -> Union[int, tf.Tensor]:
    """Returns the number of partial or completed episodes in the dataset."""

  @property
  @abc.abstractmethod
  def constructor_args_and_kwargs(self):
    """Returns args and kwargs to construct a new verion of this dataset."""

  @abc.abstractmethod
  def get_step(self, batch_size: Optional[int] = None,
               num_steps: Optional[int] = None) -> EnvStep:
    """Sample a number of steps from the environment.

    Args:
      batch_size: The desired returned batch size. Defaults to None (unbatched).
      num_steps: The number of desired contiguous steps. Defaults to None
        (single step).

    Returns:
      The step or steps collected in a single EnvStep. The elements of the
        EnvStep will have shape [batch_size, num_steps, ...].
    """

  @abc.abstractmethod
  def get_episode(self, batch_size: Optional[int] = None,
                  truncate_episode_at: Optional[int] = None) -> Tuple[
                      EnvStep, Union[np.ndarray, tf.Tensor]]:
    """Performs steps through the environment to yield full episodes.

    Args:
      batch_size: The desired returned batch size. Defaults to None (unbatched).
      truncate_episode_at: If specified, episodes are cut-off after this many
        steps. If left unspecified, episodes are only cut-off when a step is
        encountered with step_type.last() == True.

    Returns:
      env_step: An EnvStep tuple with the steps of all collected episodes
        appropriately batched. That is, if batch_size is unspecified, the
        env_step will have members of shape [T, ...], whereas if multiple
        episodes are collected, the env_step will have members of shape
        [B, T, ...].
      valid_steps: A mask (array or tensor of True/False) that tells which env
        steps are valid; for example, if two episodes are collected and one is
        shorter than the other. If batch_size is unspecified, valid_steps
        will have shape [T], whereas if multiple episodes are collected, it
        will have shape [B, T].
    """

  def save(self, directory, checkpoint=None):
    """Saves this dataset to a directory."""
    args, kwargs = self.constructor_args_and_kwargs
    constructor_info = {
        'type': type(self),
        'args': args,
        'kwargs': kwargs}

    pickle_filename = os.path.join(directory, CONSTRUCTOR_PREFIX)
    checkpoint_filename = os.path.join(directory, CHECKPOINT_PREFIX)

    with tf.io.gfile.GFile(pickle_filename, 'w') as f:
      try:
        pickle.dump(constructor_info, f)
      except pickle.PicklingError:
        raise ValueError('Dataset constructor info does not pickle: %s' %
                         constructor_info)

    if checkpoint is None:
      checkpoint = tf.train.Checkpoint(dataset=self)

    checkpoint.save(checkpoint_filename)

  @classmethod
  def load(cls, directory):
    """Loads a dataset from a directory."""
    pickle_filename = os.path.join(directory, CONSTRUCTOR_PREFIX)
    checkpoint_filename = tf.train.latest_checkpoint(directory)

    if not tf.io.gfile.exists(pickle_filename):
      raise ValueError('No file with constructor info exists: %s' %
                       pickle_filename)

    if not checkpoint_filename or not checkpoint_filename.startswith(
        os.path.join(directory, CHECKPOINT_PREFIX)):
      raise ValueError('No suitable checkpoint found in %s.' %
                       directory)

    with tf.io.gfile.GFile(pickle_filename, 'rb') as f:
      print(pickle_filename)
      constructor_info = pickle.load(f)

    dataset = constructor_info['type'](
        *constructor_info['args'],
        **constructor_info['kwargs'])

    checkpoint = tf.train.Checkpoint(dataset=dataset)
    checkpoint.restore(checkpoint_filename)

    return dataset


@six.add_metaclass(abc.ABCMeta)
class OnpolicyDataset(Dataset):
  """Abstract class for on-policy dataset.

  An on-policy dataset includes an environment and a policy. Whenever a step
  or episode is requested, the environment is sampled directly to provide this
  experience.
  """


@six.add_metaclass(abc.ABCMeta)
class OffpolicyDataset(Dataset):
  """Abstract class for off-policy dataset.

  An off-policy dataset provides steps or episodes randomly sampled from a
  potentially growing storage of experience.
  """

  @abc.abstractmethod
  def add_step(self, env_step: EnvStep):
    """Adds a potentially batched step of experience into the dataset.

    Args:
      env_step: Experience to add to the dataset. Potentially batched.
    """

  @abc.abstractmethod
  def get_all_steps(self, num_steps: Optional[int] = None,
                    limit: Optional[int] = None) -> EnvStep:
    """Gets all the non-terminal steps in the dataset.

    Args:
      num_steps: The number of desired contiguous steps. Defaults to None
        (single step).
      limit: If specified, only return at most this many steps.

    Returns:
      The steps collected in a single EnvStep.
    """

  @abc.abstractmethod
  def get_all_episodes(self, truncate_episode_at: Optional[int] = None,
                       limit: Optional[int] = None) -> Tuple[
                      EnvStep, Union[np.ndarray, tf.Tensor]]:
    """Gets all full or partial episodes in the dataset.

    Args:
      truncate_episode_at: If specified, episodes are cut-off after this many
        steps. If left unspecified, episodes are only cut-off when a step is
        encountered with step_type.last() == True.
      limit: If specified, only return at most this many episodes.

    Returns:
      env_step: An EnvStep tuple with the steps of all collected episodes.
      valid_steps: A mask (array or tensor of True/False) that tells which env
        steps are valid.
    """
