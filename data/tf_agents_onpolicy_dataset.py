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

from typing import Any, Callable, List, Optional, Tuple, Union

from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step
from tf_agents.environments import tf_environment
from tf_agents.policies import tf_policy
from tf_agents import specs
from tf_agents.utils import nest_utils

from dice_rl.data.dataset import OnpolicyDataset, EnvStep, StepType

# pytype: disable=attribute-error


class TFAgentsOnpolicyDataset(OnpolicyDataset):
  """On-policy dataset for TF environment and TF-Agents policy."""

  def __init__(self,
               tf_env: tf_environment.TFEnvironment,
               policy: tf_policy.TFPolicy,
               episode_step_limit: Optional[int] = None):
    """Creates the stepper.

    Args:
      tf_env: A TF environment.
      policy: The behavior policy.
      episode_step_limit: If specified, episodes are terminated after this many
        steps. Note that this leads to episodes with n+1 observations.
    """
    self._env = tf_env
    self._policy = policy
    self._episode_step_limit = episode_step_limit

    if self._env.batch_size not in [1, None]:
      raise ValueError('Batched environments are not supported.')

    self._spec = self._create_spec()

    self._num_steps = 0
    self._num_total_steps = 0
    self._num_episodes = 0
    self._num_total_episodes = 0
    self._start_on_next_step = True

  def _create_spec(self):
    observation_spec = self._env.observation_spec()
    action_spec = self._env.action_spec()

    tf_agents_time_step_spec = time_step.time_step_spec(observation_spec)
    step_num_spec = specs.tensor_spec.from_spec(
        specs.BoundedArraySpec([],
                               dtype=np.int64,
                               minimum=0,
                               maximum=self._episode_step_limit,
                               name='step_num'))
    return EnvStep(tf_agents_time_step_spec.step_type, step_num_spec,
                   observation_spec, action_spec,
                   tf_agents_time_step_spec.reward,
                   tf_agents_time_step_spec.discount, self._policy.info_spec,
                   {}, {})

  @property
  def spec(self) -> EnvStep:
    return self._spec

  @property
  def num_steps(self) -> Union[int, tf.Tensor]:
    return self._num_steps

  @property
  def num_total_steps(self) -> Union[int, tf.Tensor]:
    return self._num_total_steps

  @property
  def num_episodes(self) -> Union[int, tf.Tensor]:
    return self._num_episodes

  @property
  def num_total_episodes(self) -> Union[int, tf.Tensor]:
    return self._num_total_episodes

  @property
  def constructor_args_and_kwargs(self):
    args = [self._env, self._policy]
    kwargs = {'episode_step_limit': self._episode_step_limit}
    return args, kwargs

  def _start_new_episode(self):
    self._time_step = self._env.reset()
    if self._env.batch_size is not None:
      self._time_step = nest_utils.unbatch_nested_tensors(self._time_step)
    self._step_type = self._time_step.step_type
    self._discount = self._time_step.discount
    self._first_step_type = self._step_type
    self._policy_state = self._policy.get_initial_state(None)
    self._start_on_next_step = False
    self._cur_step_num = 0

  def _get_step(self) -> EnvStep:
    if self._start_on_next_step:
      self._start_new_episode()

    if StepType.is_last(self._step_type):
      # This is the last (terminating) observation of the environment.
      self._start_on_next_step = True
      self._num_total_steps += 1
      self._num_episodes += 1
      # The policy is not run on the terminal step, so we just carry over the
      # reward, action, and policy_info from the previous step.
      return EnvStep(self._step_type,
                     tf.cast(self._cur_step_num, dtype=tf.int64),
                     self._time_step.observation, self._action,
                     self._time_step.reward, self._time_step.discount,
                     self._policy_info, {}, {})

    self._action, self._policy_state, self._policy_info = self._policy.action(
        self._time_step, self._policy_state)

    # Update type of log-probs to tf.float32... a bit of a bug in TF-Agents.
    if hasattr(self._policy_info, 'log_probability'):
      self._policy_info = policy_step.set_log_probability(
          self._policy_info,
          tf.cast(self._policy_info.log_probability, tf.float32))

    # Sample action from policy.
    env_action = self._action
    if self._env.batch_size is not None:
      env_action = nest_utils.batch_nested_tensors(env_action)

    # Sample next step from environment.
    self._next_time_step = self._env.step(env_action)
    if self._env.batch_size is not None:
      self._next_time_step = nest_utils.unbatch_nested_tensors(
          self._next_time_step)
    self._next_step_type = self._next_time_step.step_type
    self._cur_step_num += 1
    if (self._episode_step_limit and
        self._cur_step_num >= self._episode_step_limit):
      self._next_step_type = tf.convert_to_tensor(  # Overwrite step type.
          value=StepType.LAST, dtype=self._first_step_type.dtype)
      self._next_step_type = tf.reshape(self._next_step_type,
                                        tf.shape(self._first_step_type))

    step = EnvStep(self._step_type,
                   tf.cast(self._cur_step_num - 1, tf.int64),
                   self._time_step.observation, self._action,
                   # Immediate reward given by next time step.
                   self._next_time_step.reward,
                   self._time_step.discount,
                   self._policy_info, {}, {})

    self._num_steps += 1
    self._num_total_steps += 1
    if StepType.is_first(self._step_type):
      self._num_total_episodes += 1

    self._time_step = self._next_time_step
    self._step_type = self._next_step_type

    return step

  def get_step(self, batch_size: Optional[int] = None,
               num_steps: Optional[int] = None) -> EnvStep:
    if batch_size is not None:
      raise ValueError('This dataset does not support batched step sampling.')

    if num_steps is None:
      return self._get_step()

    env_steps = []
    for _ in range(num_steps):
      next_step = self._get_step()
      env_steps.append(next_step)

    return nest_utils.stack_nested_tensors(env_steps)

  def _get_episode(self, truncate_episode_at: Optional[int] = None) -> List[
      EnvStep]:

    self._start_new_episode()
    env_steps = []
    while True:
      next_step = self._get_step()
      env_steps.append(next_step)
      if next_step.is_last():
        break
      if truncate_episode_at and len(env_steps) >= truncate_episode_at:
        break

    return env_steps

  def get_episode(self, batch_size: Optional[int] = None,
                  truncate_episode_at: Optional[int] = None) -> Tuple[
                      EnvStep, np.ndarray]:
    if batch_size is None:
      episode = self._get_episode(truncate_episode_at)
      mask = np.ones((len(episode),))
      return nest_utils.stack_nested_tensors(episode), mask
    if batch_size <= 0:
      raise ValueError('Invalid batch size %s.' % batch_size)

    episodes = []
    episode_lengths = []
    for _ in range(batch_size):
      next_episode = self._get_episode(truncate_episode_at)
      episodes.append(next_episode)
      episode_lengths.append(len(next_episode))

    max_length = max(episode_lengths)
    for episode in episodes:
      episode.extend([episode[-1]] * (max_length - len(episode)))

    batched_episodes = nest_utils.stack_nested_tensors(
        [nest_utils.stack_nested_tensors(episode)
         for episode in episodes])

    valid_steps = (tf.range(max_length)[None, :] <
                   tf.convert_to_tensor(episode_lengths)[:, None])

    return batched_episodes, valid_steps
