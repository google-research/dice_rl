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


import gym
import numpy as np
import tensorflow.compat.v2 as tf
import os

from gym import spaces
from gym.utils import seeding

import dice_rl.utils.common as common_utils


class Taxi(gym.Env):

  def __init__(self, length=5, tabular_obs=True):
    self._length = length
    self._tabular_obs = tabular_obs
    self._possible_passenger_loc = [
        (0, 0), (0, length - 1), (length - 1, 0),
        (length - 1, length - 1)]
    self._passenger_status = np.random.randint(16)
    self._taxi_status = 4
    self._n_state = (length ** 2) * 16 * 5
    self._n_action = 6

    if self._tabular_obs:
      self.observation_space = spaces.Discrete(self._n_state)
    else:
      self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(23,))

    self.action_space = spaces.Discrete(self._n_action)

    self.seed()
    self.reset()

  @property
  def tabular_obs(self):
    return self._tabular_obs

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    length = self._length
    self._x = self.np_random.randint(length)
    self._y = self.np_random.randint(length)
    self._passenger_status = self.np_random.randint(16)
    self._taxi_status = 4
    return self._get_obs()

  def _get_obs(self):
    length = self._length
    if self._tabular_obs:
      return (self._taxi_status +
              5 * (self._passenger_status +
                   16 * (self._x * self._length + self._y)))
    else:
      xy = np.array([self._x, self._y])
      taxi = np.equal(np.arange(5), self._taxi_status).astype(np.int32)
      passenger = np.equal(np.arange(16), self._passenger_status).astype(np.int32)
      return np.concatenate([xy, taxi, passenger], -1)

  def get_tabular_obs(self, status_obs, py):
    x = status_obs[..., 0]
    y = status_obs[..., 1]
    taxi_status = np.argmax(status_obs[..., 2:2 + 5], -1)
    if py:
      passenger_status = np.argmax(status_obs[..., 7:7 + 16], -1)
      return (taxi_status + 5 * (passenger_status + 16 *
                                 (x * self._length + y)))
    else:
      taxi_status = tf.math.argmax(status_obs[..., 2:2 + 5], axis=-1)
      passenger_status = tf.math.argmax(status_obs[..., 7:7 + 16], axis=-1)
      return (tf.cast(taxi_status, tf.float32) + 5 *
              (tf.cast(passenger_status, dtype=tf.float32) + 16 *
               (x * self._length + y)))

  def get_status_obs(self, state):
    length = self._length
    taxi_status = state % 5
    state = state / 5
    passenger_status = state % 16
    state = state / 16
    y = state % length
    x = state / length
    return x, y, passenger_status, taxi_status

  def step(self, action):
    reward = -1
    length = self._length
    if action == 0:
      if self._x < self._length - 1:
        self._x += 1
    elif action == 1:
      if self._y < self._length - 1:
        self._y += 1
    elif action == 2:
      if self._x > 0:
        self._x -= 1
    elif action == 3:
      if self._y > 0:
        self._y -= 1
    elif action == 4:  # Try to pick up
      for i in range(4):
        x, y = self._possible_passenger_loc[i]
        if x == self._x and y == self._y and (self._passenger_status & (1 << i)):
          # Pick up passenger.
          self._passenger_status -= 1 << i
          # Choose drop-off location.
          self._taxi_status = self.np_random.randint(4)
          while self._taxi_status == i:
            self._taxi_status = self.np_random.randint(4)
    elif action == 5:  # Drop-off.
      if self._taxi_status < 4:
        x, y = self._possible_passenger_loc[self._taxi_status]
        if self._x == x and self._y == y:
          reward = 20
        self._taxi_status = 4
    else:
      raise ValueError('Invalid action %s.' % action)
    self._change_passenger_status()
    done = False
    return self._get_obs(), reward, done, {}

  def _change_passenger_status(self):
    """Updates passenger locations stochastically."""
    p_generate = [0.3, 0.05, 0.1, 0.2]
    p_disappear = [0.05, 0.1, 0.1, 0.05]
    for i in range(4):
      if self._passenger_status & (1 << i):
        if self.np_random.rand() < p_disappear[i]:
          self._passenger_status -= 1 << i
      else:
        if self.np_random.rand() < p_generate[i]:
          self._passenger_status += 1 << i


def _get_taxi_policy(load_dir, file_id):
  filename = os.path.join(load_dir, 'taxi', 'pi%d.npy' % file_id)
  with tf.io.gfile.GFile(filename, 'rb') as f:
    policy = np.load(f)
  return policy


def get_taxi_policy(load_dir,
                    taxi_env,
                    alpha=1.0,
                    py=True,
                    return_distribution=True):
  """Creates a policy for solving the Taxi environment.

  Args:
    load_dir: Directory to load policy from.
    taxi_env: A Taxi environment.
    alpha: A number between 0 and 1 determining how close the policy is to the
      target (near-optimal) policy. The higher alpha, the closer it is to the
      target.
    py: Whether to return Python policy (NumPy) or TF (Tensorflow).
    return_distribution: In the case of a TF policy, whether to return the
      full action distribution.

  Returns:
    A policy_fn that takes in an observation and returns a sampled action along
      with a dictionary containing policy information (e.g., log probability).
    A spec that determines the type of objects returned by policy_info.

  Raises:
    ValueError: If alpha is not in [0, 1].
  """
  if alpha < 0 or alpha > 1:
    raise ValueError('Invalid alpha value %f' % alpha)

  baseline_policy = _get_taxi_policy(load_dir, 18)
  target_policy = _get_taxi_policy(load_dir, 19)

  policy_distribution = (
      (1 - alpha) * baseline_policy + alpha * target_policy)

  def obs_to_index_fn(observation):
    if not taxi_env.tabular_obs:
      state = taxi_env.get_tabular_obs(observation, py)
    else:
      state = observation
    if py:
      return state.astype(np.int32)
    else:
      return tf.cast(state, tf.int32)

  if py:
    return common_utils.create_py_policy_from_table(
        policy_distribution, obs_to_index_fn)
  else:
    return common_utils.create_tf_policy_from_table(
        policy_distribution, obs_to_index_fn,
        return_distribution=return_distribution)
