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

from gym import spaces
from gym.utils import seeding

import dice_rl.utils.common as common_utils


class Navigation(gym.Env):
  def __init__(self, nav_map, tabular_obs=True,
               reward_fn=None, done_fn=None):
    self._map = nav_map
    self._tabular_obs = tabular_obs
    self._reward_fn = reward_fn
    self._done_fn = done_fn
    if self._reward_fn is None:
      self._reward_fn = lambda x, y, tx, ty: float(x == tx and y == ty)
    if self._done_fn is None:
      self._done_fn = lambda x, y, tx, ty: False

    self._max_x = len(self._map)
    if not self._max_x:
      raise ValueError('Invalid map.')
    self._max_y = len(self._map[0])
    if not all(len(m) == self._max_y for m in self._map):
      raise ValueError('Invalid map.')

    self._start_x, self._start_y = self._find_initial_point()
    self._target_x, self._target_y = self._find_target_point()

    self._n_state = self._max_x * self._max_y
    self._n_action = 4

    if self._tabular_obs:
      self.observation_space = spaces.Discrete(self._n_state)
    else:
      self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(2,))

    self.action_space = spaces.Discrete(self._n_action)

    self.seed()
    self.reset()

  @property
  def nav_map(self):
    return self._map

  @property
  def n_state(self):
    return self._n_state

  @property
  def n_action(self):
    return self._n_action

  @property
  def target_location(self):
    return self._target_x, self._target_y

  @property
  def tabular_obs(self):
    return self._tabular_obs

  def _find_initial_point(self):
    for x in range(self._max_x):
      for y in range(self._max_y):
        if self._map[x][y] == 'S':
          break
      if self._map[x][y] == 'S':
        break
    else:
      return None, None

    return x, y

  def _find_target_point(self):
    for x in range(self._max_x):
      for y in range(self._max_y):
        if self._map[x][y] == 'T':
          break
      if self._map[x][y] == 'T':
        break
    else:
      raise ValueError('Target point not found in map.')

    return x, y

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    if self._start_x is not None and self._start_y is not None:
      self._x, self._y = self._start_x, self._start_y
    else:
      while True:  # Find empty grid cell.
        self._x = self.np_random.randint(self._max_x)
        self._y = self.np_random.randint(self._max_y)
        if self._map[self._x][self._y] != 'x':
          break
    return self._get_obs()

  def _get_obs(self):
    if self._tabular_obs:
      return self._x * self._max_y + self._y
    else:
      return np.array([self._x, self._y])

  def get_tabular_obs(self, status_obs):
    return self._max_y * status_obs[..., 0] + status_obs[..., 1]

  def get_xy(self, state):
    x = state / self._max_y
    y = state % self._max_y
    return x, y

  def step(self, action):
    #TODO(ofirnachum): Add stochasticity.
    last_x, last_y = self._x, self._y
    if action == 0:
      if self._x < self._max_x - 1:
        self._x += 1
    elif action == 1:
      if self._y < self._max_y - 1:
        self._y += 1
    elif action == 2:
      if self._x > 0:
        self._x -= 1
    elif action == 3:
      if self._y > 0:
        self._y -= 1

    if self._map[self._x][self._y] == 'x':
      self._x, self._y = last_x, last_y

    reward = self._reward_fn(self._x, self._y, self._target_x, self._target_y)
    done = self._done_fn(self._x, self._y, self._target_x, self._target_y)
    return self._get_obs(), reward, done, {}


class GridWalk(Navigation):
  """Walk on grid to target location."""
  def __init__(self, length=10, tabular_obs=True):
    nav_map = [[' ' for _ in range(length)]
               for _ in range(length)]
    nav_map[-1][-1] = 'T'
    self._length = length

    def reward_fn(x, y, tx, ty):
      taxi_distance = np.abs(x - tx) + np.abs(y - ty)
      return np.exp(-2 * taxi_distance / length)

    super(GridWalk, self).__init__(nav_map, tabular_obs=tabular_obs,
                                   reward_fn=reward_fn)


class FourRooms(Navigation):
  def __init__(self, tabular_obs=True):
    nav_map = [[' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               ['x', ' ', 'x', 'x', 'x', 'x', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', 'x', 'x', 'x', ' ', 'x', 'x'],
               [' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'T', ' '],
               [' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' '],
               ]
    super(FourRooms, self).__init__(nav_map, tabular_obs=tabular_obs)


def _compute_near_optimal_actions(nav_map, target_location):
  """A rough approximation to value iteration."""
  current_points = [target_location]
  chosen_actions = {target_location: 0}
  visited_points = {target_location: True}

  while current_points:
    next_points = []
    for point_x, point_y in current_points:
      for (action, (next_point_x, next_point_y)) in [
          (0, (point_x - 1, point_y)), (1, (point_x, point_y - 1)),
          (2, (point_x + 1, point_y)), (3, (point_x, point_y + 1))]:

        if (next_point_x, next_point_y) in visited_points:
          continue

        if not (next_point_x >= 0 and next_point_y >= 0 and
                next_point_x < len(nav_map) and
                next_point_y < len(nav_map[next_point_x])):
          continue

        if nav_map[next_point_x][next_point_y] == 'x':
          continue

        next_points.append((next_point_x, next_point_y))
        visited_points[(next_point_x, next_point_y)] = True
        chosen_actions[(next_point_x, next_point_y)] = action

    current_points = next_points

  return chosen_actions


def get_navigation_policy(nav_env, epsilon_explore=0.0, py=True,
                          return_distribution=True):
  """Creates a near-optimal policy for solving the navigation environment.

  Args:
    nav_env: A navigation environment.
    epsilon_explore: Probability of sampling random action as opposed to optimal
      action.
    py: Whether to return Python policy (NumPy) or TF (Tensorflow).
    return_distribution: In the case of a TF policy, whether to return the
      full action distribution.

  Returns:
    A policy_fn that takes in an observation and returns a sampled action along
      with a dictionary containing policy information (e.g., log probability).
    A spec that determines the type of objects returned by policy_info.

  Raises:
    ValueError: If epsilon_explore is not a valid probability.
  """
  if epsilon_explore < 0 or epsilon_explore > 1:
    raise ValueError('Invalid exploration value %f' % epsilon_explore)

  near_optimal_actions = _compute_near_optimal_actions(
      nav_env.nav_map, nav_env.target_location)
  policy_distribution = (
      np.ones((nav_env.n_state, nav_env.n_action)) / nav_env.n_action)
  for location, action in near_optimal_actions.items():
    tabular_id = nav_env.get_tabular_obs(np.array(location))
    policy_distribution[tabular_id] *= epsilon_explore
    policy_distribution[tabular_id, action] += 1 - epsilon_explore

  def obs_to_index_fn(observation):
    if not nav_env.tabular_obs:
      state = nav_env.get_tabular_obs(observation)
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
