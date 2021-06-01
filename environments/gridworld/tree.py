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
import gym
import numpy as np
import tensorflow.compat.v2 as tf

from gym import spaces
from gym.utils import seeding

import dice_rl.utils.common as common_utils


_Node = collections.namedtuple('_Node', ['rewards', 'transition_probabilities'])


class Tree(gym.Env):

  def __init__(self,
               branching=2,
               depth=10,
               duplicate=1,
               reward_power=3.0,
               reward_noise=1.0,
               transition_noise=0.2,
               tree_generation_seed=0,
               loop=False):
    self._branching = branching
    self._depth = depth
    self._duplicate = duplicate
    self._reward_power = reward_power
    self._reward_noise = reward_noise
    self._transition_noise = transition_noise
    self._loop = loop

    self._n_nodes = self._duplicate * (self._branching**self._depth)
    self._generate_tree(tree_generation_seed)

    self.observation_space = spaces.Discrete(self._n_nodes)
    self.action_space = spaces.Discrete(self._branching)

    self.seed()
    self.reset()

  def _generate_tree(self, seed):
    gen_random, _ = seeding.np_random(seed)

    self._nodes = []
    for node_idx in range(self._n_nodes // self._duplicate):
      rewards = gen_random.random_sample([self._branching])
      rewards = rewards ** self._reward_power
      rewards /= np.max(rewards)
      probabilities = (1 - self._transition_noise) * np.eye(
          self._branching, self._branching * self._duplicate)
      probabilities += self._transition_noise * gen_random.dirichlet(
          np.ones([self._branching * self._duplicate]), [self._branching])
      self._nodes.append(_Node(rewards, probabilities))

    return None

  @property
  def tree_nodes(self):
    return self._nodes

  @property
  def n_state(self):
    return self._n_nodes

  @property
  def n_action(self):
    return self._branching

  @property
  def duplicate(self):
    return self._duplicate

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    self._current_node_index = self.np_random.randint(self._duplicate)
    return self._get_obs()

  def _get_obs(self):
    return self._current_node_index

  def step(self, action):
    canonical_index = self._current_node_index // self._duplicate
    current_node = self._nodes[canonical_index]
    reward = current_node.rewards[action]
    reward += self.np_random.normal() * self._reward_noise
    probabilities = current_node.transition_probabilities[action]

    cum_probs = np.cumsum(probabilities, axis=-1)
    uniform_sample = self.np_random.random_sample()
    chosen_index = (uniform_sample < cum_probs).argmax(axis=-1)

    chosen_branch = chosen_index // self._duplicate
    chosen_duplicate = chosen_index % self._duplicate

    self._current_node_index = chosen_duplicate + self._duplicate * (
        self._branching * canonical_index + chosen_branch + 1)

    if self._loop:
      done = False
      if self._current_node_index >= self._n_nodes:
        self._current_node_index = self.np_random.randint(self._duplicate)
    else:
      done = ((self._current_node_index // self._duplicate + 1) *
              self._branching >= self._n_nodes // self._duplicate)

    return self._get_obs(), reward, done, {}


def _compute_near_optimal_actions(tree_nodes,
                                  branching,
                                  duplicate,
                                  root_index=0,
                                  optimal_values=None,
                                  optimal_actions=None):
  """A rough approximation to value iteration."""
  if optimal_values is None:
    optimal_values = {}
  if optimal_actions is None:
    optimal_actions = {}

  if (root_index // duplicate) * branching + 1 >= len(tree_nodes):
    optimal_values[root_index] = 0.
    return optimal_actions

  # Compute recursion.
  next_values = []
  for chosen_branch in range(branching):
    for chosen_duplicate in range(duplicate):
      next_index = chosen_duplicate + duplicate * (
          branching * (root_index // duplicate) + chosen_branch + 1)
      _compute_near_optimal_actions(
          tree_nodes,
          branching,
          duplicate,
          root_index=next_index,
          optimal_values=optimal_values,
          optimal_actions=optimal_actions)
      next_values.append(optimal_values[next_index])

  # Back up values.
  rewards = tree_nodes[root_index // duplicate].rewards
  probabilities = tree_nodes[root_index // duplicate].transition_probabilities
  qvalues = rewards + np.dot(probabilities, np.array(next_values))

  optimal_values[root_index] = np.max(qvalues)
  optimal_actions[root_index] = np.argmax(qvalues)

  if root_index == 0:
    for idx in range(1, duplicate):
      optimal_values[idx] = np.array(optimal_values[root_index])
      optimal_actions[idx] = np.array(optimal_actions[root_index])

  return optimal_actions


def get_tree_policy(tree_env, epsilon_explore=0.0, py=True,
                    return_distribution=True):
  """Creates a near-optimal policy for solving the tree environment.

  Args:
    tree_env: A tree environment.
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

  near_optimal_actions = _compute_near_optimal_actions(tree_env.tree_nodes,
                                                       tree_env.n_action,
                                                       tree_env.duplicate)
  policy_distribution = (
      np.ones((tree_env.n_state, tree_env.n_action)) / tree_env.n_action)
  for index, action in near_optimal_actions.items():
    policy_distribution[index] *= epsilon_explore
    policy_distribution[index, action] += 1 - epsilon_explore

  def obs_to_index_fn(observation):
    if py:
      return np.array(observation, dtype=np.int32)
    else:
      return tf.cast(observation, tf.int32)

  if py:
    return common_utils.create_py_policy_from_table(
        policy_distribution, obs_to_index_fn)
  else:
    return common_utils.create_tf_policy_from_table(
        policy_distribution, obs_to_index_fn,
        return_distribution=return_distribution)
