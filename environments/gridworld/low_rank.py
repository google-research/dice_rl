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


class LowRank(gym.Env):

  def __init__(self,
               num_states=100,
               num_actions=4,
               rank=10,
               generation_seed=0,
               stochastic=False):

    self._num_states = num_states
    self._num_actions = num_actions
    self._rank = rank
    self._stochastic = stochastic

    self._transitions, self._rewards = self._generate_low_rank(generation_seed)

    self.observation_space = spaces.Discrete(self._num_states)
    self.action_space = spaces.Discrete(self._num_actions)

    self.seed()
    self.reset()

  def _generate_low_rank(self, seed):
    """Generate a low-rank transition matrix.

      Args:
        seed: Generation seed.

      Returns:
        transition matrix of shape S x S' x A
        reward of size S
    """
    gen_random, _ = seeding.np_random(seed)

    if self._stochastic:
      transitions = gen_random.uniform(
          size=[self._rank * self._num_actions, self._num_states])
      transitions /= np.sum(transitions, keepdims=True, axis=-1)
    else:
      transitions = np.zeros([self._rank * self._num_actions, self._num_states])
      next_idx = gen_random.randint(
          self._num_states, size=self._rank * self._num_actions)
      transitions[np.arange(self._rank * self._num_actions), next_idx] = 1.

    transitions = np.transpose(
        np.reshape(transitions,
                   [self._rank, self._num_actions, self._num_states]),
        [0, 2, 1])
    duplicates = transitions[
        gen_random.randint(self._rank, size=self._num_states - self._rank), ...]
    transitions = np.concatenate([transitions, duplicates])
    gen_random.shuffle(transitions)

    u, s, _ = np.linalg.svd(np.reshape(transitions, [self._num_states, -1]))
    rewards = np.dot(u[:, :self._rank], gen_random.uniform(size=self._rank))
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))

    return transitions, rewards

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    self._cur_state = self.np_random.randint(self._num_states)
    return self._get_obs()

  def _get_obs(self):
    return self._cur_state

  def step(self, action):
    self._cur_state = self.np_random.choice(
        self._num_states, 1, p=self._transitions[self._cur_state, :, action])[0]

    reward = self._rewards[self._cur_state]
    done = False
    return self._get_obs(), reward, done, {}


def _compute_near_optimal_actions(num_states, num_actions, transitions,
                                  rewards):
  vals = np.zeros([num_states, 1])
  eps = 0.001
  gamma = 0.99
  while True:
    delta = 0
    for state in range(num_states):
      tmp = vals[state].copy()
      vals[state] = np.max(
          np.sum((rewards[state] + gamma * vals) * transitions[state, ...], 0))
      delta = np.max([delta, np.abs(tmp - vals[state])])
    if delta <= eps * (1 - gamma) / gamma:
      break
  pi = np.zeros(num_states, dtype=np.int32)
  for state in range(num_states):
    pi[state] = np.argmax(np.sum(vals * transitions[state, ...], 0))
  return pi


def get_low_rank_policy(env,
                        epsilon_explore=0.0,
                        py=True,
                        return_distribution=True):
  """Creates a near-optimal policy for solving the low rank environment.

  Args:
    env: A low rank environment.
    epsilon_explore: Probability of sampling random action as opposed to optimal
      action.
    py: Whether to return Python policy (NumPy) or TF (Tensorflow).
    return_distribution: In the case of a TF policy, whether to return the full
      action distribution.

  Returns:
    A policy_fn that takes in an observation and returns a sampled action along
      with a dictionary containing policy information (e.g., log probability).
    A spec that determines the type of objects returned by policy_info.

  Raises:
    ValueError: If epsilon_explore is not a valid probability.
  """
  if epsilon_explore < 0 or epsilon_explore > 1:
    raise ValueError('Invalid exploration value %f' % epsilon_explore)

  near_optimal_actions = _compute_near_optimal_actions(env._num_states,
                                                       env._num_actions,
                                                       env._transitions,
                                                       env._rewards)

  policy_distribution = (
      np.ones((env._num_states, env._num_actions)) / env._num_actions)
  for index, action in enumerate(near_optimal_actions):
    policy_distribution[index] *= epsilon_explore
    policy_distribution[index, action] += 1 - epsilon_explore

  def obs_to_index_fn(observation):
    if py:
      return np.array(observation, dtype=np.int32)
    else:
      return tf.cast(observation, tf.int32)

  if py:
    return common_utils.create_py_policy_from_table(policy_distribution,
                                                    obs_to_index_fn)
  else:
    return common_utils.create_tf_policy_from_table(
        policy_distribution,
        obs_to_index_fn,
        return_distribution=return_distribution)
