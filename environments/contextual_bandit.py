from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gym
import itertools
import numpy as np
import tensorflow.compat.v2 as tf

from gym import spaces
from gym.utils import seeding

import dice_rl.utils.common as common_utils


class ContextualBandit(gym.Env):

  def __init__(self,
               num_arms=2,
               num_rewards=None,
               generation_seed=0,
               loop=False):
    self._num_arms = num_arms
    self._num_rewards = num_rewards or self._num_arms
    assert (self._num_rewards <= self._num_arms)
    self._num_contexts = np.math.factorial(self._num_arms)
    self._loop = loop
    self._generate_bandit(generation_seed)

    self.observation_space = spaces.Discrete(self._num_contexts)
    self.action_space = spaces.Discrete(self._num_arms)

    self.seed()
    self.reset()

  def _generate_bandit(self, seed):
    gen_random, _ = seeding.np_random(seed)

    rewards = gen_random.choice(
        range(self._num_rewards),
        self._num_arms,
        replace=(self._num_rewards < self._num_arms))
    self._rewards = np.asarray(list(itertools.permutations(rewards)))

  @property
  def rewards(self):
    return self._rewards

  @property
  def num_arms(self):
    return self._num_arms

  @property
  def num_contexts(self):
    return self._num_contexts

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    self._current_context = self.np_random.randint(self._num_contexts)
    return self._get_obs()

  def _get_obs(self):
    return self._current_context

  def step(self, action):
    reward = self._rewards[self._current_context][action]
    done = not self._loop
    return self._get_obs(), reward, done, {}


def get_contextual_bandit_policy(contextual_bandit_env,
                                 epsilon_explore=0.0,
                                 py=True,
                                 return_distribution=True):
  """Creates an optimal policy for solving the contextual bandit environment.

  Args:
    contextual_bandit_env: A contextual bandit environment.
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

  optimal_action = np.argmax(contextual_bandit_env.rewards, axis=-1)
  policy_distribution = np.ones([
      contextual_bandit_env.num_contexts, contextual_bandit_env.num_arms
  ]) / contextual_bandit_env.num_arms
  policy_distribution *= epsilon_explore
  policy_distribution[np.arange(policy_distribution.shape[0]),
                      optimal_action] += 1 - epsilon_explore

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
