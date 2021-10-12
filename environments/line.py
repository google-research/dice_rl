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
from tf_agents.environments import tf_py_environment
import tensorflow_probability as tfp

tfd = tfp.distributions
from tf_agents.environments import gym_wrapper
from tf_agents import specs


class Line(gym.Env):

  def __init__(self, left=-5., right=5., random_start=True, generation_seed=0):
    self.observation_space = spaces.Box(low=left, high=right, shape=(1,))
    self.action_space = spaces.Box(low=-1., high=1., shape=(1,))
    self.random_start = random_start
    self.left = left
    self.right = right
    self.seed()
    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    if self.random_start:
      self._x = self.np_random.uniform(self.left, self.right)
    else:
      self._x = (self.left + self.right) / 2.
    return self._get_obs()

  def _get_obs(self):
    return self._x

  def step(self, action):
    self._x = self._x + action
    reward = 0.
    done = False
    if np.abs(self._x - self.left) < 0.05 or np.abs(self._x -
                                                    self.right) < 0.05:
      self.reset()
    return self._get_obs(), reward, done, {}


def get_line_env_and_policy(env_seed, scale=[0.05, 0.05]):
  """Creates a policy for the number line environment.

  Args:
    line_env: A line environment.
    action_mean_fn: Means of multi-modal actions given state.
    sigma: Standard deviation of the actions.

  Returns:
    A tf_env.
    A TFPolicy

  Raises:
    ValueError: If epsilon_explore is not a valid probability.
  """
  env = Line()
  env.seed(env_seed)
  tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))

  def policy_fn(observation, dtype=tf.float32):
    #if tf.rank(observation) < 1:
    #  observation = tf.convert_to_tensor([observation])

    #loc = [0.25, 0.75]
    loc = [-0.02 * observation**2 + 0.5, 0.02 * observation**2 - 0.5]

    #probs = [0.5, 0.5]
    probs = [0.5 + 0.1 * observation, 0.5 - 0.1 * observation]

    distribution = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=probs),
        components_distribution=tfd.Normal(loc=loc, scale=scale))

    actions = distribution.sample()
    log_probs = distribution.log_prob(actions)
    policy_info = {'log_probability': log_probs}
    return (distribution, policy_info)

  policy_info_spec = {
      'log_probability': specs.TensorSpec([], tf.float32),
  }

  tf_policy = common_utils.TFAgentsWrappedPolicy(
      tf_env.time_step_spec(),
      tf_env.action_spec(),
      policy_fn,
      policy_info_spec,
      emit_log_probability=True)

  return tf_env, tf_policy
