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

import functools
import os
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union, Mapping

import numpy as np
import tensorflow.compat.v2 as tf

from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import actor_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import gaussian_policy
from tf_agents.policies import q_policy
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.utils import nest_utils
import tensorflow_probability as tfp
import os

from dice_rl.environments import suites
from dice_rl.environments.infinite_cartpole import InfiniteCartPole
from dice_rl.environments.infinite_frozenlake import InfiniteFrozenLake
from dice_rl.environments.infinite_reacher import InfiniteReacher
from dice_rl.environments.gridworld import navigation
from dice_rl.environments.gridworld import taxi
from dice_rl.environments.gridworld import tree
from dice_rl.environments import bandit
import dice_rl.utils.common as common_lib


def get_dqn_policy(tf_env):
  q_net = q_network.QNetwork(
      tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=(100,))
  policy = q_policy.QPolicy(
      tf_env.time_step_spec(),
      action_spec=tf_env.action_spec(),
      q_network=q_net)
  return policy


def get_sac_policy(tf_env):
  actor_net = actor_distribution_network.ActorDistributionNetwork(
      tf_env.observation_spec(),
      tf_env.action_spec(),
      fc_layer_params=(256, 256),
      continuous_projection_net=tanh_normal_projection_network
      .TanhNormalProjectionNetwork)
  policy = actor_policy.ActorPolicy(
      time_step_spec=tf_env.time_step_spec(),
      action_spec=tf_env.action_spec(),
      actor_network=actor_net,
      training=False)
  return policy


def load_policy(policy, env_name, load_dir, ckpt_file=None):
  policy = greedy_policy.GreedyPolicy(policy)
  checkpoint = tf.train.Checkpoint(policy=policy)
  if ckpt_file is None:
    checkpoint_filename = tf.train.latest_checkpoint(load_dir)
  else:
    checkpoint_filename = os.path.join(load_dir, ckpt_file)
  print('Loading policy from %s.' % checkpoint_filename)
  checkpoint.restore(checkpoint_filename).assert_existing_objects_matched()
  # Unwrap greedy wrapper.
  return policy.wrapped_policy


def get_env_and_dqn_policy(env_name,
                           load_dir,
                           env_seed=0,
                           epsilon=0.0,
                           ckpt_file=None):
  gym_env = suites.load_gym(env_name)
  gym_env.seed(env_seed)
  env = tf_py_environment.TFPyEnvironment(gym_env)
  dqn_policy = get_dqn_policy(env)
  policy = load_policy(dqn_policy, env_name, load_dir, ckpt_file)
  return env, EpsilonGreedyPolicy(
      policy, epsilon=epsilon, emit_log_probability=True)


def get_env_and_policy(load_dir,
                       env_name,
                       alpha,
                       env_seed=0,
                       tabular_obs=False):
  if env_name == 'taxi':
    env = taxi.Taxi(tabular_obs=tabular_obs)
    env.seed(env_seed)
    policy_fn, policy_info_spec = taxi.get_taxi_policy(
        load_dir, env, alpha=alpha, py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'grid':
    env = navigation.GridWalk(tabular_obs=tabular_obs)
    env.seed(env_seed)
    policy_fn, policy_info_spec = navigation.get_navigation_policy(
        env, epsilon_explore=0.1 + 0.6 * (1 - alpha), py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'tree':
    env = tree.Tree(branching=2, depth=10)
    env.seed(env_seed)
    policy_fn, policy_info_spec = tree.get_tree_policy(
        env, epsilon_explore=0.1 + 0.8 * (1 - alpha), py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name.startswith('bandit'):
    num_arms = int(env_name[6:]) if len(env_name) > 6 else 2
    env = bandit.Bandit(num_arms=num_arms)
    env.seed(env_seed)
    policy_fn, policy_info_spec = bandit.get_bandit_policy(
        env, epsilon_explore=1 - alpha, py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'small_tree':
    env = tree.Tree(branching=2, depth=3, loop=True)
    env.seed(env_seed)
    policy_fn, policy_info_spec = tree.get_tree_policy(
        env, epsilon_explore=0.1 + 0.8 * (1 - alpha), py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'CartPole-v0':
    tf_env, policy = get_env_and_dqn_policy(
        env_name,
        os.path.join(load_dir, 'CartPole-v0', 'train', 'policy'),
        env_seed=env_seed,
        epsilon=0.3 + 0.15 * (1 - alpha))
  elif env_name == 'cartpole':  # Infinite-horizon cartpole.
    tf_env, policy = get_env_and_dqn_policy(
        'CartPole-v0',
        os.path.join(load_dir, 'CartPole-v0-250', 'train', 'policy'),
        env_seed=env_seed,
        epsilon=0.3 + 0.15 * (1 - alpha))
    env = InfiniteCartPole()
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
  elif env_name == 'FrozenLake-v0':
    tf_env, policy = get_env_and_dqn_policy(
        'FrozenLake-v0',
        os.path.join(load_dir, 'FrozenLake-v0', 'train', 'policy'),
        env_seed=env_seed,
        epsilon=0.2 * (1 - alpha),
        ckpt_file='ckpt-100000')
  elif env_name == 'frozenlake':  # Infinite-horizon frozenlake.
    tf_env, policy = get_env_and_dqn_policy(
        'FrozenLake-v0',
        os.path.join(load_dir, 'FrozenLake-v0', 'train', 'policy'),
        env_seed=env_seed,
        epsilon=0.2 * (1 - alpha),
        ckpt_file='ckpt-100000')
    env = InfiniteFrozenLake()
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
  elif env_name in ['Reacher-v2', 'reacher']:
    if env_name == 'Reacher-v2':
      env = suites.load_mujoco(env_name)
    else:
      env = gym_wrapper.GymWrapper(InfiniteReacher())
    env.seed(env_seed)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    sac_policy = get_sac_policy(tf_env)
    directory = os.path.join(load_dir, 'Reacher-v2', 'train', 'policy')
    policy = load_policy(sac_policy, env_name, directory)
    policy = gaussian_policy.GaussianPolicy(policy, 0.4 - 0.3 * alpha)
  elif env_name == 'HalfCheetah-v2':
    env = suites.load_mujoco(env_name)
    env.seed(env_seed)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    sac_policy = get_sac_policy(tf_env)
    directory = os.path.join(load_dir, env_name, 'train', 'policy')
    policy = load_policy(sac_policy, env_name, directory)
    policy = gaussian_policy.GaussianPolicy(policy, 0.2 - 0.1 * alpha)
  else:
    raise ValueError('Unrecognized environment %s.' % env_name)

  return tf_env, policy


def get_target_policy(load_dir, env_name, tabular_obs, alpha=1.0):
  """Gets target policy."""
  tf_env, tf_policy = get_env_and_policy(
      load_dir, env_name, alpha, tabular_obs=tabular_obs)
  return tf_policy


class EpsilonGreedyPolicy(tf_policy.TFPolicy):
  """An epsilon-greedy policy that can return distributions."""

  def __init__(self, policy, epsilon, emit_log_probability=True):
    self._wrapped_policy = policy
    self._epsilon = epsilon
    if not common_lib.is_categorical_spec(policy.action_spec):
      raise ValueError('Action spec must be categorical to define '
                       'epsilon-greedy policy.')

    super(EpsilonGreedyPolicy, self).__init__(
        policy.time_step_spec,
        policy.action_spec,
        policy.policy_state_spec,
        policy.info_spec,
        emit_log_probability=emit_log_probability)

  @property
  def wrapped_policy(self):
    return self._wrapped_policy

  def _variables(self):
    return self._wrapped_policy.variables()

  def _get_epsilon(self):
    if callable(self._epsilon):
      return self._epsilon()
    else:
      return self._epsilon

  def _distribution(self, time_step, policy_state):
    batched = nest_utils.is_batched_nested_tensors(time_step,
                                                   self._time_step_spec)
    if not batched:
      time_step = nest_utils.batch_nested_tensors(time_step)

    policy_dist_step = self._wrapped_policy.distribution(
        time_step, policy_state)
    policy_state = policy_dist_step.state
    policy_info = policy_dist_step.info
    policy_logits = policy_dist_step.action.logits_parameter()
    action_size = tf.shape(policy_logits)[-1]

    greedy_probs = tf.one_hot(tf.argmax(policy_logits, -1), action_size)
    uniform_probs = (
        tf.ones(tf.shape(policy_logits)) / tf.cast(action_size, tf.float32))
    epsilon = self._get_epsilon()
    mixed_probs = (1 - epsilon) * greedy_probs + epsilon * uniform_probs
    if not batched:
      mixed_probs = tf.squeeze(mixed_probs, 0)
      policy_state = nest_utils.unbatch_nested_tensors(policy_state)
      policy_info = nest_utils.unbatch_nested_tensors(policy_info)
    mixed_dist = tfp.distributions.Categorical(
        probs=mixed_probs, dtype=policy_dist_step.action.dtype)

    return policy_step.PolicyStep(mixed_dist, policy_state, policy_info)


class GaussianPolicy(tf_policy.TFPolicy):
  """An gaussian policy that can return distributions."""

  def __init__(self, policy, scale, emit_log_probability=True):
    self._wrapped_policy = policy
    self._scale = scale

    super(GaussianPolicy, self).__init__(
        policy.time_step_spec,
        policy.action_spec,
        policy.policy_state_spec,
        policy.info_spec,
        emit_log_probability=emit_log_probability)

  @property
  def wrapped_policy(self):
    return self._wrapped_policy

  def _variables(self):
    return self._wrapped_policy.variables()

  def _get_epsilon(self):
    if callable(self._scale):
      return self._scale()
    else:
      return self._scale

  def _distribution(self, time_step, policy_state):
    batched = nest_utils.is_batched_nested_tensors(time_step,
                                                   self._time_step_spec)
    if not batched:
      time_step = nest_utils.batch_nested_tensors(time_step)

    policy_dist_step = self._wrapped_policy.distribution(
        time_step, policy_state)
    policy_state = policy_dist_step.state
    policy_mean_action = policy_dist_step.action.mean()
    policy_info = policy_dist_step.info

    if not batched:
      policy_state = nest_utils.unbatch_nested_tensors(policy_state)
      policy_mean_action = nest_utils.unbatch_nested_tensors(policy_mean_action)
      policy_info = nest_utils.unbatch_nested_tensors(policy_info)

    gaussian_dist = tfp.distributions.MultivariateNormalDiag(
          loc=policy_mean_action,
          scale_diag=tf.ones_like(policy_mean_action) * self._scale)

    return policy_step.PolicyStep(gaussian_dist, policy_state,
                                  policy_info)


def get_env_and_policy_from_weights(env_name: str,
                                    weights: Mapping[str, np.ndarray],
                                    n_hidden: int = 300,
                                    min_log_std: float = -5,
                                    max_log_std: float = 2):
  """Return tf_env and policy from dictionary of weights.

  Assumes that the policy has 2 hidden layers with 300 units, ReLu activations,
  and outputs a normal distribution squashed by a Tanh.

  Args:
    env_name: Name of the environment.
    weights: Dictionary of weights containing keys: fc0/weight, fc0/bias,
      fc0/weight, fc0/bias, last_fc/weight, last_fc_log_std/weight,
      last_fc/bias, last_fc_log_std/bias

  Returns:
    tf_env: TF wrapped env.
    policy: TF Agents policy.
  """
  env = suites.load_mujoco(env_name)
  tf_env = tf_py_environment.TFPyEnvironment(env)
  std_transform = (
      lambda x: tf.exp(tf.clip_by_value(x, min_log_std, max_log_std)))
  actor_net = actor_distribution_network.ActorDistributionNetwork(
      tf_env.observation_spec(),
      tf_env.action_spec(),
      fc_layer_params=(n_hidden, n_hidden),
      continuous_projection_net=functools.partial(
          tanh_normal_projection_network.TanhNormalProjectionNetwork,
          std_transform=std_transform),
      activation_fn=tf.keras.activations.relu,
  )
  policy = actor_policy.ActorPolicy(
      time_step_spec=tf_env.time_step_spec(),
      action_spec=tf_env.action_spec(),
      actor_network=actor_net,
      training=False)

  # Set weights
  actor_net._encoder.layers[1].set_weights(  # pylint: disable=protected-access
      [weights['fc0/weight'].T, weights['fc0/bias']])
  actor_net._encoder.layers[2].set_weights(  # pylint: disable=protected-access
      [weights['fc1/weight'].T, weights['fc1/bias']])
  actor_net._projection_networks.layers[0].set_weights(  # pylint: disable=protected-access
      [
          np.concatenate(
              (weights['last_fc/weight'], weights['last_fc_log_std/weight']),
              axis=0).T,
          np.concatenate(
              (weights['last_fc/bias'], weights['last_fc_log_std/bias']),
              axis=0)
      ])
  return tf_env, policy
