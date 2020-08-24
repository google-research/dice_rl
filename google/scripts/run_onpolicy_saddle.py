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

from absl import app
from absl import flags

import numpy as np
import os
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import pickle

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.tree as tree
import dice_rl.environments.gridworld.taxi as taxi
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.google.rl_algos.tabular_saddle_point import TabularSaddlePoint
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_agents_onpolicy_dataset import TFAgentsOnpolicyDataset
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 1,
                     'Number of trajectories to collect at each iteration.')
flags.DEFINE_integer('max_trajectory_length', 20,
                     'Cutoff trajectory at this step.')
flags.DEFINE_bool('tabular_obs', True,
                  'Whether to use tabular observations.')
flags.DEFINE_float('gamma', 0.95,
                   'Discount factor.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 1024, 'Batch size.')


def get_onpolicy_dataset(env_name, tabular_obs, policy_fn, policy_info_spec):
  """Gets target policy."""
  if env_name == 'taxi':
    env = taxi.Taxi(tabular_obs=tabular_obs)
  elif env_name == 'grid':
    env = navigation.GridWalk(tabular_obs=tabular_obs)
  elif env_name == 'tree':
    env = tree.Tree(branching=2, depth=10)
  else:
    raise ValueError('Unknown environment: %s.' % env_name)

  tf_env = tf_py_environment.TFPyEnvironment(
      gym_wrapper.GymWrapper(env))
  tf_policy = common_utils.TFAgentsWrappedPolicy(
      tf_env.time_step_spec(), tf_env.action_spec(),
      policy_fn, policy_info_spec,
      emit_log_probability=True)

  return TFAgentsOnpolicyDataset(tf_env, tf_policy)


def get_random_policy(env_name, tabular_obs):
  if env_name == 'taxi':
    env = taxi.Taxi(tabular_obs=tabular_obs)
    policy_fn, policy_info_spec = taxi.get_taxi_policy(
        env, env, alpha=0.0, py=False)
  elif env_name == 'grid':
    env = navigation.GridWalk(tabular_obs=tabular_obs)
    policy_fn, policy_info_spec = navigation.get_navigation_policy(
        env, epsilon_explore=1.0, py=False)
  elif env_name == 'tree':
    env = tree.Tree(branching=2, depth=10)
    policy_fn, policy_info_spec = tree.get_tree_policy(
        env, epsilon_explore=1.0, py=False)
  else:
    raise ValueError('Unknown environment: %s.' % env_name)

  return policy_fn, policy_info_spec
  tf_env = tf_py_environment.TFPyEnvironment(
      gym_wrapper.GymWrapper(env))
  tf_policy = common_utils.TFAgentsWrappedPolicy(
      tf_env.time_step_spec(), tf_env.action_spec(),
      policy_fn, policy_info_spec,
      emit_log_probability=True)

  return tf_policy, policy_info_spec


def add_episodes_to_dataset(episodes, valid_ids, write_dataset):
  num_episodes = 1 if tf.rank(valid_ids) == 1 else tf.shape(valid_ids)[0]
  for ep_id in range(num_episodes):
    if tf.rank(valid_ids) == 1:
      this_valid_ids = valid_ids
      this_episode = episodes
    else:
      this_valid_ids = valid_ids[ep_id, ...]
      this_episode = tf.nest.map_structure(
          lambda t: t[ep_id, ...], episodes)

    episode_length = tf.shape(this_valid_ids)[0]
    for step_id in range(episode_length):
      this_valid_id = this_valid_ids[step_id]
      this_step = tf.nest.map_structure(
          lambda t: t[step_id, ...], this_episode)
      if this_valid_id:
        write_dataset.add_step(this_step)


def main(argv):
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  gamma = FLAGS.gamma
  assert 0 <= gamma < 1.
  learning_rate = FLAGS.learning_rate
  num_steps = FLAGS.num_steps
  batch_size = FLAGS.batch_size

  optimizer = tf.keras.optimizers.Adam(learning_rate)

  init_policy_fn, init_policy_info_spec = get_random_policy(
      env_name, tabular_obs)
  onpolicy_data = get_onpolicy_dataset(
      env_name, tabular_obs,
      init_policy_fn, init_policy_info_spec)
  onpolicy_episodes, valid_steps = onpolicy_data.get_episode(
      num_trajectory * 100, truncate_episode_at=max_trajectory_length)

  dataset = TFOffpolicyDataset(onpolicy_data.spec)
  add_episodes_to_dataset(onpolicy_episodes, valid_steps, dataset)

  algo = TabularSaddlePoint(
      dataset.spec, optimizer,
      gamma=gamma)

  losses = []
  for step in range(num_steps):
    init_batch, _ = dataset.get_episode(batch_size, truncate_episode_at=1)
    init_batch = tf.nest.map_structure(lambda t: t[:, 0, ...], init_batch)
    batch = dataset.get_step(batch_size, num_steps=2)
    loss, policy_loss = algo.train_step(init_batch, batch)
    losses.append(loss)
    if step % 100 == 0 or step == num_steps - 1:
      print('step', step, 'loss', np.mean(losses, 0))
      losses = []
      policy_fn, policy_info_spec = algo.get_policy()
      onpolicy_data = get_onpolicy_dataset(env_name, tabular_obs,
                                           policy_fn, policy_info_spec)
      onpolicy_episodes, valid_steps = onpolicy_data.get_episode(
          num_trajectory, truncate_episode_at=max_trajectory_length)
      add_episodes_to_dataset(onpolicy_episodes, valid_steps, dataset)
      print('estimated per step avg', np.mean(onpolicy_episodes.reward))

  print('Done!')


if __name__ == '__main__':
  app.run(main)
