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
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import pickle

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy

import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.taxi as taxi
import dice_rl.environments.suites as env_suites
import dice_rl.data.tf_agents_onpolicy_dataset as tf_agents_onpolicy_dataset
import dice_rl.estimators.estimator as estimator_lib
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 100,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 100,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('gamma', 0.995, 'Discount for estimators.')
flags.DEFINE_string('save_dir', None, 'Directory to save dataset from.')


def get_env_dataset(tabular_obs, epsilon_explore, limit):
  """Get on-policy dataset."""
  tf_env = env_suites.load_mujoco('Reacher-v2')
  tf_env = tf_py_environment.TFPyEnvironment(tf_env)

  tf_policy = random_tf_policy.RandomTFPolicy(
      tf_env.time_step_spec(), tf_env.action_spec(),
      emit_log_probability=True)

  dataset = tf_agents_onpolicy_dataset.TFAgentsOnpolicyDataset(
      tf_env, tf_policy,
      episode_step_limit=limit)
  return dataset


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
  seed = FLAGS.seed
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  gamma = FLAGS.gamma
  save_dir = FLAGS.save_dir

  np.random.seed(seed)
  dataset = get_env_dataset(False, 0.1, max_trajectory_length)

  write_dataset = TFOffpolicyDataset(
      dataset.spec)

  first_step = dataset.get_step()
  write_dataset.add_step(first_step)
  episode, valid_ids = dataset.get_episode()
  add_episodes_to_dataset(episode, valid_ids, write_dataset)
  episode_start, valid_steps = dataset.get_episode(truncate_episode_at=1)
  add_episodes_to_dataset(episode_start, valid_steps, write_dataset)

  episodes, valid_steps = dataset.get_episode(batch_size=num_trajectory)
  add_episodes_to_dataset(episodes, valid_steps, write_dataset)
  mask = (tf.cast(valid_steps, tf.float32) *
          (1 - tf.cast(episodes.is_last(), tf.float32)))
  episode_rewards = episodes.reward * mask
  print('avg step reward', tf.reduce_sum(episode_rewards) / tf.reduce_sum(mask))
  print('avg ep reward', tf.reduce_mean(tf.reduce_sum(episode_rewards, -1)))

  print('num steps', dataset.num_steps)
  print('num total steps', dataset.num_total_steps)
  print('num episodes', dataset.num_episodes)
  print('num total episodes', dataset.num_total_episodes)

  print('num write steps', write_dataset.num_steps)
  print('num write total steps', write_dataset.num_total_steps)
  print('num write episodes', write_dataset.num_episodes)
  print('num write total episodes', write_dataset.num_total_episodes)

  write_dataset.save(save_dir)
  new_dataset = Dataset.load(save_dir)
  print('num loaded steps', new_dataset.num_steps)
  print('num loaded total steps', new_dataset.num_total_steps)
  print('num loaded episodes', new_dataset.num_episodes)
  print('num loaded total episodes', new_dataset.num_total_episodes)

  estimate = estimator_lib.get_minibatch_average(dataset,
                                                 max_trajectory_length,
                                                 num_trajectory,
                                                 gamma=gamma)
  print('per step avg', estimate)
  estimate = estimator_lib.get_minibatch_average(dataset, num_trajectory,
                                                 by_steps=False,
                                                 gamma=gamma)
  print('per episode avg', estimate)
  estimate = estimator_lib.get_fullbatch_average(write_dataset,
                                                 gamma=gamma)
  print('per step avg on offpolicy data', estimate)
  estimate = estimator_lib.get_fullbatch_average(write_dataset,
                                                 by_steps=False,
                                                 gamma=gamma)
  print('per episode avg on offpolicy data', estimate)
  estimate = estimator_lib.get_fullbatch_average(new_dataset,
                                                 gamma=gamma)
  print('per step avg on saved and loaded offpolicy data', estimate)
  estimate = estimator_lib.get_fullbatch_average(new_dataset,
                                                 by_steps=False,
                                                 gamma=gamma)
  print('per episode avg on saved and loaded offpolicy data', estimate)


if __name__ == '__main__':
  app.run(main)
