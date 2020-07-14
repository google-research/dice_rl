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

import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.data.gym_onpolicy_dataset as gym_onpolicy_dataset

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 100,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 500,
                     'Cutoff trajectory at this step.')


def get_dataset(tabular_obs, epsilon_explore, limit):
  """Get on-policy dataset."""
  env = navigation.GridWalk(tabular_obs=tabular_obs)
  policy_fn, policy_info_spec = navigation.get_navigation_policy(
      env, epsilon_explore=0.5)
  dataset = gym_onpolicy_dataset.GymOnpolicyDataset(
      env, policy_fn, policy_info_spec, episode_step_limit=limit)
  return dataset


def main(argv):
  seed = FLAGS.seed
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length

  np.random.seed(seed)
  dataset = get_dataset(False, 0.1, max_trajectory_length)

  first_step = dataset.get_step()
  episode, _ = dataset.get_episode()
  episode_start, _ = dataset.get_episode(truncate_episode_at=1)

  print('first step', first_step)
  print('ep reward', episode.reward)
  print('ep start', episode_start)

  episodes, valid_steps = dataset.get_episode(batch_size=num_trajectory)
  mask = valid_steps * episodes.discount
  episode_rewards = episodes.reward * mask
  print('avg step reward', np.sum(episode_rewards) / np.sum(mask))
  print('avg ep reward', np.mean(np.sum(episode_rewards, -1)))

  print('num steps', dataset.num_steps)
  print('num total steps', dataset.num_total_steps)
  print('num episodes', dataset.num_episodes)
  print('num total episodes', dataset.num_total_episodes)


if __name__ == '__main__':
  app.run(main)
