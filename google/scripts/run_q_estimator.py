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

from dice_rl.environments.env_policies import get_target_policy
from dice_rl.google.estimators.tabular_qlearning import TabularQLearning
from dice_rl.estimators import estimator as estimator_lib
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
import google3.learning.deepmind.xmanager2.client.google as xm  # pylint: disable=unused-import

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'taxi', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 100,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 500,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.0,
                   'How close to target policy.')
flags.DEFINE_bool('tabular_obs', True,
                  'Whether to use tabular observations.')
#flags.DEFINE_string('load_dir', '/cns/is-d/home/sherryy/teqdice/data',
flags.DEFINE_string('load_dir', '/cns/vz-d/home/brain-ofirnachum',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir', None,
                    'Directory to save estimation results.')
flags.DEFINE_float('gamma', 0.995,
                   'Discount factor.')
flags.DEFINE_integer('limit_episodes', None,
                     'Number of episodes to take from dataset.')


def main(argv):
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  load_dir = FLAGS.load_dir
  save_dir = FLAGS.save_dir
  gamma = FLAGS.gamma
  assert 0 <= gamma < 1.
  limit_episodes = FLAGS.limit_episodes

  target_policy = get_target_policy(load_dir, env_name, tabular_obs)

  hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    TAB=tabular_obs,
                    ALPHA=alpha,
                    SEED=seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=max_trajectory_length)
  directory = os.path.join(load_dir, hparam_str)
  print('Loading dataset.')
  dataset = Dataset.load(directory)
  all_steps = dataset.get_all_steps()
  max_reward = tf.reduce_max(all_steps.reward)
  min_reward = tf.reduce_min(all_steps.reward)
  print('num loaded steps', dataset.num_steps)
  print('num loaded total steps', dataset.num_total_steps)
  print('num loaded episodes', dataset.num_episodes)
  print('num loaded total episodes', dataset.num_total_episodes)
  print('min reward', min_reward, 'max reward', max_reward)

  train_hparam_str = ('gamma{GAM}_limit{LIMIT}').format(
      GAM=gamma,
      LIMIT=limit_episodes)

  estimate = estimator_lib.get_fullbatch_average(dataset, gamma=gamma)
  print('data per step avg', estimate)

  estimator = TabularQLearning(dataset.spec, gamma, num_qvalues=200,
                               perturbation_scale=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.],
                               default_reward_value=0.0,
                               limit_episodes=limit_episodes)
  estimate = estimator.solve(dataset, target_policy)
  print('estimated per step avg', estimate)

  if save_dir is not None:
    results_dir = os.path.join(save_dir, hparam_str)
    if not tf.io.gfile.exists(results_dir):
      tf.io.gfile.makedirs(results_dir)
    results_filename = os.path.join(results_dir,
                                    'results_%s.npy' % train_hparam_str)
    with tf.io.gfile.GFile(results_filename, 'w') as f:
      np.save(f, estimate)

  print('Done!')


if __name__ == '__main__':
  app.run(main)
