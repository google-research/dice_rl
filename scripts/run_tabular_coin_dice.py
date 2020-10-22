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

"""Script for running CoinDICE with neural network function approximators.

The default parameters here should reproduce the published frozenlake results.
Make sure to generate the reacher dataset prior to running this script (see
`scripts/create_dataset.py`). Furthermore, the user will need to feed in an
appropriate `divergence_limit`, which should be set to a desired chi2 percentile
divided by the size of the offline dataset (see paper for details). For example,
if a 90% confidence interval is desired and the offline dataset is 1000
trajectories of length 100, then the divergence_limit should be 2.7055 / 100000.

For the other published results, the procedure should be the same. For best
results on Taxi, the user may want to pass in solve_for_state_action_ratio=False
to TabularCoinDice.
"""

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
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
from dice_rl.estimators.tabular_coin_dice import TabularCoinDice
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset

# BEGIN GOOGLE-INTERNAL
import google3.learning.deepmind.xmanager2.client.google as xm  # pylint: disable=unused-import
# END GOOGLE-INTERNAL

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'frozenlake', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 1000,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 100,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.0, 'How close to target policy.')
flags.DEFINE_float('divergence_limit', 1e-5, 'Divergence limit.')
flags.DEFINE_float('algae_alpha', 0.01, 'Regularizer on Df(dpi|dD).')
flags.DEFINE_bool('tabular_obs', True, 'Whether to use tabular observations.')
flags.DEFINE_string('load_dir', None,
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir', None,
                    'Directory to save estimation results.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_integer('num_steps', 1000, 'Number of training steps.')
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
  num_steps = FLAGS.num_steps
  divergence_limit = FLAGS.divergence_limit
  algae_alpha = FLAGS.algae_alpha
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

  estimate = estimator_lib.get_fullbatch_average(dataset, gamma=gamma)
  print('data per step avg', estimate)

  train_hparam_str = ('limit{LIMIT}_'
                      'gam{GAMMA}_algae{ALGAE_ALPHA}_div{DIV}').format(
                          LIMIT=limit_episodes,
                          GAMMA=gamma,
                          ALGAE_ALPHA=algae_alpha,
                          DIV=divergence_limit)

  if save_dir is not None:
    save_dir = os.path.join(save_dir, hparam_str, train_hparam_str)
    summary_writer = tf.summary.create_file_writer(logdir=save_dir)
  else:
    summary_writer = tf.summary.create_noop_writer()

  estimator = TabularCoinDice(
      dataset_spec=dataset.spec,
      gamma=gamma,
      divergence_limit=divergence_limit,
      algae_alpha=algae_alpha * np.array([1, 1]),
      limit_episodes=limit_episodes)
  estimator.prepare_dataset(dataset, target_policy)

  global_step = tf.Variable(0, dtype=tf.int64)
  tf.summary.experimental.set_step(global_step)
  with summary_writer.as_default():
    running_losses = []
    running_estimates = []
    for step in range(num_steps):
      loss = estimator.train_step(dataset, target_policy)
      running_losses.append(loss)
      global_step.assign_add(1)

      if step % 10 == 0 or step == num_steps - 1:
        print('step', step, 'losses', np.mean(running_losses, 0))
        estimate = np.mean(running_losses, 0)[0]
        for idx, est in enumerate(estimate):
          tf.summary.scalar('estimate%d' % idx, est)
        running_estimates.append(estimate)
        print('estimated confidence interval %s' % estimate)
        print('avg last 3 estimated confidence interval %s' %
              np.mean(running_estimates[-3:], axis=0))
        running_losses = []

  if save_dir is not None:
    results_filename = os.path.join(save_dir, 'results.npy')
    with tf.io.gfile.GFile(results_filename, 'w') as f:
      np.save(f, running_estimates)
  print('Done!')


if __name__ == '__main__':
  app.run(main)
