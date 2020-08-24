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
import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.taxi as taxi
from dice_rl.google.estimators.tabular_teq_dice import TabularTeQDice
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
flags.DEFINE_float('alpha', 0.0, 'How close to target policy.')
flags.DEFINE_bool('tabular_obs', True, 'Whether to use tabular observations.')
flags.DEFINE_string('load_dir', '/cns/is-d/home/sherryy/teqdice/data',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir', None, 'Directory to save estimation results.')
flags.DEFINE_float('gamma', 0.995, 'Discount factor.')
flags.DEFINE_string('step_encoding', None, 'Step encoding to use.')
flags.DEFINE_integer('max_trajectory_length_train', None,
                     'Cutoff trajectory at this step during training.')


def main(argv):
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  load_dir = FLAGS.load_dir
  save_dir = FLAGS.save_dir
  step_encoding = FLAGS.step_encoding
  gamma = FLAGS.gamma
  assert 0 <= gamma < 1.
  max_trajectory_length_train = FLAGS.max_trajectory_length_train or max_trajectory_length

  target_policy = get_target_policy(load_dir, env_name, tabular_obs)

  hparam_base = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                 'numtraj{NUM_TRAJ}').format(
                     ENV_NAME=env_name,
                     TAB=tabular_obs,
                     ALPHA=alpha,
                     SEED=seed,
                     NUM_TRAJ=num_trajectory)
  hparam_data = hparam_base + '_maxtraj{MAX_TRAJ}'.format(
      MAX_TRAJ=max_trajectory_length_train)
  hparam_out = hparam_base + '_maxtraj{MAX_TRAJ}'.format(
      MAX_TRAJ=max_trajectory_length)
  directory = os.path.join(load_dir, hparam_data)
  print('Loading dataset.')
  dataset = Dataset.load(directory)
  print('num loaded steps', dataset.num_steps)
  print('num loaded total steps', dataset.num_total_steps)
  print('num loaded episodes', dataset.num_episodes)
  print('num loaded total episodes', dataset.num_total_episodes)

  estimator = TabularTeQDice(dataset.spec, gamma, max_trajectory_length,
                             step_encoding)
  estimate = estimator.solve(dataset, target_policy)
  print('estimated per step avg', estimate)

  print('Done!')

  if save_dir is not None:
    if not tf.io.gfile.isdir(save_dir):
      tf.io.gfile.makedirs(save_dir)
    out_fname = os.path.join(
        save_dir, hparam_out + '_enc{ENC}.npy'.format(ENC=step_encoding))
    print('Saving results to', out_fname)
    with tf.io.gfile.GFile(out_fname, 'w') as f:
      np.save(f, estimate.numpy())


if __name__ == '__main__':
  app.run(main)
