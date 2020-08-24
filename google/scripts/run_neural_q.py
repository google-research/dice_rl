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
import dice_rl.environments.gridworld.tree as tree
import dice_rl.environments.gridworld.taxi as taxi
from dice_rl.google.estimators.neural_qlearning import NeuralQLearning
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
from dice_rl.data.perturbed_dataset import PerturbedDataset

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 100,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 100,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.0,
                   'How close to target policy.')
flags.DEFINE_bool('tabular_obs', False,
                  'Whether to use tabular observations.')
flags.DEFINE_string('load_dir', '/cns/vz-d/home/brain-ofirnachum',
                    'Directory to load dataset from.')
flags.DEFINE_float('gamma', 0.995,
                   'Discount factor.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('nstep_returns', 1,
                     'Use n-step returns with this many steps.')
flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')


def main(argv):
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  load_dir = FLAGS.load_dir
  gamma = FLAGS.gamma
  assert 0 <= gamma < 1.
  learning_rate = FLAGS.learning_rate
  nstep_returns = FLAGS.nstep_returns
  num_steps = FLAGS.num_steps
  batch_size = FLAGS.batch_size

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
  dataset = PerturbedDataset(dataset,
                             num_perturbations=10,
                             perturbation_scale=1.)
  #estimate = estimator_lib.get_fullbatch_average(dataset, gamma=gamma)
  #print('perturbed data per step avg', estimate)

  value_network = ValueNetwork((dataset.spec.observation, dataset.spec.action),
                               fc_layer_params=(64, 64),
                               output_dim=10)
  optimizer = tf.keras.optimizers.Adam(learning_rate)

  estimator = NeuralQLearning(dataset.spec, value_network, optimizer, gamma,
                              num_qvalues=10)
  for step in range(num_steps):
    batch = dataset.get_step(batch_size, num_steps=nstep_returns + 1)
    loss, _ = estimator.train_step(batch, target_policy)
    if step % 100 == 0 or step == num_steps - 1:
      print('step', step, 'loss', loss)
      estimate = estimator.estimate_average_reward(dataset, target_policy)
      print('estimated per step avg', estimate)

  print('Done!')


if __name__ == '__main__':
  app.run(main)
