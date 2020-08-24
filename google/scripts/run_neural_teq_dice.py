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
import sys
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import pickle

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

from dice_rl.environments.env_policies import get_target_policy
import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.tree as tree
import dice_rl.environments.gridworld.taxi as taxi
from dice_rl.google.estimators.neural_teq_dice import NeuralTeQDice
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.step_value_network import StepValueNetwork
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
import google3.learning.deepmind.xmanager2.client.google as xm

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 1000,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('num_trajectory_train', None,
                     'Number of trajectories to collect during training.')
flags.DEFINE_integer('max_trajectory_length', 100,
                     'Cutoff trajectory at this step.')
flags.DEFINE_integer('max_trajectory_length_train', None,
                     'Cutoff trajectory at this step during training.')
flags.DEFINE_float('alpha', 0.0, 'How close to target policy.')
flags.DEFINE_bool('tabular_obs', False, 'Whether to use tabular observations.')
flags.DEFINE_string('load_dir', '/cns/is-d/home/sherryy/teqdice/data/',
                    'Directory to load dataset from.')
flags.DEFINE_float('gamma', 0.995, 'Discount factor.')
flags.DEFINE_float('nu_learning_rate', 0.0001, 'Learning rate for nu.')
flags.DEFINE_float('zeta_learning_rate', 0.001, 'Learning rate for zeta.')
flags.DEFINE_float('nu_regularizer', 0.01, 'Ortho regularization on nu.')
flags.DEFINE_float('zeta_regularizer', 0.01, 'Ortho regularization on zeta.')
flags.DEFINE_float('f_exponent', 1.5, 'Exponent for f function.')
flags.DEFINE_bool('primal_form', False,
                  'Whether to use primal form of loss for nu.')
flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_string('save_dir', None,
                    'Directory to save the model and estimation results.')


class Logger(object):

  def __init__(self, filepath):
    self.terminal = sys.stdout
    self.log = tf.io.gfile.GFile(filepath, 'a+')

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)

  def flush(self):
    self.terminal.flush()
    self.log.flush()


def main(argv):
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  num_trajectory_train = FLAGS.num_trajectory_train
  if num_trajectory_train is None:
    num_trajectory_train = num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  max_trajectory_length_train = FLAGS.max_trajectory_length_train
  if max_trajectory_length_train is None:
    max_trajectory_length_train = max_trajectory_length
  alpha = FLAGS.alpha
  load_dir = FLAGS.load_dir
  gamma = FLAGS.gamma
  assert 0 <= gamma < 1.
  nu_learning_rate = FLAGS.nu_learning_rate
  zeta_learning_rate = FLAGS.zeta_learning_rate
  nu_regularizer = FLAGS.nu_regularizer
  zeta_regularizer = FLAGS.zeta_regularizer
  f_exponent = FLAGS.f_exponent
  primal_form = FLAGS.primal_form
  batch_size = FLAGS.batch_size
  num_steps = FLAGS.num_steps
  save_dir = FLAGS.save_dir
  network_dir = os.path.join(save_dir, 'networks') if save_dir else None
  estimate_dir = os.path.join(save_dir, 'estimates') if save_dir else None

  target_policy = get_target_policy(load_dir, env_name, tabular_obs)

  hparam_base = '{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}'.format(
      ENV_NAME=env_name, TAB=tabular_obs, ALPHA=alpha, SEED=seed)

  hparam_data = hparam_base + '_numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}'.format(
      NUM_TRAJ=num_trajectory if num_steps == 0 else num_trajectory_train,
      MAX_TRAJ=max_trajectory_length
      if num_steps == 0 else max_trajectory_length_train)
  hparam_net = hparam_base + '_numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}'.format(
      NUM_TRAJ=num_trajectory_train, MAX_TRAJ=max_trajectory_length_train)
  hparam_result = hparam_base + '_numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}'.format(
      NUM_TRAJ=num_trajectory, MAX_TRAJ=max_trajectory_length)

  if estimate_dir is not None:
    if not tf.io.gfile.isdir(estimate_dir):
      tf.io.gfile.makedirs(estimate_dir)
    log_file = os.path.join(estimate_dir, hparam_result + '.log')
    print("Logging to '{0}'".format(log_file))
    sys.stdout = Logger(log_file)

  directory = os.path.join(load_dir, hparam_data)
  print('Loading dataset from', directory)
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

  activation_fn = tf.nn.tanh
  kernel_initializer = tf.keras.initializers.GlorotUniform()
  hidden_dims = (64,)
  step_encoding = None
  #step_encoding = 'one_hot'
  nu_network = StepValueNetwork(
      (dataset.spec.observation, dataset.spec.action, dataset.spec.step_num),
      fc_layer_params=hidden_dims,
      activation_fn=activation_fn,
      kernel_initializer=kernel_initializer,
      last_kernel_initializer=kernel_initializer,
      max_trajectory_length_train=max_trajectory_length_train,
      step_encoding=step_encoding)
  zeta_network = StepValueNetwork(
      (dataset.spec.observation, dataset.spec.action, dataset.spec.step_num),
      fc_layer_params=hidden_dims,
      activation_fn=activation_fn,
      kernel_initializer=kernel_initializer,
      last_kernel_initializer=kernel_initializer,
      max_trajectory_length_train=max_trajectory_length_train,
      step_encoding=step_encoding)
  nu_network.create_variables()
  zeta_network.create_variables()
  try:
    nu_network.load_weights(os.path.join(network_dir, hparam_net, 'nu'))
    zeta_network.load_weights(os.path.join(network_dir, hparam_net, 'zeta'))
    print('loaded networks from', network_dir)
  except:
    print('initialized network from scratch')

  nu_optimizer = tf.keras.optimizers.Adam(nu_learning_rate)
  zeta_optimizer = tf.keras.optimizers.Adam(zeta_learning_rate)

  estimator = NeuralTeQDice(
      dataset.spec,
      nu_network,
      zeta_network,
      nu_optimizer,
      zeta_optimizer,
      gamma,
      f_exponent=f_exponent,
      primal_form=primal_form,
      nu_regularizer=nu_regularizer,
      zeta_regularizer=zeta_regularizer)

  running_losses = []
  running_estimates = []
  for step in range(num_steps):
    transitions_batch = dataset.get_step(batch_size, num_steps=2)
    initial_steps_batch, _ = dataset.get_episode(
        batch_size, truncate_episode_at=1)
    initial_steps_batch = tf.nest.map_structure(lambda t: t[:, 0, ...],
                                                initial_steps_batch)
    losses = estimator.train_step(initial_steps_batch, transitions_batch,
                                  target_policy)
    running_losses.append(losses)
    if step % 500 == 0 or step == num_steps - 1:
      print('step', step, 'losses', np.mean(running_losses, 0))
      estimate = estimator.estimate_average_reward(dataset, target_policy)
      running_estimates.append(estimate)
      print('estimated per step avg %f' % estimate)
      print('avg last 3 estimated per step avg %f' %
            np.mean(running_estimates[-3:]))
      if network_dir is not None:
        nu_network.save_weights(os.path.join(network_dir, hparam_net, 'nu'))
        zeta_network.save_weights(os.path.join(network_dir, hparam_net, 'zeta'))
        print('saved network weights to', os.path.join(network_dir, hparam_net))
      running_losses = []

  if num_steps == 0:
    estimate = estimator.estimate_average_reward(dataset, target_policy)
    running_estimates.append(estimate)
    print('eval only per step avg %f' % np.mean(running_estimates[-3:]))

  if estimate_dir is not None:
    out_fname = os.path.join(estimate_dir, hparam_result + '.npy')
    print('Saving estimation results to', out_fname)
    with tf.io.gfile.GFile(out_fname, 'w') as f:
      np.save(f, running_estimates)

  print('Done!')


if __name__ == '__main__':
  app.run(main)
