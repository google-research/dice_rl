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
from dice_rl.google.estimators.neural_robust_dice import NeuralRobustDice
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset

import google3.learning.deepmind.xmanager2.client.google as xm

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 500,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 100,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.0,
                   'How close to target policy.')
flags.DEFINE_bool('tabular_obs', False,
                  'Whether to use tabular observations.')
flags.DEFINE_integer('bootstrap_seed', None,
                     'The seed to use for bootstrapping. If None, no bootstrapping used.')
flags.DEFINE_string('load_dir', '/cns/vz-d/home/brain-ofirnachum',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir', '/cns/vz-d/home/brain-ofirnachum/robust',
                    'Directory to save results to.')
flags.DEFINE_float('gamma', 0.95,
                   'Discount factor.')
flags.DEFINE_float('nu_learning_rate', 0.0001, 'Learning rate for nu.')
flags.DEFINE_float('zeta_learning_rate', 0.001, 'Learning rate for zeta.')
flags.DEFINE_float('nu_regularizer', 0.0, 'Ortho regularization on nu.')
flags.DEFINE_float('zeta_regularizer', 0.0, 'Ortho regularization on zeta.')

flags.DEFINE_float('weight_learning_rate', 0.001, 'Learning rate for weights.')
flags.DEFINE_float('alpha_learning_rate', 0.001, 'Learning rate for alpha.')
flags.DEFINE_float('divergence_limit', 0.1, 'Divergence limit.')
flags.DEFINE_float('algae_alpha', 0.1, 'Regularizer on Df(dpi|dD).')

flags.DEFINE_float('f_exponent', 1.5, 'Exponent for f function.')
flags.DEFINE_bool('primal_form', False,
                  'Whether to use primal form of loss for nu.')
flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 512, 'Batch size.')


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


def bootstrap_dataset(dataset, seed):
  num_episodes = dataset.num_total_episodes
  np.random.seed(seed)
  sampled_episodes = np.random.choice(num_episodes, size=[num_episodes],
                                      replace=True)
  all_episodes, all_valids = dataset.get_all_episodes()
  assert len(all_valids) == num_episodes

  new_dataset = TFOffpolicyDataset(dataset.spec, dataset.capacity)
  new_episodes = tf.nest.map_structure(lambda t: tf.gather(t, sampled_episodes),
                                       all_episodes)
  new_valids = tf.gather(all_valids, sampled_episodes)

  add_episodes_to_dataset(new_episodes, new_valids, new_dataset)
  return dataset


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
  nu_learning_rate = FLAGS.nu_learning_rate
  zeta_learning_rate = FLAGS.zeta_learning_rate
  nu_regularizer = FLAGS.nu_regularizer
  zeta_regularizer = FLAGS.zeta_regularizer
  weight_learning_rate = FLAGS.weight_learning_rate
  alpha_learning_rate = FLAGS.alpha_learning_rate
  divergence_limit = FLAGS.divergence_limit
  algae_alpha = FLAGS.algae_alpha
  f_exponent = FLAGS.f_exponent
  primal_form = FLAGS.primal_form
  batch_size = FLAGS.batch_size
  num_steps = FLAGS.num_steps
  bootstrap_seed = FLAGS.bootstrap_seed

  #num_samples = 1000000
  #samples = np.random.chisquare(1, size=num_samples)
  #samples = sorted(samples)

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
  if bootstrap_seed is not None:
    dataset = bootstrap_dataset(dataset, bootstrap_seed)
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

  train_hparam_str = ('nlr{NU_LR}_zlr{Z_LR}_alr{A_LR}_batch{BATCH_SIZE}_'
                      'gam{GAMMA}_nreg{NU_REG}_zreg{Z_REG}_algae{ALGAE_ALPHA}_'
                      'prim{PRIMAL}_div{DIV}_boot{BOOT}').format(
                          NU_LR=nu_learning_rate,
                          Z_LR=zeta_learning_rate,
                          A_LR=alpha_learning_rate,
                          BATCH_SIZE=batch_size,
                          GAMMA=gamma,
                          NU_REG=nu_regularizer,
                          Z_REG=zeta_regularizer,
                          ALGAE_ALPHA=algae_alpha,
                          PRIMAL=primal_form,
                          DIV=divergence_limit,
                          BOOT=bootstrap_seed)
  if save_dir is not None:
    save_dir = os.path.join(save_dir, hparam_str, train_hparam_str)
    summary_writer = tf.summary.create_file_writer(logdir=save_dir)
  else:
    tf.summary.create_noop_writer()

  activation_fn = tf.nn.relu
  kernel_initializer = None
  #activation_fn = tf.nn.tanh
  #kernel_initializer = tf.keras.initializers.GlorotUniform()
  #hidden_dims = (64, 64)
  hidden_dims = (300, 300)
  n_intervals = 1
  nu_network = ValueNetwork((dataset.spec.observation, dataset.spec.action),
                            fc_layer_params=hidden_dims,
                            activation_fn=activation_fn,
                            kernel_initializer=kernel_initializer,
                            last_kernel_initializer=kernel_initializer,
                            output_dim=2 * 2 * n_intervals)
  zeta_network = ValueNetwork((dataset.spec.observation, dataset.spec.action),
                              fc_layer_params=hidden_dims,
                              activation_fn=activation_fn,
                              kernel_initializer=kernel_initializer,
                              last_kernel_initializer=kernel_initializer,
                              output_dim=2 * 2 * n_intervals)
  weight_network = ValueNetwork((dataset.spec.observation,  # initial state
                                 dataset.spec.observation,  # cur state
                                 dataset.spec.action,       # cur action
                                 dataset.spec.observation), # next state
                                fc_layer_params=hidden_dims,
                                activation_fn=activation_fn,
                                kernel_initializer=kernel_initializer,
                                last_kernel_initializer=kernel_initializer,
                                output_dim=2 * n_intervals)

  nu_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, beta_2=0.99)
  zeta_optimizer = tf.keras.optimizers.Adam(zeta_learning_rate, beta_2=0.99)
  weight_optimizer = tf.keras.optimizers.Adam(weight_learning_rate, beta_2=0.99)
  alpha_optimizer = tf.keras.optimizers.Adam(alpha_learning_rate, beta_2=0.99)

  estimator = NeuralRobustDice(dataset.spec,
                               nu_network, zeta_network,
                               weight_network,
                               nu_optimizer, zeta_optimizer,
                               weight_optimizer, alpha_optimizer,
                               gamma=gamma,
                               divergence_limit=divergence_limit,
                               f_exponent=f_exponent,
                               primal_form=primal_form,
                               nu_regularizer=nu_regularizer,
                               zeta_regularizer=zeta_regularizer,
                               algae_alpha=algae_alpha * np.array([1, -1]),
                               closed_form_weights=True)

  global_step = tf.Variable(0, dtype=tf.int64)
  tf.summary.experimental.set_step(global_step)

  @tf.function
  def one_step(transitions_batch, initial_steps_batch):
    global_step.assign_add(1)
    with tf.summary.record_if(tf.math.mod(global_step, 25) == 0):
      initial_steps_batch = tf.nest.map_structure(lambda t: t[:, 0, ...],
                                                  initial_steps_batch)
      losses, _ = estimator.train_step(initial_steps_batch, transitions_batch,
                                       target_policy)
    return losses

  with summary_writer.as_default():
    running_losses = []
    running_estimates = []
    for step in range(num_steps):

      transitions_batch = dataset.get_step(batch_size, num_steps=2)
      initial_steps_batch, _ = dataset.get_episode(
          batch_size, truncate_episode_at=1)
      losses = one_step(transitions_batch, initial_steps_batch)
      running_losses.append([t.numpy() for t in losses])

      if step % 500 == 0 or step == num_steps - 1:
        print('step', step, 'losses', np.mean(running_losses, 0))
        estimate = np.mean(running_losses, 0)[0]
        for idx, est in enumerate(estimate):
          tf.summary.scalar('estimate%d' % idx, est)
        running_estimates.append(estimate)
        print('estimated per step avg %s' % estimate)
        print('avg last 3 estimated per step avg %s' %
              np.mean(running_estimates[-3:], axis=0))
        running_losses = []

  if save_dir is not None:
    results_filename = os.path.join(save_dir, 'results.npy')
    with tf.io.gfile.GFile(results_filename, 'w') as f:
      np.save(f, running_estimates)
  print('Done!')


if __name__ == '__main__':
  app.run(main)
