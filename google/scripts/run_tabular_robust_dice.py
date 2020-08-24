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
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
from dice_rl.google.estimators.tabular_robust_dice import TabularRobustDice
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
flags.DEFINE_float('alpha_learning_rate', 1., 'Learning rate for alpha.')
flags.DEFINE_float('divergence_limit', 0.1, 'Divergence limit.')
flags.DEFINE_float('algae_alpha', 1.0, 'Regularizer on Df(dpi|dD).')
flags.DEFINE_bool('tabular_obs', True, 'Whether to use tabular observations.')
flags.DEFINE_string('load_dir', '/cns/pw-d/home/mudcats/dev/algae_ci/data/',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir',
                    None, #'/cns/pw-d/home/mudcats/dev/algae_ci/tabular_robust/',
                    'Directory to save estimation results.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('train_nu_zeta_per_steps', 100,
                     'Train nu_zeta per number of steps.')
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
  alpha_learning_rate = FLAGS.alpha_learning_rate
  train_nu_zeta_per_steps = FLAGS.train_nu_zeta_per_steps
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

  train_hparam_str = ('alr{A_LR}_tnzs{TNZS}_limit{LIMIT}_'
                      'gam{GAMMA}_algae{ALGAE_ALPHA}_div{DIV}').format(
                          A_LR=alpha_learning_rate,
                          TNZS=train_nu_zeta_per_steps,
                          LIMIT=limit_episodes,
                          GAMMA=gamma,
                          ALGAE_ALPHA=algae_alpha,
                          DIV=divergence_limit)

  if save_dir is not None:
    save_dir = os.path.join(save_dir, hparam_str, train_hparam_str)
    summary_writer = tf.summary.create_file_writer(logdir=save_dir)
  else:
    summary_writer = tf.summary.create_noop_writer()

  alpha_optimizer = tf.keras.optimizers.Adam(alpha_learning_rate,
                                             beta_1=0.0, beta_2=0.0)

  episodes, valid_steps = dataset.get_all_episodes(
      limit=limit_episodes)
  num_samples = tf.reduce_sum(
      tf.cast(tf.logical_and(valid_steps, episodes.discount > 0)[:, :-1],
              tf.float32))
  estimator = TabularRobustDice(
      dataset_spec=dataset.spec,
      alpha_optimizer=alpha_optimizer,
      gamma=gamma,
      divergence_limit=#divergence_limit,
      divergence_limit / num_samples,
      algae_alpha=algae_alpha * np.array([1, 1]),
      limit_episodes=limit_episodes)
  global_step = tf.Variable(0, dtype=tf.int64)
  tf.summary.experimental.set_step(global_step)

  def one_step(transitions_batch, initial_steps_batch, target_policy):
    global_step.assign_add(1)
    #initial_steps_batch = tf.nest.map_structure(lambda t: t[:, 0, ...],
    #                                            initial_steps_batch)
    #losses, _ = estimator.train_alpha(initial_steps_batch, transitions_batch,
    #                                  target_policy)
    #return losses

  with summary_writer.as_default():
    running_losses = []
    running_estimates = []
    for step in range(num_steps):
      if step % train_nu_zeta_per_steps == 0:
        # first solve for the primal nu_loss,
        print('Step: {}. Solve for an updated tabular nu/zeta.'.format(step))
        loss = estimator.solve_nu_zeta(dataset, target_policy)
        running_losses.append(loss)
      one_step(None, None, None)

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
