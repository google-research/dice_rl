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
"""Script for running tabular BayesDICE.

Make sure to generate the datasets prior to running this script (see
`scripts/create_dataset.py`). The default parameters here should reproduce
the published bandit and frozenlake results. For Taxi, pass in
solve_for_state_action_ratio=False.
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
import tensorflow_probability as tfp
import pickle

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

from dice_rl.environments.env_policies import get_target_policy
import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.taxi as taxi
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.estimators.tabular_bayes_dice import TabularBayesDice
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'frozenlake', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 5, 'Number of trajectories to collect.')
flags.DEFINE_float('alpha', 0.0,
                   'How close is the behavior policy to optimal policy.')
flags.DEFINE_integer('max_trajectory_length', 100,
                     'Cutoff trajectory at this step.')
flags.DEFINE_bool('tabular_obs', True, 'Whether to use tabular observations.')
flags.DEFINE_string('load_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('save_dir', None, 'Directory to save estimation results.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')

flags.DEFINE_integer('num_steps', 50000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
flags.DEFINE_float('zeta_learning_rate', 1e-2, 'Zeta learning rate.')
flags.DEFINE_float('nu_learning_rate', 1e-2, 'Value learning rate.')
flags.DEFINE_bool('solve_for_state_action_ratio', True,
                  'Whether to use tabular observations.')

flags.DEFINE_float('alpha_target', 1.0,
                   'How close is the target policy to optimal policy.')
flags.DEFINE_float('kl_regularizer', 1., 'LP regularizer of kl(q||p).')
flags.DEFINE_float('eps_std', 1., 'Epsilon std for reparametrization.')


def main(argv):
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  load_dir = FLAGS.load_dir
  save_dir = FLAGS.save_dir
  gamma = FLAGS.gamma
  assert 0 <= gamma < 1.
  alpha = FLAGS.alpha
  alpha_target = FLAGS.alpha_target

  num_steps = FLAGS.num_steps
  batch_size = FLAGS.batch_size
  zeta_learning_rate = FLAGS.zeta_learning_rate
  nu_learning_rate = FLAGS.nu_learning_rate
  solve_for_state_action_ratio = FLAGS.solve_for_state_action_ratio
  eps_std = FLAGS.eps_std
  kl_regularizer = FLAGS.kl_regularizer

  target_policy = get_target_policy(
      load_dir, env_name, tabular_obs, alpha=alpha_target)

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
  print('num loaded steps', dataset.num_steps)
  print('num loaded total steps', dataset.num_total_steps)
  print('num loaded episodes', dataset.num_episodes)
  print('num loaded total episodes', dataset.num_total_episodes)
  print('behavior per-step',
        estimator_lib.get_fullbatch_average(dataset, gamma=gamma))

  train_hparam_str = ('eps{EPS}_kl{KL}').format(EPS=eps_std, KL=kl_regularizer)

  if save_dir is not None:
    # Save for a specific alpha target
    target_hparam_str = hparam_str.replace(
        'alpha{}'.format(alpha), 'alpha{}_alphat{}'.format(alpha, alpha_target))
    save_dir = os.path.join(save_dir, target_hparam_str, train_hparam_str)
    summary_writer = tf.summary.create_file_writer(logdir=save_dir)
  else:
    summary_writer = tf.summary.create_noop_writer()

  estimator = TabularBayesDice(
      dataset_spec=dataset.spec,
      gamma=gamma,
      solve_for_state_action_ratio=solve_for_state_action_ratio,
      zeta_learning_rate=zeta_learning_rate,
      nu_learning_rate=nu_learning_rate,
      kl_regularizer=kl_regularizer,
      eps_std=eps_std,
  )
  estimator.prepare_dataset(dataset, target_policy)

  global_step = tf.Variable(0, dtype=tf.int64)
  tf.summary.experimental.set_step(global_step)
  with summary_writer.as_default():
    running_losses = []
    running_estimates = []
    for step in range(num_steps):
      loss = estimator.train_step()[0]
      running_losses.append(loss)
      global_step.assign_add(1)

      if step % 500 == 0 or step == num_steps - 1:
        print('step', step, 'losses', np.mean(running_losses, 0))
        estimate = estimator.estimate_average_reward(dataset, target_policy)
        tf.debugging.check_numerics(estimate, 'NaN in estimate')
        running_estimates.append(estimate)
        tf.print('est', tf.math.reduce_mean(estimate),
                 tf.math.reduce_std(estimate))

        running_losses = []

  if save_dir is not None:
    with tf.io.gfile.GFile(os.path.join(save_dir, 'results.npy'), 'w') as f:
      np.save(f, running_estimates)
    print('saved results to %s' % save_dir)

  print('Done!')


if __name__ == '__main__':
  app.run(main)
