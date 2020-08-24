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
from dice_rl.networks.policy_network import PolicyNetwork
from dice_rl.networks.value_network import ValueNetwork
from dice_rl.google.estimators.importance_sampling_ci import ImportanceSamplingCI
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
import google3.learning.deepmind.xmanager2.client.google as xm  # pylint: disable=unused-import

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'taxi', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory_data', None,
                     'Number of trajectories in data.')
flags.DEFINE_integer('num_trajectory', 100,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 500,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('policy_learning_rate', 0.001, 'Learning rate for policy.')
flags.DEFINE_float('q_learning_rate', 0.0003, 'Learning rate for Q-network.')
flags.DEFINE_bool('tabular_obs', True, 'Whether to use tabular observations.')
flags.DEFINE_string('load_dir', '/cns/vz-d/home/brain-ofirnachum',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir', None, 'Directory to save estimation results.')
flags.DEFINE_float('alpha', 0.0, 'How close to target policy.')
flags.DEFINE_string('mode', 'trajectory-wise', 'Importance sampling mode.')
flags.DEFINE_enum('ci_method', 'CH', ['CH', 'BE', 'C-BE', 'TT', 'BCa'],
                  'method for confidence interval construction.')
flags.DEFINE_float('delta', 0.9, 'Confidence interval.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_integer('num_steps', 10000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')

flags.DEFINE_bool('use_trained_policy', False, 'Whether to use trained policy.')
flags.DEFINE_bool('use_doubly_robust', False,
                  'Whether to use learn Q-values and use them for doubly robust estimation.')


def main(argv):
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  load_dir = FLAGS.load_dir
  save_dir = FLAGS.save_dir
  policy_learning_rate = FLAGS.policy_learning_rate
  q_learning_rate = FLAGS.q_learning_rate
  batch_size = FLAGS.batch_size
  mode = FLAGS.mode
  ci_method = FLAGS.ci_method
  delta = FLAGS.delta
  delta_tail = 1 - delta
  gamma = FLAGS.gamma
  num_steps = FLAGS.num_steps
  use_trained_policy = FLAGS.use_trained_policy
  use_doubly_robust = FLAGS.use_doubly_robust
  assert 0 <= gamma < 1.

  target_policy = get_target_policy(load_dir, env_name, tabular_obs)

  hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    TAB=tabular_obs,
                    ALPHA=alpha,
                    SEED=seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=max_trajectory_length)

  if FLAGS.num_trajectory_data is not None:
    hparam_str_data = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                       'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                           ENV_NAME=env_name,
                           TAB=tabular_obs,
                           ALPHA=alpha,
                           SEED=seed,
                           NUM_TRAJ=FLAGS.num_trajectory_data,
                           MAX_TRAJ=max_trajectory_length)
  else:
    hparam_str_data = hparam_str

  directory = os.path.join(load_dir, hparam_str_data)
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

  train_hparam_str = (
      'plr{P_LR}_tp{TRAINED_P}_batch{BATCH_SIZE}_mode{MODE}_CI{CI_METHOD}_UTP{USE_TRAINED_POLICY}_gam{GAMMA}_del{DELTA}'
  ).format(
      P_LR=policy_learning_rate,
      TRAINED_P=use_trained_policy,
      BATCH_SIZE=batch_size,
      MODE=mode,
      CI_METHOD=ci_method,
      USE_TRAINED_POLICY=use_trained_policy,
      GAMMA=gamma,
      DELTA=delta)

  if save_dir is not None:
    save_dir = os.path.join(save_dir, hparam_str, train_hparam_str)
    summary_writer = tf.summary.create_file_writer(logdir=save_dir)
  else:
    summary_writer = tf.summary.create_noop_writer()

  def non_negative_reward_translation(env_step):
    return env_step.reward - min_reward

  def inv_non_negative_estimate_translation(estimate):
    return estimate + min_reward

  if use_trained_policy:
    activation_fn = tf.nn.relu
    kernel_initializer = tf.keras.initializers.GlorotUniform()
    hidden_dims = (64, 64)
    policy_optimizer = tf.keras.optimizers.Adam(
        policy_learning_rate, beta_1=0.0, beta_2=0.0)
    policy_network = PolicyNetwork(
        dataset.spec.observation,
        dataset.spec.action,
        fc_layer_params=hidden_dims,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=kernel_initializer)
  else:
    policy_optimizer = None
    policy_network = None

  if use_doubly_robust:
    activation_fn = tf.nn.relu
    kernel_initializer = tf.keras.initializers.GlorotUniform()
    hidden_dims = (64, 64)
    q_optimizer = tf.keras.optimizers.Adam(q_learning_rate)
    q_network = ValueNetwork(
        (dataset.spec.observation, dataset.spec.action),
        fc_layer_params=hidden_dims,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=kernel_initializer)
  else:
    q_optimizer = None
    q_network = None

  estimator = ImportanceSamplingCI(
      dataset_spec=dataset.spec,
      policy_optimizer=policy_optimizer,
      policy_network=policy_network,
      mode=mode,
      ci_method=ci_method,
      delta_tail=delta_tail,
      gamma=gamma,
      reward_fn=non_negative_reward_translation,
      q_network=q_network,
      q_optimizer=q_optimizer)
  global_step = tf.Variable(0, dtype=tf.int64)
  tf.summary.experimental.set_step(global_step)

  # Following is for policy learning + IS confidence interval
  @tf.function
  def one_step(data_batch):
    global_step.assign_add(1)
    loss = estimator.train_step(data_batch, target_policy)
    return loss

  with summary_writer.as_default():
    running_losses = []
    running_estimates = []
    running_estimate_cis = []
    for step in range(num_steps):
      data_batch = dataset.get_step(batch_size, num_steps=2)
      loss = one_step(data_batch)
      running_losses.append(loss)

      if step % 500 == 0 or step == num_steps - 1:
        print('step', step, 'loss', np.mean(running_losses, 0))
        running_losses = []
        estimate = estimator.estimate_average_reward(
            dataset, target_policy, episode_limit=num_trajectory)
        estimate = inv_non_negative_estimate_translation(estimate)
        running_estimates.append(estimate)
        print('estimated per step avg %s' % estimate)
        print('avg last 3 estimated per step avg %s' %
              np.mean(running_estimates[-3:], axis=0))

        estimate_ci = estimator.estimate_reward_ci(dataset, target_policy)
        estimate_ci = np.array(
            [inv_non_negative_estimate_translation(ele) for ele in estimate_ci])
        running_estimate_cis.append(estimate_ci)
        print('estimated CI per step avg %s' % estimate_ci)
        print('avg last 3 estimated CI per step avg %s' %
              np.mean(running_estimate_cis[-3:], axis=0))

  if save_dir is not None:
    results_filename = os.path.join(save_dir, 'results.npy')
    with tf.io.gfile.GFile(results_filename, 'w') as f:
      np.save(f, running_estimate_cis)
  print('Done!')


if __name__ == '__main__':
  app.run(main)
