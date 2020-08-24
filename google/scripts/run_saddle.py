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

import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.tree as tree
import dice_rl.environments.gridworld.taxi as taxi
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.google.rl_algos.tabular_saddle_point import TabularSaddlePoint
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_agents_onpolicy_dataset import TFAgentsOnpolicyDataset
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 500,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 20,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', -1.0,
                   'How close to target policy.')
flags.DEFINE_bool('tabular_obs', True,
                  'Whether to use tabular observations.')
flags.DEFINE_string('load_dir', '/cns/vz-d/home/brain-ofirnachum',
                    'Directory to load dataset from.')
flags.DEFINE_float('gamma', 0.95,
                   'Discount factor.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 1024, 'Batch size.')


def get_onpolicy_dataset(env_name, tabular_obs, policy_fn, policy_info_spec):
  """Gets target policy."""
  if env_name == 'taxi':
    env = taxi.Taxi(tabular_obs=tabular_obs)
  elif env_name == 'grid':
    env = navigation.GridWalk(tabular_obs=tabular_obs)
  elif env_name == 'tree':
    env = tree.Tree(branching=2, depth=10)
  else:
    raise ValueError('Unknown environment: %s.' % env_name)

  tf_env = tf_py_environment.TFPyEnvironment(
      gym_wrapper.GymWrapper(env))
  tf_policy = common_utils.TFAgentsWrappedPolicy(
      tf_env.time_step_spec(), tf_env.action_spec(),
      policy_fn, policy_info_spec,
      emit_log_probability=True)

  return TFAgentsOnpolicyDataset(tf_env, tf_policy)


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
  num_steps = FLAGS.num_steps
  batch_size = FLAGS.batch_size

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
  print('num loaded steps', dataset.num_steps)
  print('num loaded total steps', dataset.num_total_steps)
  print('num loaded episodes', dataset.num_episodes)
  print('num loaded total episodes', dataset.num_total_episodes)

  estimate = estimator_lib.get_fullbatch_average(dataset, gamma=gamma)
  print('data per step avg', estimate)

  optimizer = tf.keras.optimizers.Adam(learning_rate)
  algo = TabularSaddlePoint(
      dataset.spec, optimizer,
      gamma=gamma)

  losses = []
  for step in range(num_steps):
    init_batch, _ = dataset.get_episode(batch_size, truncate_episode_at=1)
    init_batch = tf.nest.map_structure(lambda t: t[:, 0, ...], init_batch)
    batch = dataset.get_step(batch_size, num_steps=2)
    loss, policy_loss = algo.train_step(init_batch, batch)
    losses.append(loss)
    if step % 100 == 0 or step == num_steps - 1:
      print('step', step, 'loss', np.mean(losses, 0))
      losses = []
      policy_fn, policy_info_spec = algo.get_policy()
      onpolicy_data = get_onpolicy_dataset(env_name, tabular_obs,
                                           policy_fn, policy_info_spec)
      onpolicy_episodes, _ = onpolicy_data.get_episode(
          10, truncate_episode_at=40)
      print('estimated per step avg', np.mean(onpolicy_episodes.reward))

  print('Done!')


if __name__ == '__main__':
  app.run(main)
