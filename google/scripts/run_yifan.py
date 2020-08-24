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
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import actor_policy, greedy_policy
from tf_agents.policies.py_tf_eager_policy import SavedModelPyTFEagerPolicy

from dice_rl.environments import suites
from dice_rl.google.estimators.neural_qlearning import NeuralQLearning
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_agents_onpolicy_dataset import TFAgentsOnpolicyDataset
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
from dice_rl.data.perturbed_dataset import PerturbedDataset

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('data_name', 'eps1', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_string('data_load_dir', '/cns/vz-d/home/brain-ofirnachum',
                    'Directory to load dataset from.')
flags.DEFINE_string('policy_load_dir', '/cns/vz-d/home/brain-ofirnachum/sac2',
                    'Directory to load target policy from.')
flags.DEFINE_float('gamma', 0.995,
                   'Discount factor.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('nstep_returns', 1,
                     'Use n-step returns with this many steps.')
flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')


def get_target_policy(load_dir, env_name):
  """Gets target policy."""
  env = tf_py_environment.TFPyEnvironment(suites.load_mujoco(env_name))
  actor_net = actor_distribution_network.ActorDistributionNetwork(
      env.observation_spec(),
      env.action_spec(),
      fc_layer_params=(256, 256),
      continuous_projection_net=tanh_normal_projection_network
      .TanhNormalProjectionNetwork)
  policy = actor_policy.ActorPolicy(
      time_step_spec=env.time_step_spec(),
      action_spec=env.action_spec(),
      actor_network=actor_net,
      training=False)
  policy = greedy_policy.GreedyPolicy(policy)

  checkpoint = tf.train.Checkpoint(policy=policy)

  directory = os.path.join(load_dir, env_name, 'train/policy')
  checkpoint_filename = tf.train.latest_checkpoint(directory)
  print('Loading policy from %s' % checkpoint_filename)
  checkpoint.restore(checkpoint_filename).assert_existing_objects_matched()
  policy = policy.wrapped_policy

  return policy, env


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


def main(argv):
  env_name = FLAGS.env_name
  data_name = FLAGS.data_name
  seed = FLAGS.seed
  policy_load_dir = FLAGS.policy_load_dir
  data_load_dir = FLAGS.data_load_dir
  gamma = FLAGS.gamma
  assert 0 <= gamma < 1.
  learning_rate = FLAGS.learning_rate
  nstep_returns = FLAGS.nstep_returns
  num_steps = FLAGS.num_steps
  batch_size = FLAGS.batch_size

  target_policy, env = get_target_policy(policy_load_dir, env_name)

  directory = os.path.join(data_load_dir,
                           'yifan_%s_%s' % (env_name, data_name))
  print('Loading dataset.')
  onpolicy_dataset = TFAgentsOnpolicyDataset(env, target_policy, 1000)
  write_dataset = TFOffpolicyDataset(
      onpolicy_dataset.spec)
  batch_size = 20
  num_trajectory = 10
  for batch_num in range(1 + (num_trajectory - 1) // batch_size):
    print(batch_num)
    num_trajectory_after_batch = min(num_trajectory, batch_size * (batch_num + 1))
    num_trajectory_to_get = num_trajectory_after_batch - batch_num * batch_size
    episodes, valid_steps = onpolicy_dataset.get_episode(
        batch_size=num_trajectory_to_get)
    add_episodes_to_dataset(episodes, valid_steps, write_dataset)
  dataset = write_dataset
  """
  dataset = Dataset.load(directory)
  """
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
                             num_perturbations=None, #10,
                             perturbation_scale=1.)

  value_network = ValueNetwork((dataset.spec.observation, dataset.spec.action),
                               fc_layer_params=(64, 64),
                               output_dim=None) #10)
  optimizer = tf.keras.optimizers.Adam(learning_rate)

  estimator = NeuralQLearning(dataset.spec, value_network, optimizer, gamma,
                              num_qvalues=None, #10,
                              num_samples=1)
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
