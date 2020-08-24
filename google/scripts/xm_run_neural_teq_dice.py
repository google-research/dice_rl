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
import os

from google3.learning.deepmind.xmanager import hyper
import google3.learning.deepmind.xmanager2.client.google as xm

flags.DEFINE_string('exp_name', 'neural teqdice', 'Name of experiment.')
flags.DEFINE_string('cell', None, 'The cell to run jobs in.')
flags.DEFINE_integer('priority', 25, 'Priority of the job.')

flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_integer('num_seeds', 20, 'Number of random seed.')
flags.DEFINE_string('load_dir', '/cns/is-d/home/sherryy/teqdice/data/',
                    'Directory to load dataset from.')
flags.DEFINE_string(
    'save_dir',
    '/cns/is-d/home/sherryy/teqdice/results/r=2/dualdice_single_layer/',
    'Directory to save the results to.')
flags.DEFINE_boolean('eval_only', False, 'Whether to only run evaluation.')

FLAGS = flags.FLAGS


def get_args():
  if FLAGS.env_name == 'Reacher-v2':
    num_steps = 200000
    max_trajectory_length_train = 40
    nu_learning_rate = 0.0001
    zeta_learning_rate = 0.0001
    gamma = 0.99
    batch_size = 2048
  else:
    num_steps = 50000
    max_trajectory_length_train = 200
    nu_learning_rate = 0.0001
    zeta_learning_rate = 0.001
    gamma = 0.995
    batch_size = 512
  if FLAGS.eval_only:
    num_steps = 0
    num_trajectory_train = 1000
  else:
    max_trajectory_length_train = None
    num_trajectory_train = None
  args = [
      ('env_name', FLAGS.env_name),
      ('load_dir', FLAGS.load_dir),
      ('save_dir', FLAGS.save_dir),
      ('num_trajectory', 1000),
      ('num_steps', num_steps),
      ('nu_learning_rate', nu_learning_rate),
      ('zeta_learning_rate', zeta_learning_rate),
      ('max_trajectory_length_train', max_trajectory_length_train),
      ('num_trajectory_train', num_trajectory_train),
      ('gamma', gamma),
      ('batch_size', batch_size),
  ]
  return args


def build_experiment():
  runtime_worker = xm.Borg(
      cell=FLAGS.cell,
      priority=FLAGS.priority,
  )
  executable = xm.BuildTarget(
      '//third_party/py/dice_rl/google/scripts:run_neural_teq_dice',
      build_flags=['-c', 'opt', '--copt=-mavx'],
      args=get_args(),
      platform=xm.Platform.CPU,
      runtime=runtime_worker)

  parameters = hyper.product([
      hyper.sweep('seed', hyper.discrete(list(range(FLAGS.num_seeds)))),
      hyper.sweep(
          'max_trajectory_length',
          #hyper.discrete([5, 10, 20, 50, 100, 200]) # Grid
          hyper.discrete([5, 10, 20, 40])  # Reacher
      ),
  ])
  experiment = xm.ParameterSweep(executable, parameters)
  return experiment


def main(_):
  """Launch the experiment using the arguments from the command line."""
  description = xm.ExperimentDescription(
      FLAGS.exp_name, tags=[
          FLAGS.env_name,
      ])
  experiment = build_experiment()
  xm.launch_experiment(description, experiment)


if __name__ == '__main__':
  app.run(main)
