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

flags.DEFINE_string('exp_name', 'teqdice', 'Name of experiment.')
flags.DEFINE_string('cell', None, 'The cell to run jobs in.')
flags.DEFINE_integer('priority', 25, 'Priority to run job as.')

flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_integer('num_seeds', 20, 'Number of random seed.')
flags.DEFINE_string('load_dir', '/cns/is-d/home/sherryy/teqdice/data/',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir', '/cns/is-d/home/sherryy/teqdice/results/r=2/',
                    'Directory to save the results to.')

FLAGS = flags.FLAGS


def build_experiment():
  runtime_worker = xm.Borg(
      cell=FLAGS.cell,
      priority=FLAGS.priority,
  )
  executable = xm.BuildTarget(
      '//third_party/py/dice_rl/google/scripts:run_tabular_teq_dice',
      build_flags=['-c', 'opt', '--copt=-mavx'],
      args=[
          ('env_name', FLAGS.env_name),
          ('load_dir', FLAGS.load_dir),
          ('save_dir', os.path.join(FLAGS.save_dir, FLAGS.exp_name)),
          ('max_trajectory_length_train', 50),
          ('num_trajectory', 1000),
      ],
      platform=xm.Platform.CPU,
      runtime=runtime_worker)

  parameters = hyper.product([
      hyper.sweep('seed', hyper.discrete(list(range(FLAGS.num_seeds)))),
      hyper.sweep('step_encoding', hyper.categorical([None, 'one_hot'])),
      hyper.sweep('max_trajectory_length', hyper.discrete([5, 10, 20, 50])),
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
