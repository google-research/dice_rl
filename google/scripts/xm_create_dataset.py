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

flags.DEFINE_string('exp_name', 'create dataset', 'Name of experiment.')
flags.DEFINE_string('cell', None, 'The cell to run jobs in.')
flags.DEFINE_integer('priority', 25, 'Priority to run job as.')

flags.DEFINE_integer('num_seeds', 200, 'Number of random seed.')
flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_string('load_dir',
                    '/cns/is-d/home/sherryy/teqdice/data/third_party',
                    'Directory to load policies from.')
flags.DEFINE_string('save_dir',
                    '/cns/is-d/home/sherryy/teqdice/data/third_party',
                    'Directory to save dataset to.')
flags.DEFINE_bool('tabular_obs', False, 'Whether to use tabular observations.')

FLAGS = flags.FLAGS


def build_experiment():
  runtime_worker = xm.Borg(
      cell=FLAGS.cell,
      priority=FLAGS.priority,
  )
  executable = xm.BuildTarget(
      '//third_party/py/dice_rl/scripts:create_dataset',
      build_flags=['-c', 'opt', '--copt=-mavx'],
      args=[
          ('env_name', FLAGS.env_name),
          ('load_dir', FLAGS.load_dir),
          ('save_dir', FLAGS.save_dir),
          ('tabular_obs', FLAGS.tabular_obs),
          ('force', True),
      ],
      platform=xm.Platform.CPU,
      runtime=runtime_worker)

  parameters = hyper.product([
      hyper.sweep('alpha', hyper.discrete([0.0, 1.0])),
      hyper.sweep('seed', hyper.discrete(list(range(FLAGS.num_seeds)))),
      hyper.sweep('num_trajectory', hyper.discrete([100])),
      hyper.sweep(
          'max_trajectory_length',
          hyper.discrete([
              100,
              #5,
              #10,
              #20,
              #40,  #50, 100, 200
          ])),
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
