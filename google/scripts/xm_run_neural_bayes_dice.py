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

flags.DEFINE_string('exp_name', 'dice_family', 'Name of experiment.')
flags.DEFINE_string('cell', None, 'The cell to run jobs in.')
flags.DEFINE_integer('priority', 200, 'Priority of the job.')

flags.DEFINE_string('env_name', 'reacher', 'Environment name.')
flags.DEFINE_integer('num_seeds', 1, 'Number of random seed.')
flags.DEFINE_string('load_dir', '/cns/is-d/home/sherryy/prodice/data/',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir', '/cns/is-d/home/sherryy/prodice/results/',
                    'Directory to save the results to.')
flags.DEFINE_float('gamma', 0.99, 'Gamma value')
FLAGS = flags.FLAGS


def build_experiment():
  requirements = xm.Requirements()
  overrides = xm.BorgOverrides()
  overrides.requirements.autopilot_params = ({'min_cpu': 1})
  runtime_worker = xm.Borg(
      cell=FLAGS.cell,
      priority=FLAGS.priority,
      requirements=requirements,
      overrides=overrides,
  )
  save_dir = os.path.join(
      FLAGS.save_dir.format(CELL=FLAGS.cell),
      '{EXP}_gamma{GAMMA}'.format(EXP=FLAGS.exp_name, GAMMA=FLAGS.gamma))
  executable = xm.BuildTarget(
      '//third_party/py/dice_rl/scripts:run_neural_bayes_dice',
      build_flags=['-c', 'opt', '--copt=-mavx'],
      args=[
          ('env_name', FLAGS.env_name),
          ('gamma', FLAGS.gamma),
          ('save_dir', save_dir),
          ('load_dir', FLAGS.load_dir),
          ('num_steps', 50000),
      ],
      platform=xm.Platform.CPU,
      runtime=runtime_worker)

  parameters = hyper.product([
      hyper.sweep('seed', hyper.discrete(list(range(FLAGS.num_seeds)))),
      hyper.sweep('kl_regularizer', hyper.discrete([5.])),
      hyper.sweep('alpha', hyper.discrete([i / 10 for i in range(6)])),
      hyper.sweep('alpha_target', hyper.discrete([0.75, 0.8, 0.85, 0.9, 0.95])),
      hyper.sweep('num_trajectory', hyper.discrete([10, 25, 50, 100])),
      hyper.sweep('max_trajectory_length', hyper.discrete([100])),
  ])
  experiment = xm.ParameterSweep(executable, parameters)
  experiment = xm.WithTensorBoard(experiment, save_dir)
  return experiment


def main(_):
  """Launch the experiment using the arguments from the command line."""
  description = xm.ExperimentDescription(
      FLAGS.exp_name, tags=[
          FLAGS.env_name,
          str(FLAGS.gamma),
      ])
  experiment = build_experiment()
  xm.launch_experiment(description, experiment)


if __name__ == '__main__':
  app.run(main)
