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

flags.DEFINE_string('exp_name', 'tabular_bayes', 'Name of experiment.')
flags.DEFINE_string('cell', None, 'The cell to run jobs in.')
flags.DEFINE_integer('priority', 200, 'Priority of the job.')

flags.DEFINE_string('env_name', 'frozenlake', 'Environment name.')
flags.DEFINE_integer('num_seeds', 1, 'Number of random seed.')
flags.DEFINE_string('load_dir', '/cns/is-d/home/sherryy/prodice/data/',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir', '/cns/is-d/home/sherryy/prodice/results/',
                    'Directory to save the results to.')
FLAGS = flags.FLAGS


def build_experiment():
  requirements = xm.Requirements()
  runtime_worker = xm.Borg(
      cell=FLAGS.cell,
      priority=FLAGS.priority,
      requirements=requirements,
  )
  gamma = 0.99 if FLAGS.env_name != 'bandit' else 0.0
  save_dir = os.path.join(
      FLAGS.save_dir.format(CELL=FLAGS.cell),
      '{EXP}_gamma{GAMMA}'.format(EXP=FLAGS.exp_name, GAMMA=gamma))
  executable = xm.BuildTarget(
      '//third_party/py/dice_rl/google/scripts:run_tabular_bayes_dice',
      build_flags=['-c', 'opt', '--copt=-mavx'],
      args=[('env_name', FLAGS.env_name), ('gamma', gamma),
            ('save_dir', save_dir), ('load_dir', FLAGS.load_dir),
            ('num_steps', 50000),
            ('solve_for_state_action_ratio',
             False if FLAGS.env_name == 'taxi' else True)],
      platform=xm.Platform.CPU,
      runtime=runtime_worker)

  parameters = hyper.product([
      hyper.sweep('seed', hyper.discrete(list(range(FLAGS.num_seeds)))),
      hyper.sweep('alpha', hyper.discrete([0.])),
      hyper.sweep('alpha_target',
                  hyper.discrete([i / 100 for i in range(86, 95)])),
      hyper.sweep('kl_regularizer', hyper.discrete([5.])),
      hyper.sweep('num_trajectory', hyper.discrete([5, 10, 25, 50])),
      hyper.sweep(
          'max_trajectory_length',
          hyper.discrete([
              #1, # bandit
              #500, # taxi
              100,  # frozenlake
          ])),
  ])
  experiment = xm.ParameterSweep(executable, parameters)
  experiment = xm.WithTensorBoard(experiment, save_dir)
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
