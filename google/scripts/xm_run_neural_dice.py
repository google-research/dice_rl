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

flags.DEFINE_string('env_name', 'Reacher-v2', 'Environment name.')
flags.DEFINE_integer('num_seeds', 5, 'Number of random seed.')
flags.DEFINE_string('load_dir',
                    '/cns/is-d/home/sherryy/teqdice/data/third_party/',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir',
                    '/cns/{CELL}-d/home/sherryy/teqdice/results/r=3/',
                    'Directory to save the results to.')
flags.DEFINE_float('gamma', 0.99, 'Gamma value')
FLAGS = flags.FLAGS


def build_experiment():
  requirements = xm.Requirements(ram=10 * xm.GiB)
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
      '//third_party/py/dice_rl/scripts:run_neural_dice',
      build_flags=['-c', 'opt', '--copt=-mavx'],
      args=[
          ('env_name', FLAGS.env_name),
          ('gamma', FLAGS.gamma),
          ('save_dir', save_dir),
          ('load_dir', FLAGS.load_dir),
      ],
      platform=xm.Platform.CPU,
      runtime=runtime_worker)

  max_traj_dict = {
      'grid': 100,
      'taxi': 200,
      'Reacher-v2': 40,
      'reacher': 200,
      'cartpole': 250,
  }
  parameters = hyper.product([
      hyper.sweep('seed', hyper.discrete(list(range(FLAGS.num_seeds)))),
      hyper.sweep('zero_reward', hyper.categorical([False])),
      hyper.sweep('norm_regularizer', hyper.discrete([0.0, 1.0])),
      hyper.sweep('zeta_pos', hyper.categorical([True, False])),
      hyper.sweep('primal_form', hyper.categorical([False])),
      hyper.sweep('num_steps', hyper.discrete([200000])),
      hyper.sweep('f_exponent', hyper.discrete([2.0])),
      hyper.zipit([
          hyper.sweep('primal_regularizer', hyper.discrete([0.0, 1.0])),
          hyper.sweep('dual_regularizer', hyper.discrete([1.0, 0.0])),
      ]),
      hyper.zipit([
          hyper.sweep('nu_learning_rate', hyper.discrete([0.0001])),
          hyper.sweep('zeta_learning_rate', hyper.discrete([0.0001])),
      ]),
      hyper.sweep('alpha', hyper.discrete([0.0])),
      hyper.sweep('num_trajectory', hyper.discrete([100])),
      hyper.sweep(
          'max_trajectory_length',
          hyper.discrete([
              100  #max_traj_dict[FLAGS.env_name]
          ])),
  ])
  experiment = xm.ParameterSweep(
      executable, parameters, max_parallel_work_units=2000)
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
