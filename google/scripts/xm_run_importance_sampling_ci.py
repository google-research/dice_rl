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

flags.DEFINE_string('exp_name', 'small_tree_IS5', 'Name of experiment.')
flags.DEFINE_string('cell', None, 'The cell to run jobs in.')
flags.DEFINE_integer('num_seeds', 200, 'Number of data seeds.')

flags.DEFINE_string('env_name', 'small_tree', 'Environment name.')
flags.DEFINE_string('load_dir', '/cns/vz-d/home/brain-ofirnachum',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir',
                    '/cns/vz-d/home/brain-ofirnachum/robust/results',
                    'Directory to save the results to.')

flags.DEFINE_boolean('avx2', False, 'Whether to run with AVX2 + FMA.')
flags.DEFINE_integer('worker_ram_fs_gb', None,
                     '(optional) Amount of tmp fs ram to allocate each job.')

flags.DEFINE_bool('tabular_obs', True, 'Whether to use tabular observations.')
flags.DEFINE_string('mode', 'weighted-step-wise', 'Importance sampling mode.')
flags.DEFINE_enum('ci_method', 'BCa', ['CH', 'BE', 'C-BE', 'TT', 'BCa'],
                  'method for confidence interval construction.')

flags.DEFINE_integer('num_trajectory_data', None,
                     'Number of trajectories in data.')


FLAGS = flags.FLAGS


AVX_BUILD_FLAGS = ['-c', 'opt', '--copt=-mavx', '--experimental_deps_ok']
AVX2_BUILD_FLAGS = ['-c', 'opt', '--copt=-mavx2', '--copt=-mfma',
                                        '--experimental_deps_ok']
AVX2_CONSTRAINTS = ('platform_family!=ikaria,~platform_family!=ibis,'
                                        '~platform_family!=iota,')


def build_experiment():
  save_dir = os.path.join(FLAGS.save_dir, FLAGS.exp_name)

  requirements = xm.Requirements(ram=10 * xm.GiB)
  if FLAGS.worker_ram_fs_gb is not None:
    requirements.tmp_ram_fs_size = FLAGS.worker_ram_fs_gb * xm.GiB

  overrides = xm.BorgOverrides()
  overrides.requirements.autopilot_params = ({'min_cpu': 1})

  if FLAGS.avx2:
    overrides.requirements.constraints = AVX2_CONSTRAINTS

  runtime_worker = xm.Borg(
      cell=FLAGS.cell,
      priority=115,
      requirements=requirements,
      overrides=overrides,
  )
  executable = xm.BuildTarget(
      '//third_party/py/dice_rl/google/scripts:run_importance_sampling_ci',
      build_flags=['-c', 'opt', '--copt=-mavx'],
      args=[
          ('gfs_user', 'brain-ofirnachum'),
          ('env_name', FLAGS.env_name),
          ('load_dir', FLAGS.load_dir),
          ('num_trajectory_data', FLAGS.num_trajectory_data),
          ('save_dir', save_dir),
          ('num_steps', 10000),
          ('alpha', -1.0),
          ('ci_method', FLAGS.ci_method),
          ('mode', FLAGS.mode),
          ('tabular_obs', FLAGS.tabular_obs),
          ('use_trained_policy', False),
          ('use_doubly_robust', False),
      ],
      platform=xm.Platform.CPU,
      runtime=runtime_worker)

  parameters = hyper.product([
      hyper.sweep('seed', hyper.discrete(list(range(FLAGS.num_seeds)))),
      ## Reacher
      #hyper.sweep('gamma', [0.99]),
      #hyper.sweep('num_trajectory', [25]),
      #hyper.sweep('max_trajectory_length', [100]),
      ## FrozenLake
      # hyper.sweep('gamma', [0.99]),
      # hyper.sweep('num_trajectory', [50, 100, 200, 500, 1000]),
      # hyper.sweep('max_trajectory_length', [100]),
      ## SmallTree
       hyper.sweep('gamma', [0.0]),
       hyper.sweep('num_trajectory', [50, 100, 200]),
       hyper.sweep('max_trajectory_length', [1]),
      ## Taxi
      # hyper.sweep('gamma', [0.99]),
      # hyper.sweep('num_trajectory', [20, 50, 100]),
      # hyper.sweep('max_trajectory_length', [500]),
      ## universally needed
      hyper.sweep('delta', [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
  ])
  experiment = xm.ParameterSweep(executable, parameters)
  experiment = xm.WithTensorBoard(experiment, save_dir)
  return experiment


def main(_):
  """Launch the experiment using the arguments from the command line."""
  description = xm.ExperimentDescription(
      FLAGS.exp_name, tags=[
          FLAGS.env_name,
          FLAGS.ci_method,
          FLAGS.mode
      ])
  experiment = build_experiment()
  xm.launch_experiment(description, experiment)


if __name__ == '__main__':
  app.run(main)
