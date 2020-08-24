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

flags.DEFINE_string('exp_name', 'taxi4q500', 'Name of experiment.')
flags.DEFINE_string('cell', None, 'The cell to run jobs in.')
flags.DEFINE_integer('priority', 115, 'Priority to run job as.')

flags.DEFINE_string('env_name', 'taxi', 'Environment name.')
flags.DEFINE_integer('num_seeds', 200, 'Number of random seed.')
flags.DEFINE_string('load_dir', '/cns/vz-d/home/brain-ofirnachum',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir', '/cns/vz-d/home/brain-ofirnachum/robust/results',
                    'Directory to save the results to.')
flags.DEFINE_boolean('avx2', False, 'Whether to run with AVX2 + FMA.')
flags.DEFINE_integer('worker_ram_fs_gb', None,
                     '(optional) Amount of tmp fs ram to allocate each job.')

FLAGS = flags.FLAGS

AVX_BUILD_FLAGS = ['-c', 'opt', '--copt=-mavx', '--experimental_deps_ok']
AVX2_BUILD_FLAGS = ['-c', 'opt', '--copt=-mavx2', '--copt=-mfma',
                                        '--experimental_deps_ok']
AVX2_CONSTRAINTS = ('platform_family!=ikaria,~platform_family!=ibis,'
                                        '~platform_family!=iota,')

CHI2_PERCENTILE = {
    0: 0.,
    50: 0.4549,
    60: 0.7083,
    70: 1.0742,
    80: 1.6423,
    90: 2.7055,
    95: 3.8415,
}

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
      '//third_party/py/dice_rl/google/scripts:run_tabular_robust_dice',
      build_flags=AVX2_BUILD_FLAGS if FLAGS.avx2 else AVX_BUILD_FLAGS,
      args=[
          ('env_name', FLAGS.env_name),
          ('load_dir', FLAGS.load_dir),
          ('save_dir', save_dir),
          ('num_steps', 200000),
          ('num_trajectory', 100),
          ('max_trajectory_length', 500),
      ],
      platform=xm.Platform.CPU,
      runtime=runtime_worker)
  num_samples = 1

  parameters = hyper.product([
      hyper.sweep('seed', hyper.discrete(list(range(FLAGS.num_seeds)))),
      hyper.sweep('alpha', [0.0]),
      hyper.sweep('gamma', [0.99]),
      hyper.sweep('limit_episodes', [20, 50, 100]),
      hyper.sweep('divergence_limit', [0.5 * CHI2_PERCENTILE[p] / num_samples
                                       for p in [0, 50, 60, 70, 80, 90, 95]]),
      hyper.sweep('algae_alpha', [0.01]),
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
