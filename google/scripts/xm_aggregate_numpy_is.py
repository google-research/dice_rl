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

from google3.learning.deepmind.xmanager import hyper
import google3.learning.deepmind.xmanager2.client.google as xm

FLAGS = flags.FLAGS

flags.DEFINE_string('cell', None, 'Cell to run jobs.')
flags.DEFINE_string('mode', 'weighted_step-wise', 'IS mode.')


def build_experiment():

  requirements = xm.Requirements(ram=10 * xm.GiB,
      gpu_types=[xm.GpuType.P100])

  overrides = xm.BorgOverrides()
  # overrides.requirements.autopilot_params = ({'min_cpu': 1})
  overrides.env_vars.TMPDIR = '/tmp'

  runtime_worker = xm.Borg(
      cell=FLAGS.cell,
      priority=115,
      requirements=requirements,
      overrides=overrides,
  )

  executable = xm.BuildTarget(
      '//third_party/py/dice_rl/google/scripts:aggregate_numpy_is',
      build_flags=['-c', 'opt', '--copt=-mavx'],
      args=[
          ('gfs_user', 'mudcats'),
          ('mode', FLAGS.mode),
      ],
      platform=xm.Platform.GPU,
      runtime=runtime_worker)

  parameters = hyper.product([
      hyper.sweep('directory', [
          '/cns/pw-d/home/mudcats/dev/algae_ci/Taxi_IS/',
          '/cns/pw-d/home/mudcats/dev/algae_ci/FrozenLake_IS/',
          '/cns/pw-d/home/mudcats/dev/algae_ci/SmallTree_IS/'
      ]),
  ])

  experiment = xm.ParameterSweep(executable, parameters)

  return experiment


def main(_):
  """Launch the experiment using the arguments from the command line."""
  description = xm.ExperimentDescription(
      'Aggregate IS data.', tags=[FLAGS.mode])
  experiment = build_experiment()
  xm.launch_experiment(description, experiment)


if __name__ == '__main__':
  app.run(main)
