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
"""Makes sure that scripts/run_env.py runs without error."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import os
import mock
import tensorflow.compat.v2 as tf
from dice_rl.scripts import run_env


class RunEnvTest(tf.test.TestCase):

  def test_run_env(self):
    flags.FLAGS.num_trajectory = 10
    flags.FLAGS.max_trajectory_length = 10
    run_env.main(None)


if __name__ == '__main__':
  tf.test.main()
