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
"""Makes sure that get_env_and_policy in env_policies.py runs without error."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import os
from dice_rl.environments.env_policies import get_env_and_policy
import tensorflow.compat.v2 as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy


class EnvPoliciesTest(tf.test.TestCase):

  def _call_get_env_and_policy(self, load_dir, env_name):
    tf_env, policy = get_env_and_policy(load_dir, env_name, 0.0)
    self.assertIsInstance(tf_env, tf_py_environment.TFPyEnvironment)
    self.assertIsInstance(policy, tf_policy.TFPolicy)

  def test_get_env_and_policy(self):
    load_dir = 'testdata/'
    load_dir = os.path.join(flags.FLAGS.test_srcdir, load_dir)
    env_names = [
        'grid', 'tree', 'bandit', 'small_tree', 'reacher', 'Reacher-v2',
        'cartpole', 'CartPole-v0', 'frozenlake', 'FrozenLake-v1',
        'HalfCheetah-v2'
    ]
    for env_name in env_names:
      print(env_name)
      self._call_get_env_and_policy(load_dir, env_name)


if __name__ == '__main__':
  tf.test.main()
