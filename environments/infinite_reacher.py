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

import gym
from gym.envs.mujoco.reacher import ReacherEnv


class InfiniteReacher(ReacherEnv, gym.core.Env):

  def step(self, action):
    obs, reward, done, info = super(InfiniteReacher, self).step(action)
    if hasattr(self, 'np_random') and self.np_random.random_sample() < 0.03:
      obs = self.reset()

    return obs, reward, False, info
