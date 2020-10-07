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

import numpy as np
import tensorflow.compat.v2 as tf

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from dice_rl.data.dataset import Dataset, EnvStep, OffpolicyDataset, OnpolicyDataset, StepType
from dice_rl.utils import common as common_lib


def _default_by_steps_reward_fn(env_step):
  return env_step.reward


def _default_by_episodes_reward_fn(env_step, valid_steps, gamma):
  mask = (
      (1 - tf.cast(env_step.is_last(), tf.float32)) *
      tf.cast(valid_steps, tf.float32))
  discount = tf.pow(tf.cast(gamma, tf.float32),
                    tf.cast(env_step.step_num, tf.float32))
  return tf.reduce_sum(mask * discount * env_step.reward, -1)


def _default_by_steps_weight_fn(env_step, gamma):
  mask = 1 - tf.cast(env_step.is_last(), tf.float32)
  discount = tf.pow(tf.cast(gamma, tf.float32),
                    tf.cast(env_step.step_num, tf.float32))
  return mask * discount


def _default_by_episodes_weight_fn(env_step, valid_steps):
  return tf.ones([tf.shape(valid_steps)[0]], dtype=tf.float32)


def get_minibatch_average(dataset: Dataset,
                          batch_size: int,
                          num_batches: int = 1,
                          by_steps: bool = True,
                          truncate_episode_at: Optional[int] = None,
                          reward_fn: Callable = None,
                          weight_fn: Callable = None,
                          gamma: Union[float, tf.Tensor] = 1.0) -> Union[
                                  float, tf.Tensor]:
  """Computes average reward via randomly sampled mini-batches.

    Samples steps or episodes from the dataset and computes average reward.

    Args:
      dataset: The dataset to sample experience from.
      batch_size: The number of episodes to sample per batch.
      num_batches: The number of batches to use for estimation.
      by_steps: Whether to sample batches of steps (default) or episodes.
      truncate_episode_at: If sampling by episodes, where to truncate episodes
        from the environment, if at all.
      reward_fn: A function that takes in an EnvStep and returns the reward for
        that step. If not specified, defaults to just EnvStep.reward. When
        sampling by episode, valid_steps is also passed into reward_fn.
      weight_fn: A function that takes in an EnvStep and returns a weight for
        that step. If not specified, defaults to gamma ** step_num. When
        sampling by episode, valid_steps is also passed into reward_fn.
      gamma: The discount factor to use for the default reward/weight functions.

    Returns:
      An estimate of the average reward.
    """
  if reward_fn is None:
    if by_steps:
      reward_fn = _default_by_steps_reward_fn
    else:
      reward_fn = lambda *args: _default_by_episodes_reward_fn(
          *args, gamma=gamma)

  if weight_fn is None:
    if by_steps:
      weight_fn = lambda *args: _default_by_steps_weight_fn(*args, gamma=gamma)
    else:
      weight_fn = _default_by_episodes_weight_fn

  total_reward = 0.
  total_weight = 0.
  for _ in range(num_batches):
    if by_steps:
      if isinstance(dataset, OnpolicyDataset):
        steps = dataset.get_step(num_steps=batch_size)
      else:
        steps = dataset.get_step(batch_size)
      rewards = reward_fn(steps)
      weights = weight_fn(steps)
    else:
      episodes, valid_steps = dataset.get_episode(
          batch_size, truncate_episode_at=truncate_episode_at)
      rewards = reward_fn(episodes, valid_steps)
      weights = weight_fn(episodes, valid_steps)

    rewards = common_lib.reverse_broadcast(rewards, weights)
    weights = common_lib.reverse_broadcast(weights, rewards)
    total_reward += tf.reduce_sum(rewards * weights, axis=0)
    total_weight += tf.reduce_sum(weights, axis=0)

  return total_reward / total_weight


def get_fullbatch_average(dataset: OffpolicyDataset,
                          limit: Optional[int] = None,
                          by_steps: bool = True,
                          truncate_episode_at: Optional[int] = None,
                          reward_fn: Callable = None,
                          weight_fn: Callable = None,
                          gamma: Union[float, tf.Tensor] = 1.0) -> Union[
                                  float, tf.Tensor]:
  """Computes average reward over full dataset.

    Args:
      dataset: The dataset to sample experience from.
      limit: If specified, the maximum number of steps/episodes to take from the
        dataset.
      by_steps: Whether to sample batches of steps (default) or episodes.
      truncate_episode_at: If sampling by episodes, where to truncate episodes
        from the environment, if at all.
      reward_fn: A function that takes in an EnvStep and returns the reward for
        that step. If not specified, defaults to just EnvStep.reward. When
        sampling by episode, valid_steps is also passed into reward_fn.
      weight_fn: A function that takes in an EnvStep and returns a weight for
        that step. If not specified, defaults to gamma ** step_num. When
        sampling by episode, valid_steps is also passed into reward_fn.
      gamma: The discount factor to use for the default reward/weight functions.

    Returns:
      An estimate of the average reward.
    """
  if reward_fn is None:
    if by_steps:
      reward_fn = _default_by_steps_reward_fn
    else:
      reward_fn = lambda *args: _default_by_episodes_reward_fn(
          *args, gamma=gamma)

  if weight_fn is None:
    if by_steps:
      weight_fn = lambda *args: _default_by_steps_weight_fn(*args, gamma=gamma)
    else:
      weight_fn = _default_by_episodes_weight_fn

  if by_steps:
    steps = dataset.get_all_steps(limit=limit)
    rewards = reward_fn(steps)
    weights = weight_fn(steps)
  else:
    episodes, valid_steps = dataset.get_all_episodes(
        truncate_episode_at=truncate_episode_at, limit=limit)
    rewards = reward_fn(episodes, valid_steps)
    weights = weight_fn(episodes, valid_steps)

  rewards = common_lib.reverse_broadcast(rewards, weights)
  weights = common_lib.reverse_broadcast(weights, rewards)
  if tf.rank(weights) < 2:
    return (tf.reduce_sum(rewards * weights, axis=0) /
            tf.reduce_sum(weights, axis=0))
  return (tf.linalg.matmul(weights, rewards) /
          tf.reduce_sum(tf.math.reduce_mean(weights, axis=0)))
