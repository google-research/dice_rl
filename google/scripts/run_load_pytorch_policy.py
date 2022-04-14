"""Load a pytorch policy from weights in a pkl file."""

import pickle

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from dice_rl.data.dataset import StepType
import dice_rl.environments.env_policies as env_policies
import google3.learning.deepmind.xmanager2.client.google as xm

FLAGS = flags.FLAGS

flags.DEFINE_string('target_policy', None, 'Pickle file for target policy')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_integer('n_trajs', 100, 'Number of trajectory rollouts to do.')
flags.DEFINE_float('gamma', 0.99, 'Discount')


def get_target_policy(env_name, target_policy):
  with tf.io.gfile.GFile(target_policy, 'rb') as target_policy_file:
    weights = pickle.load(target_policy_file)
  tf_env, policy = env_policies.get_env_and_policy_from_weights(env_name,
                                                                weights)
  return tf_env, policy


@tf.function
def rollout(tf_env, policy, gamma):
  """Rollout a trajectory."""
  returns = tf.zeros((1,))
  time_step = tf_env.reset()
  t = 0
  while time_step.step_type != StepType.LAST:
    action = policy.action(time_step).action
    time_step = tf_env.step(action)
    returns += time_step.reward * tf.math.pow(gamma, tf.cast(t, tf.float32))
    t += 1
  # Add last step
  returns += time_step.reward * tf.math.pow(gamma, tf.cast(t, tf.float32))
  return t, (1 - gamma) * returns[0]


def main(_):
  tf_env, policy = get_target_policy(FLAGS.env_name, FLAGS.target_policy)
  gamma = FLAGS.gamma
  n_trajectories = FLAGS.n_trajs

  results = []
  for _ in range(n_trajectories):
    t, returns = rollout(tf_env, policy, gamma)
    results.append((t.numpy(), returns.numpy()))
  results = np.array(results)
  tf.print('Avg steps: ', np.mean(results[:, 0]),
           'Avg discounted return: ', np.mean(results[:, 1]),
           'SEM: ', np.std(results[:, 1]) / np.sqrt(n_trajectories))


if __name__ == '__main__':
  app.run(main)
