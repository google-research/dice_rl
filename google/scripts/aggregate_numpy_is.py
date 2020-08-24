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

from absl import flags

import random
import numpy as np

from pyglib.contrib.g3_multiprocessing import g3_multiprocessing

import os
import pickle
import tensorflow.compat.v2 as tf

import google3.learning.deepmind.xmanager2.client.google as xm  # pylint: disable=unused-import

# DIRECTORIES = [
#     # '/cns/pw-d/home/mudcats/dev/algae_ci/Taxi_IS/',
#     '/cns/pw-d/home/mudcats/dev/algae_ci/SmallTree_IS/',
#     '/cns/pw-d/home/mudcats/dev/algae_ci/FrozenLake_IS/',
# ]
#
OUTPUT_FILE = None
#
# MODE = ['weighted-step-wise']

flags.DEFINE_string('directory', '/cns/pw-d/home/mudcats/dev/algae_ci/FrozenLake_IS/',
                    'Environment name.')
flags.DEFINE_string('mode', 'weighted-step-wise', 'IS mode.')

FLAGS = flags.FLAGS


def read_file(full_filename):
  with tf.io.gfile.GFile(full_filename, 'rb') as f:
    confidence_interval = np.load(f)
  return confidence_interval


def get_data(root_dir, mode, mp_context):
  all_files = [
      ele for ele in tf.io.gfile.glob('%s/*/*/*.npy' % root_dir) if mode in ele
  ]
  num_paths = len(all_files)
  batch_size = 100
  data = {}
  for batch in range(1 + (num_paths - 1) // batch_size):
    batch_start = batch * batch_size
    batch_end = min(num_paths, (batch + 1) * batch_size)
    print('On batch %d, reading paths %d to %d out of %d total.' %
          (batch, batch_start, batch_end, num_paths))

    current_paths = all_files[batch_start:batch_end]
    pool = mp_context.Pool(batch_size)
    batch_data = pool.map(read_file, current_paths)
    for filename, file_data in zip(current_paths, batch_data):
      filename_elems = filename.split('/')
      env_keys = tuple([
          ele for ele in filename_elems[-3].split('_')
          if (ele.startswith('seed') or ele.startswith('maxtraj') or
              ele.startswith('numtraj') or ele.startswith('alpha'))
      ])

      exp_keys = tuple([
          ele for ele in filename_elems[-2].split('_')
          if (ele.startswith('CI') or ele.startswith('del') or
              ele.startswith('gam'))
      ])
      if env_keys not in data.keys():
        data[env_keys] = {}
      data[env_keys][exp_keys] = file_data

  return data


def main(argv):
  del argv  # Unused.
  mp_context = g3_multiprocessing.get_context('absl_forkserver')

  print('On directory: %s' % FLAGS.directory)
  print('On mode: %s' % FLAGS.mode)
  data = get_data(FLAGS.directory, FLAGS.mode, mp_context)

  if OUTPUT_FILE is None:
    rnd = int(random.random() * 1e5)
    output_file = os.path.join(FLAGS.directory, 'summary_data%d.pkl' % rnd)
  with tf.io.gfile.GFile(output_file, 'w') as f:
    pickle.dump(data, f)
  print('Dumped data to %s' % output_file)


if __name__ == '__main__':
  g3_multiprocessing.handle_main(main)
  # app.run(main)
