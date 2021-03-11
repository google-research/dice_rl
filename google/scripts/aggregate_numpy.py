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

import random

from absl import app
from absl import flags

from pyglib.contrib.g3_multiprocessing import g3_multiprocessing
from concurrent import futures
import numpy as np
import os
import pickle
import tensorflow.compat.v2 as tf

DIRECTORIES = [
    '/cns/vz-d/home/brain-ofirnachum/robust/results/boot_bandit100',
]

OUTPUT_FILE = None


def read_file(full_filename):
  with tf.io.gfile.GFile(full_filename, 'rb') as f:
    contents = np.load(f)
  return contents


def get_data(root_dir, mp_context):
  data = {}
  all_files = tf.io.gfile.glob('%s/*/*.npy' % root_dir)
  all_files = [f for f in all_files if 'gamma' in f]

  num_paths = len(all_files)
  batch_size = 100
  for batch in range(1 + (num_paths - 1) // batch_size):
    batch_start = batch * batch_size
    batch_end = min(num_paths, (batch + 1) * batch_size)
    batch_files = all_files[batch_start:batch_end]
    print('On batch %d, reading paths %d to %d out of %d total.' % (
        batch, batch_start, batch_end, num_paths))
    pool = mp_context.Pool(batch_size)
    batch_data = pool.map(read_file, batch_files)
    for filename, file_data in zip(batch_files, batch_data):
      data[filename] = file_data

  return data


def main(argv):
  del argv  # Unused.
  mp_context = g3_multiprocessing.get_context('absl_forkserver')
  for d, directory in enumerate(DIRECTORIES):
    print('On directory %d / %d: %s' % (d, len(DIRECTORIES), directory))
    data = get_data(directory, mp_context)

    output_file = OUTPUT_FILE
    if output_file is None:
      rnd = int(random.random() * 1e5)
      output_file = os.path.join(directory,
                                 'numpy_data%d.pkl' % rnd)
    with tf.io.gfile.GFile(output_file, 'w') as f:
      pickle.dump(data, f)
    print('Dumped data to %s' % output_file)


if __name__ == '__main__':
  g3_multiprocessing.handle_main(main)
  #app.run(main)
