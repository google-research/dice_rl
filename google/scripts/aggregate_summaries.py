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

from pyglib import gfile
from google3.learning.brain.google.tools.event_utils.python import event_utils

import os
import pickle

DIRECTORIES = [
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/robust8',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/reacher_boot2',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/cartpole_boot',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/cartpole100',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/reacher100',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tabrobust4',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tree2y',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tree5y',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tree6y',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tree2z1',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tree2z9',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tree7z1',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tree7z9',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tree2q9',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tree7q9',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/taxi1q500',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tree5q9',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tree6q9',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/tree8q9',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/lake6q100',
    #'/cns/vz-d/home/brain-ofirnachum/robust/results/taxi4q500',
    '/cns/vz-d/home/brain-ofirnachum/robust/results/grid20b',
    '/cns/vz-d/home/brain-ofirnachum/robust/results/grid20p',
]

OUTPUT_FILE = None
TAGS = ['estimate0', 'estimate1']


def get_paths(root_dir):
  all_paths = gfile.Glob('%s/*/*/events*' % root_dir)
  all_paths = [os.path.dirname(path) for path in all_paths]
  split_paths = [(os.path.dirname(path), path) for path in all_paths]

  sub_dirs, subsub_dirs = zip(*split_paths)
  return list(set(sub_dirs)), list(set(subsub_dirs))


def get_data(root_dir):
  sub_dirs, subsub_dirs = get_paths(root_dir)
  print('Found %d sub_dirs' % len(sub_dirs))
  print('Found %d subsub_dirs' % len(subsub_dirs))
  paths = subsub_dirs if subsub_dirs else sub_dirs
  num_paths = len(paths)
  new_data = {}
  for batch in range(1 + (num_paths - 1) // 64):
    batch_start = batch * 64
    batch_end = min(num_paths, (batch + 1) * 64)
    print('On batch %d, reading paths %d to %d out of %d total.' % (
        batch, batch_start, batch_end, num_paths))
    data = event_utils.LoadEventsFromDirectories(
        paths[batch_start:batch_end], TAGS)

    for path, summaries in data.items():
      path_elems = path.split('/')
      sub_dir = '/'.join(path_elems[:-1])
      exp_info = path_elems[-1]
      if sub_dir not in new_data:
        new_data[sub_dir] = {}
      new_data[sub_dir][exp_info] = summaries

  return new_data


def main(argv):
  del argv  # Unused.
  for d, directory in enumerate(DIRECTORIES):
    print('On directory %d / %d: %s' % (d, len(DIRECTORIES), directory))
    data = get_data(directory)

    output_file = OUTPUT_FILE
    if output_file is None:
      rnd = int(random.random() * 1e5)
      output_file = os.path.join(directory,
                                 'summary_data%d.pkl' % rnd)
    with gfile.GFile(output_file, 'w') as f:
      pickle.dump(data, f)
    print('Dumped data to %s' % output_file)


if __name__ == '__main__':
  app.run(main)
