# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import sys
from smore.cpp_sampler import libsampler
from smore.cpp_sampler import online_sampler
from collections import Counter


if __name__ == '__main__':
    db_name = sys.argv[1]
    data_folder = os.path.join(os.path.expanduser('~'), 'data/knowledge_graphs/%s' % db_name)
    with open(os.path.join(data_folder, 'stats.txt'), 'r') as f:
        num_ent = f.readline().strip().split()[-1]
        num_rel = f.readline().strip().split()[-1]
        num_ent, num_rel = int(num_ent), int(num_rel)

    kg = libsampler.KG32(num_ent, num_rel)
    kg.load(data_folder + '/train_bidir.bin')
    sampler = online_sampler.OnlineSampler(kg, [('e',('r',))], 100, (0, 0, 'w', 'wstruct', 0, True, False), [1.0])
