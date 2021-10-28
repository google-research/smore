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

import os
import sys
import numpy as np
from smore.cpp_sampler import sampler_clib
from collections import defaultdict
from tqdm import tqdm


if __name__ == "__main__":
    data_folder = sys.argv[1]
    dtype = sys.argv[2]
    phases = sys.argv[3].split(',')

    with open(os.path.join(data_folder, 'stats.txt'), 'r') as f:
        num_ent = f.readline().strip().split()[-1]
        num_rel = f.readline().strip().split()[-1]
        num_ent, num_rel = int(num_ent), int(num_rel)

    kg = sampler_clib.create_kg(num_ent, num_rel, dtype)
    list_files = [data_folder + '/%s_bidir.txt' % x for x in phases]

    print('loading from', list_files)
    kg.load_triplets_from_files(list_files, True)

    print('num ent', kg.num_ent)
    print('num rel', kg.num_rel)
    print('num edges', kg.num_edges)

    fout = data_folder + '/%s_bidir.bin' % phases[-1]
    kg.dump(fout)
