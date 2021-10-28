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
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from smore.cpp_sampler.online_sampler import OnlineSampler, query_name_dict
from collections import defaultdict
from tqdm import tqdm
from smore.cpp_sampler.sampler_clib import KGMem


def test_kg(rank, data_folder, kg_mem, barrier):
    barrier.wait()
    print('rank', rank)
    
    kg = kg_mem.create_kg()
    print('num ent', kg.num_ent)
    print('num rel', kg.num_rel)
    print('num edges', kg.num_edges)
 
    query_structures = list(query_name_dict.keys())
    sampler_type = 'naive'
    negative_sample_size = 128
    sample_mode = ('sepa', 500, 'w', 'wstruct', 500000)
    for i in range(4):
        barrier.wait()
        if i == rank:
            sampler = OnlineSampler(kg, query_structures, negative_sample_size, sample_mode, [1.0 / len(query_structures)] * len(query_structures), 
                                    sampler_type=sampler_type, num_threads=2)
            print('num ent', kg.num_ent)
            print('num rel', kg.num_rel)
            print('num edges', kg.num_edges)
    data_gen = sampler.batch_generator(1024)
    for _ in tqdm(data_gen):
        pass

if __name__ == '__main__':
    kg_mem = KGMem()
    db_name = 'FB15k-237-betae'
    data_folder = os.path.join(os.path.expanduser('~'), 'data/knowledge_graphs/%s' % db_name)
    kg_mem.load(data_folder + '/train_bidir.bin')
    kg_mem.share_memory()
    mp.set_start_method('spawn')

    procs = []
    barrier = mp.Barrier(4)
    for i in range(4):
        proc = mp.Process(target=test_kg, args=(i, data_folder, kg_mem, barrier))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
