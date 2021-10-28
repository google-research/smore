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

import torch
import sys
import numpy as np
import extlib_cuda as extlib
from smore.common.embedding.embed_rw import AsyncEmbeddingRW
import time
import random

def test_print():
    print(indices)
    print(embed)
    rw = AsyncEmbeddingRW(embed, buf_size=n_embed, gpu_id=gpu_id)
    submat = rw.read(indices)
    submat.job_handle.sync()
    print(submat)
    
    new_embed = torch.randn(n_embed, m).cuda(gpu_id)
    job = rw.add(indices, new_embed)
    print(new_embed)
    job.sync()
    print(embed)

    sys.exit()

if __name__ == '__main__':
    n = 5
    m = 2
    gpu_id = 0
    embed = torch.randn(n, m)
    n_embed = 3
    ids = list(range(n))
    random.shuffle(ids)
    ids = ids[:n_embed]
    indices = torch.LongTensor(ids)
    test_print()

    rw = AsyncEmbeddingRW(embed, buf_size=n_embed, gpu_id=gpu_id) 

    mat = torch.randn(m, m).cuda(gpu_id)
    t = time.time()
    submat = rw.embed[indices].cuda(gpu_id) 
    result1 = torch.matmul(submat, mat)
    print('t1', time.time() - t)

    rw.get_or_create_buf("default", n_embed, rw.read_buf)
    t = time.time()
    submat = rw.read(indices)
    submat.job_handle.sync()
    result2 = torch.matmul(submat, mat)
    print('t2', time.time() - t)
    print('diff', torch.sum(torch.abs(result2 - result1)))
