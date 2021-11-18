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

import ogb
from ogb.lsc import WikiKG90Mv2Dataset
from ogb.utils import url
import os
import subprocess
from smore.cpp_sampler import sampler_clib
import torch
import numpy as np
import sys


def download_candidate_set(save_dir):
    valid_url = "https://snap.stanford.edu/smore/valid.pt"
    test_url = "https://snap.stanford.edu/smore/test.pt"
    if not os.path.exists(os.path.join(save_dir, "valid.pt")):
        url.download_url(valid_url, save_dir)
    if not os.path.exists(os.path.join(save_dir, "test.pt")):
        url.download_url(test_url, save_dir)


if __name__ == '__main__':
    save_dir = sys.argv[1]
    dataset = WikiKG90Mv2Dataset(root = save_dir)
    download_candidate_set(os.path.join(save_dir, "wikikg90m-v2/eval-original"))

    save_dir = os.path.join(save_dir, "wikikg90m-v2")
    stats = torch.load(os.path.join(save_dir, 'meta.pt'))
    num_ent = stats['num_entities']
    num_rel = stats['num_relations']
    print('num ent from meta data', num_ent)
    print('num rel from meta data', num_rel)
    with open(os.path.join(save_dir, "stats.txt"), 'w') as f:
        f.write("num_entities: %d\n"%num_ent)
        f.write("num_relations: %d\n"%num_rel)

    ent_cap = num_ent
    fout = save_dir + '/train_bidir.bin'

    if not os.path.isfile(fout):
        kg = sampler_clib.create_kg(num_ent, num_rel, "uint32")
        np_file = os.path.join(save_dir, 'processed/train_hrt.npy')

        print('loading from', np_file)
        triplets = np.load(np_file)
        if ent_cap is not None:
            th = triplets[:, 0] < ent_cap
            tt = triplets[:, 2] < ent_cap
            triplets = triplets[th & tt]    

        sampler_clib.load_kg_from_numpy(kg, triplets, has_reverse_edges=True)

        print('num ent on KG', kg.num_ent)
        print('num rel on KG', kg.num_rel)
        print('num edges on KG', kg.num_edges)

        fout = save_dir + '/train_bidir.bin'
        kg.dump(fout)