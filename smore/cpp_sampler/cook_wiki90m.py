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
from smore.cpp_sampler import libsampler
from smore.cpp_sampler.sampler_clib import load_kg_from_numpy
from collections import defaultdict
import torch
from tqdm import tqdm


if __name__ == "__main__":
    data_folder = sys.argv[1]
    ent_cap = 100000

    stats = torch.load(os.path.join(data_folder, 'meta.pt'))
    num_ent = stats['num_entities'] if ent_cap is None else ent_cap
    num_rel = stats['num_relations']

    fout = data_folder + '/train_bidir.bin'

    if not os.path.isfile(fout):
        kg = libsampler.KG(num_ent, num_rel)
        np_file = os.path.join(data_folder, 'raw/train_hrt.npy')

        print('loading from', np_file)
        triplets = np.load(np_file)
        if ent_cap is not None:
            th = triplets[:, 0] < ent_cap
            tt = triplets[:, 2] < ent_cap
            triplets = triplets[th & tt]
        load_kg_from_numpy(kg, triplets, has_reverse_edges=True)

        print('num ent', kg.num_ent)
        print('num rel', kg.num_rel)
        print('num edges', kg.num_edges)

        fout = data_folder + '/train_bidir.bin'
        kg.dump(fout)
    
    for phase in ['val', 'test']:
        val_hr = os.path.join(data_folder, 'raw/%s_hr.npy' % phase)
        val_hr = np.load(val_hr)
        val_cand = os.path.join(data_folder, 'raw/%s_t_candidate.npy' % phase)
        val_cand = np.load(val_cand)

        ans_file = os.path.join(data_folder, 'raw/%s_t_correct_index.npy' % phase)
        if os.path.isfile(ans_file):
            print('loading answers')
            ans = np.load(ans_file)
        else:
            print('dummy answers')
            ans = np.zeros(val_hr.shape[0], dtype=val_hr.dtype)
        true_ent = val_cand[np.arange(ans.shape[0]), ans]
        val_cand[np.arange(ans.shape[0]), ans] = val_cand[:, 0]
        val_cand = val_cand[:, 1:]
        if ent_cap is not None:
            vh = val_hr[:, 0] < ent_cap
            va = true_ent < ent_cap
            val_hr = val_hr[vh & va]
            true_ent = true_ent[vh & va]
            val_cand = None
        print(val_hr.shape, val_cand.shape if val_cand is not None else 'None', true_ent.shape)

        d = {}
        d['head'] = val_hr[:, 0]
        d['relation'] = val_hr[:, 1]
        d['tail'] = true_ent
        d['tail_neg'] = torch.LongTensor(val_cand) if val_cand is not None else set()
        if phase == 'val':
            phase = 'valid'
        out_file = os.path.join(data_folder, 'eval-original/%s.pt' % phase)
        torch.save(d, out_file)
