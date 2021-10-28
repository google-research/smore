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
import struct
import subprocess
from tqdm import tqdm


def load_dict(fname):
    d = {}
    with open(fname, 'r') as f:
        for row in f:
            row = row.strip().split()
            x = int(row[0])
            y = row[1]
            d[y] = x
    return d


if __name__ == '__main__':
    data_folder = sys.argv[1]

    if data_folder.endswith('Freebase'):
        with open(os.path.join(data_folder, 'entity2id.txt'), 'r') as f:
            num_entities = int(f.readline().strip())
        with open(os.path.join(data_folder, 'relation2id.txt'), 'r') as f:
            num_relations = int(f.readline().strip())
        row_fn = lambda e1, r, e2: (int(e1), int(e2), int(r))
    else:
        rel_dict = load_dict(os.path.join(data_folder, 'relations.dict'))
        ent_dict = load_dict(os.path.join(data_folder, 'entities.dict'))
        num_relations = len(rel_dict)
        num_entities = len(ent_dict)
        row_fn = lambda e1, r, e2: (ent_dict[e1], rel_dict[r], ent_dict[e2])
    print('# entities', num_entities)
    print('# relations', num_relations)

    with open(os.path.join(data_folder, 'stats.txt'), 'w') as f:
        f.write('numentity: %d\nnumrelations: %d\n' % (num_entities, num_relations * 2))

    for phase in ['train', 'valid', 'test']:
        fname = os.path.join(data_folder, '%s.txt' % phase)
        fout = open(os.path.join(data_folder, '%s_bidir.txt' % phase), 'w')        
        flat_buf = []
        with open(fname, 'r') as f:
            for row in tqdm(f):
                e1, r, e2 = row.strip().split()
                e1, r, e2 = row_fn(e1, r, e2)
                fout.write('%d %d %d\n' % (e1, r * 2, e2))
                fout.write('%d %d %d\n' % (e2, r * 2 + 1, e1))                
        fout.close()

