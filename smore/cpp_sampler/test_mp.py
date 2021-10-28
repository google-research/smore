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
import ctypes
import numpy as np
import torch
from smore.cpp_sampler import libsampler
QueryTree = libsampler.QueryTree

from collections import defaultdict
from tqdm import tqdm
import torch.multiprocessing as mp
import os

query_name_dict = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    ('e', ('r', 'r', 'r', 'r')): '4p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '4i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    ((('e', ('r',)), ('e', ('r',)), ('e', ('r',))), ('r',)): '3ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                }


def build_query_tree(query_structure):
    all_relation_flag = True
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            all_relation_flag = False
            break
    if all_relation_flag:
        assert len(query_structure) == 2
        if query_structure[0] == 'e':
            prev_node = libsampler.create_qt(QueryTree.entity)
        else:
            prev_node = build_query_tree(query_structure[0])
        for i, c in enumerate(query_structure[-1]):
            if c == 'r':
                cur_op = QueryTree.relation
            else:
                assert c == 'n'
                cur_op = QueryTree.negation
            cur_root = libsampler.create_qt(QueryTree.entity_set)
            cur_root.add_child(cur_op, prev_node)
            prev_node = cur_root
        return cur_root
    else:
        last_qt = query_structure[-1]
        node_type = QueryTree.intersect
        if len(last_qt) == 1 and last_qt[0] == 'u':
            node_type = QueryTree.union
            query_structure = query_structure[:-1]
        sub_root = libsampler.create_qt(node_type)
        for c in query_structure:
            ch_node = build_query_tree(c)
            sub_root.add_child(QueryTree.no_op, ch_node)
        return sub_root


class AbstractOnlineSampler(object):
    def __init__(self, kg, query_structures, negative_sample_size, sample_mode, normalized_structure_prob, 
                 prefetch=10, num_threads=8):
        self.kg = kg
        self.query_structures = query_structures
        self.normalized_structure_prob = normalized_structure_prob
        assert len(normalized_structure_prob) == len(query_structures)
        self.negative_sample_size = negative_sample_size
        self.sample_style, self.max_to_keep, self.weighted_style, self.structure_weighted_style, self.max_n_partial_answers = sample_mode
        if self.structure_weighted_style == 'wstruct':
            assert self.normalized_structure_prob is not None        

        self.prefetch = prefetch
        self.sampler = None

    def batch_generator(self, batch_size):
        self.sampler.prefetch(batch_size * self.prefetch)

        while True:
            pos_ans = torch.LongTensor(batch_size)
            neg_ans = torch.LongTensor(batch_size, self.negative_sample_size)
            weights = torch.FloatTensor(batch_size)
            q_args = []
            q_structs = []
            self.sampler.next_batch(batch_size, 
                                    pos_ans.numpy(), neg_ans.numpy(), weights.numpy(), 
                                    q_args, q_structs)
            yield pos_ans, weights, q_args, [self.query_structures[x] for x in q_structs], neg_ans



class OldOnlineSampler(AbstractOnlineSampler):
    """Hard coded online sampler; can only handle very limited set of queries."""

    def __init__(self, kg, query_structures, negative_sample_size, sample_mode, normalized_structure_prob, sampler_type,
                 prefetch=10, num_threads=8):
        super(OldOnlineSampler, self).__init__(kg, query_structures, negative_sample_size, sample_mode, normalized_structure_prob, prefetch, num_threads)
        query_names = ','.join([query_name_dict[x] for x in query_structures])
        if sampler_type == 'naive':
            sampler_type = libsampler.OldNaiveSampler
        else:
            sampler_type = libsampler.OldSqrtSampler

        self.sampler = sampler_type(kg, query_names, normalized_structure_prob, 
                                    negative_sample_size, self.max_to_keep, self.max_n_partial_answers, num_threads)


class OnlineSampler(AbstractOnlineSampler):
    def __init__(self, kg, query_structures, negative_sample_size, sample_mode, normalized_structure_prob, sampler_type,
                 prefetch=10, num_threads=8):
        super(OnlineSampler, self).__init__(kg, query_structures, negative_sample_size, sample_mode, normalized_structure_prob, prefetch, num_threads)

        list_qt = []
        for qs in query_structures:
            list_qt.append(build_query_tree(qs))
        
        self.sampler = libsampler.NaiveSampler(kg, list_qt, normalized_structure_prob, 
                                               negative_sample_size, self.max_to_keep, self.max_n_partial_answers, num_threads)


def construct_graph(base_path, indexified_files):
    #knowledge graph
    #kb[e][rel] = set([e, e, e])
    ent_in, ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for indexified_p in indexified_files:
        with open(os.path.join(base_path, indexified_p)) as f:
            for i, line in enumerate(f):
                if len(line) == 0:
                    continue
                try:
                    e1, rel, e2 = line.split(' ')
                except:
                    try:
                        e1, rel, e2 = line.split('\t')
                    except:
                        assert False
                e1 = int(e1.strip())
                e2 = int(e2.strip())
                rel = int(rel.strip())
                ent_out[e1][rel].add(e2)
                ent_in[e2][rel].add(e1)

    return ent_in, ent_out

def gen_sample(qs, gpu_id, barrier):
    print ("entering {}".format(os.getpid()))
    target = 20
    cnt = 0
    sampler = OnlineSampler(kg, qs, negative_sample_size, sample_mode, [1.0 / len(query_structures)] * len(query_structures), 
                               sampler_type='naive',
                               num_threads=1)
 
    batch_gen = sampler.batch_generator(1024)
    for pos_ans, weights, q_args, q_structs, neg_ans in tqdm(batch_gen):
        cnt += 1
        if cnt == target:
            break

if __name__ == '__main__':
    db_name = 'FB15k'
    data_folder = os.path.join(os.path.expanduser('~'), 'data/knowledge_graphs/%s' % db_name)
    # data_folder = '/dfs/user/hyren/betae-release/data/FB15k-betae/'
    with open(os.path.join(data_folder, 'stats.txt'), 'r') as f:
        num_ent = f.readline().strip().split()[-1]
        num_rel = f.readline().strip().split()[-1]
        num_ent, num_rel = int(num_ent), int(num_rel)

    kg = libsampler.KG(num_ent, num_rel)
    kg.load(data_folder + '/train_bidir.bin')
    # kg.load_triplets(data_folder + '/train.txt', True)
    print('num ent', kg.num_ent)
    print('num rel', kg.num_rel)
    print('num edges', kg.num_edges)
    ent_in, ent_out = construct_graph(data_folder, ['train_bidir.txt'])
    # ent_in, ent_out = construct_graph(data_folder, ['train.txt'])

    # query_structures = [('e',('r',)), ('e', ('r', 'r')), ('e', ('r', 'r', 'r')), (('e', ('r',)), ('e', ('r',))), (('e', ('r',)), ('e', ('r',)), ('e', ('r',)))]
    query_structures2 = [('e', ('r', 'r'))]
    # query_structures = [(('e', ('r',)), ('e', ('r',)))]
    query_structures = [(('e', ('r',)), ('e', ('r', 'n')))]
    # query_structures = [(('e', ('r',)), ('e', ('r',)), ('u',))]
    # query_structures = list(query_name_dict.keys())

    negative_sample_size = 128
    sample_mode = ('sepa', 500, 'w', 'wstruct', 80)
    list_q = [query_structures2, query_structures]
    batch_size = 1024
    gpus = range(len(list_q))
    procs = []
    barrier = mp.Barrier(len(gpus))

    print ("begin multiprocessing")
    for i, gpu_id in enumerate(gpus):
        proc = mp.Process(target=gen_sample, args=(list_q[i], gpu_id, barrier))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    print ("finish multiprocessing")
    exit(-1)
