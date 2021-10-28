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
from smore.cpp_sampler import libsampler
from smore.cpp_sampler.online_sampler import build_query_tree
from smore.common.util import query_name_dict, name_query_dict, tuple2list, list2tuple
from collections import defaultdict
from copy import deepcopy
import pickle as cp
from tqdm import tqdm


import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='test query sampler config'
    )
    parser.add_argument('--max_num_answer', default=-1, type=int, help='max number of total answers')
    parser.add_argument('--max_num_missing_answer', default=-1, type=int, help='max number of missing answers (fp/fn)')
    parser.add_argument('--neg_samples', default=0, type=int, help='num negative samples')
    parser.add_argument('--search_bandwidth', default=-1, type=int, help='expansion bandwidth')
    parser.add_argument('--max_intermediate_answers', default=-1, type=int, help='max_intermediate_answers')
    parser.add_argument('--cpu_num', default=1, type=int, help='num search threads')
    parser.add_argument('--num_queries', default=1000, type=int, help='num queries per task')
    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('--save_path', type=str, default=None, help="test data path")
    parser.add_argument('--num_1p', type=int, default=-1, help="num 1p queries")
    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--kg_dtype', type=str, default="uint32", help="kg data type")

    parser.add_argument('--do_merge', action='store_true', help="do merge")
    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")
    return parser.parse_args(args)


def fill_query(query_tpl, query_args):
    assert isinstance(query_tpl, list)
    for i, e in enumerate(query_tpl):
        if isinstance(e, str):
            query_tpl[i] = query_args[0]
            del query_args[0]
        else:
            fill_query(e, query_args)


class TestQuerySampler(object):
    def __init__(self, args, kg_small, kg_large, query_structures):
        for attr in ['neg_samples', 'max_num_answer', 'max_num_missing_answer', 'search_bandwidth', 'max_intermediate_answers']:
            assert getattr(args, attr) >= 0

        self.query_structures = query_structures
        self.samplers = []
        fn_qt_create = libsampler.create_qt32 if args.kg_dtype == 'uint32' else libsampler.create_qt64
        for kg in [kg_small, kg_large]:
            list_qt = []
            for qs in query_structures:
                list_qt.append(build_query_tree(qs, fn_qt_create))
            sampler = sampler_clib.naive_sampler(kg_dtype)(kg, list_qt, [1.0 / len(list_qt)] * len(list_qt), False, False,
                                              0, args.search_bandwidth, args.max_intermediate_answers, args.max_num_answer, 0, [])
            self.samplers.append(sampler)

        self.query_sampler = sampler_clib.test_sampler(kg_dtype)(self.samplers[0], self.samplers[1], args.neg_samples, args.max_num_missing_answer, args.cpu_num)

    def collect_queries(self, query_type_idx, num_required, set_queries, dict_tp, dict_fp, dict_fn, dict_neg):
        self.query_sampler.launch_sampling(query_type_idx)
        query_str = self.query_structures[query_type_idx]
        query_str_template = tuple2list(query_str)

        num_gen = 0
        for _ in tqdm(range(num_required)):
            while True:
                query_args, list_tp, list_fp, list_fn, list_neg = [], [], [], [], []
                assert self.query_sampler.fetch_query(query_args, list_tp, list_fp, list_fn, list_neg) == query_type_idx

                query_tpl = deepcopy(query_str_template)
                fill_query(query_tpl, query_args)
                assert len(query_args) == 0
                query = list2tuple(query_tpl)
                if query in set_queries[query_str]:
                    continue
                break
            set_queries[query_str].add(query)
            if len(list_neg) == 0: # need to record down tp
                dict_tp[query] = set(list_tp)
            else:
                assert len(list_tp) == 0 # no need to record tp
                dict_neg[query] = list_neg
            dict_fp[query] = set(list_fp)
            dict_fn[query] = set(list_fn)
            num_gen += 1
        return num_gen


if __name__ == "__main__":
    args = parse_args()
    if args.do_merge:
        phases = []
        if args.do_train:
            phases.append('train')
        if args.do_valid:
            phases.append('valid')
        if args.do_test:
            phases.append('test')
        for phase in phases:
            fnames = os.listdir(os.path.join(args.save_path, phase))
            set_queries = defaultdict(set)
            dict_tp = defaultdict(set)
            dict_fp = defaultdict(set)
            dict_fn = defaultdict(set)
            dict_neg = defaultdict(set)
            all_data = []
            for fname in fnames:
                if not phase in fname:
                    continue
                cur_name = os.path.join(args.save_path, phase, fname)
                print('merging', fname)
                with open(cur_name, 'rb') as f:
                    d = cp.load(f)
                    if 'fn' in fname:
                        dict_fn.update(d)
                    elif 'fp' in fname:
                        dict_fp.update(d)
                    elif 'tp' in fname:
                        dict_tp.update(d)
                    elif 'negative' in fname:
                        dict_neg.update(d)
                    elif 'queries' in fname:
                        set_queries.update(d)
                    else:
                        print('unknwon file %s, skip' % fname)
            for query_structure in set_queries:
                for query in set_queries[query_structure]:
                    all_data.append([query_structure, query, dict_tp[query], dict_fn[query], dict_neg[query]])

            prefix = phase
            with open(os.path.join(args.save_path, '%s-queries.pkl' % prefix), 'wb') as f:
                cp.dump(set_queries, f, cp.HIGHEST_PROTOCOL)
            with open(os.path.join(args.save_path, '%s-tp-answers.pkl' % prefix), 'wb') as f:
                cp.dump(dict_tp, f, cp.HIGHEST_PROTOCOL)
            with open(os.path.join(args.save_path, '%s-fp-answers.pkl' % prefix), 'wb') as f:
                cp.dump(dict_fp, f, cp.HIGHEST_PROTOCOL)
            with open(os.path.join(args.save_path, '%s-fn-answers.pkl' % prefix), 'wb') as f:
                cp.dump(dict_fn, f, cp.HIGHEST_PROTOCOL)
            with open(os.path.join(args.save_path, '%s-negative_samples.pkl' % prefix), 'wb') as f:
                cp.dump(dict_neg, f, cp.HIGHEST_PROTOCOL)
            with open(os.path.join(args.save_path, 'all-%s-data.pkl' % prefix), 'wb') as f:
                cp.dump(all_data, f, cp.HIGHEST_PROTOCOL)
        sys.exit()

    with open(os.path.join(args.data_path, 'stats.txt'), 'r') as f:
        num_ent = f.readline().strip().split()[-1]
        num_rel = f.readline().strip().split()[-1]
        num_ent, num_rel = int(num_ent), int(num_rel)
    print('num ent', num_ent)
    print('num rel', num_rel)
    kg_dtype = args.kg_dtype

    for attr in ['max_num_answer', 'max_num_missing_answer', 'search_bandwidth', 'max_intermediate_answers']:
        if getattr(args, attr) <= 0:
            setattr(args, attr, num_ent)
    print(args)

    phases = []
    kg_train = sampler_clib.create_kg(num_ent, num_rel, kg_dtype)
    kg_train.load(args.data_path + '/train_bidir.bin')
    if args.do_train:
        kg_empty = sampler_clib.create_kg(num_ent, num_rel, kg_dtype)
        kg_empty.load(args.data_path + '/empty_bidir.bin')
        phases.append(('train', kg_empty, kg_train))

    kg_valid = sampler_clib.create_kg(num_ent, num_rel, kg_dtype)
    if args.do_valid:
        kg_valid.load(args.data_path + '/valid_bidir.bin')
        phases.append(('valid', kg_train, kg_valid))

    kg_test = sampler_clib.create_kg(num_ent, num_rel, kg_dtype)
    if args.do_test:
        kg_test.load(args.data_path + '/test_bidir.bin')
        phases.append(('test', kg_valid, kg_test))

    list_queries = []
    list_names = args.tasks.split('.')
    for key in list_names:
        query_struct = name_query_dict[key]
        list_queries.append(query_struct)

    for phase, kg_small, kg_large in phases:
        test_sampler = TestQuerySampler(args, kg_small, kg_large, list_queries)

        for task_id in range(len(list_queries)):
            set_queries = defaultdict(set)
            dict_tp = defaultdict(set)
            dict_fp = defaultdict(set)
            dict_fn = defaultdict(set)
            dict_neg = defaultdict(set)

            query_str = test_sampler.query_structures[task_id]
            if list_names[task_id] == '1p' and args.num_1p > 0:
                num_queries = args.num_1p
            else:
                num_queries = args.num_queries
            num_gen = test_sampler.collect_queries(task_id, num_queries, set_queries, dict_tp, dict_fp, dict_fn, dict_neg)
            print(phase, list_names[task_id], num_gen)
            prefix = '%s-%s' % (phase, list_names[task_id])
            if not os.path.exists(os.path.join(args.save_path, phase)):
                os.makedirs(os.path.join(args.save_path, phase))
            with open(os.path.join(args.save_path, phase, '%s-queries.pkl' % prefix), 'wb') as f:
                cp.dump(set_queries, f, cp.HIGHEST_PROTOCOL)
            with open(os.path.join(args.save_path, phase, '%s-tp-answers.pkl' % prefix), 'wb') as f:
                cp.dump(dict_tp, f, cp.HIGHEST_PROTOCOL)
            with open(os.path.join(args.save_path, phase, '%s-fp-answers.pkl' % prefix), 'wb') as f:
                cp.dump(dict_fp, f, cp.HIGHEST_PROTOCOL)
            with open(os.path.join(args.save_path, phase, '%s-fn-answers.pkl' % prefix), 'wb') as f:
                cp.dump(dict_fn, f, cp.HIGHEST_PROTOCOL)
            with open(os.path.join(args.save_path, phase, '%s-negative_samples.pkl' % prefix), 'wb') as f:
                cp.dump(dict_neg, f, cp.HIGHEST_PROTOCOL)
