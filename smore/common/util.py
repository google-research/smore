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
import random
import torch
import time
import os.path as osp
from collections import defaultdict
from functools import wraps
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from _thread import start_new_thread
import traceback
import logging
import os
from tqdm import tqdm
import shutil
import zipfile
import urllib.request as ur
from smore.common.config import name_query_dict, query_name_dict

GBFACTOR = float(1 << 30)

def cal_ent_loc(query_structure, idx):
    if query_structure[0] == '<':
        return cal_ent_loc(query_structure[1], idx)
    all_relation_flag = True
    ent_locations = []
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            all_relation_flag = False
            break
    if all_relation_flag:
        if query_structure[0] == 'e':
            ent_locations.append(idx)
            idx += 1
        else:
            ent_locations, idx = cal_ent_loc(query_structure[0], idx)
        for i in range(len(query_structure[-1])):
            idx += 1
    else:
        for i in range(len(query_structure)):
            if query_structure[i] == 'u':
                assert i == len(query_structure) - 1
                break
            tmp_ent_locations, idx = cal_ent_loc(query_structure[i], idx)
            ent_locations.extend(tmp_ent_locations)
    return ent_locations, idx

def cal_ent_loc_dict(query_name_dict):
    query_ent_loc_dict = {}
    for query_structure in query_name_dict:
        # if query_name_dict[query_structure] == '2u-DNF':
        #     tmp_structure = ('e', ('r',))
        # elif query_name_dict[query_structure] == 'up-DNF':
        #     tmp_structure = ('e', ('r', 'r'))
        # else:
        #     tmp_structure = query_structure
        query_ent_loc_dict[query_structure] = cal_ent_loc(query_structure, 0)[0]
    return query_ent_loc_dict
       
def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)

flatten=lambda l: sum(map(flatten, l),[]) if isinstance(l,tuple) else [l]
flatten_list=lambda l: sum(map(flatten_list, l),[]) if isinstance(l,list) else [l]

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return

def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries

def construct_graph(base_path, indexified_files):
    #knowledge graph
    #kb[e][rel] = set([e, e, e])
    ent_in, ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for indexified_p in indexified_files:
        with open(osp.join(base_path, indexified_p)) as f:
            for i, line in enumerate(f):
                if len(line) == 0:
                    continue
                e1, rel, e2 = line.split('\t')
                e1 = int(e1.strip())
                e2 = int(e2.strip())
                rel = int(rel.strip())
                ent_out[e1][rel].add(e2)
                ent_in[e2][rel].add(e1)

    return ent_in, ent_out

def tuple2filterlist(t):
    return list(tuple2filterlist(x) if type(x)==tuple else -1 if x == 'u' else -2 if x == 'n' else x for x in t)

def achieve_answer_with_constraints(query, ent_in, ent_out, max_to_keep):
    assert type(query[-1]) == list
    all_relation_flag = True
    for ele in query[-1]:
        if (type(ele) != int) or (ele == -1):
            all_relation_flag = False
            break
    if all_relation_flag:
        if type(query[0]) == int:
            ent_set = set([query[0]])
        else:
            ent_set = achieve_answer_with_constraints(query[0], ent_in, ent_out, max_to_keep)
        for i in range(len(query[-1])):
            if query[-1][i] == -2:
                assert False, 'negation not supported'
                ent_set = set(range(len(ent_in))) - ent_set
            else:
                ent_set_traverse = set()
                n_traversed = 0
                for idx, ent in enumerate(ent_set):
                    if n_traversed == max_to_keep:
                        break
                    if query[-1][i] in ent_out[ent]:
                        ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                        n_traversed += 1
                ent_set = ent_set_traverse
    else:   
        ent_set = achieve_answer_with_constraints(query[0], ent_in, ent_out, max_to_keep)
        union_flag = False
        if len(query[-1]) == 1 and query[-1][0] == -1:
            assert False, 'union not supported'
            union_flag = True
        for i in range(1, len(query)):
            if not union_flag:
                ent_set = ent_set.intersection(achieve_answer_with_constraints(query[i], ent_in, ent_out, max_to_keep))
            else:
                if i == len(query) - 1:
                    continue
                ent_set = ent_set.union(achieve_answer_with_constraints(query[i], ent_in, ent_out, max_to_keep))
    return ent_set

def fill_query(query_structure, ent_in, ent_out, answer, chill=False):
    assert type(query_structure[-1]) == list
    all_relation_flag = True
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            all_relation_flag = False
            break
    if all_relation_flag:
        r = -1
        for i in range(len(query_structure[-1]))[::-1]:
            if query_structure[-1][i] == 'n':
                query_structure[-1][i] = -2
                continue
            if chill:
                r = random.sample(ent_in[answer].keys(), 1)[0]
            else:
                found = False
                for j in range(40):
                    r_tmp = random.sample(ent_in[answer].keys(), 1)[0]
                    if r_tmp // 2 != r // 2 or r_tmp == r:
                        r = r_tmp
                        found = True
                        break
                if not found:
                    return True
            query_structure[-1][i] = r
            answer = random.sample(ent_in[answer][r], 1)[0]
            # elif query_structure[-1][i] == 'n':
            #     assert False
            #     answer = random.sample(set(range(len(ent_in))) - answer, 1)[0] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!make sure this negative entity is of same type with the current asnwer.
        if query_structure[0] == 'e':
            query_structure[0] = answer
        else:
            return fill_query(query_structure[0], ent_in, ent_out, answer, chill)
    else:
        same_structure = defaultdict(list)
        for i in range(len(query_structure)):
            same_structure[list2tuple(query_structure[i])].append(i)
        for i in range(len(query_structure)):
            if len(query_structure[i]) == 1 and query_structure[i][0] == 'u':
                assert i == len(query_structure) - 1
                query_structure[i][0] = -1
                continue
            broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer, chill)
            if broken_flag:
                return True
        if not chill:
            for structure in same_structure:
                if len(same_structure[structure]) != 1:
                    structure_set = set()
                    for i in same_structure[structure]:
                        structure_set.add(list2tuple(query_structure[i]))
                    if len(structure_set) < len(same_structure[structure]):
                        # print('same query')
                        return True
        return False

def sample_negative_bidirectional(query, ent_in, ent_out, nent):
    pass


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    if osp.exists(path) and osp.getsize(path) > 0:  # pragma: no cover
        if log:
            print('Using exist file', filename)
        return path

    if log:
        print('Downloading', url)

    if not osp.exists(folder):
        os.makedirs(folder)
    data = ur.urlopen(url)

    size = int(data.info()["Content-Length"])

    chunk_size = 1024*1024
    num_iter = int(size/chunk_size) + 2

    downloaded_size = 0

    try:
        with open(path, 'wb') as f:
            pbar = tqdm(range(num_iter))
            for i in pbar:
                chunk = data.read(chunk_size)
                downloaded_size += len(chunk)
                pbar.set_description("Downloaded {:.2f} GB".format(float(downloaded_size)/GBFACTOR))
                f.write(chunk)
    except:
        if osp.exists(path):
             os.remove(path)
        raise RuntimeError('Stopped downloading due to interruption.')


    return path

def maybe_download_dataset(data_path):
    data_name = data_path.split('/')[-1]
    if data_name in ['FB15k', 'FB15k-237', 'NELL', "FB400k"]:
        if not (osp.exists(data_path) and osp.exists(osp.join(data_path, "stats.txt"))):
            url = "https://snap.stanford.edu/betae/%s.zip" % data_name
            path = download_url(url, osp.split(osp.abspath(data_path))[0])
            extract_zip(path, osp.split(osp.abspath(data_path))[0])
            os.unlink(path)

def extract_zip(path, folder):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
    """
    print('Extracting', path)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def thread_wrapped_func(func):
    """Wrapped func for torch.multiprocessing.Process.

    With this wrapper we can use OMP threads in subprocesses
    otherwise, OMP_NUM_THREADS=1 is mandatory.

    How to use:
    @thread_wrapped_func
    def func_to_wrap(args ...):
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))



if __name__ == '__main__':

    # evaluate whether the new recursive embedding is correct or not
    a = cal_ent_loc_dict(query_name_dict)
    for qs in a:
        print (qs, a[qs])
    
    torch.manual_seed(0)
    batch_queries_dict = {
                            (('e', ('r', 'r')), ('e', ('r',))): torch.randint(low=0, high=30, size=[2,5]),
                            ('e', ('r', 'r')): torch.randint(low=0, high=30, size=[3,3]),
                            ('e', ('r',)): torch.randint(low=0, high=30, size=[3,2]),
                            (('e', ('r',)), ('e', ('r',)), ('u',)): torch.randint(low=0, high=30, size=[2,5]),
                            ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): torch.randint(low=0, high=30, size=[3,6])
                        }
    main_entity_embedding = torch.rand(30, 4)
    main_relation_embedding = torch.rand(30, 4)
    main_offset_embedding = torch.rand(30, 4)


    def embed_query_old(queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = main_entity_embedding[queries[:,idx]]
                offset_embedding = torch.zeros_like(embedding)
                idx += 1
            else:
                embedding, offset_embedding, idx = embed_query_old(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                r_embedding = main_relation_embedding[queries[:, idx]]
                r_offset_embedding = main_offset_embedding[queries[:, idx]]
                embedding += r_embedding
                offset_embedding += r_offset_embedding
                idx += 1
        else:
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = embed_query_old(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            embedding = torch.mean(torch.stack(embedding_list), dim=0)[0]
            offset_embedding = torch.max(torch.stack(offset_embedding_list), dim=0)[0]

        return embedding, offset_embedding, idx


    def embed_query_new(queries, query_structure, idx, all_ent_embedding, ent_idx):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = all_ent_embedding[:,ent_idx,:]
                offset_embedding = torch.zeros_like(embedding)
                idx += 1
                ent_idx += 1
            else:
                embedding, offset_embedding, idx, ent_idx = embed_query_new(queries, query_structure[0], idx, all_ent_embedding, ent_idx)
            for i in range(len(query_structure[-1])):
                r_embedding = main_relation_embedding[queries[:, idx]]
                r_offset_embedding = main_offset_embedding[queries[:, idx]]
                embedding += r_embedding
                offset_embedding += r_offset_embedding
                idx += 1
        else:
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx, ent_idx = embed_query_new(queries, query_structure[i], idx, all_ent_embedding, ent_idx)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            embedding = torch.mean(torch.stack(embedding_list), dim=0)[0]
            offset_embedding = torch.max(torch.stack(offset_embedding_list), dim=0)[0]

        return embedding, offset_embedding, idx, ent_idx


    def transform_union_query(queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1] # remove union -1
        elif query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(query_structure):
        if query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    all_entity_idxs, all_structure_idxs, all_embed_shapes, all_query_structures = [], [0], [], []
    print ('-'*50)
    for query_structure in batch_queries_dict:
        print (query_structure)
        print (batch_queries_dict[query_structure])
    
    cnt = 0
    for query_structure in batch_queries_dict:
        entity_locations = a[query_structure]
        all_entity_idxs.extend(batch_queries_dict[query_structure][:, entity_locations].view(-1))
        cnt += len(batch_queries_dict[query_structure])*len(entity_locations)
        all_structure_idxs.append(cnt)
        all_embed_shapes.append([len(batch_queries_dict[query_structure]), len(entity_locations)])
        all_query_structures.append(query_structure)
    all_entity_idxs = torch.LongTensor(all_entity_idxs)
    print ('-'*50)
    print (all_entity_idxs)
    print (all_structure_idxs)
    print (all_embed_shapes)
    all_ent_embedding = main_entity_embedding[all_entity_idxs]
    for i, query_structure in enumerate(all_query_structures):
        print (query_structure)
        if 'u' in query_name_dict[query_structure]:
            center_embedding, offset_embedding, _ = \
                embed_query_old(transform_union_query(batch_queries_dict[query_structure], 
                                                                query_structure), 
                                        transform_union_structure(query_structure), 
                                        0)
            # print (center_embedding.shape)
            # print (offset_embedding.shape)
            # print (transform_union_query(batch_queries_dict[query_structure], query_structure))
            if query_name_dict[query_structure] in ['2u-DNF', 'up-DNF']: 
                tmp_embed_shape = [all_embed_shapes[i][0]*2, all_embed_shapes[i][0]//2, -1]
                tmp_ent_embedding = all_ent_embedding[all_structure_idxs[i]:all_structure_idxs[i+1]].view(tmp_embed_shape)
                # print (tmp_embed_shape)
                # print (tmp_ent_embedding.shape)
                center_embedding_new, offset_embedding_new, _, _ = embed_query_new(transform_union_query(batch_queries_dict[query_structure], query_structure), 
                                        transform_union_structure(query_structure), 
                                                                                0, tmp_ent_embedding, 0)
        else:
            center_embedding, offset_embedding, _ = embed_query_old(batch_queries_dict[query_structure], 
                                                                        query_structure, 
                                                                        0)
        
            tmp_ent_embedding = all_ent_embedding[all_structure_idxs[i]:all_structure_idxs[i+1]].view(all_embed_shapes[i]+[-1])
            # print (tmp_ent_embedding.shape)
            center_embedding_new, offset_embedding_new, _, _ = embed_query_new(batch_queries_dict[query_structure], 
                                                                            query_structure, 
                                                                            0, tmp_ent_embedding, 0)
        if not torch.equal(center_embedding, center_embedding_new) or not torch.equal(offset_embedding, offset_embedding_new):
            print (center_embedding)
            print (center_embedding_new)
            print (offset_embedding)
            print (offset_embedding_new)

    
