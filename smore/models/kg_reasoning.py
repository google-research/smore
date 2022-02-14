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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from smore.common.util import flatten_list
from smore.common.embedding.sparse_embed import SparseEmbedding
from smore.common.embedding.embed_rw import EmbeddingReadOnly
from smore.common.util import cal_ent_loc_dict


class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, 
                 optim_mode, batch_size, test_batch_size=1, sparse_embeddings=None,
                 sparse_device=None, use_cuda=False, query_name_dict=None, relation_dim=None, logit_impl='native'):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1) # used in test_step
        self.query_name_dict = query_name_dict
        self.query_ent_loc_dict = cal_ent_loc_dict(self.query_name_dict)
        self.sparse_embeddings = sparse_embeddings
        self.sparse_device = sparse_device

        self.entity_dim = hidden_dim
        if relation_dim is None:
            self.relation_dim = hidden_dim
        else:
            self.relation_dim = relation_dim
        self.batch_size = batch_size
        self.has_feat = False

        # self.gamma = nn.Parameter(
        #     torch.Tensor([gamma]), 
        #     requires_grad=False
        # )
        ########################
        optim_mode = None
        ########################
        self.gamma = gamma

        self.embedding_range = (self.gamma + self.epsilon) / hidden_dim
        self.relation_embedding = SparseEmbedding(nrelation, self.relation_dim)

        self.t_read = 0
        self.t_fwd = 0
        self.t_loss = 0
        self.t_opt = 0

        self.t_fwd_prepare_data = 0
        self.t_fwd_model_call_emb = 0
        self.t_fwd_model_call_data = 0
        if logit_impl == 'native':
            self._cal_logit = self.native_cal_logit
        elif logit_impl == 'custom':
            self._cal_logit = self.custom_cal_logit
        else:
            raise ValueError('unknown logit impl %s' % logit_impl)

    def cal_logit(self, entity_embedding, entity_feat, query_embedding):
        return self._cal_logit(entity_embedding, entity_feat, query_embedding)

    def attach_feature(self, name, feat, gpu_id, is_sparse):
        if gpu_id == -1:
            device = 'cpu'
        else:
            device = 'cuda:{}'.format(gpu_id)
        if is_sparse:
            feat_read = EmbeddingReadOnly(feat, gpu_id=gpu_id)
            setattr(self, "%s_feat" % name, feat_read)
        else:
            setattr(self, "%s_feat" % name, feat.to(device))

    def init_params(self):
        self.entity_embedding.init_params(-self.embedding_range, self.embedding_range)
        self.relation_embedding.init_params(-self.embedding_range, self.embedding_range)

    def is_sparse(self, name):
        for sp_name, _ in self.named_sparse_embeddings():
            if sp_name in name:
                return True
        return False

    def named_dense_parameters(self, exclude_embedding=False):
        named_dense_params = list(self.named_dense_embedding_params())
        if len(named_dense_params):
            dense_embed_names, _ = zip(*named_dense_params)
        else:
            dense_embed_names = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if self.is_sparse(name):
                continue
            if exclude_embedding and name in dense_embed_names:
                continue
            yield name, param

    def named_sparse_embeddings(self):
        list_sparse = []
        if 'e' in self.sparse_embeddings:
            list_sparse.append(("entity_embedding", self.entity_embedding))
        if 'r' in self.sparse_embeddings:
            list_sparse.append(("relation_embedding", self.relation_embedding))
        return list_sparse

    def named_dense_embedding_params(self):
        if 'e' not in self.sparse_embeddings:
            for name, param in self.entity_embedding.named_parameters():
                yield 'entity_embedding.' + name, param
        if 'r' not in self.sparse_embeddings:
            for name, param in self.relation_embedding.named_parameters():
                yield 'relation_embedding.' + name, param

    def to_device(self, device):
        if 'e' not in self.sparse_embeddings or self.sparse_device == 'gpu':
            self.entity_embedding = self.entity_embedding.cuda(device)
        if 'r' not in self.sparse_embeddings or self.sparse_device == 'gpu':
            self.relation_embedding = self.relation_embedding.cuda(device)

    def share_memory(self):
        logging.info("Sharing memory")
        self.entity_embedding.share_memory()
        self.relation_embedding.share_memory()
        
    def cal_reg(self, positive_embedding, negative_embedding):
        reg1 = 0.0 if positive_embedding is None else torch.norm(positive_embedding, p=3)**3
        reg2 = 0.0 if negative_embedding is None else torch.norm(negative_embedding, p=3)**3
        return reg1 + reg2 + torch.norm(self.relation_embedding.embedding, p=3)**3

    def merge_embeddings(self, embedding_list, dim, expand_dim=None):
        '''
        Merge embeddings in embedding_list along dim.
        Params:
            embedding_list: a list of embeddings, each item represents the embedding of a branch.
        '''
        # [[tensor(32, 1, 8), tensor(32, 1, 8)], [tensor(32, 2, 8), tensor(32, 2, 8)]]
        merged_embedding_list = [[] for _ in range(self.num_embedding_component)]
        for embedding in embedding_list:
            for i in range(self.num_embedding_component):
                merged_embedding_list[i].append(embedding[i])
        merged_embedding_list = [torch.cat(embedding, dim=dim).unsqueeze(dim=expand_dim) if expand_dim is not None else torch.cat(embedding, dim=dim) for embedding in merged_embedding_list]
        return merged_embedding_list
    
    def entity_regularizer(self, embedding):
        return embedding

    def union(self, embedding_list):
        '''
        Take union over the embedding_list.
        Default to the DNF modeling, we stack the embeddings and do union in the last step
        Params:
            embedding_list: a list of embeddings, each item represents the embedding of a branch.
        '''
        # [[tensor(32, 1, 8), tensor(32, 1, 8)], [tensor(32, 2, 8), tensor(32, 2, 8)]]
        # [tensor(32, 3, 8), tensor(32, 3, 8)]
        return self.merge_embeddings(embedding_list, dim=1)
    
    def intersection(self, embedding_list):
        '''
        Take intersection over the embedding_list.
        Params:
            embedding_list: a list of embeddings, each item represents the embedding of a branch.
        '''
        # [[tensor(32, 1, 8), tensor(32, 1, 8)], [tensor(32, 2, 8), tensor(32, 2, 8)], [tensor(32, 3, 8), tensor(32, 3, 8)]]
        num_branches = len(embedding_list)
        num_lazy_union = [embedding[0].shape[1] for embedding in embedding_list] # [1, 2, 3]
        stacked_embedding_list = []
        for branch, embedding in enumerate(embedding_list):
            embedding = torch.cat(embedding, dim=-1) # concat the real and imaginary embedding
            for i in range(num_branches):
                if i == branch:
                    continue
                embedding = embedding.unsqueeze(i + 1)
            shape_to_tile = [1] + num_lazy_union + [1]
            shape_to_tile[branch + 1] = 1
            embedding = embedding.repeat(shape_to_tile) # [32, 1, 2, 3, 16]
            embedding = embedding.view([embedding.shape[0], -1, embedding.shape[-1]])
            stacked_embedding_list.append(embedding)
        stacked_embedding_list = torch.stack(stacked_embedding_list) # [3, 32, 6, 16]
        return self.intersection_between_stacked_embedding(stacked_embedding_list)

    def negation(self, embedding):
        raise NotImplementedError
    
    def is_all_relation(self, query_structure):
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        return all_relation_flag

    def embed_query(self, queries, query_structure, idx, device):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        if self.is_all_relation(query_structure):
            if query_structure[0] == 'e':
                embedding = self.retrieve_embedding(queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query(queries, query_structure[0], idx, device)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    embedding = self.negation(embedding)
                else:
                    embedding = self.relation_projection(embedding, queries[:, idx]) # merge
                idx += 1
        else:
            embedding_list = []
            is_union = query_structure[-1][0] == 'u'
            if is_union:
                num_branches = len(query_structure) - 1
            else:
                num_branches = len(query_structure)
            for i in range(num_branches):
                embedding, idx = self.embed_query(queries, query_structure[i], idx, device)
                embedding_list.append(embedding)
            if is_union:
                embedding = self.union(embedding_list)
                idx += 1
            else:
                embedding = self.intersection(embedding_list)
        
        return embedding, idx
    
    def embed_reverse_query(self, entity_idx, inv_rel_queries):
        tail_entities = self.entity_embedding(entity_idx)
        tail_entity_embedding = self.entity_regularizer(tail_entities).unsqueeze(1)
        return tail_entity_embedding

    def cal_reverse_logit(self, inv_query_embed, entity_embedding, entity_feat, inv_rel_queries):
        embedding = [entity_embedding]
        assert entity_feat is None
        for i in range(inv_rel_queries.shape[1] - 1, -1, -1):
            embedding = self.relation_projection(embedding, inv_rel_queries[:, i])
        query_embedding = [e.unsqueeze(1) for e in embedding]
        return self.cal_logit(inv_query_embed, entity_feat, query_embedding)

    def sync_read(self, *args):
        for arg in args:
            if arg is not None:
                arg.job_handle.sync()

    def forward(self,
                positive_sample, 
                negative_sample, 
                query_structure,
                query_matrix,
                device,
                all_neg=None,
                reg_coeff=0.):
        # #!
        # positive_sample = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7]) #!
        # batch_idxs_dict = {
        #     ('e', ('r',)): [0, 2],
        #     (('e', ('r',)), ('e', ('r',))): [1, 3],
        #     # (('e', ('r',)), ('e', ('r',)), ('u',)): [4, 5],
        #     # ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): [6, 7],
        #     # (((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)), (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('u',)), ('e', ('r',))): [7, 9],
        # }
        # #!
        # #!
        # batch_queries_dict = {
        #     ('e', ('r',)): torch.LongTensor([[5, 8], [2, 3]]),
        #     (('e', ('r',)), ('e', ('r',))): torch.LongTensor([[5, 8, 2, 3], [2, 3, 1, 6]]),
        #     # (('e', ('r',)), ('e', ('r',)), ('u',)): torch.LongTensor([[5, 8, 2, 3, -1], [2, 3, 1, 6, -1]]),
        #     # ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): torch.LongTensor([[5, 8, 2, 3, -1, 0], [2, 3, 1, 6, -1, 1]]),
        #     # (((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)), (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('u',)), ('e', ('r',))): torch.LongTensor([[5, 8, 2, 3, -1, 0, 1, 2, 3, 3, 3, 4, -1, 5, 6], [2, 3, 1, 6, -1, 1, 2, 3, 4, 5, 5, 5, -1, 7, 8]]),
        # }
        # #!

        if positive_sample is not None:
            all_positive_embedding = self.entity_embedding(positive_sample, "positive_entity")
            all_pos_ent_feat = self.entity_feat.read(positive_sample, "ent_pos_feat") if self.has_feat else None
        if negative_sample is not None:
            batch_size, negative_size = negative_sample.shape
            if all_neg is None:
                all_negative_embedding = self.entity_embedding(negative_sample.view(-1), "negative_entity")
            else:
                all_negative_embedding = all_neg
            all_neg_ent_feat = self.entity_feat.read(negative_sample.view(-1), "ent_neg_feat") if self.has_feat else None
        if query_structure[0] == '<':  # special handling for inverse query
            tail_idx, rel_queries = query_matrix.split([1, query_matrix.shape[1] - 1], dim=1)

            query_embedding = self.embed_reverse_query(tail_idx, rel_queries)
            positive_logits = negative_logits = None
            if positive_sample is not None:
                self.sync_read(all_positive_embedding, all_pos_ent_feat)
                if all_positive_embedding is not None:
                    all_positive_embedding = all_positive_embedding.unsqueeze(1)
                if all_pos_ent_feat is not None:
                    all_pos_ent_feat = all_pos_ent_feat.unsqueeze(1)
                positive_logits = self.cal_reverse_logit(query_embedding,
                                                            all_positive_embedding,
                                                            all_pos_ent_feat,
                                                            rel_queries)
            if negative_sample is not None:
                self.sync_read(all_negative_embedding, all_neg_ent_feat)
                if all_negative_embedding is not None:
                    all_negative_embedding = all_negative_embedding.view(batch_size, negative_size, -1)
                if all_neg_ent_feat is not None:
                    all_neg_ent_feat = all_neg_ent_feat.view(batch_size, negative_size, -1)
                negative_logits = self.cal_reverse_logit(query_embedding, 
                                                            all_negative_embedding,
                                                            all_neg_ent_feat,
                                                            rel_queries)
            reg_loss = self.cal_reg(all_positive_embedding, all_negative_embedding) if reg_coeff != 0 else 0
            return positive_logits, negative_logits, reg_loss

        query_embedding, end_idx = self.embed_query(query_matrix, query_structure, 0, device)
        assert end_idx == query_matrix.shape[-1]
        query_embedding = self.merge_embeddings([query_embedding], dim=0, expand_dim=-2)

        if positive_sample is not None:
            self.sync_read(all_positive_embedding, all_pos_ent_feat)
            all_positive_embedding = self.entity_regularizer(all_positive_embedding)
            positive_embedding = None if all_positive_embedding is None else all_positive_embedding.unsqueeze(1)
            positive_ent_feat = None if all_pos_ent_feat is None else all_pos_ent_feat.unsqueeze(1)
            positive_logits = self.cal_logit(positive_embedding, positive_ent_feat, query_embedding)
        else:
            positive_logits = None

        if negative_sample is not None:
            self.sync_read(all_negative_embedding, all_neg_ent_feat)
            if all_negative_embedding is not None:
                all_negative_embedding = self.entity_regularizer(all_negative_embedding.view(batch_size, negative_size, -1))
            if all_neg_ent_feat is not None:
                all_neg_ent_feat = all_neg_ent_feat.view(batch_size, negative_size, -1)
            negative_logits = self.cal_logit(all_negative_embedding, all_neg_ent_feat, query_embedding)
        else:
            negative_logits = None

        if reg_coeff != 0.:
            reg_loss = self.cal_reg(all_positive_embedding, all_negative_embedding)
        else:
            reg_loss = 0

        return positive_logits, negative_logits, reg_loss
