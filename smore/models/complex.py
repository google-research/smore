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

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pdb

from smore.models.kg_reasoning import KGReasoning
from smore.common.modules import Identity, Normalizer
from smore.common.embedding.sparse_embed import SparseEmbedding
from smore.models.featured_embedding import get_feat_embed_mod
from smore.common.torchext.ext_ops import complex_sim


class ComplexCenterSet(nn.Module):

    def __init__(self, dim, aggr=torch.max, nonlinear=True):
        super(ComplexCenterSet, self).__init__()
        self.dim = dim
        self.layers = nn.Parameter(torch.zeros(self.dim*4+4, self.dim))
        nn.init.xavier_uniform_(self.layers[:self.dim*4, :])
        self.aggr = aggr
        self.nonlinear = nonlinear

    def forward(self, embeddings):
        w1, w2, w3, w4, b1, b2, b3, b4 = torch.split(self.layers, [self.dim]*4+[1]*4, dim=0)
        x = F.relu(F.linear(embeddings, w1, b1.view(-1))) # (num_conj, batch_size, dim)
        x = F.linear(x, w2, b2.view(-1)) # (num_conj, batch_size, dim)
        if self.nonlinear:
            x = F.relu(x)
        if self.aggr in [torch.max, torch.min]:
            x = self.aggr(x, dim=0)[0]
        elif self.aggr in [torch.mean, torch.sum]:
            x = self.aggr(x, dim=0)
        x = F.relu(F.linear(x, w3, b3.view(-1)))
        x = F.linear(x, w4, b4.view(-1))

        return x


def Aggr(aggr_str):
    if aggr_str == 'Max':
        return torch.max
    elif aggr_str == 'Min':
        return torch.min
    elif aggr_str == 'Sum':
        return torch.sum
    elif aggr_str == 'Mean':
        return torch.mean
    else:
        assert False

pi = 3.1415926

class ComplexReasoning(KGReasoning):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, 
                 optim_mode, batch_size, test_batch_size=1, sparse_embeddings=None,
                 sparse_device='gpu', use_cuda=False, query_name_dict=None, complex_mode=None, logit_impl='native'):
        super(ComplexReasoning, self).__init__(nentity=nentity, nrelation=nrelation, hidden_dim=hidden_dim, 
                                           gamma=gamma, optim_mode=optim_mode, batch_size=batch_size, test_batch_size=test_batch_size,
                                           sparse_embeddings=sparse_embeddings, sparse_device=sparse_device, use_cuda=use_cuda, query_name_dict=query_name_dict, relation_dim=hidden_dim*2,
                                           logit_impl=logit_impl)
        self.geo = 'complex'
        self.entity_dim = hidden_dim * 2
        self.entity_embedding = SparseEmbedding(nentity, self.entity_dim)

        self.center_net = ComplexCenterSet(self.entity_dim, aggr=Aggr(complex_mode[0]), nonlinear=complex_mode[1])
        if len(complex_mode) == 3 and complex_mode[-1] == 'norm':
            self.embed_norm = Normalizer()
            self.need_norm = True
        else:
            self.embed_norm = Identity
            self.need_norm = False
        self.num_embedding_component = 2
        self.init_params()

    def to_device(self, device):
        super(ComplexReasoning, self).to_device(device)
        self.center_net = self.center_net.to(device)
        self.empty_logit_tensor = torch.tensor([]).to(device)

    def init_params(self):
        super(ComplexReasoning, self).init_params()

    def share_memory(self):
        super(ComplexReasoning, self).share_memory()
        self.center_net.share_memory()

    def _rel_proj(self, cur_embedding, relation_embedding):
        re_relation, im_relation = torch.chunk(relation_embedding, 2, dim=1)
        re_relation, im_relation = self.embed_norm(re_relation).unsqueeze(1), self.embed_norm(im_relation).unsqueeze(1)
        re_embedding, im_embedding = cur_embedding
        new_re_embedding = re_embedding * re_relation - im_embedding * im_relation
        new_im_embedding = re_embedding * im_relation + im_embedding * re_relation
        re_embedding = new_re_embedding
        im_embedding = new_im_embedding
        re_embedding, im_embedding = self.embed_norm(re_embedding), self.embed_norm(im_embedding)
        return [re_embedding, im_embedding]

    def relation_projection(self, cur_embedding, relation_ids):
        relation_embedding = self.relation_embedding(relation_ids)
        return self._rel_proj(cur_embedding, relation_embedding)

    def retrieve_embedding(self, entity_ids):
        '''
        Retrieve the entity embeddings given the entity indices
        Params:
            entity_ids: a list of entities indices
        '''
        embedding = self.entity_embedding(entity_ids)
        re_embedding, im_embedding = torch.chunk(embedding, 2, dim=-1)
        re_embedding, im_embedding = self.embed_norm(re_embedding), self.embed_norm(im_embedding)
        return [re_embedding.unsqueeze(1), im_embedding.unsqueeze(1)] # [num_queries, 1, embedding_dim]
    
    def intersection_between_stacked_embedding(self, stacked_embedding_list):
        embedding = self.center_net(stacked_embedding_list) # [32, 6, 16]
        re_embedding, im_embedding = torch.chunk(embedding, 2, dim=-1)
        re_embedding, im_embedding = self.embed_norm(re_embedding), self.embed_norm(im_embedding)
        return [re_embedding, im_embedding]

    def custom_cal_logit(self, entity_embedding, entity_feat, query_embedding):
        assert entity_feat is None
        query_re_embedding, query_im_embedding = query_embedding
        if self.need_norm:
            entity_re_embedding, entity_im_embedding = torch.chunk(entity_embedding, 2, dim=-1)
            entity_re_embedding, entity_im_embedding = self.embed_norm(entity_re_embedding), self.embed_norm(entity_im_embedding)
            entity_embedding = torch.cat((entity_re_embedding, entity_im_embedding), dim=-1).contiguous()
        logit = complex_sim(entity_embedding, query_re_embedding, query_im_embedding)
        logit = torch.max(logit, dim=1)[0]
        return logit

    def retrieve_relation_repr(self, relation_ids):
        return self.relation_embedding(relation_ids)

    def embed_reverse_query(self, entity_idx, inv_rel_queries):
        re_embedding, im_embedding = self.retrieve_embedding(entity_idx.view(-1))
        for i in range(inv_rel_queries.shape[1]):
            relation_embedding = self.retrieve_relation_repr(inv_rel_queries[:, i])
            re_relation, im_relation = torch.chunk(relation_embedding, 2, dim=1)
            re_relation, im_relation = self.embed_norm(re_relation).unsqueeze(1), self.embed_norm(im_relation).unsqueeze(1)
            new_re_embedding = re_embedding * re_relation + im_embedding * im_relation
            new_im_embedding =  - re_embedding * im_relation + im_embedding * re_relation
            re_embedding, im_embedding = self.embed_norm(new_re_embedding), self.embed_norm(new_im_embedding)
        return [re_embedding.unsqueeze(1), im_embedding.unsqueeze(1)]

    def cal_reverse_logit(self, inv_query_embed, entity_embedding, entity_feat, inv_rel_queries):
        assert entity_feat is None
        return self.cal_logit(entity_embedding, entity_feat, inv_query_embed)

    def native_cal_logit(self, entity_embedding, entity_feat, query_embedding):
        assert entity_feat is None
        query_re_embedding, query_im_embedding = query_embedding
        entity_re_embedding, entity_im_embedding = torch.chunk(entity_embedding.unsqueeze(1), 2, dim=-1)
        if self.need_norm:
            entity_re_embedding, entity_im_embedding = self.embed_norm(entity_re_embedding), self.embed_norm(entity_im_embedding)
        if entity_re_embedding.shape[0] == query_re_embedding.shape[0]:
            logit = (entity_re_embedding * query_re_embedding + entity_im_embedding * query_im_embedding).sum(dim=-1)
            logit = torch.max(logit, dim=1)[0]
        else:
            bsize = query_re_embedding.shape[0]
            nent = entity_re_embedding.shape[2]
            embed_dim = query_re_embedding.shape[-1]
            entity_re_embedding = entity_re_embedding.view(nent, embed_dim)
            entity_im_embedding = entity_im_embedding.view(nent, embed_dim)
            query_re_embedding = query_re_embedding.view(bsize, embed_dim)
            query_im_embedding = query_im_embedding.view(bsize, embed_dim)
            logit = torch.mm(query_re_embedding, entity_re_embedding.t()) + torch.mm(query_im_embedding, entity_im_embedding.t())
        return logit


class ComplexFeatured(ComplexReasoning):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 optim_mode, batch_size, test_batch_size=1, sparse_embeddings=None,
                 sparse_device='gpu', use_cuda=False, query_name_dict=None, model_config=None,logit_impl='native'):
        feature_mod = get_feat_embed_mod(model_config[0:1], hidden_dim * 2, hidden_dim * 2)
        if not feature_mod.embedding_needed:
            nentity = nrelation = 0
        super(ComplexFeatured, self).__init__(nentity=nentity, nrelation=nrelation, hidden_dim=hidden_dim,
                                              gamma=gamma, optim_mode=optim_mode, batch_size=batch_size, test_batch_size=test_batch_size,
                                              sparse_embeddings=sparse_embeddings, sparse_device=sparse_device, use_cuda=use_cuda,
                                              query_name_dict=query_name_dict, complex_mode=model_config[1:],logit_impl=logit_impl)
        self.feature_mod = feature_mod
        self.has_feat = True

    def to_device(self, device):
        super(ComplexFeatured, self).to_device(device)
        self.feature_mod = self.feature_mod.cuda(device)

    def share_memory(self):
        super(ComplexFeatured, self).share_memory()
        self.feature_mod.share_memory()

    def retrieve_relation_repr(self, relation_ids):
        relation_embedding = self.relation_embedding(relation_ids)
        relation_feat = self.relation_feat[relation_ids]
        embedding = self.feature_mod.forward_relation(relation_embedding, relation_feat)
        return embedding
 
    def retrieve_embedding(self, entity_ids):
        '''
        Retrieve the entity embeddings given the entity indices
        Params:
            entity_ids: a list of entities indices
        '''
        embedding = self.entity_embedding(entity_ids)
        feat = self.entity_feat.read(entity_ids)
        embedding = self.feature_mod.forward_entity(embedding, feat)
        re_embedding, im_embedding = torch.chunk(embedding, 2, dim=-1)
        re_embedding, im_embedding = self.embed_norm(re_embedding), self.embed_norm(im_embedding)
        return [re_embedding.unsqueeze(1), im_embedding.unsqueeze(1)] # [num_queries, 1, embedding_dim]

    def relation_projection(self, cur_embedding, relation_ids):
        embedding = self.retrieve_relation_repr(relation_ids)
        return self._rel_proj(cur_embedding, embedding)

    def cal_logit(self, entity_embedding, entity_feat, query_embedding):
        embedding = self.feature_mod.forward_entity(entity_embedding, entity_feat)
        return super(ComplexFeatured, self).cal_logit(embedding, None, query_embedding)

    def cal_reverse_logit(self, inv_query_embed, entity_embedding, entity_feat, inv_rel_queries):
        embedding = self.feature_mod.forward_entity(entity_embedding, entity_feat)
        return super(ComplexFeatured, self).cal_logit(embedding, None, inv_query_embed)
