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
from smore.common.torchext.ext_ops import distmult_sim


class DistmultCenterSet(nn.Module):

    def __init__(self, dim, aggr=torch.max, nonlinear=True):
        super(DistmultCenterSet, self).__init__()
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

class DistmultReasoning(KGReasoning):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, 
                 optim_mode, batch_size, test_batch_size=1, sparse_embeddings=None,
                 sparse_device='gpu', use_cuda=False, query_name_dict=None, distmult_mode=None, logit_impl='native'):
        super(DistmultReasoning, self).__init__(nentity=nentity, nrelation=nrelation, hidden_dim=hidden_dim, 
                                           gamma=gamma, optim_mode=optim_mode, batch_size=batch_size, test_batch_size=test_batch_size,
                                           sparse_embeddings=sparse_embeddings, sparse_device=sparse_device, use_cuda=use_cuda, query_name_dict=query_name_dict,
                                           logit_impl=logit_impl)
        self.geo = 'distmult'
        self.entity_embedding = SparseEmbedding(nentity, self.entity_dim)

        self.center_net = DistmultCenterSet(self.entity_dim, aggr=Aggr(distmult_mode[0]), nonlinear=distmult_mode[1])
        if len(distmult_mode) == 3 and distmult_mode[-1] == 'norm':
            self.embed_norm = Normalizer()
            self.need_norm = True
        else:
            self.embed_norm = Identity
            self.need_norm = False
        self.num_embedding_component = 1
        self.init_params()

    def to_device(self, device):
        super(DistmultReasoning, self).to_device(device)
        self.center_net = self.center_net.to(device)
        self.empty_logit_tensor = torch.tensor([]).to(device)

    def init_params(self):
        super(DistmultReasoning, self).init_params()

    def share_memory(self):
        super(DistmultReasoning, self).share_memory()
        self.center_net.share_memory()

    def relation_projection(self, cur_embedding, relation_ids):
        relation_embedding = self.relation_embedding(relation_ids).unsqueeze(1)
        relation_embedding = self.embed_norm(relation_embedding)
        embedding = cur_embedding[0] * relation_embedding
        embedding = self.embed_norm(embedding)
        return [embedding]

    def retrieve_embedding(self, entity_ids):
        '''
        Retrieve the entity embeddings given the entity indices
        Params:
            entity_ids: a list of entities indices
        '''
        embedding = self.embed_norm(self.entity_embedding(entity_ids))
        return [embedding.unsqueeze(1)] # [num_queries, 1, embedding_dim]
    
    def intersection_between_stacked_embedding(self, stacked_embedding_list):
        embedding = self.embed_norm(self.center_net(stacked_embedding_list)) # [32, 6, 16]
        return [embedding]

    def native_cal_logit(self, entity_embedding, entity_feat, query_embedding):
        assert entity_feat is None
        entity_embedding = entity_embedding.unsqueeze(1)
        if self.need_norm:
            entity_embedding = self.embed_norm(entity_embedding)
        logit = (query_embedding[0] * entity_embedding).sum(-1)
        logit = torch.max(logit, dim=1)[0]
        return logit
