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
from smore.common.modules import Identity
from smore.common.embedding.sparse_embed import SparseEmbedding
from smore.common.torchext.ext_ops import rotate_dist


pi = 3.1415926

class RotateCenterSet(nn.Module):

    def __init__(self, dim, aggr=torch.max, nonlinear=True):
        super(RotateCenterSet, self).__init__()
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


class RotateReasoning(KGReasoning):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, 
                 optim_mode, batch_size, test_batch_size=1, sparse_embeddings=None,
                 sparse_device='gpu', use_cuda=False, query_name_dict=None, rotate_mode=None, logit_impl='native'):
        super(RotateReasoning, self).__init__(nentity=nentity, nrelation=nrelation, hidden_dim=hidden_dim, 
                                           gamma=gamma, optim_mode=optim_mode, batch_size=batch_size, test_batch_size=test_batch_size,
                                           sparse_embeddings=sparse_embeddings, sparse_device=sparse_device, use_cuda=use_cuda, query_name_dict=query_name_dict,
                                           logit_impl=logit_impl)
        self.geo = 'rotate'
        self.entity_dim = hidden_dim * 2
        self.entity_embedding = SparseEmbedding(nentity, self.entity_dim)

        self.center_net = RotateCenterSet(self.entity_dim, aggr=Aggr(rotate_mode[0]), nonlinear=rotate_mode[1])
        self.num_embedding_component = 2 # real and imaginary
        self.init_params()

    def to_device(self, device):
        super(RotateReasoning, self).to_device(device)
        self.center_net = self.center_net.to(device)

    def init_params(self):
        super(RotateReasoning, self).init_params()

    def share_memory(self):
        super(RotateReasoning, self).share_memory()
        self.center_net.share_memory()

    def relation_projection(self, cur_embedding, relation_ids):
        relation_embedding = self.relation_embedding(relation_ids).unsqueeze(1)
        phase_relation = relation_embedding / (self.embedding_range / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        re_embedding, im_embedding = cur_embedding
        new_re_embedding = re_embedding * re_relation - im_embedding * im_relation
        new_im_embedding = re_embedding * im_relation + im_embedding * re_relation
        return [new_re_embedding, new_im_embedding]
    
    def retrieve_embedding(self, entity_ids):
        '''
        Retrieve the entity embeddings given the entity indices
        Params:
            entity_ids: a list of entities indices
        '''
        embedding = self.entity_embedding(entity_ids)
        re_embedding, im_embedding = torch.chunk(embedding, 2, dim=1)
        return [re_embedding.unsqueeze(1), im_embedding.unsqueeze(1)] # [num_queries, 1, embedding_dim]
    
    def intersection_between_stacked_embedding(self, stacked_embedding_list):
        embedding = self.center_net(stacked_embedding_list) # [32, 6, 16]
        return torch.chunk(embedding, 2, dim=-1)

    def native_cal_logit(self, entity_embedding, entity_feat, query_embedding):
        assert entity_feat is None
        query_re_embedding, query_im_embedding = query_embedding
        entity_re_embedding, entity_im_embedding = torch.chunk(entity_embedding.unsqueeze(1), 2, dim=-1)
        re_distance = entity_re_embedding - query_re_embedding
        im_distance = entity_im_embedding - query_im_embedding
        distance = torch.stack([re_distance, im_distance], dim=0)
        distance = distance.norm(dim=0)
        logit = self.gamma - distance.sum(dim=-1)
        logit = torch.max(logit, dim=1)[0]
        return logit

    def custom_cal_logit(self, entity_embedding, entity_feat, query_embedding):
        assert entity_feat is None
        query_re_embedding, query_im_embedding = query_embedding
        logit = self.gamma - rotate_dist(entity_embedding, query_re_embedding, query_im_embedding)
        logit = torch.max(logit, dim=1)[0]
        return logit
