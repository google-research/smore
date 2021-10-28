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
from smore.common.torchext.ext_ops import box_dist_in, box_dist_out


class BoxOffsetIntersection(nn.Module):
    
    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layers = nn.Parameter(torch.zeros(self.dim*2+2, self.dim))
        nn.init.xavier_uniform_(self.layers[:self.dim*2, :])

    def forward(self, embeddings):
        w1, w2, b1, b2 = torch.split(self.layers, [self.dim, self.dim, 1, 1], dim=0)
        layer1_act = F.relu(F.linear(embeddings, w1, b1.view(-1)))
        layer1_mean = torch.mean(layer1_act, dim=0) 
        gate = torch.sigmoid(F.linear(layer1_mean, w2, b2.view(-1)))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layers = nn.Parameter(torch.zeros(self.dim*2+2, self.dim))
        nn.init.xavier_uniform_(self.layers[:self.dim*2, :])

    def forward(self, embeddings):
        w1, w2, b1, b2 = torch.split(self.layers, [self.dim, self.dim, 1, 1], dim=0)
        layer1_act = F.relu(F.linear(embeddings, w1, b1.view(-1))) # (num_conj, dim)
        attention = F.softmax(F.linear(layer1_act, w2, b2.view(-1)), dim=0) # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class BoxReasoning(KGReasoning):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, 
                 optim_mode, batch_size, test_batch_size=1, sparse_embeddings=None,
                 sparse_device='gpu', use_cuda=False, query_name_dict=None, box_mode=None,logit_impl='native'):
        super(BoxReasoning, self).__init__(nentity=nentity, nrelation=nrelation, hidden_dim=hidden_dim, 
                                           gamma=gamma, optim_mode=optim_mode, batch_size=batch_size, test_batch_size=test_batch_size,
                                           sparse_embeddings=sparse_embeddings, sparse_device=sparse_device, use_cuda=use_cuda, query_name_dict=query_name_dict,
                                           logit_impl=logit_impl)
        self.geo = 'box'
        self.entity_embedding = SparseEmbedding(nentity, self.entity_dim)
        activation, cen = box_mode
        self.cen = cen # hyperparameter that balances the in-box distance and the out-box distance
        if activation == 'none':
            self.func = Identity
        elif activation == 'relu':
            self.func = F.relu
        elif activation == 'softplus':
            self.func = F.softplus

        self.offset_embedding = SparseEmbedding(nrelation, self.entity_dim)
        self.center_net = CenterIntersection(self.entity_dim)
        self.offset_net = BoxOffsetIntersection(self.entity_dim)
        self.num_embedding_component = 2
        self.init_params()

    def named_sparse_embeddings(self):
        list_sparse = super(BoxReasoning, self).named_sparse_embeddings()
        if 'r' in self.sparse_embeddings:
            list_sparse.append(("offset_embedding", self.offset_embedding))
        return list_sparse

    def named_dense_embedding_params(self):
        pgen = super(BoxReasoning, self).named_dense_embedding_params()
        for name, param in pgen:
            yield name, param
        if 'r' not in self.sparse_embeddings:
            for name, param in self.offset_embedding.named_parameters():
                yield name, param

    def to_device(self, device):
        super(BoxReasoning, self).to_device(device)
        self.center_net = self.center_net.to(device)
        self.offset_net = self.offset_net.to(device)
        self.zero_offset_tensor = torch.zeros([self.batch_size, 1, self.entity_dim]).to(device)
        self.empty_logit_tensor = torch.tensor([]).to(device)
        if 'r' not in self.sparse_embeddings or self.sparse_device == 'gpu':
            self.offset_embedding = self.offset_embedding.cuda(device)

    def init_params(self):
        super(BoxReasoning, self).init_params()
        self.offset_embedding.init_params(0, self.embedding_range)

    def share_memory(self):
        super(BoxReasoning, self).share_memory()
        self.center_net.share_memory()
        self.offset_net.share_memory()
        self.offset_embedding.share_memory()

    def relation_projection(self, cur_embedding, relation_ids):
        relation_embedding = self.relation_embedding(relation_ids).unsqueeze(1)
        offset_embedding = self.offset_embedding(relation_ids).unsqueeze(1)
        return [cur_embedding[0] + relation_embedding, cur_embedding[1] + self.func(offset_embedding)]

    def retrieve_embedding(self, entity_ids):
        '''
        Retrieve the entity embeddings given the entity indices
        Params:
            entity_ids: a list of entities indices
        '''
        embedding = self.entity_embedding(entity_ids)
        offset_embedding = torch.zeros_like(embedding).to(embedding.device)
        return [embedding.unsqueeze(1), offset_embedding.unsqueeze(1)]
    
    def intersection_between_stacked_embedding(self, stacked_embedding_list):
        embedding, offset_embedding = torch.chunk(stacked_embedding_list, 2, dim=-1)
        embedding = self.center_net(embedding) # [32, 6, 16]
        offset_embedding = self.offset_net(offset_embedding)
        return [embedding, offset_embedding]

    def native_cal_logit(self, entity_embedding, entity_feat, query_embedding):
        assert entity_feat is None
        query_center_embedding, query_offset_embedding = query_embedding
        delta = (entity_embedding.unsqueeze(1) - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        logit = torch.max(logit, dim=1)[0]
        return logit

    def custom_cal_logit(self, entity_embedding, entity_feat, query_embedding):
        assert entity_feat is None
        query_center_embedding, query_offset_embedding = query_embedding
        d1 = box_dist_out(entity_embedding, query_center_embedding, query_offset_embedding)
        d2 = box_dist_in(entity_embedding, query_center_embedding, query_offset_embedding)
        logit = self.gamma - d1 - self.cen * d2
        logit = torch.max(logit, dim=1)[0]
        return logit
