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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from smore.common.embedding.sparse_embed import SparseEmbedding
from smore.models.kg_reasoning import KGReasoning
from smore.models.featured_embedding import get_feat_embed_mod
from smore.models.box import CenterIntersection
from smore.common.torchext.ext_ops import l1_dist, l2_dist


class VecReasoning(KGReasoning):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, 
                 optim_mode, batch_size, test_batch_size=1, sparse_embeddings=None, 
                 sparse_device='gpu', use_cuda=False, query_name_dict=None, model_config=None, logit_impl='native'):
        super(VecReasoning, self).__init__(nentity=nentity, nrelation=nrelation, hidden_dim=hidden_dim, 
                                           gamma=gamma, optim_mode=optim_mode, batch_size=batch_size, test_batch_size=test_batch_size,
                                           sparse_embeddings=sparse_embeddings, sparse_device=sparse_device, use_cuda=use_cuda, 
                                           query_name_dict=query_name_dict, logit_impl=logit_impl)
        self.geo = 'vec'
        self.entity_embedding = SparseEmbedding(nentity, self.entity_dim)
        self.center_net = CenterIntersection(self.entity_dim)
        self.num_embedding_component = 1
        for c in model_config:
            if c == 'l2':
                self.dist_func = l2_dist
            if c == 'l1':
                self.dist_func = l1_dist
        self.init_params()

    def to_device(self, device):
        super(VecReasoning, self).to_device(device)
        self.center_net = self.center_net.cuda(device)

    def share_memory(self):
        super(VecReasoning, self).share_memory()
        self.center_net.share_memory()

    def relation_projection(self, cur_embedding, relation_ids):
        relation_embedding = self.relation_embedding(relation_ids).unsqueeze(1)
        return [cur_embedding[0] + relation_embedding]
    
    def retrieve_embedding(self, entity_ids):
        '''
        Retrieve the entity embeddings given the entity indices
        Params:
            entity_ids: a list of entities indices
        '''
        embedding = self.entity_embedding(entity_ids)
        return [embedding.unsqueeze(1)] # [num_queries, 1, embedding_dim]
    
    def retrieve_relation_embedding(self, relation_ids):
        return self.relation_embedding(relation_ids).unsqueeze(1)
    
    def intersection_between_stacked_embedding(self, stacked_embedding_list):
        embedding = self.center_net(stacked_embedding_list) # [32, 6, 16]
        return [embedding]

    def native_cal_logit(self, entity_embedding, entity_feat, query_embedding):
        assert entity_feat is None
        distance = entity_embedding.unsqueeze(1) - query_embedding[0]
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        logit = torch.max(logit, dim=1)[0]
        return logit

    def custom_cal_logit(self, entity_embedding, entity_feat, query_embedding):
        assert entity_feat is None
        l1_d = self.dist_func(entity_embedding, query_embedding[0])
        logit = self.gamma - l1_d
        logit = torch.max(logit, dim=1)[0]
        return logit

    def embed_reverse_query(self, entity_idx, inv_rel_queries):
        embedding = self.retrieve_embedding(entity_idx.view(-1))[0]
        for i in range(inv_rel_queries.shape[1]):
            embedding = embedding - self.retrieve_relation_embedding(inv_rel_queries[:, i])
        return [embedding.unsqueeze(1)]

    def cal_reverse_logit(self, inv_query_embed, entity_embedding, entity_feat, inv_rel_queries):
        return self.cal_logit(entity_embedding, entity_feat, inv_query_embed)


class VecFeatured(VecReasoning):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 optim_mode, batch_size, test_batch_size=1, sparse_embeddings=None,
                 sparse_device='gpu', use_cuda=False, query_name_dict=None, model_config=None, logit_impl='native'):
        feature_mod = get_feat_embed_mod(model_config[0:1], hidden_dim, hidden_dim)
        if not feature_mod.embedding_needed:
            nentity = nrelation = 0
        super(VecFeatured, self).__init__(nentity=nentity, nrelation=nrelation, hidden_dim=hidden_dim,
                                          gamma=gamma, optim_mode=optim_mode, batch_size=batch_size, test_batch_size=test_batch_size,
                                          sparse_embeddings=sparse_embeddings, sparse_device=sparse_device, use_cuda=use_cuda,
                                          query_name_dict=query_name_dict, model_config=model_config, logit_impl=logit_impl)
        self.feature_mod = feature_mod
        self.has_feat = True

    def to_device(self, device):
        super(VecFeatured, self).to_device(device)
        self.feature_mod = self.feature_mod.cuda(device)

    def share_memory(self):
        super(VecFeatured, self).share_memory()
        self.feature_mod.share_memory()

    def retrieve_relation_embedding(self, relation_ids):
        relation_embedding = self.relation_embedding(relation_ids)
        if relation_embedding is not None:
            relation_embedding = relation_embedding.unsqueeze(1)
        relation_feat = self.relation_feat[relation_ids].unsqueeze(1)
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
        return [embedding.unsqueeze(1)] # [num_queries, 1, embedding_dim]

    def relation_projection(self, cur_embedding, relation_ids):
        embedding = self.retrieve_relation_embedding(relation_ids)
        return [cur_embedding[0] + embedding]

    def cal_logit(self, entity_embedding, entity_feat, query_embedding):
        embedding = self.feature_mod.forward_entity(entity_embedding, entity_feat)
        return super(VecFeatured, self).cal_logit(embedding, None, query_embedding)

    def cal_reverse_logit(self, inv_query_embed, entity_embedding, entity_feat, inv_rel_queries):
        embedding = self.feature_mod.forward_entity(entity_embedding, entity_feat)
        return super(VecFeatured, self).cal_logit(embedding, None, inv_query_embed)
