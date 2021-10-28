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
import math

from smore.models.kg_reasoning import KGReasoning
from smore.common.modules import Regularizer
from smore.common.embedding.sparse_embed import SparseEmbedding
from smore.common.torchext.dist_func.beta_dist import BetaDist, beta_kl, beta_l2, beta_fisher_approx, naive_beta_fisher_approx, naive_beta_kl, naive_beta_l2
import torch.distributions.kl as torch_kl


class BetaIntersection(nn.Module):

    def __init__(self, dim, norm_mode, norm_param_flag):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.norm_mode = norm_mode
        self.norm_param_flag = norm_param_flag
        if norm_mode == 'None' or not norm_param_flag:
            self.layers = nn.Parameter(torch.zeros(self.dim * 2 + self.dim + 2, 2 * self.dim))
            if norm_mode == 'batch':
                self.register_buffer('running_mean', torch.zeros(2 * self.dim))
                self.register_buffer('running_var', torch.ones(2 * self.dim))
                self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        elif norm_mode in ['layer', 'batch']:
            self.layers = nn.Parameter(torch.zeros(self.dim * 2 + self.dim + 4, 2 * self.dim))
            nn.init.ones_(self.layers[-2, :])
            if norm_mode == 'batch':
                self.register_buffer('running_mean', torch.zeros(2 * self.dim))
                self.register_buffer('running_var', torch.ones(2 * self.dim))
                self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        nn.init.xavier_uniform_(self.layers[:self.dim*2 + self.dim, :])
        

    def forward(self, alpha_embeddings, beta_embeddings):
        if self.norm_mode == 'None' or not self.norm_param_flag:
            w1, w2, b1, b2 = torch.split(self.layers, [self.dim * 2, self.dim, 1, 1], dim=0)
            if self.norm_mode in ['layer', 'batch']:
                w3 = None
                b3 = None
            if self.norm_mode == 'batch':
                exponential_average_factor = 0.1
                if self.training:
                    if self.num_batches_tracked is not None:
                        self.num_batches_tracked += 1
                    bn_training = True
                else:
                    bn_training = (self.running_mean is None) and (self.running_var is None)
        elif self.norm_mode in ['layer', 'batch']:
            w1, w2, b1, b2, w3, b3 = torch.split(self.layers, [self.dim * 2, self.dim, 1, 1, 1, 1], dim=0)
            w3 = w3[0]
            b3 = b3[0]
            if self.norm_mode == 'batch':
                exponential_average_factor = 0.1
                if self.training:
                    if self.num_batches_tracked is not None:
                        self.num_batches_tracked += 1
                    bn_training = True
                else:
                    bn_training = (self.running_mean is None) and (self.running_var is None)
            # print(w3)

        b2 = b2.view(2, -1)[0]
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        if self.norm_mode == 'None':
            layer1_act = F.relu(F.linear(all_embeddings, w1, b1.view(-1))) # (num_conj, batch_size, 2 * dim)
        elif self.norm_mode == 'layer':
            layer1_act = F.relu(F.layer_norm(F.linear(all_embeddings, w1, b1.view(-1)), [2 * self.dim], w3, b3, 1e-5)) # (num_conj, batch_size, 2 * dim)
        elif self.norm_mode == 'batch':
            # print(F.linear(all_embeddings, w1, b1.view(-1)).shape)
            embedding_shape = list(all_embeddings.shape)
            layer1_act = F.relu(F.batch_norm(F.linear(all_embeddings, w1, b1.view(-1)).view([embedding_shape[0]*embedding_shape[1], -1]), self.running_mean, self.running_var, w3, b3, bn_training, exponential_average_factor, 1e-5)) # (num_conj, batch_size, 2 * dim)
            layer1_act = layer1_act.view([embedding_shape[0], embedding_shape[1], -1])
        attention = F.softmax(F.linear(layer1_act, w2, b2), dim=0) # (num_conj, batch_size, dim)
        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding


class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers, norm_mode, norm_param_flag):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.norm_mode = norm_mode
        self.norm_param_flag = norm_param_flag

        assert num_layers >= 1
        w_offset = self.entity_dim + self.relation_dim + (num_layers - 1) * self.hidden_dim + self.entity_dim
        self.num_last_bias = math.ceil(self.entity_dim / self.hidden_dim)
        assert self.num_last_bias * self.hidden_dim >= self.entity_dim

        if norm_mode == 'None' or not self.norm_param_flag:
            self.layers = nn.Parameter(torch.zeros(w_offset + num_layers + self.num_last_bias, self.hidden_dim))
            self.nrows = [self.entity_dim + self.relation_dim] + [self.hidden_dim] * (self.num_layers - 1) + [self.entity_dim] + [1] * num_layers + [self.num_last_bias]
            self.bias_start = self.num_layers + 1
            if norm_mode == 'batch':
                self.register_buffer('running_means', torch.zeros([num_layers, self.hidden_dim]))
                self.register_buffer('running_vars', torch.ones([num_layers, self.hidden_dim]))
                self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        elif norm_mode in ['layer', 'batch']:
            self.layers = nn.Parameter(torch.zeros(w_offset + num_layers + self.num_last_bias + num_layers * 2, self.hidden_dim))
            nn.init.ones_(self.layers[w_offset + num_layers + self.num_last_bias:w_offset + num_layers + self.num_last_bias + num_layers])
            self.nrows = [self.entity_dim + self.relation_dim] + [self.hidden_dim] * (self.num_layers - 1) + [self.entity_dim] + [1] * num_layers + [self.num_last_bias] + [1] * num_layers + [1] * num_layers
            self.bias_start = self.num_layers + 1
            self.norm_weight_start = 2 * self.num_layers + 2
            self.norm_bias_start = 3 * self.num_layers + 2
            if norm_mode == 'batch':
                self.register_buffer('running_means', torch.zeros([num_layers, self.hidden_dim]))
                self.register_buffer('running_vars', torch.ones([num_layers, self.hidden_dim]))
                self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        nn.init.xavier_uniform_(self.layers[:w_offset, :])

        assert sum(self.nrows) == self.layers.shape[0]
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)

        params = torch.split(self.layers, self.nrows, dim=0)
        if self.norm_mode == 'batch':
            exponential_average_factor = 0.1
            if self.training:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                bn_training = True
            else:
                bn_training = (self.running_means is None) and (self.running_vars is None)
        w0, b0 = params[0], params[self.bias_start].view(-1)
        
        if self.norm_mode == 'None':
            x = F.relu(torch.matmul(x, w0) + b0)
        elif self.norm_mode == 'layer':
            if not self.norm_param_flag:
                x = F.relu(F.layer_norm(torch.matmul(x, w0) + b0, [self.hidden_dim], None, None, 1e-5))
            else:
                norm_w0, norm_b0 = params[self.norm_weight_start].view(-1), params[self.norm_bias_start].view(-1)
                x = F.relu(F.layer_norm(torch.matmul(x, w0) + b0, [self.hidden_dim], norm_w0, norm_b0, 1e-5))
        elif self.norm_mode == 'batch':
            if not self.norm_param_flag:
                norm_w0, norm_b0 = None, None
            else:
                norm_w0, norm_b0 = params[self.norm_weight_start].view(-1), params[self.norm_bias_start].view(-1)
            x = F.relu(F.batch_norm(torch.matmul(x, w0) + b0, self.running_means[0], self.running_vars[0], norm_w0, norm_b0, bn_training, exponential_average_factor, 1e-5))
            # print(norm_w0)

        for i in range(1, self.num_layers):
            wi, bi = params[i], params[self.bias_start + i].view(-1)
            if self.norm_mode == 'None':
                x = F.relu(F.linear(x, wi, bi))
            elif self.norm_mode == 'layer':
                if not self.norm_param_flag:
                    x = F.relu(F.layer_norm(F.linear(x, wi, bi), [self.hidden_dim], None, None, 1e-5))
                else:
                    norm_wi, norm_bi = params[self.norm_weight_start + i].view(-1), params[self.norm_bias_start + i].view(-1)
                    x = F.relu(F.layer_norm(F.linear(x, wi, bi), [self.hidden_dim], norm_wi, norm_bi, 1e-5))
            elif self.norm_mode == 'batch':
                if not self.norm_param_flag:
                    norm_wi, norm_bi = None, None
                else:
                    norm_wi, norm_bi = params[self.norm_weight_start + i].view(-1), params[self.norm_bias_start + i].view(-1)
                x = F.relu(F.batch_norm(F.linear(x, wi, bi), self.running_means[i], self.running_vars[i], norm_wi, norm_bi, bn_training, exponential_average_factor, 1e-5))

        w_last, b_last = params[self.num_layers], params[-1].view(-1)
        if b_last.shape[0] != self.entity_dim:
            b_last = b_last[:self.entity_dim]
        x = F.linear(x, w_last, b_last)
        x = self.projection_regularizer(x)

        return x


class BetaReasoning(KGReasoning):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, 
                 optim_mode, batch_size, test_batch_size=1, sparse_embeddings=None,
                 sparse_device='gpu', use_cuda=False, query_name_dict=None, beta_mode=None, logit_impl='native'):
        super(BetaReasoning, self).__init__(nentity=nentity, nrelation=nrelation, hidden_dim=hidden_dim, 
                                           gamma=gamma, optim_mode=optim_mode, batch_size=batch_size, test_batch_size=test_batch_size,
                                           sparse_embeddings=sparse_embeddings, sparse_device=sparse_device, use_cuda=use_cuda, 
                                           query_name_dict=query_name_dict,logit_impl=logit_impl)
        self.geo = 'beta'

        self.entity_embedding = SparseEmbedding(nentity, self.entity_dim * 2) # alpha and beta
        self.entity_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings are positive
        self.projection_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings after relation projection are positive

        if len(beta_mode) == 2:
            hidden_dim, num_layers = beta_mode
            dist_mode = 'kl'
            self.entity_range = self.embedding_range
            norm_mode = 'None'
            norm_param_flag = False
        elif len(beta_mode) == 3:
            hidden_dim, num_layers, dist_mode = beta_mode
            self.entity_range = self.embedding_range
            norm_mode = 'None'
            norm_param_flag = False
        elif len(beta_mode) == 4:
            hidden_dim, num_layers, dist_mode, self.entity_range = beta_mode
            norm_mode = 'None'
            norm_param_flag = False
        elif len(beta_mode) == 6:
            hidden_dim, num_layers, dist_mode, self.entity_range, norm_mode, norm_param_flag = beta_mode

        assert self.entity_range > 0

        assert dist_mode in ['kl', 'fisher', 'l2', 'naive_fisher']
        assert norm_mode in ['None', 'layer', 'batch']
        if dist_mode == 'kl':
            self.dist_func = beta_kl
        elif dist_mode == 'fisher':
            self.dist_func = beta_fisher_approx
        elif dist_mode == 'naive_fisher':
            self.dist_func = naive_beta_fisher_approx
        elif dist_mode == 'l2':
            self.dist_func = beta_l2
        self.center_net = BetaIntersection(self.entity_dim, norm_mode, norm_param_flag)
        self.projection_net = BetaProjection(self.entity_dim * 2, 
                                            self.relation_dim, 
                                            hidden_dim, 
                                            self.projection_regularizer, 
                                            num_layers,
                                            norm_mode,
                                            norm_param_flag)
        self.num_embedding_component = 2
        self.init_params()

    def init_params(self):
        self.entity_embedding.init_params(-self.entity_range, self.entity_range)
        self.relation_embedding.init_params(-self.embedding_range, self.embedding_range)

    def to_device(self, device):
        super(BetaReasoning, self).to_device(device)
        self.center_net = self.center_net.to(device)
        self.projection_net = self.projection_net.to(device)

    def share_memory(self):
        super(BetaReasoning, self).share_memory()
        self.center_net.share_memory()
        self.projection_net.share_memory()

    def relation_projection(self, cur_embedding, relation_ids):
        relation_embedding = self.relation_embedding(relation_ids).unsqueeze(1)
        num_lazy_union = cur_embedding[0].shape[1]
        relation_embedding = relation_embedding.repeat([1, num_lazy_union, 1])
        embedding = torch.cat(cur_embedding, dim=-1)
        embedding = self.projection_net(embedding, relation_embedding)
        return torch.chunk(embedding, 2, dim=-1)
    
    def negation(self, cur_embedding):
        return [1./embedding for embedding in cur_embedding]

    def retrieve_embedding(self, entity_ids):
        '''
        Retrieve the entity embeddings given the entity indices
        Params:
            entity_ids: a list of entities indices
        '''
        embedding = self.entity_regularizer(self.entity_embedding(entity_ids))
        alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        return [alpha_embedding.unsqueeze(1), beta_embedding.unsqueeze(1)]
    
    def intersection_between_stacked_embedding(self, stacked_embedding_list):
        alpha_embedding, beta_embedding = torch.chunk(stacked_embedding_list, 2, dim=-1)
        return self.center_net(alpha_embedding, beta_embedding)

    def native_cal_logit(self, entity_embedding, entity_feat, query_embedding):
        assert entity_feat is None
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding.unsqueeze(1), 2, dim=-1)
        entity_dist = BetaDist(alpha_embedding, beta_embedding)
        query_dist = BetaDist(*query_embedding)
        kld = torch_kl._kl_beta_beta(entity_dist, query_dist)
        logit = self.gamma - torch.norm(kld, p=1, dim=-1)
        logit = torch.max(logit, dim=1)[0]
        return logit

    def custom_cal_logit(self, entity_embedding, entity_feat, query_embedding):
        assert entity_feat is None
        kld = self.dist_func(entity_embedding, BetaDist(*query_embedding))
        logit = self.gamma - kld
        logit = torch.max(logit, dim=1)[0]
        return logit
