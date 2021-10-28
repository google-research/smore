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


class FeatOnlyMod(nn.Module):
    def __init__(self, feat_dim, ent_dim, rel_dim):
        super(FeatOnlyMod, self).__init__()
        self.embedding_needed = False
        self.entity_proj = nn.Linear(feat_dim, ent_dim)
        self.relation_proj = nn.Linear(feat_dim, rel_dim)

    def forward_entity(self, entity_embedding, entity_feature):
        ent_embed = self.entity_proj(entity_feature.float())
        return ent_embed
    
    def forward_relation(self, relation_embedding, relation_feat):
        rel_embed = self.relation_proj(relation_feat.float())
        return rel_embed


class FeatConcatMod(nn.Module):
    def __init__(self, feat_dim, ent_dim, rel_dim):
        super(FeatConcatMod, self).__init__()
        self.embedding_needed = True
        self.entity_proj = nn.Linear(ent_dim + feat_dim, ent_dim)
        self.relation_proj = nn.Linear(rel_dim + feat_dim, rel_dim)

    def forward_entity(self, entity_embedding, entity_feature):
        ent_embed = self.entity_proj(torch.cat((entity_embedding, entity_feature), dim=-1))
        return ent_embed
    
    def forward_relation(self, relation_embedding, relation_feat):
        rel_embed = self.relation_proj(torch.cat((relation_embedding, relation_feat), dim=-1))
        return rel_embed


def get_feat_embed_mod(model_config, ent_dim, rel_dim):
    mod_config = None
    for cfg in model_config:
        if cfg.startswith('feat'):
            mod_config = cfg[5:]
            break
    assert mod_config is not None
    mod, feat_dim = mod_config.split('-')
    feat_dim = int(feat_dim)
    if mod == 'concat':
        return FeatConcatMod(feat_dim, ent_dim, rel_dim)
    elif mod == 'only':
        return FeatOnlyMod(feat_dim, ent_dim, rel_dim)
    else:
        raise ValueError('unknown mod %s' % mod)
