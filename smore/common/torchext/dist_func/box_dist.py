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

import torch
import extlib_cuda as extlib
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BoxDistFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, entity_embed, center_embed, offset_embed, dist_type):
        assert entity_embed.shape[-1] == center_embed.shape[-1] == offset_embed.shape[-1]
        assert entity_embed.shape[0] % center_embed.shape[0] == 0 or center_embed.shape[0] % entity_embed.shape[0] == 0
        assert entity_embed.shape[1] % center_embed.shape[1] == 0 or center_embed.shape[1] % entity_embed.shape[1] == 0
        assert center_embed.shape == offset_embed.shape
        out_rows = max(entity_embed.shape[0], center_embed.shape[0])
        out_cols = max(entity_embed.shape[1], center_embed.shape[1])
        with torch.no_grad():
            dst = entity_embed.new(out_rows, out_cols).contiguous()
            ctx.dist_type = dist_type
            ctx.save_for_backward(entity_embed.data, center_embed.data, offset_embed.data)
            extlib.box_dist_forward(entity_embed, center_embed, offset_embed, dst, dist_type)
            return dst

    @staticmethod
    def backward(ctx, grad_out):
        with torch.no_grad():
            entity_embed, center_embed, offset_embed = ctx.saved_tensors
            grad_entity = grad_out.new(entity_embed.shape).zero_()
            grad_center = grad_out.new(center_embed.shape).zero_()
            grad_offset = grad_out.new(offset_embed.shape).zero_()
        
            extlib.box_dist_backward(grad_out, entity_embed, center_embed, offset_embed, grad_entity, grad_center, grad_offset, ctx.dist_type)
            return grad_entity, grad_center, grad_offset, None


def box_dist(entity_embed, center_embed, offset_embed, dist_type):
    if entity_embed.dim() != center_embed.dim():
        assert center_embed.dim() == 4
        assert entity_embed.dim() == 3
        l_dist = []
        for i in range(center_embed.shape[1]):
            c = center_embed[:, i, :, :].contiguous()
            o = offset_embed[:, i, :, :].contiguous()
            d = BoxDistFunc.apply(entity_embed, c, o, dist_type)
            l_dist.append(d)
        d = torch.stack(l_dist, dim=1)
        return d
    else:
        assert entity_embed.dim() == 3 and center_embed.dim() == 3 and offset_embed.dim() == 3
        return BoxDistFunc.apply(entity_embed, center_embed, offset_embed, dist_type)

def box_dist_out(entity_embed, center_embed, offset_embed):
    return box_dist(entity_embed, center_embed, offset_embed, 'out')

def box_dist_in(entity_embed, center_embed, offset_embed):
    return box_dist(entity_embed, center_embed, offset_embed, 'in')

def box_fast_logit(entity_embed, center_embed, offset_embed):
    d1 = box_dist_out(entity_embed, center_embed, offset_embed)
    d2 = box_dist_in(entity_embed, center_embed, offset_embed)
    logit = 24 - d1 - 1 * d2
    return logit

def naive_box_dist_out(entity_embedding, query_center_embedding, query_offset_embedding):
    delta = (entity_embedding - query_center_embedding).abs()
    distance_out = F.relu(delta - query_offset_embedding)
    dist = torch.norm(distance_out, p=1, dim=-1)
    return dist


def naive_box_dist_in(entity_embedding, query_center_embedding, query_offset_embedding):
    delta = (entity_embedding - query_center_embedding).abs()
    distance_in = torch.min(delta, query_offset_embedding)
    dist = torch.norm(distance_in, p=1, dim=-1)
    return dist


def naive_logit(entity_embedding, query_center_embedding, query_offset_embedding):
    delta = (entity_embedding - query_center_embedding).abs()
    distance_out = F.relu(delta - query_offset_embedding)
    distance_in = torch.min(delta, query_offset_embedding)
    logit = 24  - torch.norm(distance_out, p=1, dim=-1) - 1 * torch.norm(distance_in, p=1, dim=-1)
    return logit


def test_box():
    entity = Parameter(torch.randn(30, 1, 400).cuda())
    center = Parameter(torch.randn(1, 20, 400).cuda())
    offset = Parameter(torch.randn(1, 20, 400).cuda())

    l2 = box_fast_logit(entity, center, offset)
    loss = torch.sum(l2 ** 2) * 3.14
    print(loss.item())
    loss.backward()
    e2 = entity.grad.clone()
    c2 = center.grad.clone()
    o2 = offset.grad.clone()
    entity.grad = center.grad = offset.grad = None

    l1 = naive_logit(entity, center, offset)
    loss = torch.sum(l1 ** 2) * 3.14
    print(loss.item())
    loss.backward()
    e1 = entity.grad.clone()
    c1 = center.grad.clone()
    o1 = offset.grad.clone()
    print(torch.mean(torch.abs(e1 - e2)))
    print(torch.mean(torch.abs(c1 - c2)))
    print(torch.mean(torch.abs(o1 - o2)))


if __name__ == '__main__':
    import numpy as np
    import random
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    test_box()

