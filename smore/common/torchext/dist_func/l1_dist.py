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


class L1DistFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, entity_embed, center_embed):
        assert entity_embed.is_contiguous()
        assert center_embed.is_contiguous()
        assert entity_embed.shape[-1] == center_embed.shape[-1]
        assert entity_embed.shape[0] % center_embed.shape[0] == 0 or center_embed.shape[0] % entity_embed.shape[0] == 0
        assert entity_embed.shape[1] % center_embed.shape[1] == 0 or center_embed.shape[1] % entity_embed.shape[1] == 0
        
        out_rows = max(entity_embed.shape[0], center_embed.shape[0])
        out_cols = max(entity_embed.shape[1], center_embed.shape[1])
        with torch.no_grad():
            dst = entity_embed.new(out_rows, out_cols).contiguous()
            ctx.save_for_backward(entity_embed.data, center_embed.data)
            extlib.l1_dist_forward(entity_embed, center_embed, dst)
            return dst

    @staticmethod
    def backward(ctx, grad_out):
        with torch.no_grad():
            entity_embed, center_embed = ctx.saved_tensors
            grad_entity = grad_out.new(entity_embed.shape).zero_()
            grad_center = grad_out.new(center_embed.shape).zero_()
            extlib.l1_dist_backward(grad_out, entity_embed, center_embed, grad_entity, grad_center)
            return grad_entity, grad_center


def l1_dist(entity_embed, center_embed):
    if entity_embed.dim() != center_embed.dim():
        assert center_embed.dim() == 4
        assert entity_embed.dim() == 3
        l_dist = []
        for i in range(center_embed.shape[1]):
            center = center_embed[:, i, :, :].contiguous()
            d = L1DistFunc.apply(entity_embed, center)
            l_dist.append(d)
        d = torch.stack(l_dist, dim=1)
        return d
    else:
        assert entity_embed.dim() == 3 and center_embed.dim() == 3
        return L1DistFunc.apply(entity_embed, center_embed)


def naive_l1(entity_embedding, query_embedding):
    distance = entity_embedding - query_embedding
    return torch.norm(distance, p=1, dim=-1)


def test_l1():
    entity = Parameter(torch.randn(30, 1, 400).cuda())
    center = Parameter(torch.randn(1, 20, 400).cuda())

    l2 = l1_dist(entity, center)
    loss = torch.sum(l2 ** 2) * 3.14
    print(loss.item())
    loss.backward()
    e2 = entity.grad.clone()
    c2 = center.grad.clone()

    print('\n========\n')
    entity.grad = center.grad = None

    l1 = naive_l1(entity, center)
    loss = torch.sum(l1 ** 2) * 3.14
    print(loss.item())
    loss.backward()
    e1 = entity.grad.clone()
    c1 = center.grad.clone()

    print(torch.mean(torch.abs(e1 - e2)))
    print(torch.mean(torch.abs(c1 - c2)))


if __name__ == '__main__':
    import numpy as np
    import random
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    test_l1()
