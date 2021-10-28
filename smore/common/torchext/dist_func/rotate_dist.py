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


class RotateDistFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, entity_embed, re_embed, im_embed):
        assert entity_embed.is_contiguous()
        assert re_embed.is_contiguous()
        assert im_embed.is_contiguous()
        assert entity_embed.shape[-1] == re_embed.shape[-1] * 2 == im_embed.shape[-1] * 2
        assert entity_embed.shape[0] % re_embed.shape[0] == 0 or re_embed.shape[0] % re_embed.shape[0] == 0
        assert entity_embed.shape[1] % re_embed.shape[1] == 0 or re_embed.shape[1] % entity_embed.shape[1] == 0
        assert re_embed.shape == im_embed.shape
        out_rows = max(entity_embed.shape[0], re_embed.shape[0])
        out_cols = max(entity_embed.shape[1], re_embed.shape[1])
        with torch.no_grad():
            dst = entity_embed.new(out_rows, out_cols).contiguous()
            ctx.save_for_backward(entity_embed.data, re_embed.data, im_embed.data)
            extlib.rotate_dist_forward(entity_embed, re_embed, im_embed, dst)
            return dst

    @staticmethod
    def backward(ctx, grad_out):
        with torch.no_grad():
            entity_embed, re_embed, im_embed = ctx.saved_tensors
            grad_entity = grad_out.new(entity_embed.shape).zero_()
            grad_re = grad_out.new(re_embed.shape).zero_()
            grad_im = grad_out.new(im_embed.shape).zero_()
            extlib.rotate_dist_backward(grad_out, entity_embed, re_embed, im_embed, grad_entity, grad_re, grad_im)
            return grad_entity, grad_re, grad_im


def rotate_dist(entity_embed, re_embed, im_embed):
    if entity_embed.dim() != re_embed.dim():
        assert re_embed.dim() == 4
        assert entity_embed.dim() == 3
        l_dist = []
        for i in range(re_embed.shape[1]):
            re = re_embed[:, i, :, :].contiguous()
            im = im_embed[:, i, :, :].contiguous()
            d = RotateDistFunc.apply(entity_embed, re, im)
            l_dist.append(d)
        d = torch.stack(l_dist, dim=1)
        return d
    else:
        assert entity_embed.dim() == 3 and re_embed.dim() == 3 and im_embed.dim() == 3
        return RotateDistFunc.apply(entity_embed, re_embed, im_embed)


def naive_rotate_dist(entity_embedding, query_re_embedding, query_im_embedding):
    entity_re_embedding, entity_im_embedding = torch.chunk(entity_embedding, 2, dim=-1)
    re_distance = entity_re_embedding - query_re_embedding
    im_distance = entity_im_embedding - query_im_embedding
    distance = torch.stack([re_distance, im_distance], dim=0)
    distance = distance.norm(dim=0)
    logit = distance.sum(dim=-1)
    return logit


def test_rotate():
    entity = Parameter(torch.randn(30, 1, 400).cuda())
    re = Parameter(torch.randn(1, 20, 200).cuda())
    im = Parameter(torch.randn(1, 20, 200).cuda())

    l2 = rotate_dist(entity, re, im)
    loss = torch.sum(l2 ** 2) * 3.14
    print(loss.item())
    loss.backward()
    e2 = entity.grad.clone()
    r2 = re.grad.clone()
    i2 = im.grad.clone()

    print('\n========\n')
    entity.grad = re.grad = im.grad = None

    l1 = naive_rotate_dist(entity, re, im)
    loss = torch.sum(l1 ** 2) * 3.14
    print(loss.item())
    loss.backward()
    e1 = entity.grad.clone()
    r1 = re.grad.clone()
    i1 = im.grad.clone()

    print(torch.mean(torch.abs(e1 - e2)))
    print(torch.mean(torch.abs(r1 - r2)))
    print(torch.mean(torch.abs(i1 - i2)))


if __name__ == '__main__':
    import numpy as np
    import random
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    test_rotate()
