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

from collections import namedtuple
BetaDist = namedtuple('BetaDist', ['concentration1', 'concentration0'])

class BetaDistFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, entity_embed, re_embed, im_embed, dist_name):
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
            ctx.dist_name = dist_name
            ctx.save_for_backward(entity_embed.data, re_embed.data, im_embed.data)
            extlib.beta_dist_forward(entity_embed, re_embed, im_embed, dst, dist_name)
            return dst

    @staticmethod
    def backward(ctx, grad_out):
        with torch.no_grad():
            entity_embed, re_embed, im_embed = ctx.saved_tensors
            grad_entity = grad_out.new(entity_embed.shape).zero_()
            grad_re = grad_out.new(re_embed.shape).zero_()
            grad_im = grad_out.new(im_embed.shape).zero_()
            extlib.beta_dist_backward(grad_out, entity_embed, re_embed, im_embed, grad_entity, grad_re, grad_im, ctx.dist_name)
            return grad_entity, grad_re, grad_im, None


def beta_dist(entity_embed, query_dist, dist_name):
    re_embed, im_embed = query_dist.concentration1, query_dist.concentration0
    if entity_embed.dim() != re_embed.dim():
        assert re_embed.dim() == 4
        assert entity_embed.dim() == 3
        l_dist = []
        for i in range(re_embed.shape[1]):
            re = re_embed[:, i, :, :].contiguous()
            im = im_embed[:, i, :, :].contiguous()
            d = BetaDistFunc.apply(entity_embed, re, im, dist_name)
            l_dist.append(d)
        d = torch.stack(l_dist, dim=1)
        return d
    else:
        assert entity_embed.dim() == 3 and re_embed.dim() == 3 and im_embed.dim() == 3
        return BetaDistFunc.apply(entity_embed, re_embed, im_embed, dist_name)


def beta_kl(entity_embed, query_dist):
    return beta_dist(entity_embed, query_dist, "kl")


def beta_l2(entity_embed, query_dist):
    return beta_dist(entity_embed, query_dist, "l2")

def beta_fisher_approx(entity_embed, query_dist):
    return beta_dist(entity_embed, query_dist, "fisher_approx")

def naive_beta_kl(entity_embedding, query_dist):
    alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
    entity_dist = BetaDist(alpha_embedding, beta_embedding)
    kld = torch.distributions.kl._kl_beta_beta(entity_dist, query_dist)
    return torch.norm(kld, p=1, dim=-1)


def naive_beta_l2(entity_embedding, query_dist):
    alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
    d1 = (alpha_embedding - query_dist.concentration1) ** 2
    d2 = (beta_embedding - query_dist.concentration0) ** 2
    d = torch.sum(d1 + d2, dim=-1) * 0.5
    return d

def naive_beta_fisher_approx(entity_embedding, query_dist):
    alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
    d1 = (alpha_embedding - query_dist.concentration1)
    d2 = (beta_embedding - query_dist.concentration0)
    with torch.no_grad():
        tri_alpha = alpha_embedding.polygamma(1)
        tri_beta = beta_embedding.polygamma(1)
        tri_sum = -(alpha_embedding + beta_embedding).polygamma(1)
    t1 = (tri_alpha + tri_sum) * (d1 ** 2)
    t2 = 2 * tri_sum * d1 * d2
    t3 = (tri_beta + tri_sum) * (d2 ** 2)
    return 0.5 * torch.sum(t1 + t2 + t3, dim=-1)

def test_beta(dist_name):
    from smore.common.modules import Regularizer
    reg = Regularizer(1, 0.05, 1e9)
    entity = Parameter(reg(torch.randn(30, 1, 400)).data.cuda())
    re = Parameter(reg(torch.randn(1, 20, 200)).data.cuda())
    im = Parameter(reg(torch.randn(1, 20, 200)).data.cuda())
    query_dist = BetaDist(re, im)

    if dist_name == 'kl':
        fast_d = beta_kl
        slow_d = naive_beta_kl
    elif dist_name == 'l2':
        fast_d = beta_l2
        slow_d = naive_beta_l2
    elif dist_name == 'fisher_approx':
        fast_d = beta_fisher_approx
        slow_d = naive_beta_fisher_approx        
    else:
        raise NotImplementedError

    l2 = fast_d(entity, query_dist)
    loss = torch.sum(l2 ** 2) * 3.14
    print(loss.item())
    loss.backward()
    e2 = entity.grad.clone()
    r2 = re.grad.clone()
    i2 = im.grad.clone()

    print('\n========\n')
    entity.grad = re.grad = im.grad = None

    l1 = slow_d(entity, query_dist)
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
    test_beta('fisher_approx')
