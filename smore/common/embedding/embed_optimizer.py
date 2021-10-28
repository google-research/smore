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

import time
import math
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.multiprocessing import Queue, Pipe
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_scatter import scatter
from functools import partial
from abc import ABC, abstractmethod
from smore.common.embedding.sparse_embed import SparseEmbedding
from smore.common.embedding.embed_rw import EmbeddingRW
from smore.common.util import eval_tuple
from smore.common.consts import eps


def merge_grad_indices(idx_list):
    concatenated_indices = torch.cat(idx_list, dim=0)
    merged_indices, inverse_indices = torch.unique(concatenated_indices, return_inverse=True)
    return merged_indices, inverse_indices


def merge_grad_list(grad_list, inverse_indices, reduce_type="sum"):
    concatenated_grad = torch.cat(grad_list, dim=0)
    merged_grad = scatter(concatenated_grad, inverse_indices, dim=0, reduce=reduce_type)
    return merged_grad


class EmbedLookupFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, embed_opt, dummy_tensor, indices, name):
        ctx.embed_opt = embed_opt
        ctx.indices = indices
        ctx.idx_fwd = embed_opt.n_fwd
        embed_opt.n_fwd += 1
        with torch.no_grad():
            submat = embed_opt.embed_rw.read(indices, name)
        return submat

    @staticmethod
    def backward(ctx, grad_output):
        ctx.embed_opt.accumulate_grad(ctx.indices, grad_output, ctx.idx_fwd)
        return None, None, None, None


class EmbeddingOptimizer(ABC):

    @classmethod
    @abstractmethod
    def prepare_embed(cls, args, optim_mode, embed, fn_register):
        pass

    @abstractmethod
    def share_memory(self):
        pass

    @abstractmethod
    def prepare_forward(self, indices, name):
        pass

    @abstractmethod
    def update_step(self, indices, inverse_indices, grad, lr):
        pass

    def zero_grad(self):
        self.grad_list = []
        self.n_fwd = 0
        self.has_grad_list = []
        self.grad_stats = []

    def filter_stats(self, list_stat_full):
        list_st = []
        for i in self.has_grad_list:
            e = list_stat_full[i]
            e.job_handle.sync()
            list_st.append(e)
        return list_st

    @classmethod
    def prepare_optimizers(cls, args, list_sparse_embeds):
        if not args.share_optim_stats:  # no need to have global optimizer stats
            return
        optim_mode = eval_tuple(args.optim_mode)
        for e in list_sparse_embeds:
            assert isinstance(e, SparseEmbedding)
            fn_register = lambda n, t: e.register_tensor(n, t)
            cls.prepare_embed(args, optim_mode, e, fn_register)

    def set_local_rw(self, name, tensor, gpu_id):
        cur_rw = EmbeddingRW(tensor, gpu_id=gpu_id)
        setattr(self, name + '_rw', cur_rw)

    def __init__(self, args, sp_embed, gpu_id):
        self.args = args
        self.grad_stats = []
        self.embed_rw = EmbeddingRW(sp_embed.embedding, gpu_id=gpu_id)
        self.sparse_embed = sp_embed
        sp_embed.fwd_delegate = self.forward

        self.dummy_tensor = nn.Parameter(torch.randn(1, 1))
        self.grad_list = []
        self.n_fwd = 0
        self.step = 0
        self.gpu_id = gpu_id

        optim_mode = eval_tuple(args.optim_mode)
        self.optim_mode = optim_mode
        assert self.optim_mode[0] in ['fast', 'aggr']
        self.optimizer_name = optim_mode[1]
        self.optimizer_device = optim_mode[2]
        assert self.optimizer_device in ['gpu', 'cpu']
        self.squeeze = self.optim_mode[3]
        self.queue_size = self.optim_mode[4]

        self.share_optim_stats = args.share_optim_stats
        fn_local_rw = partial(self.set_local_rw, gpu_id=self.gpu_id)
        if not self.share_optim_stats:
            self.prepare_embed(args, optim_mode, sp_embed, fn_local_rw)
        else:
            for name, buf in sp_embed.named_buffers(recurse=False):
                fn_local_rw(name, buf)

    def accumulate_grad(self, indices, grad_submat, fwd_idx):
        self.grad_list.append((indices, grad_submat))
        self.has_grad_list.append(fwd_idx)

    def forward(self, indices, name):
        if self.sparse_embed.training and self.optimizer_device == 'cpu':
            self.prepare_forward(indices, name)
        return EmbedLookupFunc.apply(self, self.dummy_tensor, indices, name)

    def apply_grad(self, lr):
        self.step += 1
        if len(self.grad_list) == 0:
            return
        merged_indices, inverse_indices = merge_grad_indices([indices for indices, _ in self.grad_list])
        grad_list = [grad for _, grad in self.grad_list]
        inverse_indices = inverse_indices.to(grad_list[0].device)
        merged_grad = merge_grad_list(grad_list, inverse_indices)
        self.update_step(merged_indices, inverse_indices, merged_grad, lr)


class SGDEmbedOptimizer(EmbeddingOptimizer):

    @classmethod
    def prepare_embed(cls, args, optim_mode, embed, fn_register):
        pass

    def share_memory(self):
        pass

    def zero_grad(self):
        super(SGDEmbedOptimizer, self).zero_grad()

    def prepare_forward(self, indices, name):
        pass

    def update_step(self, indices, inverse_indices, grad, lr):
        with torch.no_grad():
            # calculate final grad and update embed_mat
            tmp = -lr * grad
            self.embed_rw.write(indices, tmp, name="embed_gd", additive=True)


class AdamEmbedOptimizer(EmbeddingOptimizer):

    @classmethod
    def prepare_embed(cls, args, optim_mode, embed, fn_register):
        fn_register("state_sum_fo", torch.zeros(embed.embedding.size()))
        squeeze = optim_mode[3]
        if squeeze:
            fn_register("state_sum_sq", torch.zeros(embed.embedding.size(0), 1))
        else:
            fn_register("state_sum_sq", torch.zeros(embed.embedding.size()))

    def share_memory(self):
        self.state_sum_fo_rw.embed.share_memory_()
        self.state_sum_sq_rw.embed.share_memory_()

    def prepare_forward(self, indices, name):
        fo = self.state_sum_fo_rw.read(indices, name)
        sq = self.state_sum_sq_rw.read(indices, name)
        self.grad_stats.append((fo, sq))

    def update_step(self, indices, inverse_indices, grad, lr):
        beta1 = 0.9
        beta2 = 0.999 # this might further improve the performance if we keep track of the optimization steps for each embedding rather than using a global optimization step for all embeddings
        # indices, grad on gpu
        stats_device = self.state_sum_sq_rw.embed.device
        with torch.no_grad():
            if self.optimizer_device == 'gpu':
                indices = indices.to(stats_device)
            # for indices, grad in zip(indices_list, grad_list):
            bias_correction1 = 1 - beta1 ** self.step
            bias_correction2 = 1 - beta2 ** self.step
            indices = indices.to(stats_device)
            if self.squeeze:
                grad_sum_sq = torch.mean(grad * grad, dim=1, keepdim=True) # gpu

            # retreive old momentum
            if self.optimizer_device == 'cpu':
                list_fo_full, list_sq_full = zip(*self.grad_stats)
                list_fo = self.filter_stats(list_fo_full)
                grad_avg_fo = merge_grad_list(list_fo, inverse_indices, reduce_type="max")
                list_sq = self.filter_stats(list_sq_full)
                grad_avg_sq = merge_grad_list(list_sq, inverse_indices, reduce_type="max")
            else:
                grad_avg_fo = self.state_sum_fo_rw.read(indices)
                grad_avg_sq = self.state_sum_sq_rw.read(indices)
            bak_fo = grad_avg_fo.clone()
            bak_sq = grad_avg_sq.clone()

            # update momentum
            if self.squeeze:
                grad_avg_sq.mul_(beta2).add_(grad_sum_sq * (1-beta2))
            else:
                grad_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
            grad_avg_fo.mul_(beta1).add_(grad, alpha=1-beta1)

            # calculate avg
            denom = (grad_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            # calculate final grad and update embed_mat
            tmp = -lr/bias_correction1 * grad_avg_fo/denom
            self.embed_rw.write(indices, tmp, name="embed_gd", additive=True)

            # copy new momentum to state_sum
            self.state_sum_fo_rw.write(indices, grad_avg_fo - bak_fo, name="update_fo", additive=True)
            self.state_sum_sq_rw.write(indices, grad_avg_sq - bak_sq, name="update_sq", additive=True)


class RMSpropEmbedOptimizer(EmbeddingOptimizer):

    @classmethod
    def prepare_embed(cls, args, optim_mode, embed, fn_register):
        squeeze = optim_mode[3]
        if squeeze:
            fn_register("state_sum", torch.zeros(embed.embedding.size(0)))
        else:
            fn_register("state_sum", torch.zeros(embed.embedding.size()))

    def share_memory(self):
        self.state_sum.share_memory_()

    def __init__(self, args, sp_embed, gpu_id):
        super(RMSpropEmbedOptimizer, self).__init__(args, sp_embed, gpu_id)

    def update_step(self, indices, grad, lr):
        alpha = 0.99
        # indices, grad on gpu
        with torch.no_grad():
            # for indices, grad in zip(indices_list, grad_list):
            indices = indices.to(self.state_sum.device)
            if self.squeeze:
                grad_sum = (grad * grad).mean(1) # gpu
            if self.optimizer_device == 'cpu':
                grad = grad.to(self.state_sum.device)
                if self.squeeze:
                    grad_sum = grad_sum.to(self.state_sum.device)

            # retreive old momentum
            square_avg = self.state_sum[indices] 
            if self.optimizer_device == 'gpu':
                square_avg = square_avg.to(grad.device)

            # update momentum
            if self.squeeze:
                square_avg.mul_(alpha).add_(grad_sum * (1-alpha))
            else:
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1-alpha)

            # calculate avg
            avg = square_avg.sqrt().add_(eps)
            if self.squeeze:
                avg = avg.unsqueeze(1)

            # copy new momentum to state_sum
            if self.optimizer_device == 'gpu':
                square_avg = square_avg.to(self.state_sum.device)
            self.state_sum.index_copy_(0, indices, square_avg)

            # calculate final grad and update embed_mat
            tmp = -lr * grad/avg
            if self.optimizer_device == 'gpu':
                tmp = tmp.to(self.sp_embed.device)
            self.sp_embed.embedding.index_add_(0, indices, tmp)


class AdaGradEmbedOptimizer(EmbeddingOptimizer):

    @classmethod
    def prepare_embed(cls, args, optim_mode, embed, fn_register):        
        squeeze = optim_mode[3]
        if squeeze:
            fn_register("state_sum", torch.zeros(embed.embedding.size(0), 1))
        else:
            fn_register("state_sum", torch.zeros(embed.embedding.size()))

    def share_memory(self):
        self.state_sum_rw.embed.share_memory_()

    def prepare_forward(self, indices, name):
        fo = self.state_sum_rw.read(indices, name)
        self.grad_stats.append(fo)

    def update_step(self, indices, inverse_indices, grad, lr):
        # indices, grad on gpu
        stats_device = self.state_sum_rw.embed.device
        with torch.no_grad():
            if self.optimizer_device == 'gpu':
                indices = indices.to(stats_device)
            if self.squeeze:
                grad_sum = torch.mean(grad * grad, dim=1, keepdim=True) # gpu
            else:
                grad_sum = grad * grad

            if self.optimizer_device == 'cpu':
                list_sq = self.filter_stats(self.grad_stats)
                square_avg = merge_grad_list(list_sq, inverse_indices, reduce_type="max")
            else:
                square_avg = self.state_sum_rw.read(indices)
            square_avg = square_avg + grad_sum
            avg = square_avg.sqrt().add_(eps)
            tmp = -lr * grad/avg
            self.embed_rw.write(indices, tmp, name="embed_gd", additive=True)
            self.state_sum_rw.write(indices, grad_sum, name="update_sq", additive=True)


def get_optim_class(args):
    optim_mode = eval_tuple(args.optim_mode)
    optimizer_name = optim_mode[1]
    if optimizer_name == 'adam':
        return AdamEmbedOptimizer
    elif optimizer_name == 'rmsprop':
        return RMSpropEmbedOptimizer
    elif optimizer_name == 'adagrad':
        return AdaGradEmbedOptimizer
    elif optimizer_name == 'sgd':
        return SGDEmbedOptimizer
    else:
        raise NotImplementedError
