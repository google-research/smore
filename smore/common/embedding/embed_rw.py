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


import torch
import torch.nn as nn
import extlib_cuda as extlib
from abc import ABC, abstractmethod


class DummyJob(object):
    def sync(self):
        pass


class EmbeddingReadOnly(object):
    def __init__(self, embed, gpu_id=-1):
        super(EmbeddingReadOnly, self).__init__()
        self.embed = embed.data
        self.gpu_id = gpu_id
        self.embed_dim = embed.shape[1]
        if gpu_id == -1:
            self.device = 'cpu'
        else:
            self.device = 'cuda:{}'.format(gpu_id)
        if self.embed.is_cuda:
            assert self.embed.device == torch.device(self.device)
        self.dummy_job = DummyJob()
        self.embed.job_handle = self.dummy_job
        self.read_thread_pool = extlib.ThreadPool(1)
        assert gpu_id != -1
        self.stream = torch.cuda.Stream(gpu_id)
        self.default_stream = torch.cuda.default_stream(gpu_id)
        self.read_buf = {}
        self.read_src_cache = {}
        self.last_write_jobs = {}

    def get_or_create_buf(self, name, buf_size, buf_dict, gpu_buf=True):
        if name in buf_dict and buf_dict[name][0].shape[0] >= buf_size:
            return buf_dict[name]
        buf = torch.zeros(buf_size, self.embed_dim, dtype=self.embed.dtype).pin_memory()
        if gpu_buf:
            out = torch.zeros(buf_size, self.embed_dim, dtype=self.embed.dtype).cuda(self.device)
        else:
            out = None
        buf_dict[name] = (buf, out)
        return (buf, out)

    def read(self, indices, name=None):
        if indices is None:
            t = self.embed.to(self.device)
            t.job_handle = self.dummy_job
            return t
        if not self.embed.is_cuda:
            if name is not None and indices.numel() != self.embed.shape[0]:  # TODO: make it more explicit
                return self.async_read(indices, name)
            for key in self.last_write_jobs:
                self.last_write_jobs[key].sync()
        indices = indices.view(-1)
        submat = self.embed[indices].to(self.device)
        submat.job_handle = self.dummy_job
        return submat

    def async_read(self, indices, name):
        assert not indices.is_cuda
        indices = indices.view(-1)
        buf, out = self.get_or_create_buf(name, indices.shape[0], self.read_buf)
        self.read_src_cache[name] = indices
        with torch.cuda.stream(self.stream):
            job_handle = extlib.async_read(self.read_thread_pool,
                                           indices,
                                           self.embed,
                                           buf,
                                           out)
            submat = out[:indices.shape[0]]
            submat.job_handle = job_handle
        return submat


class EmbeddingRW(EmbeddingReadOnly):
    def __init__(self, embed, gpu_id=-1):
        super(EmbeddingRW, self).__init__(embed, gpu_id)
        self.write_thread_pool = self.read_thread_pool
        self.write_buf = {}        
        self.write_src_cache = {}

    def write(self, indices, values, name=None, additive=False):
        if self.embed.is_cuda:
            indices = indices.to(self.embed.device)
        elif name is not None:
            return self.async_write(indices, values, name, additive)
        if additive:
            self.embed.index_add_(0, indices, values.to(self.embed.device))
        else:
            self.embed.index_copy_(0, indices, values.to(self.embed.device))
        return self.dummy_job

    def async_write(self, indices, values, name, additive=False):
        src_event = extlib.CudaEvent(self.gpu_id)
        src_event.record()
        indices = indices.view(-1)
        assert indices.is_cuda == False and values.is_cuda
        assert values.shape[0] == indices.shape[0] and values.shape[1] == self.embed_dim
        if name in self.last_write_jobs:
            self.last_write_jobs[name].sync()
        buf, _ = self.get_or_create_buf(name, indices.shape[0], self.write_buf, gpu_buf=False)
        if buf.shape[0] > indices.shape[0]:
            buf = buf[:indices.shape[0]]
        with torch.cuda.stream(self.stream):
            job_handle = extlib.async_write(self.write_thread_pool, 
                                            indices,
                                            self.embed,
                                            buf,
                                            values,
                                            src_event,
                                            additive)
        self.write_src_cache[name] = (src_event, indices, values)
        self.last_write_jobs[name] = job_handle
        return job_handle
 
    def add(self, indices, values, name="default"):
        return self.write(indices, values, name, additive=True)