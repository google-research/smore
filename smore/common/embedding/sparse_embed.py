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

from smore.common.embedding.embed_rw import DummyJob

class SparseEmbedding(nn.Module):
    """Sparse embedding"""

    def __init__(self, num_rows, num_cols):
        super(SparseEmbedding, self).__init__()
        self.num_rows = num_rows
        self.embedding = nn.Parameter(torch.zeros(num_rows, num_cols))
        self.fwd_delegate = None
        self.dummy_job = DummyJob()

    @property
    def device(self):
        return self.embedding.device

    @property
    def shape(self):
        return self.embedding.shape

    def init_params(self, lb, ub):
        nn.init.uniform_(
            tensor=self.embedding, 
            a=lb,
            b=ub
        )

    def register_tensor(self, name, tensor_obj):
        self.register_buffer(name, tensor_obj)
        if self.embedding.is_shared():
            tensor_obj.share_memory_()

    def forward(self, indices, name=None):
        if self.num_rows == 0:
            return None
        if self.fwd_delegate is not None:
            return self.fwd_delegate(indices, name)
        else:
            if indices is None:
                mat = self.embedding
            else:
                mat = self.embedding[indices]
            mat.job_handle = self.dummy_job
            return mat
#           return torch.index_select(self.embedding, dim=0, index=indices.to(self.embedding.device))
