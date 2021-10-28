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
import itertools
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple


class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class Normalizer():
    def __call__(self, embedding):
        return F.normalize(embedding, p=2, dim=-1)

class Normalizer_with_const():
    def __init__(self, const):
        self.const = const
    def __call__(self, embedding):
        return F.normalize(embedding, p=2, dim=-1) * self.const

def Identity(x):
    return x
