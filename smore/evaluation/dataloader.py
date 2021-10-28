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


from torch.utils.data import Dataset
import numpy as np
import torch
from smore.common.util import list2tuple, tuple2list, flatten
from collections import defaultdict
from smore.common.util import fill_query, achieve_answer_with_constraints, list2tuple, tuple2list, flatten, flatten_list, sample_negative_bidirectional


class TestDataset(Dataset):
    def __init__(self, nentity, nrelation):
        self.nentity = nentity
        self.nrelation = nrelation
        self.all_entity_idx = torch.LongTensor(range(self.nentity)).unsqueeze(0)

    @staticmethod
    def collate_fn(data):
        if data[0][0] is None:
            negative_sample = None
        else:
            negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]
        query_unflatten = [_[2] for _ in data]
        query_structure = [_[3] for _ in data]
        easy_answers = [_[4] for _ in data]
        hard_answers = [_[5] for _ in data]
        return negative_sample, query, query_unflatten, query_structure, easy_answers, hard_answers


class MultihopTestDataset(TestDataset):
    def __init__(self, data, nentity, nrelation):
        """
        Args:
            data: a list of (query_structure, query, easy_answers, hard_answers, negatives) tuples
        """
        super(MultihopTestDataset, self).__init__(nentity, nrelation)
        self.data = data
        self.test_all = len(data[0][4]) == 0

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        query = self.data[idx][1]
        query_structure = self.data[idx][0]
        easy_answers = self.data[idx][2]
        hard_answers = self.data[idx][3]
        if self.test_all:
            neg_samples = None
        else:
            neg_samples = torch.LongTensor(list(hard_answers) + list(self.data[idx][4]))
        return neg_samples, flatten(query), query, query_structure, easy_answers, hard_answers

    def subset(self, pos, num):
        data = self.data[pos : pos + num]
        return MultihopTestDataset(data, self.nentity, self.nrelation)


class Test1pBatchDataset(TestDataset):
    def __init__(self, data, nentity, nrelation):
        """
        Args:
            data: a dict of {'head', 'relation', 'tail', 'tail_neg' (optional)}
        """
        super(Test1pBatchDataset, self).__init__(nentity, nrelation)
        self.data = data

    def __len__(self):
        return self.data['head'].shape[0]

    def __getitem__(self, idx):
        head = self.data['head'][idx]
        rel = self.data['relation'][idx]
        tail = self.data['tail'][idx]
        if 'tail_neg' in self.data:
            neg_samples = torch.LongTensor([tail] + self.data['tail_neg'][idx].tolist())
        else:
            neg_samples = None
        return neg_samples, head, rel, tail

    def subset(self, pos, num):
        d = {}
        for key in self.data:
            if len(self.data[key]):
                d[key] = self.data[key][pos : pos + num]
            else:
                d[key] = set()
        return Test1pBatchDataset(d, self.nentity, self.nrelation)

    @staticmethod
    def collate_fn(list_samples):
        list_negs, list_head, list_rel, list_tail = zip(*list_samples)
        if list_negs[0] is not None:
            neg_samples = torch.stack(list_negs)
        else:
            neg_samples = None
        list_head = torch.LongTensor(list_head)
        list_rel = torch.LongTensor(list_rel)
        query_structure = ('e', ('r',))
        query = torch.cat((list_head, list_rel), dim=-1)
        list_tail = torch.LongTensor(list_tail)
        return list_head, list_rel, list_tail, neg_samples


class Test1pDataset(TestDataset):
    def __init__(self, data, nentity, nrelation):
        """
        Args:
            data: a dict of {'head', 'relation', 'tail', 'tail_neg'}
        """
        super(Test1pDataset, self).__init__(nentity, nrelation)
        self.data = data
        self.test_all = len(data['tail_neg']) == 0

    def __len__(self):
        return self.data['head'].shape[0]

    def __getitem__(self, idx):        
        query_structure = ('e', ('r',))
        head = self.data['head'][idx]
        rel = self.data['relation'][idx]
        tail = self.data['tail'][idx]
        query = (head, (rel,))

        easy_answers = set()

        if self.test_all:
            neg_samples = None
        else:
            neg_samples = torch.LongTensor([tail] + self.data['tail_neg'][idx].tolist())
        return neg_samples, flatten(query), query, query_structure, easy_answers, set([tail])

    def subset(self, pos, num):
        d = {}
        for key in self.data:
            if len(self.data[key]):
                d[key] = self.data[key][pos : pos + num]
            else:
                d[key] = set()
        return Test1pDataset(d, self.nentity, self.nrelation)
