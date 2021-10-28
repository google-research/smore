#!/bin/bash

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

#dbname=ogbl-wikikg2
#dbname=FB15k-237-betae
dbname=Freebase

max_num_answer=0
max_num_missing_answer=100
neg_samples=1000
search_bandwidth=0
max_intermediate_answers=0

num_gen=10000

tasks=2u-DNF.up-DNF

data_path=$HOME/data/knowledge_graphs/$dbname

save_path=$data_path/eval-miss-$max_num_missing_answer-neg-$neg_samples

if [ ! -e $save_path ];
then
    mkdir -p $save_path
fi

python test_sampler.py \
    --data_path $data_path \
    --save_path $save_path \
    --max_num_answer $max_num_answer \
    --num_queries $num_gen \
    --tasks $tasks \
    --neg_samples $neg_samples \
    --search_bandwidth $search_bandwidth \
    --max_num_missing_answer $max_num_missing_answer \
    --max_intermediate_answers $max_intermediate_answers \
    --cpu_num 16 \
    --do_valid \
    --do_test \
    --do_merge
