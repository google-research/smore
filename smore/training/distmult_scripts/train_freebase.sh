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

data_name=Freebase
data_folder=$HOME/data/knowledge_graphs/$data_name
eval_path=$data_folder/eval-miss-100-neg-1000

export CUDA_VISIBLE_DEVICES=4,5,6,7

python ../main_train.py --do_train --do_test --gpus '0.1.2.3' \
 --data_path $data_folder --eval_path $eval_path \
 -n 1024 -b 1024 -d 400 -g 2 \
 -a 0.5 -adv \
 -lr 0.00005 --max_steps 1000001 --geo distmult --valid_steps 20000 \
 -distmultm '(Mean,True,norm)' --tasks '1p.2p.3p.2i.3i.ip.pi.2u.up' --training_tasks '1p.2p.3p.2i.3i' \
 --lr_schedule none \
 --filter_test \
 --sampler_type sqrt-2 \
 --share_negative \
 --save_checkpoint_steps 1000000 \
 --share_optim_stats \
 --cpu_num 6 \
 --online_sample --prefix '../logs' --online_sample_mode '(50000,0,u,u,0)' \
 --train_online_mode '(single,3000,e,True,before)' --optim_mode '(aggr,adam,cpu,True,5)' --online_weighted_structure_prob '(1,1,1,1,1,1,1)' --print_on_screen \
 --port 29501
