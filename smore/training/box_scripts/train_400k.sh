#!/bin/bash

# Copyright 2022 SMORE Authors
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

data_name=FB400k-50-multi
data_folder=$HOME/data/knowledge_graphs/$data_name
eval_path=$data_folder/eval-original

export CUDA_VISIBLE_DEVICES=0,1,2,3

python ../main_train.py --do_train --do_test --gpus '0.1.2.3' \
 --data_path $data_folder --eval_path $eval_path \
 -n 1024 -b 512 -d 1600 -g 2 \
 -a 0.5 -adv \
 -lr 0.00005 --max_steps 600001 --geo box --valid_steps 10000 \
 -boxm '(none,0.02)' --tasks '1p.2p.3p.2i.3i.ip.pi' --training_tasks '1p.2p.3p.2i.3i' \
 --online_sample --prefix '../logs' --online_sample_mode '(5000,0,w,u,80)' \
 --lr_schedule step \
 --sampler_type naive \
 --share_negative \
 --share_optim_stats \
 --cpu_num 6 \
 --save_checkpoint_steps 100000 \
 --train_online_mode '(single,3000,e,True,before)' --optim_mode '(aggr,adam,cpu,False,5)' --online_weighted_structure_prob '(1,1,1,1,1)' --print_on_screen \
 --port 29500
