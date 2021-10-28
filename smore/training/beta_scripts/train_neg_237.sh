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

data_name=FB15k-237-betae
data_folder=$HOME/data/knowledge_graphs/$data_name
eval_path=$data_folder/eval-original

export CUDA_VISIBLE_DEVICES=0,1,2,3

#beta
python ../main_train.py --do_train --do_test --gpus '0.1.2.3' \
 --data_path $data_folder --eval_path $eval_path \
 -n 1024 -b 512 -d 400 -g 60 \
 -a 0.5 -adv \
 -lr 0.0001 --max_steps 450001 --geo beta --valid_steps 15000 \
 -betam "(1600,2)" --tasks '1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up' --training_tasks '1p.2p.3p.2i.3i.2in.3in.inp.pin.pni' \
 --save_checkpoint_steps 150000 \
 --share_negative \
 --lr_schedule none \
 --sampler_type naive \
 --filter_test \
 --share_optim_stats \
 --port 29500 \
 --online_sample --prefix '../logs' --online_sample_mode '(500,0,w,wstruct,120)' \
 --train_online_mode '(single,3000,e,True,before)' --optim_mode '(aggr,adam,cpu,False,5)' --online_weighted_structure_prob '(20,20,20,10,10,1,1,1,1,1)' --print_on_screen \
 $@
