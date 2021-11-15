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

data_name=wikikg90m-v2
data_folder=$data_folder # set wikikgv2 save_dir here, note the folder should contain "wikikg90m" substring
eval_path=$data_folder/eval-original

export CUDA_VISIBLE_DEVICES=0,1,2,3

python ../main_train.py --do_train --do_valid --do_test --gpus '0.1.2.3' \
 --data_path $data_folder --eval_path $eval_path \
 -n 1024 -b 512 -d 200 -g 10.0 \
 -a 1.0 -adv \
 -lr 0.1 --max_steps 10000001 --geo vec -vecm '(l2,)' --valid_steps 200000 \
 --tasks '1p' --training_tasks '1p.-1p' \
 --share_negative \
 --sampler_type naive \
 --logit_impl custom \
 --save_checkpoint_steps 1000000 \
 --eval_link_pred \
 --share_optim_stats \
 --cpu_num 4 \
 --online_sample --prefix '../logs' --online_sample_mode '(0,0,u,u,0,True,False)' --reg_coeff 1e-9 \
 --train_online_mode '(single,3000,e,True,before)' --optim_mode '(aggr,adagrad,cpu,False,5)' --online_weighted_structure_prob '(1,1,1,1,1,1,1)' --print_on_screen \
 --port 29520 \
 $@

