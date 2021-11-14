# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

data_name=wikikg90m-v2
data_folder=/home/ssm-user/hyren/license-srkg/mkg/training/large-logs/dummy-wikikg/wikikg90m-v2 # set wikikgv2 save_dir here, note the folder should contain "wikikg90m" substring
eval_path=$data_folder/eval-original
feature_folder=$data_folder/processed

export CUDA_VISIBLE_DEVICES=4

python ../main_train.py --do_valid --gpus '0' \
 --data_path $data_folder --eval_path $eval_path \
 -n 10 -b 32 -d 25 -g 8.0 \
 -a 1.0 -adv \
 -lr 0.1 --max_steps 10 --geo ComplexFeatured \
 --model_config '(feat-concat-500,Mean,True)' --valid_steps 5 \
 --tasks '1p' --training_tasks '1p' \
 --share_negative \
 --logit_impl custom \
 --sampler_type naive \
 --save_checkpoint_steps 5 \
 --eval_link_pred \
 --share_optim_stats \
 --dense_learning_rate 1e-4 \
 --cpu_num 4 \
 --online_sample --prefix '/home/ssm-user/hyren/license-srkg/mkg/training/large-logs' --online_sample_mode '(0,0,u,u,0,True,False)' --reg_coeff 1e-9 \
 --train_online_mode '(single,3000,e,True,before)' --optim_mode '(aggr,adagrad,cpu,False,5)' --online_weighted_structure_prob '(1,1,1,1,1,1,1)' --print_on_screen \
 --checkpoint_path="/home/ssm-user/hyren/license-srkg/mkg/training/large-logs/wikikg90m-v2/1p-1p/ComplexFeatured/g-8.0-mode-(feat-concat-500,Mean,True)-adv-1.0-reg-1e-09-ngpu-0.1.2.3-os-(0,0,u,u,0,True,False)-dataset-(single,3000,e,True,before)-opt-(aggr,adagrad,cpu,False,5)-sharen-naive-lr_none/2021.11.14-09:02:39" \
 --feature_folder=$feature_folder \
 $@


#  --checkpoint_path="/home/ssm-user/hyren/license-srkg/mkg/training/large-logs/wikikg90m-v2/1p-1p/ComplexFeatured/g-8.0-mode-(feat-concat-768,Mean,True)-adv-1.0-reg-1e-09-ngpu-0.1.2.3-os-(0,0,u,u,0,True,False)-dataset-(single,3000,e,True,before)-opt-(aggr,adagrad,cpu,False,5)-sharen-naive-lr_none/2021.09.18-23:58:09" \