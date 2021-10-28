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

import argparse

query_name_dict = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    ('e', ('r', 'r', 'r', 'r')): '4p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '4i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    ((('e', ('r',)), ('e', ('r',)), ('e', ('r',))), ('r',)): '3ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM',
                    ('<', ('e',('r',))): '-1p',
                    ('<', ('e',('r', 'r'))): '-2p',
                    ('<', ('e',('r', 'r', 'r'))): '-3p',
                }

name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")

    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('--eval_path', type=str, default=None, help="KG eval data path")
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=500, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=12.0, type=float, help="margin in the loss")
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int, help="batch size of queries")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('--dense_learning_rate', default=None, type=float)

    parser.add_argument('-cpu', '--cpu_num', default=6, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default=None, type=str, help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=100000, type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None, type=int, help="no need to set manually, will configure automatically")
    
    parser.add_argument('--save_checkpoint_steps', default=50000, type=int, help="save checkpoints every xx steps")
    parser.add_argument('--valid_steps', default=10000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    parser.add_argument('--geo', default=None, type=str, choices=['vec', 'box', 'beta', 'rotate', 'distmult', 'complex',
                                                                  'VecFeatured', 'VecReasoning', 'ComplexReasoning', 'ComplexFeatured'], help='the reasoning model, vec for GQE, box for Query2box, beta for BetaE')
    parser.add_argument('--model_config', default=None, type=str, help='model config')

    parser.add_argument('--lr_schedule', default='none', type=str, choices=['none', 'step'], help='learning rate scheduler')
    parser.add_argument('--kg_dtype', default='uint32', type=str, choices=['uint32', 'uint64'], help='data type of kg')

    parser.add_argument('--print_on_screen', action='store_true')
    
    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--training_tasks', default=None, type=str, help="training tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-betam', '--beta_mode', default="(1600,2)", type=str, help='(hidden_dim,num_layer) for BetaE relational projection')
    parser.add_argument('-boxm', '--box_mode', default="(none,0.02)", type=str, help='(offset activation,center_reg) for Query2box, center_reg balances the in_box dist and out_box dist')
    parser.add_argument('-rotatem', '--rotate_mode', default="(Mean,True)", type=str, help='(intersection aggr,nonlinearity) for Rotate')
    parser.add_argument('-complexm', '--complex_mode', default="(Mean,True)", type=str, help='(intersection aggr,nonlinearity) for Rotate')
    parser.add_argument('-vecm', '--vec_mode', default="(l2,)", type=str, help='(dist,) for transE')
    parser.add_argument('-distmultm', '--distmult_mode', default="(Mean,True)", type=str, help='(intersection aggr,nonlinearity) for Rotate')
    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    parser.add_argument('--share_optim_stats', action='store_true', default=False)
    parser.add_argument('--filter_test', action='store_true', default=False)
    parser.add_argument('--online_sample', action='store_true', default=False)
    parser.add_argument('--sampler_type', type=str, default='naive', help="type of sampler, choose from [naive, sqrt, nosearch, mix_0.x]")

    parser.add_argument('--share_negative', action='store_true', default=False)
    parser.add_argument('--online_sample_mode', default="(500,-1,w,wstruct,80,True,True)", type=str, 
                help='(0,0,w/u,wstruct/u,0,True,True) or (relation_bandwidth,max_num_of_intermediate,w/u,wstruct/u,max_num_of_partial_answer,weighted_ans,weighted_neg)')
    parser.add_argument('--online_weighted_structure_prob', default="(70331,141131,438875)", type=str, 
                help='(same,0,w/u,wstruct/u)')
    
    parser.add_argument('--gpus', default='-1', type=str, help="gpus")
    parser.add_argument('--logit_impl', default='native', type=str, help="logit implentation", choices=['native', 'custom'])
    parser.add_argument('--port', default='29500', type=str, help="dist port")
    parser.add_argument('--train_online_mode', default="(single,500,er,False,before)", type=str, 
                help='(mix/single,sync_steps,er/e/r/n trained on cpu,async flag, before/after)')
    parser.add_argument('--optim_mode', default="(fast,adagrad,cpu,True,5)", type=str, 
                help='(fast/aggr,adagrad/rmsprop,cpu/gpu,True/False,queue_size)')

    parser.add_argument('--reg_coeff', default=0., type=float, help="margin in the loss")
    parser.add_argument('--neg_sample_size_eval', type=int, default=0, help='neg sample size for eval; set 0 to eval against all')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size for eval')
    parser.add_argument('--eval_link_pred', action='store_true', default=False)
    parser.add_argument('--feature_folder', default=None, type=str, help="folder to entity and relation features")
    return parser
    # return parser.parse_args(args)
