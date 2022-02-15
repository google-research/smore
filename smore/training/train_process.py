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
import json
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader
import random
import math
import collections
import itertools
import time
from tqdm import tqdm
import os
import pdb
from torch.multiprocessing import Queue, Pipe
import torch.multiprocessing as mp
from _thread import start_new_thread
import traceback
from functools import wraps
from tensorboardX import SummaryWriter, GlobalSummaryWriter
from torch_scatter import scatter

from smore.cpp_sampler.online_sampler import OnlineSampler
from smore.common.util import name_query_dict, query_name_dict, thread_wrapped_func, log_metrics, tuple2filterlist, eval_tuple, flatten_query
from smore.common.embedding.embed_optimizer import get_optim_class
from smore.evaluation.dataloader import TestDataset

eps = 1e-8

def test_step_1p(model, args, train_sampler, test_dataloader, result_buffer, train_step, device, phase):
    model.eval()
    rank = dist.get_rank()
    logs = collections.defaultdict(collections.Counter)
    with torch.no_grad():
        for list_heads, rels, list_tails, batch_neg_samples in tqdm(test_dataloader, disable=(not args.print_on_screen) or rank):
            q_structs = [name_query_dict['1p'], name_query_dict['-1p']]
            ht = [(list_heads, list_tails), (list_tails, list_heads)]

            for (entities, hard_answers), query_structure in zip(ht, q_structs):
                assert batch_neg_samples is None and args.neg_sample_size_eval > 0
                n1 = train_sampler.sample_entities(True, args.neg_sample_size_eval)
                n2 = train_sampler.sample_entities(False, args.neg_sample_size_eval)
                negative_samples = torch.cat((n1, n2), dim=-1).unsqueeze(0)

                queries = torch.cat((entities.view(-1, 1), rels.view(-1, 1)), dim=-1)

                positive_logits, negative_logits, _ = model(hard_answers, negative_samples, query_structure, queries, device=device)
                joint_logits = torch.cat((positive_logits.view(-1, 1), negative_logits), dim=-1)
                argsort = torch.argsort(joint_logits, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                ranking = ranking.scatter_(1,
                                            argsort,
                                            torch.arange(joint_logits.shape[1]).to(torch.float).repeat(argsort.shape[0], 1).to(device))
                ans_ranking = ranking[:, 0].cpu().numpy()
                for idx in range(joint_logits.shape[0]):
                    cur_rank = ans_ranking[idx] + 1
                    mrr = 1.0 / cur_rank
                    h1 = int(cur_rank <= 1)
                    h3 = int(cur_rank <= 3)
                    h10 = int(cur_rank <= 10)
                    h1m = cur_rank == 1
                    logs[query_structure]['MRR'] += mrr
                    logs[query_structure]['HITS1'] += h1
                    logs[query_structure]['HITS3'] += h3
                    logs[query_structure]['HITS10'] += h10
                    logs[query_structure]['HITS1max'] += h1m
                    logs[query_structure]['num_hard_answer'] += 1
                    logs[query_structure]['num_queries'] += 1

    result_buffer.put((logs, train_step))


def test_step_mp(model, args, train_sampler, test_dataloader, result_buffer, train_step, device, phase):
    if args.eval_link_pred and args.eval_batch_size > 1:
        return test_step_1p(model, args, train_sampler, test_dataloader, result_buffer, train_step, device, phase)
    model.eval()
    rank = dist.get_rank()
    step = 0
    total_steps = len(test_dataloader)
    logs = collections.defaultdict(collections.Counter)
    all_embed = None
    negative_sample_bias = None

    with torch.no_grad():
        for negative_sample, queries, queries_unflatten, query_structures, easy_answers, hard_answers in tqdm(test_dataloader, disable=(not args.print_on_screen) or rank):
            assert len(queries_unflatten) == 1 # batch size == 1
            query_structure = query_structures[0]
            query_mat = torch.LongTensor(queries)

            test_all = negative_sample is None
            if negative_sample is None:  # test against all entities
                if all_embed is None:
                    all_embed = model.entity_embedding(None)
                negative_sample = test_dataloader.dataset.all_entity_idx
                if negative_sample_bias is None:
                    negative_sample_bias = torch.zeros([1, args.nentity]).to(device)
            else:
                negative_sample_bias = torch.where(negative_sample == -1, -1e6, 0.).to(device)
                negative_sample = torch.where(negative_sample == -1, 0, negative_sample)

            _, negative_logit, _= model(None, negative_sample, query_structure, query_mat, device=device, all_neg=all_embed)
            negative_logit += negative_sample_bias
            argsort = torch.argsort(negative_logit, dim=1, descending=True)
            ranking = argsort.clone().to(torch.float)
            ranking = ranking.scatter_(1,
                                       argsort,
                                       torch.arange(negative_logit.shape[1]).to(torch.float).repeat(argsort.shape[0], 1).to(device))

            for idx, (query, query_structure, easy_answer, hard_answer) in enumerate(zip(queries_unflatten,
                                                                                         query_structures,
                                                                                         easy_answers,
                                                                                         hard_answers)):
                num_hard = len(hard_answer)
                if (num_hard == 1 and list(hard_answer)[0] == -1): # the ground truth is not in the candidate set, mrr should be zero
                    mrr = h1 = h3 = h10 = h1m = 0
                else:
                    if test_all:
                        num_easy = len(easy_answer)
                        assert len(hard_answer.intersection(easy_answer)) == 0
                        cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    else:
                        num_easy = 0
                        cur_ranking = ranking[idx, :num_hard]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(device)
                    cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                    cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers
                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()
                    h1m = ((cur_ranking[0] == 1).to(torch.float)).item()

                logs[query_structure]['MRR'] += mrr
                logs[query_structure]['HITS1'] += h1
                logs[query_structure]['HITS3'] += h3
                logs[query_structure]['HITS10'] += h10
                logs[query_structure]['HITS1max'] += h1m
                logs[query_structure]['num_hard_answer'] += 1
                logs[query_structure]['num_queries'] += 1

            if step % args.test_log_steps == 0:
                logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

            step += 1

    result_buffer.put((logs, train_step))


def train_step_mp(model, dense_optimizers, embedding_optimizers, train_iterator, args, step, lr, device, world_size):
    model.train()
    for optimizer in dense_optimizers:
        optimizer.zero_grad()
    for embedding_name in embedding_optimizers:
        embedding_optimizers[embedding_name].zero_grad()

    t1 = time.time()
    positive_sample, negative_sample, is_negative_mat, subsampling_weight, batch_queries, query_structures = next(train_iterator)

    if "cuda" in device:
        subsampling_weight = subsampling_weight.cuda(device)
        if is_negative_mat is not None:
            is_negative_mat = is_negative_mat.cuda(device)

    t2 = time.time()
    positive_logit, negative_logit, reg_loss = model(positive_sample, 
                                                     negative_sample,
                                                     query_structures[0],
                                                     batch_queries,
                                                     device=device,
                                                     reg_coeff=args.reg_coeff)
    t3 = time.time()
    if is_negative_mat is not None:
        negative_logit = negative_logit * is_negative_mat
    if args.negative_adversarial_sampling:
        negative_score = (F.softmax(negative_logit * args.adversarial_temperature, dim=1).detach() \
                                        * F.logsigmoid(-negative_logit)).sum(dim=1)
    else:
        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
    positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
    if eval_tuple(args.online_sample_mode)[2] == 'w':
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()
    else:
        assert eval_tuple(args.online_sample_mode)[2] == 'u'
        positive_sample_loss = -torch.mean(positive_score)
        negative_sample_loss = -torch.mean(negative_score)

    loss = (positive_sample_loss + negative_sample_loss + reg_loss*args.reg_coeff) / 2 / world_size
    if args.reg_coeff != 0.:
        pass
    loss.backward()
    # print ("after loss backward")
    t4 = time.time()
    for embedding_name in embedding_optimizers:
        embedding_optimizers[embedding_name].apply_grad(lr)
    for _, param in model.named_dense_parameters():
        if param.grad is None:  # should be None for everyone
            continue
            # param.grad = param.data.new(param.data.shape).zero_()
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
    for optimizer in dense_optimizers:
        optimizer.step()

    t5 = time.time()
    model.t_read += t2 - t1
    model.t_fwd += t3 - t2
    model.t_loss += t4 - t3
    model.t_opt += t5 - t4
    log = {
        'positive_sample_loss': positive_sample_loss.item(),
        'negative_sample_loss': negative_sample_loss.item(),
        'loss': loss.item(),
    }
    
    log['msg'] = "step: {}, t_read: {:.5f}, t_fwd: {:.5f}, t_loss: {:.5f}, t_opt: {:.5f}".format(
                step,
                model.t_read/(step+1),
                model.t_fwd/(step+1),
                model.t_loss/(step+1),
                model.t_opt/(step+1))

    if args.reg_coeff != 0 and reg_loss > 0:
        log['reg_loss'] = reg_loss.item()
    return log


def save_model(model, optimizers, embedding_optimizers, save_variable_list, args):
    for name in embedding_optimizers:
        assert embedding_optimizers[name].share_optim_stats
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)
    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
    }, os.path.join(args.save_path, 'checkpoint'))


def train_func(args, kg_mem, opt_stats, model, eval_dict, training_tasks, ro_feat, gpu_id):
    kg = kg_mem.create_kg()
    assert kg.num_ent == args.nentity and kg.num_rel == args.nrelation
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    training_logs = []
    init_step = opt_stats['init_step']
    warm_up_steps = opt_stats['warm_up_steps']

    data_loaders = {}
    for key in eval_dict:
        data_loaders[key] = DataLoader(
            eval_dict[key].data,
            batch_size=args.eval_batch_size,
            num_workers=1,
            collate_fn=eval_dict[key].data.collate_fn
        )

    if gpu_id != -1:
        device = "cuda:{}".format(gpu_id)
    else:
        device = 'cpu'
    if "cuda" in device:
        model.to_device(device)

    embedding_optimizers = {}
    EmbedOpt = get_optim_class(args)
    for name, embed in model.named_sparse_embeddings():
        embedding_optimizers[name] = EmbedOpt(args, embed, gpu_id)
    if ro_feat is not None:
        model.attach_feature("entity", ro_feat["entity"], gpu_id, is_sparse=True)
        model.attach_feature("relation", ro_feat["relation"], gpu_id, is_sparse=False)

    old_sampler = None
    init_sampler_type = 'nosearch' if args.sampler_type.startswith('mix') else args.sampler_type
    if args.sampler_type.startswith('mix'):
        pct = float(args.sampler_type.split('_')[-1])
        switch_step = int(args.max_steps * pct)
    else:
        switch_step = args.max_steps + 1

    train_sampler = OnlineSampler(kg, training_tasks, args.negative_sample_size, eval_tuple(args.online_sample_mode), args.normalized_structure_prob, 
                                  sampler_type=init_sampler_type,
                                  share_negative=args.share_negative,
                                  same_in_batch=args.train_dataset_mode=='single',
                                  num_threads=args.cpu_num)

    if not args.do_train:
        dist.barrier(device_ids=[rank])
        for phase in eval_dict:
            logging.info('[{}] Evaluating on {} Dataset...'.format(os.getpid(), phase))
            d = eval_dict[phase]
            test_step_mp(model, args, train_sampler, data_loaders[phase], d.buffer, init_step, device, phase)
            eval_dict[phase].buffer.put((None, None))
        return

    current_learning_rate = opt_stats['current_learning_rate']
    dense_lr = args.dense_learning_rate or current_learning_rate
    dense_param_names, dense_params = zip(*list(model.named_dense_parameters(exclude_embedding=True)))
    dense_optimizer = torch.optim.Adam(dense_params, lr=dense_lr)
    logging.info("[{}], dense params: {}".format(os.getpid(), dense_param_names))
    optimizers = [dense_optimizer]

    dense_named_embeds = list(model.named_dense_embedding_params())
    if len(dense_named_embeds):
        dense_embed_names, dense_embeds = zip(*dense_named_embeds)
        logging.info("[{}], dense embed params: {}".format(os.getpid(), dense_embed_names))

        optim_def = eval_tuple(args.optim_mode)[1]
        if optim_def == 'adam':
            dense_embed_optimizer = torch.optim.Adam(dense_embeds, lr=current_learning_rate)
        elif optim_def == 'adagrad':
            dense_embed_optimizer = torch.optim.Adagrad(dense_embeds, lr=current_learning_rate)
        else:
            raise NotImplementedError
        optimizers.append(dense_embed_optimizer)

    if opt_stats['optimizer_stats'] is not None:
        list_stats_dict = opt_stats['optimizer_stats']
        for i in range(len(list_stats_dict)):
            optimizers[i].load_state_dict(list_stats_dict[i])

    train_sampler.set_seed(1)
    if rank == 0:
        train_sampler.print_queries()
    train_iterator = train_sampler.batch_generator(args.batch_size)
    dist.barrier(device_ids=[rank])
    valid_stages = list(range(init_step, args.max_steps, args.valid_steps)) + [args.max_steps]
    for stage in range(len(valid_stages) - 1):
        step_start = valid_stages[stage]
        step_end = valid_stages[stage + 1]
        pbar = tqdm(range(step_start, step_end), disable=(not args.print_on_screen) or rank)
        if step_start >= switch_step and old_sampler is None:
            old_sampler = (train_sampler, train_iterator)
            train_sampler = OnlineSampler(kg, training_tasks, args.negative_sample_size, eval_tuple(args.online_sample_mode), args.normalized_structure_prob, 
                                  sampler_type='sqrt',
                                  share_negative=args.share_negative,
                                  same_in_batch=args.train_dataset_mode=='single',
                                  num_threads=args.cpu_num)
            train_sampler.set_seed(1)
            train_iterator = train_sampler.batch_generator(args.batch_size)
            print('sampler switched to sqrt')
        for step in pbar:
            log = train_step_mp(model, optimizers, embedding_optimizers, train_iterator, args, step, current_learning_rate, device=device, world_size=world_size)
            log_msg = log['msg']
            pbar.set_description(log_msg)
            del log['msg']
            training_logs.append(log)

            if args.lr_schedule == 'step':
                if step >= warm_up_steps:            
                    current_learning_rate = current_learning_rate / 5
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                    for param_group in dense_embed_optimizer.param_groups:
                        param_group['lr'] = current_learning_rate
                    warm_up_steps = warm_up_steps * 1.5

            if step and step % args.save_checkpoint_steps == 0:
                if rank == 0:
                    save_variable_list = {
                            'step': step, 
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps
                    }
                    save_model(model, optimizers, embedding_optimizers, save_variable_list, args)
                dist.barrier(device_ids=[rank])

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                log_metrics('[{}] Training average'.format(os.getpid()), step, metrics)
                training_logs = []

        dist.barrier(device_ids=[rank])
        for phase in eval_dict:
            logging.info('[{}] Evaluating on {} Dataset...'.format(os.getpid(), phase))
            d = eval_dict[phase]
            test_step_mp(model, args, train_sampler, data_loaders[phase], d.buffer, step, device, phase)
        dist.barrier(device_ids=[rank])

    logging.info("Finish training! Now cleaning up")
    dist.barrier(device_ids=[rank])
    for phase in eval_dict:
        eval_dict[phase].buffer.put((None, None))


@thread_wrapped_func
def train_mp(args, kg_mem, opt_stats, model, eval_dict, training_tasks, ro_feat, gpu_id):
    if len(args.gpus) > 1:
        torch.set_num_threads(1)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group('nccl', rank=gpu_id, world_size=len(args.gpus))
    train_func(args, kg_mem, opt_stats, model, eval_dict, training_tasks, ro_feat, gpu_id)


def async_aggr(args, result_buffer, writer_buffer, mode):
    all_result_dicts = collections.defaultdict(list)
    num_received = 0
    num_finished = 0
    num_workers = len(args.gpus)

    while True:
        if num_finished == num_workers and num_received == 0:  # no more write jobs
            writer_buffer.put((None, None, -1))
            return
        result_dict, step = result_buffer.get()
        if result_dict is None:
            num_finished += 1
            assert num_finished <= num_workers
            continue
        for query_structure in result_dict:
            all_result_dicts[query_structure].extend([result_dict[query_structure]])
        num_received += 1
        if num_received == num_workers:
            metrics = collections.defaultdict(lambda: collections.defaultdict(int))
            for query_structure in all_result_dicts:
                num_queries = sum([result_dict['num_queries'] for result_dict in all_result_dicts[query_structure]])
                for metric in all_result_dicts[query_structure][0].keys():
                    if metric in ['num_hard_answer', 'num_queries']:
                        continue
                    metrics[query_structure][metric] = sum([result_dict[metric] for result_dict in all_result_dicts[query_structure]]) / num_queries
                metrics[query_structure]['num_queries'] = num_queries
            average_metrics = collections.defaultdict(float)
            all_metrics = collections.defaultdict(float)

            num_query_structures = 0
            num_queries = 0
            for query_structure in metrics:
                qname = query_name_dict[query_structure] if query_structure in query_name_dict else str(query_structure)
                log_metrics(mode+" "+qname, step, metrics[query_structure])
                for metric in metrics[query_structure]:
                    all_metrics["_".join([qname, metric])] = metrics[query_structure][metric]
                    if metric != 'num_queries':
                        average_metrics["_".join([metric, "qs"])] += metrics[query_structure][metric]
                        average_metrics["_".join([metric, "q"])] += metrics[query_structure][metric] * metrics[query_structure]['num_queries']
                num_queries += metrics[query_structure]['num_queries']
                num_query_structures += 1

            for metric in average_metrics:
                if '_qs' in metric:
                    average_metrics[metric] /= num_query_structures
                else:
                    average_metrics[metric] /= num_queries
                all_metrics["_".join(["average", metric])] = average_metrics[metric]
            for metric in all_metrics:
                print(metric, all_metrics[metric])
            log_metrics('%s average'%mode, step, average_metrics)

            writer_buffer.put((dict(metrics), dict(average_metrics), step))
            all_result_dicts = collections.defaultdict(list)
            num_received = 0
