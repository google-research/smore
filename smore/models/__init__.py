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

from .box import BoxReasoning
from .vec import VecReasoning
from .beta import BetaReasoning
from .rotate import RotateReasoning
from .complex import ComplexReasoning
from .distmult import DistmultReasoning
from smore.models import model_list
from smore.common.util import eval_tuple


def build_model(args, nentity, nrelation, query_name_dict):
    has_neg = False
    tasks = args.tasks.split('.')
    for task in tasks:
        if 'n' in task:
            has_neg = True
            break

    if args.evaluate_union == 'DM':
        assert args.geo == 'beta'
    if args.geo == 'box':
        assert not has_neg, "Q2B cnanot handle queries with negation"
        model = BoxReasoning(nentity=nentity,
                             nrelation=nrelation,
                             hidden_dim=args.hidden_dim,
                             gamma=args.gamma,
                             use_cuda = args.cuda,
                             box_mode=eval_tuple(args.box_mode),
                             batch_size = args.batch_size,
                             test_batch_size=args.test_batch_size,
                             sparse_embeddings=args.sparse_embeddings,
                             sparse_device=args.sparse_device,
                             query_name_dict = query_name_dict,
                             optim_mode = args.optim_mode,
                             logit_impl=args.logit_impl)
    elif args.geo == 'rotate':
        assert not has_neg, "Rotate cnanot handle queries with negation"
        model = RotateReasoning(nentity=nentity,
                             nrelation=nrelation,
                             hidden_dim=args.hidden_dim,
                             gamma=args.gamma,
                             use_cuda = args.cuda,
                             rotate_mode=eval_tuple(args.rotate_mode),
                             batch_size = args.batch_size,
                             test_batch_size=args.test_batch_size,
                             sparse_embeddings=args.sparse_embeddings,
                             sparse_device=args.sparse_device,
                             query_name_dict = query_name_dict,
                             optim_mode = args.optim_mode,
                             logit_impl=args.logit_impl)
    elif args.geo == 'complex':
        assert not has_neg, "Rotate cnanot handle queries with negation"
        model = ComplexReasoning(nentity=nentity,
                             nrelation=nrelation,
                             hidden_dim=args.hidden_dim,
                             gamma=args.gamma,
                             use_cuda = args.cuda,
                             complex_mode=eval_tuple(args.complex_mode),
                             batch_size = args.batch_size,
                             test_batch_size=args.test_batch_size,
                             sparse_embeddings=args.sparse_embeddings,
                             sparse_device=args.sparse_device,
                             query_name_dict = query_name_dict,
                             optim_mode = args.optim_mode,
                             logit_impl=args.logit_impl)
    elif args.geo == 'distmult':
        assert not has_neg, "Rotate cnanot handle queries with negation"
        model = DistmultReasoning(nentity=nentity,
                             nrelation=nrelation,
                             hidden_dim=args.hidden_dim,
                             gamma=args.gamma,
                             use_cuda = args.cuda,
                             distmult_mode=eval_tuple(args.distmult_mode),
                             batch_size = args.batch_size,
                             test_batch_size=args.test_batch_size,
                             sparse_embeddings=args.sparse_embeddings,
                             sparse_device=args.sparse_device,
                             query_name_dict = query_name_dict,
                             optim_mode = args.optim_mode,
                             logit_impl=args.logit_impl)
    elif args.geo == 'beta':
        model = BetaReasoning(nentity=nentity,
                             nrelation=nrelation,
                             hidden_dim=args.hidden_dim,
                             gamma=args.gamma,
                             use_cuda = args.cuda,
                             beta_mode=eval_tuple(args.beta_mode),
                             batch_size = args.batch_size,
                             test_batch_size=args.test_batch_size,
                             sparse_embeddings=args.sparse_embeddings,
                             sparse_device=args.sparse_device,
                             query_name_dict = query_name_dict,
                             optim_mode = args.optim_mode,
                             logit_impl=args.logit_impl)
    elif args.geo == 'vec':
        assert not has_neg, "GQE cnanot handle queries with negation"
        model = VecReasoning(nentity=nentity,
                             nrelation=nrelation,
                             hidden_dim=args.hidden_dim,
                             gamma=args.gamma,
                             use_cuda = args.cuda,
                             model_config=eval_tuple(args.vec_mode),
                             batch_size = args.batch_size,
                             test_batch_size=args.test_batch_size,
                             sparse_embeddings=args.sparse_embeddings,
                             sparse_device=args.sparse_device,
                             query_name_dict = query_name_dict,
                             optim_mode = args.optim_mode,
                             logit_impl=args.logit_impl)
    else:
        mod_class = getattr(model_list, args.geo)
        model = mod_class(nentity=nentity,
                          nrelation=nrelation,
                          hidden_dim=args.hidden_dim,
                          gamma=args.gamma,
                          use_cuda = args.cuda,
                          batch_size = args.batch_size,
                          test_batch_size=args.test_batch_size,
                          sparse_embeddings=args.sparse_embeddings,
                          sparse_device=args.sparse_device,
                          query_name_dict = query_name_dict,
                          optim_mode = args.optim_mode,
                          model_config=eval_tuple(args.model_config),
                          logit_impl=args.logit_impl)
    return model
