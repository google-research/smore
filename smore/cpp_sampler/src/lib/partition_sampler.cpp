// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils.h"
#include "sampler.h"
#include "knowledge_graph.h"
#include "query.h"

template<typename Dtype>
PartitionSampler<Dtype>::PartitionSampler(KG<Dtype>* _kg, py::list _query_trees, py::list _query_prob, bool _share_negative, bool _same_in_batch,
                           bool _weighted_answer_sampling, bool _weighted_negative_sampling,
                           Dtype _negative_sample_size, Dtype _rel_bandwidth, Dtype _max_to_keep, Dtype _max_n_partial_answers,
                           int num_threads, py::list no_search_list) : NoSearchSampler<Dtype>(_kg, _query_trees, _query_prob, _share_negative, _same_in_batch, 
                                                    _weighted_answer_sampling, _weighted_negative_sampling,
                                                    _negative_sample_size, _rel_bandwidth, _max_to_keep,
                                                    _max_n_partial_answers, num_threads, no_search_list) {}

template<typename Dtype>
bool PartitionSampler<Dtype>::sample_actual_query(QueryTree<Dtype>* qt, Dtype answer, bool inverse, std::vector<Dtype>& ent_list)
{
    qt->answer = answer;
    ent_list.push_back(answer);
    if (qt->node_type == QueryNodeType::intersect || qt->node_type == QueryNodeType::union_set)
    {
        qt->hash_code = qt->node_type;
        assert(qt->children.size() > 1u); // it is a non-trivial interesect/join
        for (auto& ch : qt->children)
        {
            assert(ch.first == QueryEdgeType::no_op);
            auto* subtree = ch.second;
            if (!this->sample_actual_query(subtree, answer, inverse, ent_list))
                return false;
            hash_combine(qt->hash_code, subtree->hash_code);
        }

        for (size_t i = 0; i + 1 < qt->children.size(); ++i)  // assume the number of branches is small
        {
            auto& code_first = qt->children[i].second->hash_code;
            for (size_t j = i + 1; j < qt->children.size(); ++j)
                if (code_first == qt->children[j].second->hash_code)
                    return false;
        }
        return true;
    } else {
        if (qt->node_type == QueryNodeType::entity) // we have successfully instantiated the query at this branch
        {
            qt->hash_code = answer;
            return true;
        }
        assert(qt->children.size() == 1u); // it should have a single relation/negation child.
        auto& ch = qt->children[0];
        auto e_type = ch.first;
        assert(e_type != QueryEdgeType::no_op); // doesn't make sense to have no-op here

        Dtype r, prev;
        qt->hash_code = 0;
        if (e_type == QueryEdgeType::relation)
        {
            auto* edge_set = inverse ? this->kg->ent_out : this->kg->ent_in;
            if (!inverse && this->kg->in_degree(answer) == 0)
                return false;
            if (inverse && this->kg->out_degree(answer) == 0)
                return false;
            sample_rand_neighbor(edge_set, answer, r, prev);
            ch.second->parent_r = r;
            qt->hash_code = r;
        } else { // negation
            assert(e_type == QueryEdgeType::negation);
            prev = this->sample_entity(this->weighted_negative_sampling, inverse ? this->am_out : this->am_in);
            ch.second->parent_r = this->kg->num_rel;
            qt->hash_code = this->kg->num_rel;
        }
        bool ch_result = this->sample_actual_query(ch.second, prev, inverse, ent_list);
        hash_combine(qt->hash_code, ch.second->hash_code);
        return ch_result;
    }
}

template<typename Dtype>
QuerySample<Dtype>* PartitionSampler<Dtype>::gen_sample(int query_type, const Dtype* list_neg_candidates)
{
    QueryTree<Dtype>* qt = this->query_trees[query_type]->copy_backbone();
    std::vector<Dtype> ent_list;
    while (true) {
        Dtype ans = this->sample_entity(this->weighted_answer_sampling, qt->is_inverse ? this->am_out : this->am_in);
        ent_list.clear();
        if (this->sample_actual_query(qt, ans, qt->is_inverse, ent_list) && this->verify_sampled_query(qt, ans, qt->is_inverse))
            break;
    }
    bool is_same_partition = true;
    for (size_t i = 1; i < ent_list.size(); ++i)
        if (this->kg->partition_ids[ent_list[i]] != this->kg->partition_ids[ent_list[0]])
        {
            is_same_partition = false;
            break;
        }
    
    QuerySample<Dtype>* sample = new QuerySample<Dtype>(query_type);
    sample->query_args.clear();
    qt->get_query_args(sample->query_args);
    if (!is_same_partition)
    {
        for (size_t i = 0; i < sample->query_args.size(); ++i)
            sample->query_args[i] = this->kg->num_ent;
    }
    sample->positive_answer = qt->answer;
    sample->sample_weight = 1.0;
    this->negative_sampling(sample, qt, list_neg_candidates);
    delete qt;
    return sample;
}

template class PartitionSampler<unsigned>;
template class PartitionSampler<uint64_t>;

