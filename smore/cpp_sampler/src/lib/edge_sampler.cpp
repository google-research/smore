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

#include "edge_sampler.h"
#include "ThreadPool.h"
#include <iostream>


template<typename Dtype>
EdgeSampler<Dtype>::EdgeSampler(KG<Dtype>* _kg, py::list _query_specs, py::list _query_prob, 
    bool _share_negative, bool _same_in_batch,
    bool _weighted_answer_sampling, bool _weighted_negative_sampling,
    Dtype _negative_sample_size, Dtype _rel_bandwidth, Dtype _max_to_keep, Dtype _max_n_partial_answers,
    int num_threads, py::list no_search_list) : ISampler<Dtype>(_kg, _query_prob, _share_negative, _same_in_batch,
                                        _weighted_answer_sampling, _weighted_negative_sampling,
                                        _negative_sample_size, _rel_bandwidth, _max_to_keep, 
                                        _max_n_partial_answers, num_threads)
{
    assert(this->share_negative);
    this->query_specs.clear();
    for (size_t i = 0; i < py::len(_query_specs); ++i)
        this->query_specs.push_back(py::cast<std::string>(_query_specs[i]));
    no_search_set.clear();
    for (size_t i = 0; i < py::len(no_search_list); ++i)
        no_search_set.insert(py::cast<int>(no_search_list[i]));        
    this->batch_sample_buf.clear();
}

template<typename Dtype>
QuerySample<Dtype>* EdgeSampler<Dtype>::gen_sample(int query_type, const Dtype* list_neg_candidates)
{
    std::cerr << "not implemented" << std::endl;
    return nullptr;
}

template<typename Dtype>
void EdgeSampler<Dtype>::print_queries()
{
    for (auto& name : this->query_specs)
        std::cerr << name << std::endl;
}


template<typename Dtype>
int EdgeSampler<Dtype>::gen_batch_sample(int query_type, long long* positive_samples, long long* negative_samples, 
                                         float* sample_weights, float* is_negative, long long* query_args)
{
    bool is_inverse = this->query_specs[query_type] == "-1p";
    auto* edge_backwd = is_inverse ? this->kg->ent_out : this->kg->ent_in;
    auto* edge_fwd = is_inverse ? this->kg->ent_in : this->kg->ent_out;

    for (Dtype i = 0; i < this->negative_sample_size; ++i)
        negative_samples[i] = this->sample_entity(this->weighted_negative_sampling, this->am_out);
    for (auto i = 0; i < this->batch_size; ++i)
    {
        Dtype tail = this->sample_entity(this->weighted_answer_sampling, is_inverse ? this->am_out : this->am_in);
        Dtype r, head;
        sample_rand_neighbor(edge_backwd, tail, r, head);
        auto ch_range = get_ch_range(edge_fwd, head, r, this->rel_bandwidth);
        Dtype num_answers = (Dtype)(ch_range.second - ch_range.first);
//        auto* cur_is_neg = is_negative + i * this->negative_sample_size;
//        for (Dtype j = 0; j < this->negative_sample_size; ++j)
//        {
//            bool is_positive = std::binary_search(ch_range.first, ch_range.second, negative_samples[j]);
//            cur_is_neg[j] = !is_positive;
//        }
        positive_samples[i] = tail;
        sample_weights[i] = sqrt(1.0 / (num_answers + 1.0));
        query_args[i * 2] = head;
        query_args[i * 2 + 1] = r;
    }
    return this->batch_size;
}


template<typename Dtype>
void EdgeSampler<Dtype>::prefetch(int _batch_size, int num_batches)
{
    this->batch_size = _batch_size;
}


template<typename Dtype>
int EdgeSampler<Dtype>::next_batch(py::array_t<long long, py::array::c_style | py::array::forcecast> _positive_samples, 
                                    py::array_t<long long, py::array::c_style | py::array::forcecast> _negative_samples, 
                                    py::array_t<float, py::array::c_style | py::array::forcecast> _sample_weights, 
                                    py::array_t<float, py::array::c_style | py::array::forcecast> _is_negative,
                                    py::array_t<long long, py::array::c_style | py::array::forcecast> _query_args)
{
    for (size_t i = 0; i < this->batch_sample_buf.size(); ++i)
        this->batch_sample_buf[i].get();
    this->batch_sample_buf.clear();

    long long* positive_samples = _positive_samples.mutable_unchecked<1>().mutable_data(0);
    long long* negative_samples = _negative_samples.mutable_unchecked<2>().mutable_data(0, 0);
    float* is_negative = _is_negative.mutable_unchecked<2>().mutable_data(0, 0);
    assert (this->share_negative);
    float* sample_weights = _sample_weights.mutable_unchecked<1>().mutable_data(0);
    long long* query_args = _query_args.mutable_unchecked<2>().mutable_data(0, 0);

    int q_type = this->query_dist(this->query_rand_engine);

    this->batch_sample_buf.emplace_back(
        this->thread_pool->enqueue([=]{
            return this->gen_batch_sample(q_type, positive_samples, negative_samples, sample_weights, is_negative,
                                          query_args);
        })
    );
    return q_type;
}


template class EdgeSampler<unsigned>;
template class EdgeSampler<uint64_t>;
