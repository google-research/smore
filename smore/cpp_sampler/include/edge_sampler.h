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

#ifndef EDGE_SAMPLER_H
#define EDGE_SAMPLER_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "sampler.h"

template<typename Dtype>
class EdgeSampler : public ISampler<Dtype> {
    public:
    EdgeSampler(KG<Dtype>* _kg, py::list query_specs, py::list _query_prob, 
                bool share_negative, bool _same_in_batch,
                bool weighted_answer_sampling, bool weighted_negative_sampling,
                Dtype negative_sample_size, Dtype rel_bandwidth, Dtype max_to_keep, Dtype max_n_partial_answers,
                int num_threads, py::list no_search_list);

    virtual QuerySample<Dtype>* gen_sample(int query_type, const Dtype* list_neg_candidates) override;
    virtual void prefetch(int _batch_size, int num_batches) override;
    virtual int next_batch(py::array_t<long long, py::array::c_style | py::array::forcecast> _positive_samples, 
                           py::array_t<long long, py::array::c_style | py::array::forcecast> _negative_samples, 
                           py::array_t<float, py::array::c_style | py::array::forcecast> _sample_weights, 
                           py::array_t<float, py::array::c_style | py::array::forcecast> _is_negative,
                           py::array_t<long long, py::array::c_style | py::array::forcecast> _query_args) override;
    
    int gen_batch_sample(int query_type, long long* positive_samples, long long* negative_samples, 
                         float* sample_weights, float* is_negative,
                         long long* query_args);

    virtual void print_queries();
    std::vector<std::string> query_specs;
    std::set<int> no_search_set;
    std::vector< std::future<int> > batch_sample_buf;
};


#endif
