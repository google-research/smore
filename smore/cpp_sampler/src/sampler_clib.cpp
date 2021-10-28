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

#include "sampler_clib.h"
#include "knowledge_graph.h"
#include <string>


template<typename Dtype>
void _load_binary_file(const char* fname, void* _mem_ptr, unsigned long long n_ints)
{
    Dtype* buf_ptr = static_cast<Dtype*>(_mem_ptr);
    FILE* fin = fopen(fname, "rb");
    assert(fread(buf_ptr, sizeof(Dtype), n_ints, fin) == n_ints);
    fclose(fin);
}


void load_binary_file(const char* fname, void* _mem_ptr, unsigned long long n_ints, const char* dtype)
{
    if (std::string(dtype) == "uint32")
        _load_binary_file<unsigned>(fname, _mem_ptr, n_ints);
    else
        _load_binary_file<uint64_t>(fname, _mem_ptr, n_ints);
}


template<typename Dtype>
void _load_kg_from_binary(void* _kg_ptr, void* _mem_ptr, unsigned long long n_ints)
{
    KG<Dtype>* kg = static_cast<KG<Dtype>*>(_kg_ptr);
    Dtype* buf_ptr = static_cast<Dtype*>(_mem_ptr);
    kg->load_from_binary(buf_ptr, n_ints);
}


void load_kg_from_binary(void* _kg_ptr, void* _mem_ptr, unsigned long long n_ints, const char* dtype)
{
    if (std::string(dtype) == "uint32")
        _load_kg_from_binary<unsigned>(_kg_ptr, _mem_ptr, n_ints);
    else
        _load_kg_from_binary<uint64_t>(_kg_ptr, _mem_ptr, n_ints);
}


template<typename Dtype>
void _load_kg_from_numpy(void* _kg_ptr, void* _triple_ptr, long long n_triplets, bool has_reverse_edges)
{
    KG<Dtype>* kg = static_cast<KG<Dtype>*>(_kg_ptr);
    kg->load_from_numpy(_triple_ptr, n_triplets, has_reverse_edges);
}


void load_kg_from_numpy(void* _kg_ptr, void* _triple_ptr, long long n_triplets, bool has_reverse_edges, const char* dtype)
{
    if (std::string(dtype) == "uint32")
        _load_kg_from_numpy<unsigned>(_kg_ptr, _triple_ptr, n_triplets, has_reverse_edges);
    else
        _load_kg_from_numpy<uint64_t>(_kg_ptr, _triple_ptr, n_triplets, has_reverse_edges);
}
