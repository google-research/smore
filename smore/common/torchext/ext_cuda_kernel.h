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

#ifndef EXT_CUDA_KERNEL_H
#define EXT_CUDA_KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

template<typename scalar_t>
class DistImpl
{
public:

    DistImpl(std::string _dist_name);
    void forward(cudaStream_t stream, const scalar_t* ent_ptr, const scalar_t* center_ptr, const scalar_t* offset_ptr, scalar_t* dst_ptr, size_t bsize, size_t num, size_t be, size_t ne, size_t bc, size_t nc, size_t dim);
    void backward(cudaStream_t stream, const scalar_t* gout_ptr, const scalar_t* ent_ptr, const scalar_t* center_ptr, const scalar_t* offset_ptr, scalar_t* gent_ptr, scalar_t* gcenter_ptr, scalar_t* goff_otr, size_t bsize, size_t num, size_t be, size_t ne, size_t bc, size_t nc, size_t dim);
    std::string dist_name;
};


#endif
