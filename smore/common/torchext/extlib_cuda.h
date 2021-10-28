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

#ifndef EXTLIB_CUDA_H
#define EXTLIB_CUDA_H

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <vector>
#include <string>

#include "extlib.h"

class AsyncGPUJob
{
public:
    AsyncGPUJob(int8_t _device_idx);
    ~AsyncGPUJob();

    void sync();

    std::shared_future<cudaEvent_t>* job_ptr;
    int8_t device_idx;
};

class CudaEvent
{
public:
    CudaEvent(int8_t device_id);
    ~CudaEvent();
    void record();
    cudaEvent_t event;
    cudaStream_t stream;
    int8_t cur_device;
    bool initialized;
};

AsyncGPUJob* async_read(ThreadPool* thread_pool, torch::Tensor indices, torch::Tensor src, torch::Tensor buf, torch::Tensor dst);

AsyncGPUJob* async_write(ThreadPool* thread_pool, torch::Tensor indices, torch::Tensor dst, torch::Tensor buf, torch::Tensor src, CudaEvent* src_event, bool additive);


#endif
