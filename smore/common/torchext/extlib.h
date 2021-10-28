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

#ifndef EXTLIB_H
#define EXTLIB_H

#include <torch/extension.h>
#include <vector>
#include <future>
#include "ThreadPool.h"

namespace py = pybind11;

class AsyncJob
{
public:
    AsyncJob() : job_ptr(nullptr) {};
    ~AsyncJob() {
        if (job_ptr != nullptr)
            delete job_ptr;        
    };

    void sync() {
        if (job_ptr == nullptr)
            return;
        job_ptr->get();
        job_ptr = nullptr;
    };

    std::shared_future<int>* job_ptr;
};


#endif
