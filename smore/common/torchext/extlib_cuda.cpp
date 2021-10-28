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

#include "extlib_cuda.h"
#include "ext_cuda_kernel.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <future>
#include <cstdlib>
#include <iostream>

typedef c10::cuda::CUDAGuard CUDAGuard;

void EXT_ASSERT(bool result)
{
    if (!result)
        exit(EXIT_FAILURE);
}

void EXT_ASSERT(bool result, const std::string& message)
{
    if (!result)
    {
        std::cerr << message << std::endl;
        exit(EXIT_FAILURE);
    }
}


void EXT_ASSERT(cudaError_t err, const std::string& message)
{
    if (err != cudaSuccess)
    {
        std::cerr << err << " " << message << std::endl;
        exit(EXIT_FAILURE);
    }
}


AsyncGPUJob::AsyncGPUJob(int8_t _device_idx) : job_ptr(nullptr), device_idx(_device_idx) {}

AsyncGPUJob::~AsyncGPUJob()
{
    if (job_ptr != nullptr)
    {
        CUDAGuard g(device_idx);
        auto event = job_ptr->get();
        if (event != (cudaEvent_t)NULL)
            EXT_ASSERT(cudaEventDestroy(event), "delete event failure");
        delete job_ptr;
    }
}

void AsyncGPUJob::sync()
{
    CUDAGuard g(device_idx);
    if (job_ptr == nullptr)
        return;
    auto event = job_ptr->get();
    if (event != (cudaEvent_t)NULL)
    {
        auto stream = at::cuda::getCurrentCUDAStream(this->device_idx);
        EXT_ASSERT(cudaStreamWaitEvent(stream, event, 0), "wait async job failure");
        EXT_ASSERT(cudaEventDestroy(event), "delete event failure after async job");
    }
    delete job_ptr;
    job_ptr = nullptr;
}

AsyncGPUJob* async_read(ThreadPool* thread_pool, torch::Tensor indices, torch::Tensor src, torch::Tensor buf, torch::Tensor dst)
{
    EXT_ASSERT(buf.is_pinned(), "buf is not pinned");
    EXT_ASSERT(dst.is_contiguous(), "dst is not contiguous");
    auto cur_device = dst.get_device();
    auto stream = at::cuda::getCurrentCUDAStream(cur_device);

    size_t n_rows = indices.sizes()[0];
    size_t embed_dim = src.sizes()[1];
    std::shared_future<cudaEvent_t>* job_ptr = new std::shared_future<cudaEvent_t>(
        thread_pool->enqueue([=]{
            CUDAGuard g(cur_device);
            cudaEvent_t event;
            EXT_ASSERT(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cuda read event creation failure");
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.type(), "async_read", ([&]{
                scalar_t* src_ptr = src.data<scalar_t>();
                scalar_t* dst_ptr = dst.data<scalar_t>();
                scalar_t* buf_ptr = buf.data<scalar_t>();
                auto idx_acc = indices.accessor<int64_t, 1>();
                for (size_t i = 0; i < n_rows; ++i)
                    memcpy(buf_ptr + i * embed_dim, src_ptr + idx_acc[i] * embed_dim, sizeof(scalar_t) * embed_dim);
                EXT_ASSERT(cudaMemcpyAsync(dst_ptr, buf_ptr, sizeof(scalar_t) * n_rows * embed_dim, cudaMemcpyHostToDevice, stream), "host2device failure");
                EXT_ASSERT(cudaEventRecord(event, stream), "event record failure");
            }));
            return event;
        })
    );
    AsyncGPUJob* job = new AsyncGPUJob(cur_device);
    job->job_ptr = job_ptr;
    return job;
}


AsyncGPUJob* async_write(ThreadPool* thread_pool, torch::Tensor indices, torch::Tensor dst, torch::Tensor buf, torch::Tensor src, CudaEvent* src_event,  bool additive)
{
    EXT_ASSERT(buf.is_pinned(), "buf is not pinned");
    EXT_ASSERT(src.is_contiguous(), "src is not contiguous");
    auto cur_device = src.get_device();
    auto stream = at::cuda::getCurrentCUDAStream(cur_device);

    size_t n_rows = indices.sizes()[0];
    size_t embed_dim = dst.sizes()[1];
    std::shared_future<cudaEvent_t>* job_ptr = new std::shared_future<cudaEvent_t>(
        thread_pool->enqueue([=]{
            CUDAGuard g(cur_device);
            AT_DISPATCH_FLOATING_TYPES(dst.type(), "async_write", ([&]{
                scalar_t* src_ptr = src.data_ptr<scalar_t>();
                scalar_t* buf_ptr = buf.data_ptr<scalar_t>();
                EXT_ASSERT(cudaStreamWaitEvent(stream.stream(), src_event->event, 0), "failure while waiting for src to be finished");
                EXT_ASSERT(cudaMemcpyAsync(buf_ptr, src_ptr, sizeof(scalar_t) * n_rows * embed_dim, cudaMemcpyDeviceToHost, stream), "device2host failure");
                stream.synchronize();
                if (additive)
                    dst.index_add_(0, indices, buf);
                else
                    dst.index_copy_(0, indices, buf);
            }));
            return (cudaEvent_t)NULL;
        })
    );
    AsyncGPUJob* job = new AsyncGPUJob(cur_device);
    job->job_ptr = job_ptr;
    return job;
}


CudaEvent::CudaEvent(int8_t _cur_device) : cur_device(_cur_device)
{
    CUDAGuard g(cur_device);
    initialized = false;
    stream = at::cuda::getCurrentCUDAStream(cur_device).stream();
}

CudaEvent::~CudaEvent()
{
    CUDAGuard g(cur_device);
    if (initialized)
        EXT_ASSERT(cudaEventDestroy(event), "event destructor failure");
}

void CudaEvent::record()
{
    CUDAGuard g(cur_device);
    if (!initialized) {
        EXT_ASSERT(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event creation failure");
        initialized = true;
    }
    EXT_ASSERT(cudaEventRecord(event, stream), "event record failure");
}


void dist_forward(const std::string dname, torch::Tensor entity_embed, torch::Tensor center_embed, torch::Tensor offset_embed, torch::Tensor dst, bool has_second)
{
    auto be = entity_embed.sizes()[0];
    auto ne = entity_embed.sizes()[1];
    auto bc = center_embed.sizes()[0];
    auto nc = center_embed.sizes()[1];
    
    auto embed_dim = center_embed.sizes()[2];
    auto cur_device = entity_embed.get_device();
    CUDAGuard g(cur_device);
    auto stream = at::cuda::getCurrentCUDAStream(cur_device);
    auto bsize = dst.sizes()[0];
    auto num = dst.sizes()[1];

    std::string func_name = "dist_" + dname;
    AT_DISPATCH_FLOATING_TYPES(entity_embed.type(), "dist_forward", ([&]{
        scalar_t* ent_ptr = entity_embed.data<scalar_t>();
        scalar_t* center_ptr = center_embed.data<scalar_t>();
        scalar_t* off_ptr = has_second ? offset_embed.data<scalar_t>() : nullptr;
        scalar_t* dst_ptr = dst.data<scalar_t>();
        DistImpl<scalar_t> dist(dname);
        dist.forward(stream, ent_ptr, center_ptr, off_ptr, dst_ptr, bsize, num, be, ne, bc, nc, embed_dim);
        EXT_ASSERT(cudaGetLastError(), func_name + " forward error");
    }));
}

void box_dist_forward(torch::Tensor entity_embed, torch::Tensor center_embed, torch::Tensor offset_embed, torch::Tensor dst, const std::string& dist_type)
{
    dist_forward("box_" + dist_type, entity_embed, center_embed, offset_embed, dst, true);
}

void rotate_dist_forward(torch::Tensor entity_embed, torch::Tensor re_embed, torch::Tensor im_embed, torch::Tensor dst)
{
    dist_forward("rotate", entity_embed, re_embed, im_embed, dst, true);
}

void l1_dist_forward(torch::Tensor entity_embed, torch::Tensor center_embed, torch::Tensor dst)
{
    dist_forward("l1", entity_embed, center_embed, center_embed, dst, false);
}

void l2_dist_forward(torch::Tensor entity_embed, torch::Tensor center_embed, torch::Tensor dst)
{
    dist_forward("l2", entity_embed, center_embed, center_embed, dst, false);
}

void complex_dist_forward(torch::Tensor entity_embed, torch::Tensor re_embed, torch::Tensor im_embed, torch::Tensor dst)
{
    dist_forward("complex", entity_embed, re_embed, im_embed, dst, true);
}

void beta_dist_forward(torch::Tensor entity_embed, torch::Tensor re_embed, torch::Tensor im_embed, torch::Tensor dst, const std::string& dist_type)
{
    dist_forward("beta_" + dist_type, entity_embed, re_embed, im_embed, dst, true);
}

void distmult_dist_forward(torch::Tensor entity_embed, torch::Tensor center_embed, torch::Tensor dst)
{
    dist_forward("distmult", entity_embed, center_embed, center_embed, dst, false);
}

void dist_backward(const std::string dname, torch::Tensor grad_out, torch::Tensor entity_embed, torch::Tensor center_embed, 
                   torch::Tensor offset_embed, torch::Tensor grad_entity, torch::Tensor grad_center, torch::Tensor grad_offset,
                   bool has_second)
{
    auto be = entity_embed.sizes()[0];
    auto ne = entity_embed.sizes()[1];
    auto bc = center_embed.sizes()[0];
    auto nc = center_embed.sizes()[1];

    auto embed_dim = center_embed.sizes()[2];
    auto cur_device = grad_out.get_device();
    CUDAGuard g(cur_device);
    auto stream = at::cuda::getCurrentCUDAStream(cur_device);
    auto bsize = grad_out.sizes()[0];
    auto num = grad_out.sizes()[1];
    std::string func_name = "dist_" + dname;
    AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "dist_backward", ([&]{
        scalar_t* gout_ptr = grad_out.data_ptr<scalar_t>();
        scalar_t* ent_ptr = entity_embed.data_ptr<scalar_t>();
        scalar_t* center_ptr = center_embed.data_ptr<scalar_t>();
        scalar_t* off_ptr = has_second ? offset_embed.data_ptr<scalar_t>() : nullptr;

        scalar_t* gent_ptr = grad_entity.data_ptr<scalar_t>();
        scalar_t* gcenter_ptr = grad_center.data_ptr<scalar_t>();
        scalar_t* goff_ptr = has_second ? grad_offset.data_ptr<scalar_t>() : nullptr;
        DistImpl<scalar_t> dist(dname);
        dist.backward(stream, gout_ptr, ent_ptr, center_ptr, off_ptr, gent_ptr, gcenter_ptr, goff_ptr, bsize, num, be, ne, bc, nc, embed_dim);
        EXT_ASSERT(cudaGetLastError(), func_name + " dist backward error");
    }));
}

void box_dist_backward(torch::Tensor grad_out, torch::Tensor entity_embed, torch::Tensor center_embed, torch::Tensor offset_embed,
                       torch::Tensor grad_entity, torch::Tensor grad_center, torch::Tensor grad_offset, const std::string& dist_type)
{
    dist_backward("box_" + dist_type, grad_out, entity_embed, center_embed, offset_embed, grad_entity, grad_center, grad_offset, true);
}

void rotate_dist_backward(torch::Tensor grad_out, torch::Tensor entity_embed, torch::Tensor re_embed, torch::Tensor im_embed, 
                          torch::Tensor grad_entity, torch::Tensor grad_re, torch::Tensor grad_im)
{
    dist_backward("rotate", grad_out, entity_embed, re_embed, im_embed, grad_entity, grad_re, grad_im, true);
}

void l1_dist_backward(torch::Tensor grad_out, torch::Tensor entity_embed, torch::Tensor center_embed,
                      torch::Tensor grad_entity, torch::Tensor grad_center)
{
    dist_backward("l1", grad_out, entity_embed, center_embed, center_embed, grad_entity, grad_center, grad_center, false);
}

void l2_dist_backward(torch::Tensor grad_out, torch::Tensor entity_embed, torch::Tensor center_embed,
                      torch::Tensor grad_entity, torch::Tensor grad_center)
{
    dist_backward("l2", grad_out, entity_embed, center_embed, center_embed, grad_entity, grad_center, grad_center, false);
}

void complex_dist_backward(torch::Tensor grad_out, torch::Tensor entity_embed, torch::Tensor re_embed, torch::Tensor im_embed, 
                          torch::Tensor grad_entity, torch::Tensor grad_re, torch::Tensor grad_im)
{
    dist_backward("complex", grad_out, entity_embed, re_embed, im_embed, grad_entity, grad_re, grad_im, true);
}

void beta_dist_backward(torch::Tensor grad_out, torch::Tensor entity_embed, torch::Tensor re_embed, torch::Tensor im_embed, 
                          torch::Tensor grad_entity, torch::Tensor grad_re, torch::Tensor grad_im, const std::string& dist_type)
{
    dist_backward("beta_" + dist_type, grad_out, entity_embed, re_embed, im_embed, grad_entity, grad_re, grad_im, true);
}

void distmult_dist_backward(torch::Tensor grad_out, torch::Tensor entity_embed, torch::Tensor center_embed,
                      torch::Tensor grad_entity, torch::Tensor grad_center)
{
    dist_backward("distmult", grad_out, entity_embed, center_embed, center_embed, grad_entity, grad_center, grad_center, false);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    py::class_<ThreadPool>(mod, "ThreadPool")
        .def(py::init<size_t>());

    py::class_<CudaEvent>(mod, "CudaEvent")
        .def(py::init<int8_t>())
        .def("record", &CudaEvent::record);

    py::class_<AsyncGPUJob>(mod, "AsyncGPUJob")
        .def(py::init<int8_t>())
        .def("sync", &AsyncGPUJob::sync);

    mod.def("async_read", &async_read, "Async embedding read");
    mod.def("async_write", &async_write, "Async embedding write");

    mod.def("box_dist_forward", &box_dist_forward, "box dist forward");
    mod.def("box_dist_backward", &box_dist_backward, "box dist backward");

    mod.def("rotate_dist_forward", &rotate_dist_forward, "rotate dist forward");
    mod.def("rotate_dist_backward", &rotate_dist_backward, "rotate dist backward");

    mod.def("l1_dist_forward", &l1_dist_forward, "l1 dist forward");
    mod.def("l1_dist_backward", &l1_dist_backward, "l1 dist backward");

    mod.def("l2_dist_forward", &l2_dist_forward, "l2 dist forward");
    mod.def("l2_dist_backward", &l2_dist_backward, "l2 dist backward");

    mod.def("complex_dist_forward", &complex_dist_forward, "complex dist forward");
    mod.def("complex_dist_backward", &complex_dist_backward, "complex dist backward");

    mod.def("beta_dist_forward", &beta_dist_forward, "beta dist forward");
    mod.def("beta_dist_backward", &beta_dist_backward, "beta dist backward");

    mod.def("distmult_dist_forward", &distmult_dist_forward, "distmult dist forward");
    mod.def("distmult_dist_backward", &distmult_dist_backward, "distmult dist backward");    
}
