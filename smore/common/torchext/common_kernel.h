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

#ifndef COMMON_KERNEL_H
#define COMMON_KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>

const int kWarpSize = 32;
const int kWarpPerBlock = 4;


template<typename T>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
      val = val + __shfl_xor_sync(0xffffffff, val, mask);
    }
  return val;
}

template<typename scalar_t, class Dist>
__global__ void dist_forward_kernel(size_t mul, Dist f_dist, const scalar_t* ent_ptr, const scalar_t* first_ptr, const scalar_t* second_ptr, scalar_t* dst, size_t bsize, size_t num, size_t be, size_t ne, size_t bc, size_t nc, size_t dim)
{
    const size_t global_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    const size_t num_global_warps = gridDim.x * blockDim.y;
    const int lane_id = threadIdx.x;
    for (size_t warp_id = global_warp_id; warp_id < bsize * num; warp_id += num_global_warps)
    {
        size_t dim_0 = warp_id / num;
        size_t dim_1 = warp_id % num;
        if (dim_0 >= bsize)
            continue;
        scalar_t s = 0;

        const scalar_t* pe = ent_ptr + (dim_0 % be) * ne * dim * mul + (dim_1 % ne) * dim * mul;
        const scalar_t* pc = first_ptr + (dim_0 % bc) * nc * dim + (dim_1 % nc) * dim;
        const scalar_t* po = second_ptr + (dim_0 % bc) * nc * dim + (dim_1 % nc) * dim;
        for (int i = lane_id; i < dim; i += kWarpSize)
            s += f_dist.forward(pe, pc, po, i);
        scalar_t dist = WarpAllReduce(s);
        if (lane_id == 0)
            dst[warp_id] = dist;
    }
}


template<typename scalar_t, class Dist>
__global__ void dist_backward_ent_kernel(size_t mul, Dist f_dist, const scalar_t* gout_ptr, const scalar_t* ent_ptr, const scalar_t* first_ptr, const scalar_t* second_ptr, scalar_t* gent_ptr, size_t bsize, size_t num, size_t be, size_t ne, size_t bc, size_t nc, size_t dim)
{
    const size_t dim_0 = blockIdx.x;
    const size_t dim_1 = blockIdx.y;
    const int d = blockIdx.z * blockDim.y + threadIdx.y;
    const int lane_id = threadIdx.x;
    const scalar_t* pe = ent_ptr + dim_0 * ne * dim * mul + dim_1 * dim * mul;
    scalar_t* g_pe = gent_ptr + dim_0 * ne * dim * mul + dim_1 * dim * mul;
    scalar_t t1 = 0, t2 = 0;

    scalar_t local_gpe_l = 0, local_gpe_r = 0;
    if (d < (int)dim)
    {
        int num_job_i = bsize / be;
        int num_job_j = num / ne;
        for (int job_id = lane_id; job_id < num_job_i * num_job_j; job_id += kWarpSize)
        {
            int i = (job_id / num_job_j) * be + dim_0;
            int j = (job_id % num_job_j) * ne + dim_1;
            scalar_t gout = gout_ptr[i * num + j];
            const scalar_t* pc = first_ptr + (i % bc) * nc * dim + (j % nc) * dim;
            const scalar_t* po = second_ptr + (i % bc) * nc * dim + (j % nc) * dim;
            f_dist.backward(gout, pe, pc, po, d, local_gpe_l, local_gpe_r, t1, t2);
        }
    }
    scalar_t all_gpe_l = WarpAllReduce(local_gpe_l);
    scalar_t all_gpe_r = WarpAllReduce(local_gpe_r);
    if (lane_id == 0 && d < dim)
        f_dist.update_ent(g_pe, d, all_gpe_l, all_gpe_r);
}


template<typename scalar_t, class Dist>
__global__ void dist_backward_query_kernel(size_t mul, Dist f_dist, const scalar_t* gout_ptr, const scalar_t* ent_ptr, const scalar_t* first_ptr, const scalar_t* second_ptr, scalar_t* gfirst_ptr, scalar_t* gsecond_ptr, size_t bsize, size_t num, size_t be, size_t ne, size_t bc, size_t nc, size_t dim)
{
    const size_t dim_0 = blockIdx.x;
    const size_t dim_1 = blockIdx.y;
    const size_t d = blockIdx.z * blockDim.y + threadIdx.y;
    const int lane_id = threadIdx.x;
    const scalar_t* pc = first_ptr + dim_0 * nc * dim + dim_1 * dim;
    const scalar_t* po = second_ptr + dim_0 * nc * dim + dim_1 * dim;
    scalar_t* g_pc = gfirst_ptr + dim_0 * nc * dim + dim_1 * dim;
    scalar_t* g_po = gsecond_ptr + dim_0 * nc * dim + dim_1 * dim;

    scalar_t t = 0;
    scalar_t local_gpc = 0, local_gpo = 0;
    if (d < dim)
    {
        int num_job_i = bsize / bc;
        int num_job_j = num / nc;
        for (int job_id = lane_id; job_id < num_job_i * num_job_j; job_id += kWarpSize)
        {
            int i = (job_id / num_job_j) * bc + dim_0;
            int j = (job_id % num_job_j) * nc + dim_1;
            scalar_t gout = gout_ptr[i * num + j];
            const scalar_t* pe = ent_ptr + (i % be) * ne * dim * mul + (j % ne) * dim * mul;
            f_dist.backward(gout, pe, pc, po, d, t, t, local_gpc, local_gpo);
        }
    }
    scalar_t all_gpc = WarpAllReduce(local_gpc);
    scalar_t all_gpo = WarpAllReduce(local_gpo);
    if (lane_id == 0 && d < dim)
        f_dist.update_query(g_pc, g_po, d, all_gpc, all_gpo);
}


template<typename scalar_t, class Dist>
void forward_impl(size_t mul, Dist dist, cudaStream_t stream, const scalar_t* ent_ptr, const scalar_t* center_ptr, const scalar_t* offset_ptr, scalar_t* dst_ptr, size_t bsize, size_t num, size_t be, size_t ne, size_t bc, size_t nc, size_t dim)
{
    size_t tot_warp_needed = bsize * num;
    dim3 grid_spec(tot_warp_needed / kWarpPerBlock + 1);
    dim3 block_spec(kWarpSize, kWarpPerBlock);
    dist_forward_kernel<<<grid_spec, block_spec, 0, stream>>>(mul, dist, ent_ptr, center_ptr, offset_ptr, dst_ptr, bsize, num, be, ne, bc, nc, dim);
}


template<typename scalar_t, class Dist>
void backward_impl(size_t mul, Dist dist, cudaStream_t stream, const scalar_t* gout_ptr, const scalar_t* ent_ptr, const scalar_t* center_ptr, const scalar_t* offset_ptr, scalar_t* gent_ptr, scalar_t* gcenter_ptr, scalar_t* goff_ptr, size_t bsize, size_t num, size_t be, size_t ne, size_t bc, size_t nc, size_t dim)
{
    int num_z = dim / kWarpPerBlock + 1;
    dim3 grid_spec_ent(be, ne, num_z);
    dim3 grid_spec_query(bc, nc, num_z);
    dim3 block_spec(kWarpSize, kWarpPerBlock);
    dist_backward_ent_kernel<<<grid_spec_ent, block_spec, 0, stream>>>(mul, dist, gout_ptr, ent_ptr, center_ptr, offset_ptr, gent_ptr, bsize, num, be, ne, bc, nc, dim);
    dist_backward_query_kernel<<<grid_spec_query, block_spec, 0, stream>>>(mul, dist, gout_ptr, ent_ptr, center_ptr, offset_ptr, gcenter_ptr, goff_ptr, bsize, num, be, ne, bc, nc, dim);
}


#endif
