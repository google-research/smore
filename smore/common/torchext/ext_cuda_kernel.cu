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

#include "ext_cuda_kernel.h"
#include "cuda_math_ext.h"
#include "common_kernel.h"
#include "dist_kernel.h"

#include <iostream>

template<typename scalar_t>
DistImpl<scalar_t>::DistImpl(std::string _dist_name) : dist_name(_dist_name) {}

template<typename scalar_t>
void DistImpl<scalar_t>::forward(cudaStream_t stream, const scalar_t* ent_ptr, const scalar_t* center_ptr, const scalar_t* offset_ptr, scalar_t* dst_ptr, size_t bsize, size_t num, size_t be, size_t ne, size_t bc, size_t nc, size_t dim)
{
    if (dist_name == "box_out")
        forward_impl(1, DistOut<scalar_t>(), stream, ent_ptr, center_ptr, offset_ptr, dst_ptr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "box_in")
        forward_impl(1, DistIn<scalar_t>(), stream, ent_ptr, center_ptr, offset_ptr, dst_ptr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "rotate")
        forward_impl(2, DistRotate<scalar_t>(dim), stream, ent_ptr, center_ptr, offset_ptr, dst_ptr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "l1")
        forward_impl(1, DistL1<scalar_t>(), stream, ent_ptr, center_ptr, offset_ptr, dst_ptr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "l2")
        forward_impl(1, DistL2<scalar_t>(), stream, ent_ptr, center_ptr, offset_ptr, dst_ptr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "complex")
        forward_impl(2, DistComplex<scalar_t>(dim), stream, ent_ptr, center_ptr, offset_ptr, dst_ptr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "beta_kl")
        forward_impl(2, DistBetaKL<scalar_t>(dim), stream, ent_ptr, center_ptr, offset_ptr, dst_ptr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "distmult")
        forward_impl(1, Distmult<scalar_t>(), stream, ent_ptr, center_ptr, offset_ptr, dst_ptr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "beta_l2")
        forward_impl(2, DistBetaL2<scalar_t>(dim), stream, ent_ptr, center_ptr, offset_ptr, dst_ptr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "beta_fisher_approx")
        forward_impl(2, DistBetaFisherApprox<scalar_t>(dim), stream, ent_ptr, center_ptr, offset_ptr, dst_ptr, bsize, num, be, ne, bc, nc, dim);
    else
        std::cerr << "unknown dist " << dist_name << std::endl;
}

template<typename scalar_t>
void DistImpl<scalar_t>::backward(cudaStream_t stream, const scalar_t* gout_ptr, const scalar_t* ent_ptr, const scalar_t* center_ptr, const scalar_t* offset_ptr, scalar_t* gent_ptr, scalar_t* gcenter_ptr, scalar_t* goff_otr, size_t bsize, size_t num, size_t be, size_t ne, size_t bc, size_t nc, size_t dim)
{
    if (dist_name == "box_out")
        backward_impl(1, DistOut<scalar_t>(), stream, gout_ptr, ent_ptr, center_ptr, offset_ptr, gent_ptr, gcenter_ptr, goff_otr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "box_in")
        backward_impl(1, DistIn<scalar_t>(), stream, gout_ptr, ent_ptr, center_ptr, offset_ptr, gent_ptr, gcenter_ptr, goff_otr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "rotate")
        backward_impl(2, DistRotate<scalar_t>(dim), stream, gout_ptr, ent_ptr, center_ptr, offset_ptr, gent_ptr, gcenter_ptr, goff_otr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "l1")
        backward_impl(1, DistL1<scalar_t>(), stream, gout_ptr, ent_ptr, center_ptr, offset_ptr, gent_ptr, gcenter_ptr, goff_otr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "l2")
        backward_impl(1, DistL2<scalar_t>(), stream, gout_ptr, ent_ptr, center_ptr, offset_ptr, gent_ptr, gcenter_ptr, goff_otr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "complex")
        backward_impl(2, DistComplex<scalar_t>(dim), stream, gout_ptr, ent_ptr, center_ptr, offset_ptr, gent_ptr, gcenter_ptr, goff_otr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "beta_kl")
        backward_impl(2, DistBetaKL<scalar_t>(dim), stream, gout_ptr, ent_ptr, center_ptr, offset_ptr, gent_ptr, gcenter_ptr, goff_otr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "distmult")
        backward_impl(1, Distmult<scalar_t>(), stream, gout_ptr, ent_ptr, center_ptr, offset_ptr, gent_ptr, gcenter_ptr, goff_otr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "beta_l2")
        backward_impl(2, DistBetaL2<scalar_t>(dim), stream, gout_ptr, ent_ptr, center_ptr, offset_ptr, gent_ptr, gcenter_ptr, goff_otr, bsize, num, be, ne, bc, nc, dim);
    else if (dist_name == "beta_fisher_approx")
        backward_impl(2, DistBetaFisherApprox<scalar_t>(dim), stream, gout_ptr, ent_ptr, center_ptr, offset_ptr, gent_ptr, gcenter_ptr, goff_otr, bsize, num, be, ne, bc, nc, dim);
    else
        std::cerr << "unknown dist " << dist_name << std::endl;
}

template class DistImpl<float>;
template class DistImpl<double>;
