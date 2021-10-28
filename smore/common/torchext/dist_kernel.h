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

#ifndef DIST_KERNEL_H
#define DIST_KERNEL_H

#include "cuda_math_ext.h"

template<typename scalar_t>
class BoxDist {
public:
    BoxDist() {}

    __device__ inline void update_ent(scalar_t* gpe, const int& d, const scalar_t& all_gpe_l, const scalar_t& all_gpe_r)
    {
        gpe[d] = all_gpe_l;
    }

    __device__ inline void update_query(scalar_t* g_pc, scalar_t* g_po, const int& d, const scalar_t& all_gpc, const scalar_t& all_gpo)    
    {
        g_pc[d] = all_gpc;
        g_po[d] = all_gpo;
    }
};

template<typename scalar_t>
class HalfDist {
public:
    HalfDist(int _dim) : dim(_dim) {}

    __device__ inline void update_ent(scalar_t* gpe, const int& d, const scalar_t& all_gpe_l, const scalar_t& all_gpe_r)
    {
        gpe[d] = all_gpe_l;
        gpe[d + dim] = all_gpe_r;
    }

    __device__ inline void update_query(scalar_t* g_pc, scalar_t* g_po, const int& d, const scalar_t& all_gpc, const scalar_t& all_gpo)    
    {
        g_pc[d] = all_gpc;
        g_po[d] = all_gpo;
    }

    int dim;
};


template<typename scalar_t>
class DistOut : public BoxDist<scalar_t> {
public:
    DistOut() {}
    
    __device__ inline scalar_t forward(const scalar_t* pe, const scalar_t* pc, const scalar_t* po, const int& d)
    {
        return cuda_relu(cuda_fabs(pe[d] - pc[d]) - po[d]);
    }

    __device__ inline void backward(const scalar_t& gout, const scalar_t* pe, const scalar_t* pc, const scalar_t* po, const int& d, scalar_t& ge, scalar_t& tmp, scalar_t& gc, scalar_t& go)
    {
        scalar_t e = pe[d], c = pc[d], o = po[d];
        scalar_t before_relu = cuda_fabs(e - c) - o;
        if (before_relu >= 0) {
            go -= gout;
            scalar_t sign = e - c > 0 ? 1 : -1;
            ge += sign * gout;
            gc += -sign * gout;
        }
    }
};


template<typename scalar_t>
class DistIn : public BoxDist<scalar_t> {
public:
    DistIn() {}
    
    __device__ inline scalar_t forward(const scalar_t* pe, const scalar_t* pc, const scalar_t* po, const int& d)
    {
        return cuda_fabs(cuda_min(cuda_fabs(pe[d] - pc[d]), po[d]));
    }

    __device__ inline void backward(const scalar_t& gout, const scalar_t* pe, const scalar_t* pc, const scalar_t* po, const int& d, scalar_t& ge, scalar_t& tmp, scalar_t& gc, scalar_t& go)
    {
        scalar_t e = pe[d], c = pc[d], o = po[d];
        scalar_t diff_abs = cuda_fabs(e - c);
        scalar_t outer_sign = cuda_min(diff_abs, o) > 0 ? 1 : -1;
        if (diff_abs < o) {
            scalar_t inner_sign = e - c > 0 ? 1 : -1;
            ge += outer_sign * inner_sign * gout;
            gc += -outer_sign * inner_sign * gout;
        } else {
            go += outer_sign * gout;
        }
    }
};


template<typename scalar_t>
class DistRotate : public HalfDist<scalar_t> {
public:
    DistRotate(int _dim) : HalfDist<scalar_t>(_dim) {}
    
    __device__ inline scalar_t forward(const scalar_t* pe, const scalar_t* pr, const scalar_t* pi, const int& d)
    {
        return cuda_sqrt(cuda_sqr(pe[d] - pr[d]) + cuda_sqr(pe[d + this->dim] - pi[d]));
    }

    __device__ inline void backward(const scalar_t& gout, const scalar_t* pe, const scalar_t* pre, const scalar_t* pim, const int& d, scalar_t& g_pe_l, scalar_t& g_pe_r, scalar_t& g_pre, scalar_t& g_pim)
    {
        scalar_t denorm = cuda_rsqrt(cuda_sqr(pe[d] - pre[d]) + cuda_sqr(pe[d + this->dim] - pim[d]));
        g_pe_l += (pe[d] - pre[d]) * denorm * gout;
        g_pe_r += (pe[d + this->dim] - pim[d]) * denorm * gout;
        g_pre += -(pe[d] - pre[d]) * denorm * gout;
        g_pim += -(pe[d + this->dim] - pim[d]) * denorm * gout;
    }
};


template<typename scalar_t>
class DistL1 {
public:
    DistL1() {}

    __device__ inline scalar_t forward(const scalar_t* pe, const scalar_t* pc, const scalar_t* tmp, const int& d)
    {
        return cuda_fabs(pe[d] - pc[d]);
    }

    __device__ inline void backward(const scalar_t& gout, const scalar_t* pe, const scalar_t* pc, const scalar_t* t1, const int& d, scalar_t& g_pe_l, scalar_t& g_pe_r, scalar_t& g_pc, scalar_t& t2)
    {
        scalar_t sign = pe[d] - pc[d] > 0 ? 1 : -1;
        g_pe_l += sign * gout;
        g_pc += -sign * gout;
    }

    __device__ inline void update_ent(scalar_t* gpe, const int& d, const scalar_t& all_gpe_l, const scalar_t& all_gpe_r)
    {
        gpe[d] = all_gpe_l;
    }

    __device__ inline void update_query(scalar_t* g_pc, scalar_t* t1, const int& d, const scalar_t& all_gpc, const scalar_t& t2)
    {
        g_pc[d] = all_gpc;
    }
};


template<typename scalar_t>
class DistL2 {
public:
    DistL2() {}

    __device__ inline scalar_t forward(const scalar_t* pe, const scalar_t* pc, const scalar_t* tmp, const int& d)
    {
        return cuda_sqr(pe[d] - pc[d]);
    }

    __device__ inline void backward(const scalar_t& gout, const scalar_t* pe, const scalar_t* pc, const scalar_t* t1, const int& d, scalar_t& g_pe_l, scalar_t& g_pe_r, scalar_t& g_pc, scalar_t& t2)
    {
        scalar_t diff = gout * (pe[d] - pc[d]) * 2.0;
        g_pe_l += diff;
        g_pc += -diff;
    }

    __device__ inline void update_ent(scalar_t* gpe, const int& d, const scalar_t& all_gpe_l, const scalar_t& all_gpe_r)
    {
        gpe[d] = all_gpe_l;
    }

    __device__ inline void update_query(scalar_t* g_pc, scalar_t* t1, const int& d, const scalar_t& all_gpc, const scalar_t& t2)
    {
        g_pc[d] = all_gpc;
    }
};


template<typename scalar_t>
class DistComplex : public HalfDist<scalar_t> {
public:
    DistComplex(int _dim) : HalfDist<scalar_t>(_dim) {}

    __device__ inline scalar_t forward(const scalar_t* pe, const scalar_t* pr, const scalar_t* pi, const int& d)
    {
        return pe[d] * pr[d] + pe[d + this->dim] * pi[d];
    }

    __device__ inline void backward(const scalar_t& gout, const scalar_t* pe, const scalar_t* pre, const scalar_t* pim, const int& d, scalar_t& g_pe_l, scalar_t& g_pe_r, scalar_t& g_pre, scalar_t& g_pim)
    {
        g_pe_l += pre[d] * gout;
        g_pe_r += pim[d] * gout;
        g_pre += pe[d] * gout;
        g_pim += pe[d + this->dim] * gout;
    }
};


template<typename scalar_t>
class DistBetaKL : public HalfDist<scalar_t> {
public:
    DistBetaKL(int _dim) : HalfDist<scalar_t>(_dim) {}

    __device__ inline scalar_t forward(const scalar_t* pe, const scalar_t* pr, const scalar_t* pi, const int& d)
    {
        scalar_t sum_params_p = pe[d] + pe[d + this->dim];
        scalar_t sum_params_q = pr[d] + pi[d];
        scalar_t t1 = cuda_lgamma(pr[d]) + cuda_lgamma(pi[d]) + cuda_lgamma(sum_params_p);
        scalar_t t2 = cuda_lgamma(pe[d]) + cuda_lgamma(pe[d + this->dim]) + cuda_lgamma(sum_params_q);
        scalar_t t3 = (pe[d] - pr[d]) * calc_digamma(pe[d]);
        scalar_t t4 = (pe[d + this->dim] - pi[d]) * calc_digamma(pe[d + this->dim]);
        scalar_t t5 = (sum_params_q - sum_params_p) * calc_digamma(sum_params_p);
        return t1 - t2 + t3 + t4 + t5;
    }

    __device__ inline void backward(const scalar_t& gout, const scalar_t* pe, const scalar_t* pr, const scalar_t* pi, const int& d, scalar_t& g_pe_l, scalar_t& g_pe_r, scalar_t& g_pre, scalar_t& g_pim)
    {
        scalar_t sum_params_p = pe[d] + pe[d + this->dim];
        scalar_t sum_params_q = pr[d] + pi[d];
        scalar_t dpel = 0, dper = 0, dp_i = 0, dp_r = 0, dsp = 0, dsq = 0;
        scalar_t digamma_sp = calc_digamma(sum_params_p);
        scalar_t digamma_pe_l = calc_digamma(pe[d]);
        scalar_t digamma_pe_r = calc_digamma(pe[d + this->dim]);

        // t5
        scalar_t dt5_l = gout * digamma_sp;
        dsp += gout * (sum_params_q - sum_params_p) * calc_trigamma(sum_params_p);
        dsp += -dt5_l;
        dsq += dt5_l;

        // t4
        scalar_t dt4_l = gout * digamma_pe_r;
        dper += gout * (pe[d + this->dim] - pi[d]) * calc_trigamma(pe[d + this->dim]);
        dper += dt4_l;
        dp_i += -dt4_l;

        // t3
        scalar_t dt3_l = gout * digamma_pe_l;
        dpel += gout * (pe[d] - pr[d]) * calc_trigamma(pe[d]);
        dpel += dt3_l;
        dp_r += -dt3_l;

        // t2
        dpel += -gout * digamma_pe_l;
        dper += -gout * digamma_pe_r;
        dsq += -gout * calc_digamma(sum_params_q);

        // t1
        dp_r += gout * calc_digamma(pr[d]);
        dp_i += gout * calc_digamma(pi[d]);
        dsp += gout * digamma_sp;

        g_pe_l += dpel + dsp;
        g_pe_r += dper + dsp;
        g_pre += dp_r + dsq;
        g_pim += dp_i + dsq;
    }
};


template<typename scalar_t>
class DistBetaL2 : public HalfDist<scalar_t> {
public:
    DistBetaL2(int _dim) : HalfDist<scalar_t>(_dim) {}

    __device__ inline scalar_t forward(const scalar_t* pe, const scalar_t* pr, const scalar_t* pi, const int& d)
    {
        return 0.5 * (cuda_sqr(pe[d] - pr[d]) + cuda_sqr(pe[d + this->dim] - pi[d]));
    }

    __device__ inline void backward(const scalar_t& gout, const scalar_t* pe, const scalar_t* pr, const scalar_t* pi, const int& d, scalar_t& g_pe_l, scalar_t& g_pe_r, scalar_t& g_pre, scalar_t& g_pim)
    {
        scalar_t t1 = pe[d] - pr[d], t2 = pe[d + this->dim] - pi[d];
        g_pe_l += gout * t1;
        g_pe_r += gout * t2;
        g_pre += -gout * t1;
        g_pim += -gout * t2;
    }
};


template<typename scalar_t>
class DistBetaFisherApprox : public HalfDist<scalar_t> {
public:
    DistBetaFisherApprox(int _dim) : HalfDist<scalar_t>(_dim) {}

    __device__ inline scalar_t forward(const scalar_t* pe, const scalar_t* pr, const scalar_t* pi, const int& d)
    {
        scalar_t psi_alpha = calc_trigamma(pe[d]), psi_beta = calc_trigamma(pe[d + this->dim]), neg_psi_ab = -calc_trigamma(pe[d] + pe[d + this->dim]);
        scalar_t da = pe[d] - pr[d], db = pe[d + this->dim] - pi[d];

        scalar_t t1 = da * da * (psi_alpha + neg_psi_ab);
        scalar_t t2 = 2 * neg_psi_ab * da * db;
        scalar_t t3 = db * db * (psi_beta + neg_psi_ab);

        return 0.5 * (t1 + t2 + t3);
    }

    __device__ inline void backward(const scalar_t& gout, const scalar_t* pe, const scalar_t* pr, const scalar_t* pi, const int& d, scalar_t& g_pe_l, scalar_t& g_pe_r, scalar_t& g_pre, scalar_t& g_pim)
    {
        scalar_t psi_alpha = calc_trigamma(pe[d]), psi_beta = calc_trigamma(pe[d + this->dim]), neg_psi_ab = -calc_trigamma(pe[d] + pe[d + this->dim]);
        scalar_t da = pe[d] - pr[d], db = pe[d + this->dim] - pi[d];

        scalar_t dadb_mul_neg = neg_psi_ab * (da + db);

        scalar_t gl = gout * (psi_alpha * da + dadb_mul_neg);
        scalar_t gr = gout * (psi_beta * db + dadb_mul_neg);
        g_pe_l += gl;
        g_pe_r += gr;
        g_pre += -gl;
        g_pim += -gr;
    }
};


template<typename scalar_t>
class Distmult {
public:
    Distmult() {}

    __device__ inline scalar_t forward(const scalar_t* pe, const scalar_t* pc, const scalar_t* tmp, const int& d)
    {
        return pe[d] * pc[d];
    }

    __device__ inline void backward(const scalar_t& gout, const scalar_t* pe, const scalar_t* pc, const scalar_t* t1, const int& d, scalar_t& g_pe_l, scalar_t& g_pe_r, scalar_t& g_pc, scalar_t& t2)
    {
        g_pe_l += pc[d] * gout;
        g_pc += pe[d] * gout;
    }

    __device__ inline void update_ent(scalar_t* gpe, const int& d, const scalar_t& all_gpe_l, const scalar_t& all_gpe_r)
    {
        gpe[d] = all_gpe_l;
    }

    __device__ inline void update_query(scalar_t* g_pc, scalar_t* t1, const int& d, const scalar_t& all_gpc, const scalar_t& t2)
    {
        g_pc[d] = all_gpc;
    }
};


#endif
