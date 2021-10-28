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

#ifndef CUDA_MATH_EXT_H
#define CUDA_MATH_EXT_H

#include <cuda.h>
#include <cuda_runtime.h>

#define EPS 1e-20


__device__ inline float cuda_fabs(const float& src)
{
    return fabsf(src);
}

__device__ inline double cuda_fabs(const double& src)
{
    return fabs(src);
}

__device__ inline float cuda_sqr(const float& src)
{
    return src * src;
}

__device__ inline double cuda_sqr(const double& src)
{
    return src * src;
}

__device__ inline float cuda_sin(const float& src)
{
    return sinf(src);
}

__device__ inline double cuda_sin(const double& src)
{
    return sin(src);
}

__device__ inline float cuda_tan(const float& src)
{
    return tanf(src);
}

__device__ inline double cuda_tan(const double& src)
{
    return tan(src);
}

__device__ inline float cuda_log(const float& src)
{
    return logf(src);
}

__device__ inline double cuda_log(const double& src)
{
    return log(src);
}

__device__ inline float cuda_trunc(const float& src)
{
    return truncf(src);
}

__device__ inline double cuda_trunc(const double& src)
{
    return trunc(src);
}

__device__ inline float cuda_lgamma(const float& src)
{
    return lgammaf(src);
}

__device__ inline double cuda_lgamma(const double& src)
{
    return lgamma(src);
}

template<typename T>
__device__ inline float cuda_relu(const T& src)
{
    return src > 0 ? src : 0;
}

__device__ inline float cuda_sqrt(const float& src)
{
    return sqrtf(src + EPS);
}

__device__ inline double cuda_sqrt(const double& src)
{
    return sqrt(src + EPS);
}

__device__ inline float cuda_rsqrt(const float& src)
{
    return rsqrtf(src + EPS);
}

__device__ inline double cuda_rsqrt(const double& src)
{
    return rsqrt(src + EPS);
}

__device__ inline float cuda_min(const float& x, const float& y)
{
    return fminf(x, y);
}

__device__ inline double cuda_min(const double& x, const double& y)
{
    return fmin(x, y);
}

template <typename scalar_t>
static inline __device__ scalar_t calc_digamma(scalar_t in) {
  // [C++ Standard Reference: Gamma Function] https://en.cppreference.com/w/cpp/numeric/math/tgamma
  using accscalar_t = scalar_t;
  static const double PI_f64 = 3.14159265358979323846;
  const accscalar_t PSI_10 = 2.25175258906672110764;
  const accscalar_t A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  accscalar_t x = static_cast<accscalar_t>(in);
  if (x == 0) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    return std::copysign(static_cast<scalar_t>(INFINITY), -x);
  }

  bool x_is_integer = x == cuda_trunc(x);
  accscalar_t result = 0;
  if (x < 0) {
    if (x_is_integer) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      return static_cast<scalar_t>(NAN);
    }
    // Rounding errors in tan's input can really affect the output
    // for extreme values, so we always perform this computation in double.
    result = static_cast<accscalar_t>(- PI_f64 / cuda_tan(PI_f64 * static_cast<double>(x)));
    x = 1 - x;
  }

  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    return static_cast<scalar_t>(result + PSI_10);
  }

  accscalar_t y = 0;
  if (x < 1.0e17) {
    accscalar_t z = 1 / (x * x);

    accscalar_t polevl_result = 0;
    for (int i = 0; i <= 6; i++) {
      polevl_result = polevl_result * z + A[i];
    }
    y = z * polevl_result;
  }

  return static_cast<scalar_t>(cuda_log(x) - (static_cast<accscalar_t>(0.5) / x) - y + result);
}


template <typename scalar_t>
static inline __device__ scalar_t calc_trigamma(scalar_t in) {
  using accscalar_t = scalar_t;
  const accscalar_t PI = 3.14159265358979323846;
  accscalar_t x = static_cast<accscalar_t>(in);
  accscalar_t sign = +1;
  accscalar_t result = 0;
  if (x < 0.5f) {
    sign = -1;
    accscalar_t sin_pi_x = cuda_sin(PI * x);
    result -= (PI * PI) / (sin_pi_x * sin_pi_x);
    x = 1 - x;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (x * x);
    x += 1;
  }
  const accscalar_t one = static_cast<scalar_t>(1);
  const accscalar_t ixx = 1 / (x*x);
  result += (1 + 1 / (2*x) + ixx * (one/6 - ixx * (one/30 - ixx * (one/42)))) / x;
  return static_cast<scalar_t>(sign * result);
}


#endif