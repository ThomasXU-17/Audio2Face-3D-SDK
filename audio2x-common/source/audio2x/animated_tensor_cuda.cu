// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
#include "audio2x/internal/animated_tensor_cuda.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/error.h"

#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

namespace {

// From https://developer.nvidia.com/blog/lerp-faster-cuda/
template <typename T>
__host__ __device__
inline T lerp(T v0, T v1, T t) {
    return fma(t, v1, fma(-t, v0, v0));
}

class linear_interpolator {
public:
  linear_interpolator(float t) : _t{t} {}
  __host__ __device__ float operator()(float a, float b) { return lerp(a, b, _t); }

private:
  float _t;
};

}

std::error_code nva2x::cuda::Lerp(
  float* destination, const float* a, const float* b, float t, std::size_t size, cudaStream_t cudaStream
) {
  thrust::device_ptr<float> destinationPtr = thrust::device_pointer_cast<float>(destination);
  thrust::device_ptr<const float> aPtr = thrust::device_pointer_cast<const float>(a);
  thrust::device_ptr<const float> bPtr = thrust::device_pointer_cast<const float>(b);
  thrust::transform(
    thrust::cuda::par_nosync.on(cudaStream),
    aPtr,
    aPtr + size,
    bPtr,
    destinationPtr,
    linear_interpolator(t)
    );

  A2X_CUDA_CHECK_ERROR(cudaGetLastError(), ErrorCode::eCudaThrustError);
  return ErrorCode::eSuccess;
}
