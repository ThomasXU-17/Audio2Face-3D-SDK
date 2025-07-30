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
#include "audio2x/internal/tensor_cuda.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/error.h"

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>

template<typename Scalar>
std::error_code nva2x::cuda::FillOnDevice<Scalar>(
  Scalar* destination, std::size_t size, Scalar value, cudaStream_t cudaStream
) {
  thrust::fill_n(
    thrust::cuda::par_nosync.on(cudaStream),
    thrust::device_pointer_cast(destination),
    size,
    value
  );
  A2X_CUDA_CHECK_ERROR(cudaGetLastError(), ErrorCode::eCudaThrustError);
  return ErrorCode::eSuccess;
}

template std::error_code nva2x::cuda::FillOnDevice<float>(float* destination, std::size_t size, float value, cudaStream_t cudaStream);
template std::error_code nva2x::cuda::FillOnDevice<int64_t>(int64_t* destination, std::size_t size, int64_t value, cudaStream_t cudaStream);
template std::error_code nva2x::cuda::FillOnDevice<uint64_t>(uint64_t* destination, std::size_t size, uint64_t value, cudaStream_t cudaStream);
template std::error_code nva2x::cuda::FillOnDevice<bool>(bool* destination, std::size_t size, bool value, cudaStream_t cudaStream);
