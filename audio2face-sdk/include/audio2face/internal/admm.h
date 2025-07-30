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
#pragma once

#include <cuda_runtime_api.h>

#include <cublas_v2.h>

#include <system_error>

namespace nva2f {

std::error_code admm_init(
    float *z, float *u, float *Atb,
    float *b_vec,
    float *amat, float *amat_inv,
    float *lower, float *upper,
    int batch_size, int num_poses, cublasHandle_t cublas_handle, cudaStream_t cuda_stream);

std::error_code admm_update(
    float *z, float *u,
    float *z_out, float *u_out,
    float *admmWeights, float *ATb, float *admmMatInvDevice,
    float *lower, float *upper,
    int batch_size, int num_poses, cudaStream_t cuda_stream);

} // namespace nva2f
