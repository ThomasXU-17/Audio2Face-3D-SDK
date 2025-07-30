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

#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2face/internal/admm.h"
#include "audio2face/error.h"
#include "audio2x/internal/tensor.h"
#include "audio2x/internal/nvtx_trace.h"
#include "audio2x/error.h"

namespace nva2f {

__global__ void ADMMupdateKernel(
    float* u_out, float* z_out,
    float* u, float* z,
    float* w, float* Atb, float *admm_mat_inv,
    float alpha, float* lower, float* upper,
    size_t batch_size, size_t num_poses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_poses)
    {
        int i = idx % num_poses;
        int batch_offset = idx - i;

        // Do the x variable update:
        // x = np.linalg.multi_dot((W.T, W, z - u)) + Atb
        float x_i = 0;
        for( int j=0; j < num_poses; j++ )
        {
            // rhs[idx] = w[i] * w[i] * (z[idx] - u[idx]) + Atb[idx];
            int jj = batch_offset + j;
            float rhs_j = w[j] * w[j] * (z[jj] - u[jj]) + Atb[jj];
            x_i += admm_mat_inv[j + num_poses * i] * rhs_j;
        }

        // Do the z variable update:
        // z = np.clip(u + x, lower, upper)
        z_out[idx] = u[idx] + x_i;
        if( z_out[idx] < lower[i])
        {
            z_out[idx] = lower[i];
        }
        if( z_out[idx] > upper[i])
        {
            z_out[idx] = upper[i];
        }

        // Do the u variable update:
        // u = u + alpha * (x - z)
        u_out[idx] = u[idx] + alpha * (x_i - z_out[idx]);
    }
}

// z = np.clip(z, lower, upper)
__global__ void ClipKernel(
    float* z, float* lower, float* upper, size_t batch_size, size_t num_poses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_poses)
    {
        int batch_relative_idx = idx % num_poses;
        if( z[idx] < lower[batch_relative_idx])
        {
            z[idx] = lower[batch_relative_idx];
        }
        if( z[idx] > upper[batch_relative_idx])
        {
            z[idx] = upper[batch_relative_idx];
        }
    }
}

std::error_code admm_init(
    float *z, float *u, float *Atb,
    float *b_vec,
    float *amat, float *amat_inv,
    float *lower, float *upper,
    int batch_size, int num_poses, cublasHandle_t cublas_handle, cudaStream_t cuda_stream)
{
    NVTX_TRACE("admm_init");

    // Atb = np.dot(A.T, b)
    float cu_alpha = 1;
    float cu_beta = 0;
    CUBLAS_CHECK_ERROR(
        cublasSgemm(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            num_poses, batch_size, num_poses,
            &cu_alpha,
            amat, num_poses,
            b_vec, num_poses,
            &cu_beta,
            Atb, num_poses
        ),
        ErrorCode::eCublasMatMatFailed
    );
    CHECK_NO_ERROR(nva2x::FillOnDevice(nva2x::GetDeviceTensorFloatView(u, num_poses * batch_size), 0.0f, cuda_stream));

    // solve A z = b:
    CUBLAS_CHECK_ERROR(
        cublasSgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            num_poses, batch_size, num_poses,
            &cu_alpha,
            amat_inv, num_poses,
            b_vec, num_poses,
            &cu_beta,
            z, num_poses
        ),
        ErrorCode::eCublasMatMatFailed
    );

    // project constraints:
    dim3 numBlocks(IDIVUP(static_cast<unsigned int>(batch_size * num_poses), 1024u));
    dim3 numThreads(1024u);
    ClipKernel<<<numBlocks, numThreads, 0, cuda_stream>>>(
        z, lower, upper, batch_size, num_poses);
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    return nva2x::ErrorCode::eSuccess;
}

std::error_code admm_update(
    float *z, float *u,
    float *z_out, float *u_out,
    float *admmWeights, float *ATb, float *admmMatInvDevice,
    float *lower, float *upper,
    int batch_size, int num_poses, cudaStream_t cuda_stream)
{
    NVTX_TRACE("admm_update");
    dim3 numBlocks(IDIVUP(static_cast<unsigned int>(batch_size * num_poses), 1024u));
    dim3 numThreads(1024u);
    const float alpha = 1.9f;

    ADMMupdateKernel<<<numBlocks, numThreads, 0, cuda_stream>>>(
        u_out, z_out,
        u, z,
        admmWeights, ATb, admmMatInvDevice,
        alpha, lower, upper,
        batch_size, num_poses
    );
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f
