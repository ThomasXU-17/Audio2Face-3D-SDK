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

#include "audio2face/internal/mask_extraction.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime_api.h>



namespace nva2f
{

// Kernel to copy elements from src to dst based on indices
__global__ void copyIndicesKernel(float* dst, const float* src, const int* indices, int numIndices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numIndices) {
        int srcIdx = indices[idx];
        dst[idx] = src[srcIdx];
    }
}

std::error_code CopyIndices(float* dst, const float* src, const int* indices, int numIndices, cudaStream_t stream) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numIndices + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    copyIndicesKernel<<<blocksPerGrid, threadsPerBlock, 0,stream>>>(dst,src, indices, numIndices);

    // Check for any errors launching the kernel
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);
    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f
