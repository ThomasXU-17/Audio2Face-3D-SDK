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
#include "audio2x/error.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime_api.h>

namespace nva2f {

// helper kernels that are used by the pure gpu solver

__global__ void setUpperByCancelPairs(float* upper, float* weights, int* first, int* second, int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num) {
        int i = first[idx];
        int j = second[idx];
        upper[weights[i] >= weights[j] ? j : i] = 1e-10f;
    }
}

std::error_code SetUpperByCancelPairs(float* upper, float* weights, int* first, int* second, int num, cudaStream_t stream) {
    int threadsPerBlock = num;
    int blocksPerGrid = 1;

    // Launch the kernel
    setUpperByCancelPairs<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(upper, weights, first, second, num);

    // Check for any errors launching the kernel
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);
    return nva2x::ErrorCode::eSuccess;
}

__global__ void unmapActiveBlendshapes(float* fullWeights, float* solvedWeights, int* activeShapeMap, int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num) {
        fullWeights[activeShapeMap[idx]] = solvedWeights[idx];
    }
}

std::error_code UnmapActiveBlendshapes(float* fullWeights, float* solvedWeights, int* activeShapeMap, int num, cudaStream_t stream) {
    int threadsPerBlock = num;
    int blocksPerGrid = 1;

    // Launch the kernel
    unmapActiveBlendshapes<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(fullWeights, solvedWeights, activeShapeMap, num);

    // Check for any errors launching the kernel
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);
    return nva2x::ErrorCode::eSuccess;
}

// Kernel to apply blendshape multipliers and offsets
__global__ void applyMultipliersAndOffsets(float* weights, const float* multipliers, const float* offsets, int numWeights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWeights) {
        weights[idx] = weights[idx] * multipliers[idx] + offsets[idx];
    }
}

std::error_code ApplyBlendshapeMultiplersAndOffsets(float* weights, const float* multipliers, const float* offsets, int numWeights, cudaStream_t stream) {
    int threadsPerBlock = numWeights;
    int blocksPerGrid = 1;

    // Launch the kernel
    applyMultipliersAndOffsets<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(weights, multipliers, offsets, numWeights);

    // Check for any errors launching the kernel
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);
    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f
