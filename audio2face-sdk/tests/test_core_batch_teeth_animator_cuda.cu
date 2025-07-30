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
#include "test_core_batch_teeth_animator_cuda.h"

#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"

#include <cassert>

#include <cuda_runtime_api.h>

#include "tbtSVD/SVD.h"


namespace {


__global__ void ComputeRigidXformKernel(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t nbPoints,
  std::size_t nbTracks
 ) {
  const auto track_idx = blockIdx.x;
  // Keep 1 thread per block.
  if (threadIdx.x != 0) {
    return;
  }

  assert(nbPoints == 5);

  // Compute the pose of the jaw.
  const auto lowerTeethStrength = params[track_idx * paramsStride + 0];
  const auto lowerTeethHeightOffset = params[track_idx * paramsStride + 1];
  const auto lowerTeethDepthOffset = params[track_idx * paramsStride + 2];

  inputDeltas += track_idx * inputDeltasStride + inputDeltasOffset;
  __shared__ float pose[5*3];
  for (std::size_t i = 0; i < nbPoints; ++i) {
    pose[i*3 + 0] = neutralPose[i*3 + 0] + inputDeltas[i*3 + 0] * lowerTeethStrength;
    pose[i*3 + 1] = neutralPose[i*3 + 1] + inputDeltas[i*3 + 1] * lowerTeethStrength + lowerTeethHeightOffset;
    pose[i*3 + 2] = neutralPose[i*3 + 2] + inputDeltas[i*3 + 2] * lowerTeethStrength + lowerTeethDepthOffset;
  }

  __shared__ float a_mean[3];
  __shared__ float a_delta[5*3];
  for (int component_idx = 0; component_idx < 3; ++component_idx) {
    float average = pose[component_idx];
    for (std::size_t vertex_idx = 1; vertex_idx < nbPoints; ++vertex_idx) {
      average += pose[vertex_idx * 3 + component_idx];
    }
    average /= nbPoints;
    a_mean[component_idx] = average;
    for (std::size_t vertex_idx = 0; vertex_idx < nbPoints; ++vertex_idx) {
      a_delta[vertex_idx * 3 + component_idx] = pose[vertex_idx * 3 + component_idx] - average;
    }
  }

  // OPTME: This is from the neutral pase, and should be precomputed.
  __shared__ float b_mean[3];
  __shared__ float b_delta[5*3];
  for (int component_idx = 0; component_idx < 3; ++component_idx) {
    float average = neutralPose[component_idx];
    for (std::size_t vertex_idx = 1; vertex_idx < nbPoints; ++vertex_idx) {
      average += neutralPose[vertex_idx * 3 + component_idx];
    }
    average /= nbPoints;
    b_mean[component_idx] = average;
    for (std::size_t vertex_idx = 0; vertex_idx < nbPoints; ++vertex_idx) {
      b_delta[vertex_idx * 3 + component_idx] = neutralPose[vertex_idx * 3 + component_idx] - average;
    }
  }

  __shared__ float H[9];
  for (int row_idx = 0; row_idx < 3; ++row_idx) {
    for (int col_idx = 0; col_idx < 3; ++col_idx) {
      float value = 0;
      for (std::size_t vertex_idx = 0; vertex_idx < nbPoints; ++vertex_idx) {
        value += b_delta[vertex_idx * 3 + row_idx] * a_delta[vertex_idx * 3 + col_idx];
      }
      // Store matrix in row major since it will help with dot products below.
      H[row_idx * 3 + col_idx] = value;
    }
  }

  // Compute SVD.
  __shared__ float U[9];
  __shared__ float V[9];
  const auto A = SVD::Mat3x3::fromPtr(H, 0, 1);
  const auto usv = SVD::svd(A);
  usv.U.toPtr(U, 0, 1);
  usv.V.toPtr(V, 0, 1);

  // Re-use H to store rotation results.
  for (int row_idx = 0; row_idx < 3; ++row_idx) {
    for (int col_idx = 0; col_idx < 3; ++col_idx) {
      const auto u = U + 3 * col_idx;
      const auto v = V + 3 * row_idx;
      H[col_idx * 3 + row_idx] = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
    }
  }

  // Compute determinant of H
  const float det = H[0*3 + 0] * (H[1*3 + 1] * H[2*3 + 2] - H[1*3 + 2] * H[2*3 + 1])
                  - H[0*3 + 1] * (H[1*3 + 0] * H[2*3 + 2] - H[1*3 + 2] * H[2*3 + 0])
                  + H[0*3 + 2] * (H[1*3 + 0] * H[2*3 + 1] - H[1*3 + 1] * H[2*3 + 0]);
  H[3*2 + 0] /= det;
  H[3*2 + 1] /= det;
  H[3*2 + 2] /= det;

  __shared__ float tt[3];
  tt[0] = a_mean[0] - (b_mean[0] * H[3*0 + 0] + b_mean[1] * H[3*1 + 0] + b_mean[2] * H[3*2 + 0]);
  tt[1] = a_mean[1] - (b_mean[0] * H[3*0 + 1] + b_mean[1] * H[3*1 + 1] + b_mean[2] * H[3*2 + 1]);
  tt[2] = a_mean[2] - (b_mean[0] * H[3*0 + 2] + b_mean[1] * H[3*1 + 2] + b_mean[2] * H[3*2 + 2]);

  // Store the result.
  const auto output = result + track_idx * resultStride + resultOffset;
  output[0*4 + 0] = H[0*3 + 0];
  output[0*4 + 1] = H[0*3 + 1];
  output[0*4 + 2] = H[0*3 + 2];
  output[0*4 + 3] = 0;
  output[1*4 + 0] = H[1*3 + 0];
  output[1*4 + 1] = H[1*3 + 1];
  output[1*4 + 2] = H[1*3 + 2];
  output[1*4 + 3] = 0;
  output[2*4 + 0] = H[2*3 + 0];
  output[2*4 + 1] = H[2*3 + 1];
  output[2*4 + 2] = H[2*3 + 2];
  output[2*4 + 3] = 0;
  output[3*4 + 0] = tt[0];
  output[3*4 + 1] = tt[1];
  output[3*4 + 2] = tt[2];
  output[3*4 + 3] = 1;
}


__global__ void ComputeRigidXformKernelLocal(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t nbPoints,
  std::size_t nbTracks
 ) {
  const auto track_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (track_idx >= nbTracks) {
    return;
  }

  assert(nbPoints == 5);

  // Compute the pose of the jaw.
  const auto lowerTeethStrength = params[track_idx * paramsStride + 0];
  const auto lowerTeethHeightOffset = params[track_idx * paramsStride + 1];
  const auto lowerTeethDepthOffset = params[track_idx * paramsStride + 2];

  inputDeltas += track_idx * inputDeltasStride + inputDeltasOffset;
  float pose[5*3];
  for (std::size_t i = 0; i < nbPoints; ++i) {
    pose[i*3 + 0] = neutralPose[i*3 + 0] + inputDeltas[i*3 + 0] * lowerTeethStrength;
    pose[i*3 + 1] = neutralPose[i*3 + 1] + inputDeltas[i*3 + 1] * lowerTeethStrength + lowerTeethHeightOffset;
    pose[i*3 + 2] = neutralPose[i*3 + 2] + inputDeltas[i*3 + 2] * lowerTeethStrength + lowerTeethDepthOffset;
  }

  float a_mean[3];
  float a_delta[5*3];
  for (int component_idx = 0; component_idx < 3; ++component_idx) {
    float average = pose[component_idx];
    for (std::size_t vertex_idx = 1; vertex_idx < nbPoints; ++vertex_idx) {
      average += pose[vertex_idx * 3 + component_idx];
    }
    average /= nbPoints;
    a_mean[component_idx] = average;
    for (std::size_t vertex_idx = 0; vertex_idx < nbPoints; ++vertex_idx) {
      a_delta[vertex_idx * 3 + component_idx] = pose[vertex_idx * 3 + component_idx] - average;
    }
  }

  // OPTME: This is from the neutral pase, and should be precomputed.
  float b_mean[3];
  float b_delta[5*3];
  for (int component_idx = 0; component_idx < 3; ++component_idx) {
    float average = neutralPose[component_idx];
    for (std::size_t vertex_idx = 1; vertex_idx < nbPoints; ++vertex_idx) {
      average += neutralPose[vertex_idx * 3 + component_idx];
    }
    average /= nbPoints;
    b_mean[component_idx] = average;
    for (std::size_t vertex_idx = 0; vertex_idx < nbPoints; ++vertex_idx) {
      b_delta[vertex_idx * 3 + component_idx] = neutralPose[vertex_idx * 3 + component_idx] - average;
    }
  }

  float H[9];
  for (int row_idx = 0; row_idx < 3; ++row_idx) {
    for (int col_idx = 0; col_idx < 3; ++col_idx) {
      float value = 0;
      for (std::size_t vertex_idx = 0; vertex_idx < nbPoints; ++vertex_idx) {
        value += b_delta[vertex_idx * 3 + row_idx] * a_delta[vertex_idx * 3 + col_idx];
      }
      // Store matrix in row major since it will help with dot products below.
      H[row_idx * 3 + col_idx] = value;
    }
  }

  // Compute SVD.
  float U[9];
  float V[9];
  const auto A = SVD::Mat3x3::fromPtr(H, 0, 1);
  const auto usv = SVD::svd(A);
  usv.U.toPtr(U, 0, 1);
  usv.V.toPtr(V, 0, 1);

  // Re-use H to store rotation results.
  for (int row_idx = 0; row_idx < 3; ++row_idx) {
    for (int col_idx = 0; col_idx < 3; ++col_idx) {
      const auto u = U + 3 * col_idx;
      const auto v = V + 3 * row_idx;
      H[col_idx * 3 + row_idx] = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
    }
  }

  // Compute determinant of H
  const float det = H[0*3 + 0] * (H[1*3 + 1] * H[2*3 + 2] - H[1*3 + 2] * H[2*3 + 1])
                  - H[0*3 + 1] * (H[1*3 + 0] * H[2*3 + 2] - H[1*3 + 2] * H[2*3 + 0])
                  + H[0*3 + 2] * (H[1*3 + 0] * H[2*3 + 1] - H[1*3 + 1] * H[2*3 + 0]);
  H[3*2 + 0] /= det;
  H[3*2 + 1] /= det;
  H[3*2 + 2] /= det;

  float tt[3];
  tt[0] = a_mean[0] - (b_mean[0] * H[3*0 + 0] + b_mean[1] * H[3*1 + 0] + b_mean[2] * H[3*2 + 0]);
  tt[1] = a_mean[1] - (b_mean[0] * H[3*0 + 1] + b_mean[1] * H[3*1 + 1] + b_mean[2] * H[3*2 + 1]);
  tt[2] = a_mean[2] - (b_mean[0] * H[3*0 + 2] + b_mean[1] * H[3*1 + 2] + b_mean[2] * H[3*2 + 2]);

  // Store the result.
  const auto output = result + track_idx * resultStride + resultOffset;
  output[0*4 + 0] = H[0*3 + 0];
  output[0*4 + 1] = H[0*3 + 1];
  output[0*4 + 2] = H[0*3 + 2];
  output[0*4 + 3] = 0;
  output[1*4 + 0] = H[1*3 + 0];
  output[1*4 + 1] = H[1*3 + 1];
  output[1*4 + 2] = H[1*3 + 2];
  output[1*4 + 3] = 0;
  output[2*4 + 0] = H[2*3 + 0];
  output[2*4 + 1] = H[2*3 + 1];
  output[2*4 + 2] = H[2*3 + 2];
  output[2*4 + 3] = 0;
  output[3*4 + 0] = tt[0];
  output[3*4 + 1] = tt[1];
  output[3*4 + 2] = tt[2];
  output[3*4 + 3] = 1;
}


__global__ void ComputeRigidXformKernelParallel(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t nbPoints,
  std::size_t nbTracks
 ) {
  assert(warpSize == test::kExpectedWarpSize);
  const int totalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t trackIdx = totalIdx / test::kExpectedWarpSize;
  if (trackIdx >= nbTracks) {
    return;
  }
  const std::size_t idx = totalIdx % test::kExpectedWarpSize;

  const float* inputDeltasPtr = inputDeltas + trackIdx * inputDeltasStride + inputDeltasOffset;

  // Compute the current pose of the jaw.
  assert(nbPoints * 3 <= test::kExpectedWarpSize);
  const float* paramsPtr = params + trackIdx * paramsStride;
  const float lowerTeethStrength = paramsPtr[0];
  const float* lowerTeethOffsets = paramsPtr + 1;
  float targetPose = 0.0f;
  if (idx < nbPoints * 3) {
    targetPose = neutralPose[idx] + inputDeltasPtr[idx] * lowerTeethStrength + lowerTeethOffsets[idx % 3];
  }

  // Compute the average of the pose.
  constexpr const int kReductionStartOffset = 4;
  constexpr const unsigned int kMask = 0xffffffff;
  assert(kReductionStartOffset * 2 >= nbPoints);
  // After this computation, targetPoseMean will contain the average of the pose,
  // where the x value is in thread 0, y value is in thread 1, and z value is in thread 2.
  float targetPoseMean = targetPose;
  for (int offset = kReductionStartOffset; offset > 0; offset /= 2) {
    targetPoseMean += __shfl_down_sync(kMask, targetPoseMean, offset * 3);
  }
  targetPoseMean /= nbPoints;

  // Broadcast the pose mean to all threads.
  targetPoseMean = __shfl_sync(kMask, targetPoseMean, idx % 3);

  // Compute the delta of the pose.
  const float targetPoseDelta = targetPose - targetPoseMean;

  // The mean of the neutral pose is stored at the end of the neutral pose array.
  const float* neutralPoseMeanPtr = neutralPose + nbPoints * 3;
  const float sourcePoseMean = neutralPoseMeanPtr[idx % 3];
  float sourcePoseDelta = 0.0f;
  if (idx < nbPoints * 3) {
    sourcePoseDelta = neutralPose[idx] - sourcePoseMean;
  }

  // Compute the 3x3 H matrix.
  float h = 0.0f;
  // The H matrix is stored in row major order since it will help with dot products below.
  auto readH = [&h](int row, int col) {
    return __shfl_sync(kMask, h, row * 3 + col);
  };
  {
    const auto rowIdx = idx / 3;
    const auto colIdx = idx % 3;
    for (std::size_t vertexIdx = 0; vertexIdx < nbPoints; ++vertexIdx) {
      const auto targetDelta = __shfl_sync(kMask, targetPoseDelta, vertexIdx * 3 + colIdx);
      const auto sourceDelta = __shfl_sync(kMask, sourcePoseDelta, vertexIdx * 3 + rowIdx);
      h += targetDelta * sourceDelta;
    }

    // Compute the SVD.
    float H[9];
    for (int i = 0; i < 9; ++i) {
      H[i] = __shfl_sync(kMask, h, i);
    }
    float U[9];
    float V[9];
    const auto A = SVD::Mat3x3::fromPtr(H, 0, 1);
    const auto usv = SVD::svd(A);
    usv.U.toPtr(U, 0, 1);
    usv.V.toPtr(V, 0, 1);

    // Re-use h to store rotation results.
    const auto u = U + 3 * colIdx;
    const auto v = V + 3 * rowIdx;
    h = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];

    // Compute the determinant of H.
    const auto det = readH(0,0) * (readH(1,1) * readH(2,2) - readH(1,2) * readH(2,1))
                   - readH(0,1) * (readH(1,0) * readH(2,2) - readH(1,2) * readH(2,0))
                   + readH(0,2) * (readH(1,0) * readH(2,1) - readH(1,1) * readH(2,0));

    // Only divide the last 3 elements.
    if (idx >= 6) {
      h /= det;
    }
  }

  // Store output, output matrix is column major.
  const auto output = result + trackIdx * resultStride + resultOffset;

  // Store final rotation.
  if (idx < 9) {
    const auto rowIdx = idx / 3;
    const auto colIdx = idx % 3;
    output[colIdx*4 + rowIdx] = h;
  }

  // Store final translation.
  const float sourceMean[3] = {
    __shfl_sync(kMask, sourcePoseMean, 0),
    __shfl_sync(kMask, sourcePoseMean, 1),
    __shfl_sync(kMask, sourcePoseMean, 2),
  };
  const float translation = targetPoseMean -
    (sourceMean[0] * readH(idx,0) + sourceMean[1] * readH(idx,1) + sourceMean[2] * readH(idx,2));
  if (idx < 3) {
    output[3*4 + idx] = translation;
    output[idx*4 + 3] = 0.0f;
  }
  if (idx < 1) {
    output[3*4 + 3] = 1.0f;
  }
}


__global__ void ComputeRigidXformKernelEmpty(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t nbPoints,
  std::size_t nbTracks
 ) {
}


} // Anonymous namespace.


std::error_code test::ComputeRigidXform(
    float* result, std::size_t resultOffset, std::size_t resultStride,
    const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
    const float* neutralPose,
    const float* params, std::size_t paramsStride,
    std::size_t nbPoints,
    std::size_t nbTracks,
    cudaStream_t cudaStream
) {
  dim3 numBlocks(static_cast<unsigned int>(nbTracks));
  dim3 numThreads(32u);

  ComputeRigidXformKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
    result, resultOffset, resultStride,
    inputDeltas, inputDeltasOffset, inputDeltasStride,
    neutralPose,
    params, paramsStride,
    nbPoints,
    nbTracks
    );
  A2F_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::ComputeRigidXformLocal(
    float* result, std::size_t resultOffset, std::size_t resultStride,
    const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
    const float* neutralPose,
    const float* params, std::size_t paramsStride,
    std::size_t nbPoints,
    std::size_t nbTracks,
    cudaStream_t cudaStream
) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(nbTracks), 32u));
  dim3 numThreads(32u);

  ComputeRigidXformKernelLocal<<<numBlocks, numThreads, 0, cudaStream>>>(
    result, resultOffset, resultStride,
    inputDeltas, inputDeltasOffset, inputDeltasStride,
    neutralPose,
    params, paramsStride,
    nbPoints,
    nbTracks
    );
  A2F_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::ComputeRigidXformParallel(
    float* result, std::size_t resultOffset, std::size_t resultStride,
    const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
    const float* neutralPose,
    const float* params, std::size_t paramsStride,
    std::size_t nbPoints,
    std::size_t nbTracks,
    cudaStream_t cudaStream
) {
  static constexpr const unsigned int kNumThreadsPerBlock = kExpectedWarpSize * 1u;
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(nbTracks) * kExpectedWarpSize, kNumThreadsPerBlock));
  dim3 numThreads(kNumThreadsPerBlock);

  ComputeRigidXformKernelParallel<<<numBlocks, numThreads, 0, cudaStream>>>(
    result, resultOffset, resultStride,
    inputDeltas, inputDeltasOffset, inputDeltasStride,
    neutralPose,
    params, paramsStride,
    nbPoints,
    nbTracks
    );
  A2F_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::ComputeRigidXformEmpty(
    float* result, std::size_t resultOffset, std::size_t resultStride,
    const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
    const float* neutralPose,
    const float* params, std::size_t paramsStride,
    std::size_t nbPoints,
    std::size_t nbTracks,
    cudaStream_t cudaStream
) {
  dim3 numBlocks(static_cast<unsigned int>(nbTracks));
  dim3 numThreads(32u);

  ComputeRigidXformKernelEmpty<<<numBlocks, numThreads, 0, cudaStream>>>(
    result, resultOffset, resultStride,
    inputDeltas, inputDeltasOffset, inputDeltasStride,
    neutralPose,
    params, paramsStride,
    nbPoints,
    nbTracks
    );
  A2F_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}
