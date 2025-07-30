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
#include "test_core_batch_tongue_animator_cuda.h"

#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"

#include <cassert>

#include <cuda_runtime_api.h>


namespace {

__global__ void AnimatorTongueAnimateBatched(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  float tongueStrength, float tongueHeightOffset, float tongueDepthOffset,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < poseSize * nbTracks) {
    const std::size_t trackIdx = idx / poseSize;
    const std::size_t elemIdx = idx % poseSize;

    float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
    const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

    const float neutralPoseValue = neutralPose[elemIdx];

    result = neutralPoseValue + inputDelta * tongueStrength;
    if (elemIdx % 3 == 1) {
      result += tongueHeightOffset;
    }
    if (elemIdx % 3 == 2) {
      result += tongueDepthOffset;
    }
  }
}

__global__ void AnimatorTongueAnimateBatchedParamsPacked(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < poseSize * nbTracks) {
    const std::size_t trackIdx = idx / poseSize;
    const std::size_t elemIdx = idx % poseSize;

    float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
    const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

    const float neutralPoseValue = neutralPose[elemIdx];

    const float tongueStrength = params[trackIdx * paramsStride + 0];
    const float tongueHeightOffset = params[trackIdx * paramsStride + 1];
    const float tongueDepthOffset = params[trackIdx * paramsStride + 2];

    result = neutralPoseValue + inputDelta * tongueStrength;
    if (elemIdx % 3 == 1) {
      result += tongueHeightOffset;
    }
    if (elemIdx % 3 == 2) {
      result += tongueDepthOffset;
    }
  }
}

__global__ void AnimatorTongueAnimateBatchedParamsPackedVertices(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t numVertices, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numVertices * nbTracks) {
    const std::size_t trackIdx = idx / numVertices;
    const std::size_t elemIdx = idx % numVertices;

    float* result = results + trackIdx * resultsStride + resultsOffset + elemIdx * 3;
    const float* inputDelta = inputDeltas + trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx * 3;

    const float* neutralPoseValue = neutralPose + elemIdx * 3;

    const float tongueStrength = params[trackIdx * paramsStride + 0];
    const float tongueHeightOffset = params[trackIdx * paramsStride + 1];
    const float tongueDepthOffset = params[trackIdx * paramsStride + 2];

    result[0] = neutralPoseValue[0] + inputDelta[0] * tongueStrength;
    result[1] = neutralPoseValue[1] + inputDelta[1] * tongueStrength + tongueHeightOffset;
    result[2] = neutralPoseValue[2] + inputDelta[2] * tongueStrength + tongueDepthOffset;
  }
}

__global__ void AnimatorTongueAnimateBatchedParamsPackedFull(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < poseSize * nbTracks) {
    const std::size_t trackIdx = idx / poseSize;
    const std::size_t elemIdx = idx % poseSize;

    float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
    const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

    const float neutralPoseValue = neutralPose[elemIdx];

    const float tongueStrength = params[trackIdx * paramsStride + 3];
    const float offset = params[trackIdx * paramsStride + elemIdx % 3];

    result = neutralPoseValue + inputDelta * tongueStrength + offset;
  }
}

} // Anonymous namespace.


std::error_code test::AnimateBatched(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  float tongueStrength, float tongueHeightOffset, float tongueDepthOffset,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorTongueAnimateBatched<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        neutralPose,
        tongueStrength, tongueHeightOffset, tongueDepthOffset,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::AnimateBatchedParamsPacked(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorTongueAnimateBatchedParamsPacked<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        neutralPose,
        params, paramsStride,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::AnimateBatchedParamsPackedVertices(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  A2F_CHECK_ERROR_WITH_MSG(poseSize % 3 == 0, "Invalid pose size", nva2x::ErrorCode::eInvalidValue);

  const auto numVertices = poseSize / 3;

  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(numVertices * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorTongueAnimateBatchedParamsPackedVertices<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        neutralPose,
        params, paramsStride,
        numVertices, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::AnimateBatchedParamsPackedFull(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorTongueAnimateBatchedParamsPackedFull<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        neutralPose,
        params, paramsStride,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}
