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
#include "test_core_batch_eyes_animator_cuda.h"

#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"

#include <cassert>

#include <cuda_runtime_api.h>


namespace {


__global__ void ComputeEyesRotationKernel(
  float* outputEyesRotation, std::size_t outputEyesRotationOffset, std::size_t outputEyesRotationStride,
  const float* inputEyesRotationResult, std::size_t inputEyesRotationResultOffset, std::size_t inputEyesRotationResultStride,
  const float* params, std::size_t paramsStride,
  const float* saccadeRot, std::size_t saccadeRotSize,
  float dt,
  float* liveTime,
  std::size_t nbTracks
 ) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto trackIdx = idx / 6;
  if (trackIdx >= nbTracks) {
    return;
  }

  const auto output = outputEyesRotation + trackIdx * outputEyesRotationStride + outputEyesRotationOffset;

  const auto elementIdx = idx % 6;
  const auto eyeIdx = elementIdx / 3;
  const auto axisIdx = elementIdx % 3;

  float value;
  if (axisIdx < 2) {
    const auto inputEyes = inputEyesRotationResult + trackIdx * inputEyesRotationResultStride + inputEyesRotationResultOffset;
    const auto input = inputEyes[eyeIdx*2 + axisIdx];

    const auto eyeParams = params + trackIdx * paramsStride;
    const auto eyeballsStrength = eyeParams[0];
    const auto saccadeStrength = eyeParams[1];
    const auto offsets = eyeParams + 2;
    const auto eyeOffset = offsets[eyeIdx*2 + axisIdx];
    const auto saccadeSeed = eyeParams[6];

    auto& outLiveTime = liveTime[trackIdx];

    const auto maxNbFrames = saccadeRotSize / 2;
    auto wrap = [max=static_cast<float>(maxNbFrames)](float value) {
      float result = fmod(value, max);
      if (result < 0.0f) {
        result += max;
      }
      return result;
    };

    // Compute frame index from live time and seed
    float inLiveTime = outLiveTime;
    float totalTime = saccadeSeed + inLiveTime;
    totalTime = wrap(totalTime);
    const auto frameIdx = static_cast<int>(totalTime);
    assert(frameIdx >= 0);
    assert(frameIdx < maxNbFrames);
    const float saccade = saccadeRot[2 * frameIdx + axisIdx];

    // Using explicit intrinsics to prevent fma and get the same results as the CPU version
    value = eyeOffset + __fmul_rn(input, eyeballsStrength) + __fmul_rn(saccadeStrength, saccade);

    if (elementIdx == 0) {
      // Increment live time
      // Assume the saccade information was meant for 30 FPS, otherwise there would be more
      // movement with a higher FPS.
      constexpr float fps = 30.0f;
      const float increment = dt * fps;

      // Increment live time and wrap so it doesn't grow indefinitely
      inLiveTime += increment;
      inLiveTime = wrap(inLiveTime);
      outLiveTime = inLiveTime;
    }
  }
  else {
    assert(axisIdx == 2);
    value = 0.0f;
  }

  output[elementIdx] = value;
}


__global__ void ComputeEyesRotationEmptyKernel(
  float* outputEyesRotation, std::size_t outputEyesRotationOffset, std::size_t outputEyesRotationStride,
  const float* inputEyesRotationResult, std::size_t inputEyesRotationResultOffset, std::size_t inputEyesRotationResultStride,
  const float* params, std::size_t paramsStride,
  const float* saccadeRot, std::size_t saccadeRotSize,
  float dt,
  float* liveTime,
  std::size_t nbTracks
 ) {
}


} // Anonymous namespace.


std::error_code test::ComputeEyesRotation(
    float* outputEyesRotation, std::size_t outputEyesRotationOffset, std::size_t outputEyesRotationStride,
    const float* inputEyesRotationResult, std::size_t inputEyesRotationResultOffset, std::size_t inputEyesRotationResultStride,
    const float* params, std::size_t paramsStride,
    const float* saccadeRot, std::size_t saccadeRotSize,
    float dt,
    float* liveTime,
    std::size_t nbTracks,
    cudaStream_t cudaStream
) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(nbTracks * 6), 1024u));
  dim3 numThreads(1024u);

  ComputeEyesRotationKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
    outputEyesRotation, outputEyesRotationOffset, outputEyesRotationStride,
    inputEyesRotationResult, inputEyesRotationResultOffset, inputEyesRotationResultStride,
    params, paramsStride,
    saccadeRot, saccadeRotSize,
    dt,
    liveTime,
    nbTracks
    );
  A2F_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::ComputeEyesRotationEmpty(
    float* outputEyesRotation, std::size_t outputEyesRotationOffset, std::size_t outputEyesRotationStride,
    const float* inputEyesRotationResult, std::size_t inputEyesRotationResultOffset, std::size_t inputEyesRotationResultStride,
    const float* params, std::size_t paramsStride,
    const float* saccadeRot, std::size_t saccadeRotSize,
    float dt,
    float* liveTime,
    std::size_t nbTracks,
    cudaStream_t cudaStream
) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(nbTracks * 6), 1024u));
  dim3 numThreads(1024u);

  ComputeEyesRotationEmptyKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
    outputEyesRotation, outputEyesRotationOffset, outputEyesRotationStride,
    inputEyesRotationResult, inputEyesRotationResultOffset, inputEyesRotationResultStride,
    params, paramsStride,
    saccadeRot, saccadeRotSize,
    dt,
    liveTime,
    nbTracks
    );
  A2F_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}
