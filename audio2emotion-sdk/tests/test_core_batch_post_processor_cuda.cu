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
#include "test_core_batch_post_processor_cuda.h"

#include "audio2emotion/internal/logger.h"
#include "audio2emotion/internal/macros.h"
#include "audio2x/error.h"

#include <cassert>

#include <cuda_runtime_api.h>

#include <cuda/std/limits>


namespace {

// This kernel uses one thread per track, it processes the arrays very naively,
// which is fine since the arrays are small.
// It also does everything in global memory, which could be optimized.
__global__ void PostProcessGPUKernel(
  float* outputEmotions, std::size_t outputEmotionsOffset, std::size_t outputEmotionsStride, std::size_t outputEmotionsSize,
  const float* inputEmotions, std::size_t inputEmotionsOffset, std::size_t inputEmotionsStride, std::size_t inputEmotionsSize,
  const std::int64_t* emotionCorrespondence,
  const float* postProcessParams, std::size_t postProcessParamsStride,
  const float* preferredEmotions,
  float* stateAndWorkBuffers, std::size_t stateAndWorkBuffersStride,
  std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nbTracks) {
    return;
  }

  const std::size_t trackIdx = idx;

  float* output = outputEmotions + trackIdx * outputEmotionsStride + outputEmotionsOffset;
  const float* input = inputEmotions + trackIdx * inputEmotionsStride + inputEmotionsOffset;
  const float* params = postProcessParams + trackIdx * postProcessParamsStride;
  float* stateAndWorkBuffer = stateAndWorkBuffers + trackIdx * stateAndWorkBuffersStride;

  float* a2eEmotions = stateAndWorkBuffer + 1;
  float* a2fEmotions = stateAndWorkBuffer + 1 + inputEmotionsSize;
  float* prevEmotion = stateAndWorkBuffer + 1 + inputEmotionsSize + outputEmotionsSize;
  float* prevBlendedEmotion = stateAndWorkBuffer + 1 + inputEmotionsSize + outputEmotionsSize * 2;

  //
  // Initialization.
  //
  const bool firstFrame = stateAndWorkBuffer[0] > 0.0f;
  const float* initialEmotion = params + 8;
  if (firstFrame) {
    for (std::size_t i = 0; i < outputEmotionsSize; ++i) {
      a2fEmotions[i] = initialEmotion[i];
    }
    stateAndWorkBuffer[0] = 0.0f;
  }

  //
  // Softmax and contrast.
  //
  const float emotionContrast = params[1];
  for (std::size_t i = 0; i < inputEmotionsSize; ++i) {
    a2eEmotions[i] = input[i] * emotionContrast;
  }

  float maxElem = a2eEmotions[0];
  for (std::size_t i = 1; i < inputEmotionsSize; ++i) {
    const float v = a2eEmotions[i];
    if (v > maxElem) {
      maxElem = v;
    }
  }

  float expSum = 0.0f;
  for (std::size_t i = 0; i < inputEmotionsSize; ++i) {
    const float v = std::exp(a2eEmotions[i] - maxElem);
    a2eEmotions[i] = v;
    expSum += v;
  }

  for (std::size_t i = 0; i < inputEmotionsSize; ++i) {
    a2eEmotions[i] /= expSum;
  }

  //
  // Nullify unmapped emotions.
  //
  for (std::size_t i = 0; i < inputEmotionsSize; ++i) {
    if (emotionCorrespondence[i] == -1) {
      a2eEmotions[i] = 0.0f;
    }
  }

  //
  // Apply max emotions.
  //
  const std::uint32_t maxEmotions = *reinterpret_cast<const std::uint32_t*>(&params[0]);
  const std::size_t emotionsToZero = inputEmotionsSize - maxEmotions;
  if (emotionsToZero > 0) {
    // Any value above 1.0f is fine.
    static constexpr float kMaxFloat = 2.0f;
    for (std::size_t k = 0; k < emotionsToZero; ++k) {
      float minElem = a2eEmotions[0];
      std::size_t minIndex = 0;
      for (std::size_t i = 1; i < inputEmotionsSize; ++i) {
        const float v = a2eEmotions[i];
        if (minElem == kMaxFloat || (v != kMaxFloat && v < minElem)) {
          minElem = v;
          minIndex = i;
        }
      }
      a2eEmotions[minIndex] = kMaxFloat;
    }
    for (std::size_t i = 0; i < inputEmotionsSize; ++i) {
      if (a2eEmotions[i] == kMaxFloat) {
        a2eEmotions[i] = 0.0f;
      }
    }
  }

  //
  // Apply emotion correspondence.
  //
  for (std::size_t i = 0; i < inputEmotionsSize; ++i) {
    const auto j = emotionCorrespondence[i];
    if (j == -1) {
      continue;
    }
    a2fEmotions[j] = a2eEmotions[i];
  }

  //
  // Blend with either the beginning emotion or the previous one.
  //
  const float liveBlendCoef = params[2];
  const float* blendSource = firstFrame ? initialEmotion : prevEmotion;
  for (std::size_t i = 0; i < outputEmotionsSize; ++i) {
    a2fEmotions[i] = (1.0f - liveBlendCoef) * a2fEmotions[i] + liveBlendCoef * blendSource[i];
    prevEmotion[i] = a2fEmotions[i];
  }

  //
  // Blend with preferred emotion.
  //
  const bool enablePreferredEmotion = params[3] > 0.0f;
  if (enablePreferredEmotion) {
    const float preferredEmotionStrength = params[4];
    const float* preferredEmotion = preferredEmotions + trackIdx * outputEmotionsSize;
    for (std::size_t i = 0; i < outputEmotionsSize; ++i) {
      a2fEmotions[i] = (1.0f - preferredEmotionStrength) * a2fEmotions[i] + preferredEmotionStrength * preferredEmotion[i];
    }
  }

  //
  // Apply transition.
  //
  if (!firstFrame) {
    const float liveTransitionTime = params[5];
    const float fixedDt = params[6];

    // ensure the transition time is at least 1e-3 seconds.
    const float transitionTime = max(liveTransitionTime, 1e-3f);
    // ensure w is at most 1.
    const float w = min(fixedDt / transitionTime, 1.0f);

    for (std::size_t i = 0; i < outputEmotionsSize; ++i) {
      a2fEmotions[i] = w * a2fEmotions[i] + (1.0f - w) * prevBlendedEmotion[i];
    }
  }
  for (std::size_t i = 0; i < outputEmotionsSize; ++i) {
    prevBlendedEmotion[i] = a2fEmotions[i];
  }

  //
  // Apply strength.
  //
  const float emotionStrength = params[7];
  for (std::size_t i = 0; i < outputEmotionsSize; ++i) {
    a2fEmotions[i] *= emotionStrength;
  }

  // Copy to output.
  //
  for (std::size_t i = 0; i < outputEmotionsSize; ++i) {
    output[i] = a2fEmotions[i];
  }
}


// This kernel tries to use shared memory to use parallelism when possible,
// while still using a single thread for reductions.
__global__ void PostProcessGPUKernelShared(
  float* outputEmotions, std::size_t outputEmotionsOffset, std::size_t outputEmotionsStride, std::size_t outputEmotionsSize,
  const float* inputEmotions, std::size_t inputEmotionsOffset, std::size_t inputEmotionsStride, std::size_t inputEmotionsSize,
  const std::int64_t* emotionCorrespondence,
  const float* postProcessParams, std::size_t postProcessParamsStride,
  const float* preferredEmotions,
  float* stateAndWorkBuffers, std::size_t stateAndWorkBuffersStride,
  std::size_t nbTracks
  ) {
  const std::size_t trackIdx = blockIdx.x;
  const std::size_t idx = threadIdx.x;

  float* output = outputEmotions + trackIdx * outputEmotionsStride + outputEmotionsOffset;
  const float* input = inputEmotions + trackIdx * inputEmotionsStride + inputEmotionsOffset;
  const float* params = postProcessParams + trackIdx * postProcessParamsStride;
  float* stateAndWorkBuffer = stateAndWorkBuffers + trackIdx * stateAndWorkBuffersStride;

  float* a2fEmotions = stateAndWorkBuffer + 1;
  float* prevEmotion = stateAndWorkBuffer + 1 + outputEmotionsSize;
  float* prevBlendedEmotion = stateAndWorkBuffer + 1 + outputEmotionsSize * 2;

  constexpr const std::size_t kA2EBufferSize = 6;
  constexpr const std::size_t kA2FBufferSize = 10;
  assert(kA2EBufferSize == inputEmotionsSize);
  assert(kA2FBufferSize == outputEmotionsSize);

  __shared__ float a2eEmotionsShared[kA2EBufferSize];
  __shared__ float a2fEmotionsShared[kA2FBufferSize];

  //
  // Initialization.
  //
  const bool firstFrame = stateAndWorkBuffer[0] > 0.0f;
  const float* initialEmotion = params + 8;
  if (idx < kA2FBufferSize) {
    if (firstFrame) {
      a2fEmotionsShared[idx] = initialEmotion[idx];
    }
    else {
      a2fEmotionsShared[idx] = a2fEmotions[idx];
    }
  }

  //
  // Softmax and contrast.
  //
  const float emotionContrast = params[1];
  if (idx < kA2EBufferSize) {
    a2eEmotionsShared[idx] = input[idx] * emotionContrast;
  }

  __syncthreads();

  if (idx == 0) {
    float maxElem = a2eEmotionsShared[0];
    for (std::size_t i = 1; i < kA2EBufferSize; ++i) {
      const float v = a2eEmotionsShared[i];
      if (v > maxElem) {
        maxElem = v;
      }
    }

    float expSum = 0.0f;
    for (std::size_t i = 0; i < kA2EBufferSize; ++i) {
      const float v = std::exp(a2eEmotionsShared[i] - maxElem);
      a2eEmotionsShared[i] = v;
      expSum += v;
    }

    for (std::size_t i = 0; i < kA2EBufferSize; ++i) {
      a2eEmotionsShared[i] /= expSum;
    }
  }

  __syncthreads();

  //
  // Nullify unmapped emotions.
  //
  if (idx < kA2EBufferSize) {
    if (emotionCorrespondence[idx] == -1) {
      a2eEmotionsShared[idx] = 0.0f;
    }
  }

  __syncthreads();

  //
  // Apply max emotions.
  //
  // Any value above 1.0f is fine because this is applied right after softmax.
  static constexpr float kMaxFloat = 2.0f;
  if (idx == 0) {
    const std::uint32_t maxEmotions = *reinterpret_cast<const std::uint32_t*>(&params[0]);
    const std::size_t emotionsToZero = inputEmotionsSize - maxEmotions;
    for (std::size_t k = 0; k < emotionsToZero; ++k) {
      float minElem = a2eEmotionsShared[0];
      std::size_t minIndex = 0;
      for (std::size_t i = 1; i < kA2EBufferSize; ++i) {
        const float v = a2eEmotionsShared[i];
        if (minElem == kMaxFloat || (v != kMaxFloat && v < minElem)) {
          minElem = v;
          minIndex = i;
        }
      }
      a2eEmotionsShared[minIndex] = kMaxFloat;
    }
  }

  __syncthreads();

  if (idx < kA2EBufferSize) {
    if (a2eEmotionsShared[idx] == kMaxFloat) {
      a2eEmotionsShared[idx] = 0.0f;
    }

    //
    // Apply emotion correspondence.
    //
    const auto j = emotionCorrespondence[idx];
    if (j != -1) {
      a2fEmotionsShared[j] = a2eEmotionsShared[idx];
    }
  }

  __syncthreads();

  if (idx < kA2FBufferSize) {
    //
    // Blend with either the beginning emotion or the previous one.
    //
    const float liveBlendCoef = params[2];
    const float* blendSource = firstFrame ? initialEmotion : prevEmotion;
    a2fEmotionsShared[idx] = (1.0f - liveBlendCoef) * a2fEmotionsShared[idx] + liveBlendCoef * blendSource[idx];
    prevEmotion[idx] = a2fEmotionsShared[idx];

    //
    // Blend with preferred emotion.
    //
    const bool enablePreferredEmotion = params[3] > 0.0f;
    if (enablePreferredEmotion) {
      const float preferredEmotionStrength = params[4];
      const float* preferredEmotion = preferredEmotions + trackIdx * outputEmotionsSize;
      a2fEmotionsShared[idx] = (1.0f - preferredEmotionStrength) * a2fEmotionsShared[idx] + preferredEmotionStrength * preferredEmotion[idx];
    }

    //
    // Apply transition.
    //
    if (!firstFrame) {
      const float liveTransitionTime = params[5];
      const float fixedDt = params[6];

      // ensure the transition time is at least 1e-3 seconds.
      const float transitionTime = max(liveTransitionTime, 1e-3f);
      // ensure w is at most 1.
      const float w = min(fixedDt / transitionTime, 1.0f);

      a2fEmotionsShared[idx] = w * a2fEmotionsShared[idx] + (1.0f - w) * prevBlendedEmotion[idx];
    }

    prevBlendedEmotion[idx] = a2fEmotionsShared[idx];

    //
    // Apply strength.
    //
    const float emotionStrength = params[7];
    a2fEmotionsShared[idx] *= emotionStrength;

    output[idx] = a2fEmotionsShared[idx];
    a2fEmotions[idx] = a2fEmotionsShared[idx];
  }

  if (idx == 0) {
    stateAndWorkBuffer[0] = 0.0f;
  }
}


// This kernel tries to use one warp per track, using maximum parallelism where possible.
__global__ void PostProcessGPUKernelLocal(
  float* outputEmotions, std::size_t outputEmotionsOffset, std::size_t outputEmotionsStride, std::size_t outputEmotionsSize,
  const float* inputEmotions, std::size_t inputEmotionsOffset, std::size_t inputEmotionsStride, std::size_t inputEmotionsSize,
  const std::int64_t* a2eEmotionCorrespondence, const std::int64_t* a2fEmotionCorrespondence,
  const float* postProcessParams, std::size_t postProcessParamsStride,
  const float* preferredEmotions,
  float* stateAndWorkBuffers, std::size_t stateAndWorkBuffersStride,
  std::size_t nbTracks
  ) {
  const std::size_t trackIdx = blockIdx.x;
  const std::size_t idx = threadIdx.x;

  float* output = outputEmotions + trackIdx * outputEmotionsStride + outputEmotionsOffset;
  const float* input = inputEmotions + trackIdx * inputEmotionsStride + inputEmotionsOffset;
  const float* params = postProcessParams + trackIdx * postProcessParamsStride;
  float* stateAndWorkBuffer = stateAndWorkBuffers + trackIdx * stateAndWorkBuffersStride;

  float* a2fEmotions = stateAndWorkBuffer + 1;
  float* prevEmotion = stateAndWorkBuffer + 1 + outputEmotionsSize;
  float* prevBlendedEmotion = stateAndWorkBuffer + 1 + outputEmotionsSize * 2;

  //
  // Initialization.
  //
  const bool firstFrame = stateAndWorkBuffer[0] > 0.0f;
  const float* initialEmotion = params + 8;
  float a2fEmotionLocal = 0.0f;
  if (idx < outputEmotionsSize) {
    a2fEmotionLocal = firstFrame ? initialEmotion[idx] : a2fEmotions[idx];
  }

  //
  // Softmax and contrast.
  //
  const float emotionContrast = params[1];
  float a2eEmotionLocal = cuda::std::numeric_limits<float>::lowest();
  if (idx < inputEmotionsSize) {
    a2eEmotionLocal = input[idx] * emotionContrast;
  }

  // Get the max element.
  constexpr const int kReductionStartOffset = 4;
  assert(kReductionStartOffset * 2 >= inputEmotionsSize);
  constexpr const unsigned int kMask = 0xffffffff;
  float maxElem = a2eEmotionLocal;
  for (int offset = kReductionStartOffset; offset > 0; offset /= 2) {
    maxElem = fmaxf(maxElem, __shfl_down_sync(kMask, maxElem, offset));
  }
  maxElem = __shfl_sync(kMask, maxElem, 0);

  // Get the exp sum.
  float v = 0.0f;
  if (idx < inputEmotionsSize) {
    v = std::exp(a2eEmotionLocal - maxElem);
  }
  float expSum = v;
  for (int offset = kReductionStartOffset; offset > 0; offset /= 2) {
    expSum += __shfl_down_sync(kMask, expSum, offset);
  }
  expSum = __shfl_sync(kMask, expSum, 0);

  // Normalize.
  if (idx < inputEmotionsSize) {
    a2eEmotionLocal = v / expSum;
  }
  else {
    // Next operation is max emotions which looks fir the smallest ones.
    a2eEmotionLocal = cuda::std::numeric_limits<float>::max();
  }

  //
  // Nullify unmapped emotions.
  //
  if (idx < inputEmotionsSize) {
    if (a2eEmotionCorrespondence[idx] == -1) {
      a2eEmotionLocal = 0.0f;
    }
  }

  //
  // Apply max emotions.
  //
  const std::uint32_t maxEmotions = *reinterpret_cast<const std::uint32_t*>(&params[0]);
  const std::size_t emotionsToZero = inputEmotionsSize - maxEmotions;
  // Any value above 1.0f is fine because this is applied right after softmax.
  static constexpr float kMaxFloat = 2.0f;
  for (std::size_t k = 0; k < emotionsToZero; ++k) {
    float minElem = a2eEmotionLocal;
    std::size_t minIndex = idx;
    for (int offset = kReductionStartOffset; offset > 0; offset /= 2) {
      const float v = __shfl_down_sync(kMask, minElem, offset);
      const std::size_t i = __shfl_down_sync(kMask, minIndex, offset);
      if (minElem == kMaxFloat || (v != kMaxFloat && v < minElem)) {
        minElem = v;
        minIndex = i;
      }
    }

    const std::size_t indexToReset = __shfl_sync(kMask, minIndex, 0);
    if (idx == indexToReset) {
      a2eEmotionLocal = kMaxFloat;
    }

    __syncwarp();
  }
  if (a2eEmotionLocal == kMaxFloat) {
    a2eEmotionLocal = 0.0f;
  }

  //
  // Apply emotion correspondence.
  //
  std::int64_t j = -1;
  auto indexToQuery = idx;
  if (idx < outputEmotionsSize) {
    j = a2fEmotionCorrespondence[idx];
    if (j >= 0) {
      indexToQuery = j;
    }
  }
  const float movedA2eEmotionLocal = __shfl_sync(kMask, a2eEmotionLocal, indexToQuery);
  if (j != -1) {
    a2fEmotionLocal = movedA2eEmotionLocal;
  }

  if (idx < outputEmotionsSize) {
    //
    // Blend with either the beginning emotion or the previous one.
    //
    const float liveBlendCoef = params[2];
    const float* blendSource = firstFrame ? initialEmotion : prevEmotion;
    a2fEmotionLocal = (1.0f - liveBlendCoef) * a2fEmotionLocal + liveBlendCoef * blendSource[idx];
    prevEmotion[idx] = a2fEmotionLocal;

    //
    // Blend with preferred emotion.
    //
    const bool enablePreferredEmotion = params[3] > 0.0f;
    if (enablePreferredEmotion) {
      const float preferredEmotionStrength = params[4];
      const float* preferredEmotion = preferredEmotions + trackIdx * outputEmotionsSize;
      a2fEmotionLocal = (1.0f - preferredEmotionStrength) * a2fEmotionLocal + preferredEmotionStrength * preferredEmotion[idx];
    }

    //
    // Apply transition.
    //
    if (!firstFrame) {
      const float liveTransitionTime = params[5];
      const float fixedDt = params[6];

      // ensure the transition time is at least 1e-3 seconds.
      const float transitionTime = max(liveTransitionTime, 1e-3f);
      // ensure w is at most 1.
      const float w = min(fixedDt / transitionTime, 1.0f);

      a2fEmotionLocal = w * a2fEmotionLocal + (1.0f - w) * prevBlendedEmotion[idx];
    }

    prevBlendedEmotion[idx] = a2fEmotionLocal;

    //
    // Apply strength.
    //
    const float emotionStrength = params[7];
    a2fEmotionLocal *= emotionStrength;

    //
    // Copy to state and output.
    //
    a2fEmotions[idx] = a2fEmotionLocal;
    output[idx] = a2fEmotionLocal;
  }

  if (idx == 0) {
    stateAndWorkBuffer[0] = 0.0f;
  }
}

} // Anonymous namespace.


std::error_code test::PostProcessGPU(
  float* outputEmotions, std::size_t outputEmotionsOffset, std::size_t outputEmotionsStride, std::size_t outputEmotionsSize,
  const float* inputEmotions, std::size_t inputEmotionsOffset, std::size_t inputEmotionsStride, std::size_t inputEmotionsSize,
  const std::int64_t* emotionCorrespondence,
  const float* postProcessParams, std::size_t postProcessParamsStride,
  const float* preferredEmotion,
  float* stateAndWorkBuffers, std::size_t stateAndWorkBuffersStride,
  std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(nbTracks), 1024u));
  dim3 numThreads(1024u);

  PostProcessGPUKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
        outputEmotions, outputEmotionsOffset, outputEmotionsStride, outputEmotionsSize,
        inputEmotions, inputEmotionsOffset, inputEmotionsStride, inputEmotionsSize,
        emotionCorrespondence,
        postProcessParams, postProcessParamsStride,
        preferredEmotion,
        stateAndWorkBuffers, stateAndWorkBuffersStride,
        nbTracks
        );
  A2E_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::PostProcessGPUShared(
  float* outputEmotions, std::size_t outputEmotionsOffset, std::size_t outputEmotionsStride, std::size_t outputEmotionsSize,
  const float* inputEmotions, std::size_t inputEmotionsOffset, std::size_t inputEmotionsStride, std::size_t inputEmotionsSize,
  const std::int64_t* emotionCorrespondence,
  const float* postProcessParams, std::size_t postProcessParamsStride,
  const float* preferredEmotion,
  float* stateAndWorkBuffers, std::size_t stateAndWorkBuffersStride,
  std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(static_cast<unsigned int>(nbTracks));
  dim3 numThreads(32u);

  PostProcessGPUKernelShared<<<numBlocks, numThreads, 0, cudaStream>>>(
        outputEmotions, outputEmotionsOffset, outputEmotionsStride, outputEmotionsSize,
        inputEmotions, inputEmotionsOffset, inputEmotionsStride, inputEmotionsSize,
        emotionCorrespondence,
        postProcessParams, postProcessParamsStride,
        preferredEmotion,
        stateAndWorkBuffers, stateAndWorkBuffersStride,
        nbTracks
        );
  A2E_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::PostProcessGPULocal(
  float* outputEmotions, std::size_t outputEmotionsOffset, std::size_t outputEmotionsStride, std::size_t outputEmotionsSize,
  const float* inputEmotions, std::size_t inputEmotionsOffset, std::size_t inputEmotionsStride, std::size_t inputEmotionsSize,
  const std::int64_t* a2eEmotionCorrespondence, const std::int64_t* a2fEmotionCorrespondence,
  const float* postProcessParams, std::size_t postProcessParamsStride,
  const float* preferredEmotion,
  float* stateAndWorkBuffers, std::size_t stateAndWorkBuffersStride,
  std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(static_cast<unsigned int>(nbTracks));
  dim3 numThreads(32u);

  PostProcessGPUKernelLocal<<<numBlocks, numThreads, 0, cudaStream>>>(
        outputEmotions, outputEmotionsOffset, outputEmotionsStride, outputEmotionsSize,
        inputEmotions, inputEmotionsOffset, inputEmotionsStride, inputEmotionsSize,
        a2eEmotionCorrespondence, a2fEmotionCorrespondence,
        postProcessParams, postProcessParamsStride,
        preferredEmotion,
        stateAndWorkBuffers, stateAndWorkBuffersStride,
        nbTracks
        );
  A2E_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}
