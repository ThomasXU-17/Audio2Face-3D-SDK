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
#include "test_core_batch_skin_animator_cuda.h"

#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2face/error.h"
#include "audio2x/error.h"

#include <cassert>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <cuda_runtime_api.h>


namespace {

__global__ void AnimatorSkinGetYKernel(float* poseY, const float* pose, size_t numVertices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices)
    {
        poseY[idx] = pose[1 + idx * 3];
    }
}

__global__ void AnimatorSkinGetYKernelPacked(float* poseY, const float* animatorData, size_t animatorDataStride, size_t numVertices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices)
    {
        poseY[idx] = animatorData[(1 + idx * 3) * animatorDataStride + 2];
    }
}

__global__ void AnimatorSkinGetFaceMaskLowerKernel(
    float* faceMaskLower, size_t numVertices, float min, float maxSubMin, float faceMaskLevel, float faceMaskSoftness)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices)
    {
        faceMaskLower[idx] =
            1.0f / (1.0f + expf(-(faceMaskLevel - (faceMaskLower[idx] - min) / maxSubMin) / faceMaskSoftness));
    }
}


__device__ void ComputeSkinPostProcessing(
  float& result,
  float inputDelta,
  float eyeClosePoseDelta, float lipOpenPoseDelta, float neutralPose, float faceMaskLower,
  float lower1, float lower2,
  float upper1, float upper2,
  float& outLower1, float& outLower2,
  float& outUpper1, float& outUpper2,
  float skinStrength, float eyelidOpenOffset, float blinkOffset, float blinkStrength, float lipOpenOffset,
  float lowerFaceAlpha, float upperFaceAlpha,
  float lowerFaceStrength, float upperFaceStrength,
  bool initialized
  ) {
  // Step 1.
  const float delta =
    skinStrength * inputDelta +
    eyeClosePoseDelta * (-eyelidOpenOffset + blinkOffset * blinkStrength) +
    lipOpenPoseDelta * lipOpenOffset;

  // Smoothing.
  if (initialized && lowerFaceAlpha > 0.0f) {
    outLower1 = lower1 + (delta - lower1) * lowerFaceAlpha;
    outLower2 = lower2 + (outLower1 - lower2) * lowerFaceAlpha;
  }
  else {
    outLower1 = delta;
    outLower2 = delta;
  }

  if (initialized && upperFaceAlpha > 0.0f) {
    outUpper1 = upper1 + (delta - upper1) * upperFaceAlpha;
    outUpper2 = upper2 + (outUpper1 - upper2) * upperFaceAlpha;
  }
  else {
    outUpper1 = delta;
    outUpper2 = delta;
  }

  // Step 2.
  result =
    neutralPose +
    outUpper2 * upperFaceStrength * (1.0f - faceMaskLower) +
    outLower2 * lowerFaceStrength * faceMaskLower;
}

__device__ void ComputeSkinPostProcessing(
  float& result,
  float inputDelta,
  float eyeClosePoseDelta, float lipOpenPoseDelta, float neutralPose, float faceMaskLower,
  float& lower1, float& lower2,
  float& upper1, float& upper2,
  float skinStrength, float eyelidOpenOffset, float blinkOffset, float blinkStrength, float lipOpenOffset,
  float lowerFaceAlpha, float upperFaceAlpha,
  float lowerFaceStrength, float upperFaceStrength,
  bool initialized
  ) {
  // Step 1.
  const float delta =
    skinStrength * inputDelta +
    eyeClosePoseDelta * (-eyelidOpenOffset + blinkOffset * blinkStrength) +
    lipOpenPoseDelta * lipOpenOffset;

  // Smoothing.
  if (initialized && lowerFaceAlpha > 0.0f) {
    lower1 += (delta - lower1) * lowerFaceAlpha;
    lower2 += (lower1 - lower2) * lowerFaceAlpha;
  }
  else {
    lower1 = delta;
    lower2 = delta;
  }

  if (initialized && upperFaceAlpha > 0.0f) {
    upper1 += (delta - upper1) * upperFaceAlpha;
    upper2 += (upper1 - upper2) * upperFaceAlpha;
  }
  else {
    upper1 = delta;
    upper2 = delta;
  }

  // Step 2.
  result =
    neutralPose +
    upper2 * upperFaceStrength * (1.0f - faceMaskLower) +
    lower2 * lowerFaceStrength * faceMaskLower;
}


__global__ void AnimatorSkinAnimateFusedKernel(
  float* results,
  const float* inputDeltas,
  const float* eyeClosePoseDelta, const float* lipOpenPoseDelta, const float* neutralPose, const float* faceMaskLower,
  float* interpLower, float* interpUpper,
  float skinStrength, float eyelidOpenOffset, float blinkOffset, float blinkStrength, float lipOpenOffset,
  float lowerFaceAlpha, float upperFaceAlpha,
  float lowerFaceStrength, float upperFaceStrength,
  std::size_t poseSize
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < poseSize) {
    const std::size_t elemIdx = idx;

    float& result = results[elemIdx];
    const float inputDelta = inputDeltas[elemIdx];

    const float eyeClosePoseDeltaValue = eyeClosePoseDelta[elemIdx];
    const float lipOpenPoseDeltaValue = lipOpenPoseDelta[elemIdx];
    const float neutralPoseValue = neutralPose[elemIdx];
    const float faceMaskLowerValue = faceMaskLower[elemIdx / 3];

    float& lower1 = interpLower[elemIdx];
    float& lower2 = interpLower[elemIdx + poseSize];
    float& upper1 = interpUpper[elemIdx];
    float& upper2 = interpUpper[elemIdx + poseSize];

    ComputeSkinPostProcessing(
      result,
      inputDelta,
      eyeClosePoseDeltaValue, lipOpenPoseDeltaValue, neutralPoseValue, faceMaskLowerValue,
      lower1, lower2,
      upper1, upper2,
      skinStrength, eyelidOpenOffset, blinkOffset, blinkStrength, lipOpenOffset,
      lowerFaceAlpha, upperFaceAlpha, lowerFaceStrength, upperFaceStrength,
      true
    );
  }
}


__global__ void AnimatorSkinAnimateFusedKernelBatched(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* eyeClosePoseDelta, const float* lipOpenPoseDelta, const float* neutralPose, const float* faceMaskLower,
  float* interpLower, float* interpUpper,
  float skinStrength, float eyelidOpenOffset, float blinkOffset, float blinkStrength, float lipOpenOffset,
  float lowerFaceAlpha, float upperFaceAlpha,
  float lowerFaceStrength, float upperFaceStrength,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < poseSize * nbTracks) {
    const std::size_t trackIdx = idx / poseSize;
    const std::size_t elemIdx = idx % poseSize;

    float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
    const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

    const float eyeClosePoseDeltaValue = eyeClosePoseDelta[elemIdx];
    const float lipOpenPoseDeltaValue = lipOpenPoseDelta[elemIdx];
    const float neutralPoseValue = neutralPose[elemIdx];
    const float faceMaskLowerValue = faceMaskLower[elemIdx / 3];

    static constexpr std::size_t kInterpolatorDegree = 2;
    float& lower1 = interpLower[poseSize * (kInterpolatorDegree * trackIdx + 0) + elemIdx];
    float& lower2 = interpLower[poseSize * (kInterpolatorDegree * trackIdx + 1) + elemIdx];
    float& upper1 = interpUpper[poseSize * (kInterpolatorDegree * trackIdx + 0) + elemIdx];
    float& upper2 = interpUpper[poseSize * (kInterpolatorDegree * trackIdx + 1) + elemIdx];

    ComputeSkinPostProcessing(
      result,
      inputDelta,
      eyeClosePoseDeltaValue, lipOpenPoseDeltaValue, neutralPoseValue, faceMaskLowerValue,
      lower1, lower2,
      upper1, upper2,
      skinStrength, eyelidOpenOffset, blinkOffset, blinkStrength, lipOpenOffset,
      lowerFaceAlpha, upperFaceAlpha, lowerFaceStrength, upperFaceStrength,
      true
    );
  }
}


__global__ void AnimatorSkinAnimateFusedKernelBatchedParams(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* eyeClosePoseDelta, const float* lipOpenPoseDelta, const float* neutralPose, const float* faceMaskLower,
  float* interpLower, float* interpUpper,
  const float* skinStrengths, const float* eyelidOpenOffsets, const float* blinkOffsets, const float* blinkStrengths, const float* lipOpenOffsets,
  const float* lowerFaceAlphas, const float* upperFaceAlphas,
  const float* lowerFaceStrengths, const float* upperFaceStrengths,
  const bool* initializeds,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < poseSize * nbTracks) {
    const std::size_t trackIdx = idx / poseSize;
    const std::size_t elemIdx = idx % poseSize;

    float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
    const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

    const float eyeClosePoseDeltaValue = eyeClosePoseDelta[elemIdx];
    const float lipOpenPoseDeltaValue = lipOpenPoseDelta[elemIdx];
    const float neutralPoseValue = neutralPose[elemIdx];
    const float faceMaskLowerValue = faceMaskLower[elemIdx / 3];

    static constexpr std::size_t kInterpolatorDegree = 2;
    float& lower1 = interpLower[poseSize * (kInterpolatorDegree * trackIdx + 0) + elemIdx];
    float& lower2 = interpLower[poseSize * (kInterpolatorDegree * trackIdx + 1) + elemIdx];
    float& upper1 = interpUpper[poseSize * (kInterpolatorDegree * trackIdx + 0) + elemIdx];
    float& upper2 = interpUpper[poseSize * (kInterpolatorDegree * trackIdx + 1) + elemIdx];

    const float skinStrength = skinStrengths[trackIdx];
    const float eyelidOpenOffset = eyelidOpenOffsets[trackIdx];
    const float blinkOffset = blinkOffsets[trackIdx];
    const float blinkStrength = blinkStrengths[trackIdx];
    const float lipOpenOffset = lipOpenOffsets[trackIdx];
    const float lowerFaceAlpha = lowerFaceAlphas[trackIdx];
    const float upperFaceAlpha = upperFaceAlphas[trackIdx];
    const float lowerFaceStrength = lowerFaceStrengths[trackIdx];
    const float upperFaceStrength = upperFaceStrengths[trackIdx];
    const bool initialized = initializeds[trackIdx];

    ComputeSkinPostProcessing(
      result,
      inputDelta,
      eyeClosePoseDeltaValue, lipOpenPoseDeltaValue, neutralPoseValue, faceMaskLowerValue,
      lower1, lower2,
      upper1, upper2,
      skinStrength, eyelidOpenOffset, blinkOffset, blinkStrength, lipOpenOffset,
      lowerFaceAlpha, upperFaceAlpha, lowerFaceStrength, upperFaceStrength,
      initialized
    );
  }
}


__global__ void AnimatorSkinAnimateFusedKernelBatchedParamsPacked(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* eyeClosePoseDelta, const float* lipOpenPoseDelta, const float* neutralPose, const float* faceMaskLower,
  float* interpLower, float* interpUpper,
  const float* params, std::size_t paramsStride,
  const bool* initializeds,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < poseSize * nbTracks) {
    const std::size_t trackIdx = idx / poseSize;
    const std::size_t elemIdx = idx % poseSize;

    float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
    const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

    const float eyeClosePoseDeltaValue = eyeClosePoseDelta[elemIdx];
    const float lipOpenPoseDeltaValue = lipOpenPoseDelta[elemIdx];
    const float neutralPoseValue = neutralPose[elemIdx];
    const float faceMaskLowerValue = faceMaskLower[elemIdx / 3];

    static constexpr std::size_t kInterpolatorDegree = 2;
    float& lower1 = interpLower[poseSize * (kInterpolatorDegree * trackIdx + 0) + elemIdx];
    float& lower2 = interpLower[poseSize * (kInterpolatorDegree * trackIdx + 1) + elemIdx];
    float& upper1 = interpUpper[poseSize * (kInterpolatorDegree * trackIdx + 0) + elemIdx];
    float& upper2 = interpUpper[poseSize * (kInterpolatorDegree * trackIdx + 1) + elemIdx];

    const float skinStrength = params[trackIdx * paramsStride + 0];
    const float eyelidOpenOffset = params[trackIdx * paramsStride + 1];
    const float blinkOffset = params[trackIdx * paramsStride + 2];
    const float blinkStrength = params[trackIdx * paramsStride + 3];
    const float lipOpenOffset = params[trackIdx * paramsStride + 4];
    const float lowerFaceAlpha = params[trackIdx * paramsStride + 5];
    const float upperFaceAlpha = params[trackIdx * paramsStride + 6];
    const float lowerFaceStrength = params[trackIdx * paramsStride + 7];
    const float upperFaceStrength = params[trackIdx * paramsStride + 8];
    const bool initialized = initializeds[trackIdx];

    ComputeSkinPostProcessing(
      result,
      inputDelta,
      eyeClosePoseDeltaValue, lipOpenPoseDeltaValue, neutralPoseValue, faceMaskLowerValue,
      lower1, lower2,
      upper1, upper2,
      skinStrength, eyelidOpenOffset, blinkOffset, blinkStrength, lipOpenOffset,
      lowerFaceAlpha, upperFaceAlpha, lowerFaceStrength, upperFaceStrength,
      initialized
    );
  }
}


__global__ void AnimatorSkinAnimatePacked(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const bool* initializeds,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < poseSize * nbTracks) {
    const std::size_t trackIdx = idx / poseSize;
    const std::size_t elemIdx = idx % poseSize;

    float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
    const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

    const float eyeClosePoseDeltaValue = animatorData[elemIdx * animatorDataStride + 0];
    const float lipOpenPoseDeltaValue = animatorData[elemIdx * animatorDataStride + 1];
    const float neutralPoseValue = animatorData[elemIdx * animatorDataStride + 2];
    const float faceMaskLowerValue = faceMaskLower[elemIdx / 3];

    float& lower1 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 0];
    float& lower2 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 1];
    float& upper1 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 2];
    float& upper2 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 3];

    const float skinStrength = params[trackIdx * paramsStride + 0];
    const float eyelidOpenOffset = params[trackIdx * paramsStride + 1];
    const float blinkOffset = params[trackIdx * paramsStride + 2];
    const float blinkStrength = params[trackIdx * paramsStride + 3];
    const float lipOpenOffset = params[trackIdx * paramsStride + 4];
    const float lowerFaceAlpha = params[trackIdx * paramsStride + 5];
    const float upperFaceAlpha = params[trackIdx * paramsStride + 6];
    const float lowerFaceStrength = params[trackIdx * paramsStride + 7];
    const float upperFaceStrength = params[trackIdx * paramsStride + 8];
    const bool initialized = initializeds[trackIdx];

    ComputeSkinPostProcessing(
      result,
      inputDelta,
      eyeClosePoseDeltaValue, lipOpenPoseDeltaValue, neutralPoseValue, faceMaskLowerValue,
      lower1, lower2,
      upper1, upper2,
      skinStrength, eyelidOpenOffset, blinkOffset, blinkStrength, lipOpenOffset,
      lowerFaceAlpha, upperFaceAlpha, lowerFaceStrength, upperFaceStrength,
      initialized
    );
  }
}


__global__ void AnimatorSkinAnimatePackedInOut(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  const float* interpData, float* outInterpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const bool* initializeds,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < poseSize * nbTracks) {
    const std::size_t trackIdx = idx / poseSize;
    const std::size_t elemIdx = idx % poseSize;

    float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
    const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

    const float eyeClosePoseDeltaValue = animatorData[elemIdx * animatorDataStride + 0];
    const float lipOpenPoseDeltaValue = animatorData[elemIdx * animatorDataStride + 1];
    const float neutralPoseValue = animatorData[elemIdx * animatorDataStride + 2];
    const float faceMaskLowerValue = faceMaskLower[elemIdx / 3];

    float lower1 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 0];
    float lower2 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 1];
    float upper1 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 2];
    float upper2 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 3];
    float& outLower1 = outInterpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 0];
    float& outLower2 = outInterpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 1];
    float& outUpper1 = outInterpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 2];
    float& outUpper2 = outInterpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 3];

    const float skinStrength = params[trackIdx * paramsStride + 0];
    const float eyelidOpenOffset = params[trackIdx * paramsStride + 1];
    const float blinkOffset = params[trackIdx * paramsStride + 2];
    const float blinkStrength = params[trackIdx * paramsStride + 3];
    const float lipOpenOffset = params[trackIdx * paramsStride + 4];
    const float lowerFaceAlpha = params[trackIdx * paramsStride + 5];
    const float upperFaceAlpha = params[trackIdx * paramsStride + 6];
    const float lowerFaceStrength = params[trackIdx * paramsStride + 7];
    const float upperFaceStrength = params[trackIdx * paramsStride + 8];
    const bool initialized = initializeds[trackIdx];

    ComputeSkinPostProcessing(
      result,
      inputDelta,
      eyeClosePoseDeltaValue, lipOpenPoseDeltaValue, neutralPoseValue, faceMaskLowerValue,
      lower1, lower2,
      upper1, upper2,
      outLower1, outLower2,
      outUpper1, outUpper2,
      skinStrength, eyelidOpenOffset, blinkOffset, blinkStrength, lipOpenOffset,
      lowerFaceAlpha, upperFaceAlpha, lowerFaceStrength, upperFaceStrength,
      initialized
    );
  }
}


__global__ void AnimatorSkinAnimatePackedControl(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const std::uint64_t* initializedActives,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= poseSize * nbTracks) {
    return;
  }

  const std::size_t trackIdx = idx / poseSize;

  const bool active = (initializedActives[trackIdx / 32] >> (2*(trackIdx % 32))) & 0b01;
  if (!active) {
    return;
  }

  const std::size_t elemIdx = idx % poseSize;
  const bool initialized = (initializedActives[trackIdx / 32] >> (2*(trackIdx % 32))) & 0b10;

  float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
  const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

  const float eyeClosePoseDeltaValue = animatorData[elemIdx * animatorDataStride + 0];
  const float lipOpenPoseDeltaValue = animatorData[elemIdx * animatorDataStride + 1];
  const float neutralPoseValue = animatorData[elemIdx * animatorDataStride + 2];
  const float faceMaskLowerValue = faceMaskLower[elemIdx / 3];

  float& lower1 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 0];
  float& lower2 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 1];
  float& upper1 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 2];
  float& upper2 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 3];

  const float skinStrength = params[trackIdx * paramsStride + 0];
  const float eyelidOpenOffset = params[trackIdx * paramsStride + 1];
  const float blinkOffset = params[trackIdx * paramsStride + 2];
  const float blinkStrength = params[trackIdx * paramsStride + 3];
  const float lipOpenOffset = params[trackIdx * paramsStride + 4];
  const float lowerFaceAlpha = params[trackIdx * paramsStride + 5];
  const float upperFaceAlpha = params[trackIdx * paramsStride + 6];
  const float lowerFaceStrength = params[trackIdx * paramsStride + 7];
  const float upperFaceStrength = params[trackIdx * paramsStride + 8];

  ComputeSkinPostProcessing(
    result,
    inputDelta,
    eyeClosePoseDeltaValue, lipOpenPoseDeltaValue, neutralPoseValue, faceMaskLowerValue,
    lower1, lower2,
    upper1, upper2,
    skinStrength, eyelidOpenOffset, blinkOffset, blinkStrength, lipOpenOffset,
    lowerFaceAlpha, upperFaceAlpha, lowerFaceStrength, upperFaceStrength,
    initialized
  );
}

__global__ void AnimatorSkinAnimatePackedControl_Set(
  std::uint64_t* initializedActives, std::size_t initializedActivesSize, std::size_t index, std::uint64_t value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= initializedActivesSize) {
    return;
  }

  initializedActives[idx] = value;
}


__global__ void AnimatorSkinAnimatePackedControl2(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const std::uint64_t* initializeds,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= poseSize * nbTracks) {
    return;
  }

  const std::size_t trackIdx = idx / poseSize;

  const std::size_t elemIdx = idx % poseSize;
  const bool initialized = (initializeds[trackIdx / 64] >> (trackIdx % 64)) & 0b1;

  float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
  const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

  const float eyeClosePoseDeltaValue = animatorData[elemIdx * animatorDataStride + 0];
  const float lipOpenPoseDeltaValue = animatorData[elemIdx * animatorDataStride + 1];
  const float neutralPoseValue = animatorData[elemIdx * animatorDataStride + 2];
  const float faceMaskLowerValue = faceMaskLower[elemIdx / 3];

  float& lower1 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 0];
  float& lower2 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 1];
  float& upper1 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 2];
  float& upper2 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 3];

  const float skinStrength = params[trackIdx * paramsStride + 0];
  const float eyelidOpenOffset = params[trackIdx * paramsStride + 1];
  const float blinkOffset = params[trackIdx * paramsStride + 2];
  const float blinkStrength = params[trackIdx * paramsStride + 3];
  const float lipOpenOffset = params[trackIdx * paramsStride + 4];
  const float lowerFaceAlpha = params[trackIdx * paramsStride + 5];
  const float upperFaceAlpha = params[trackIdx * paramsStride + 6];
  const float lowerFaceStrength = params[trackIdx * paramsStride + 7];
  const float upperFaceStrength = params[trackIdx * paramsStride + 8];

  ComputeSkinPostProcessing(
    result,
    inputDelta,
    eyeClosePoseDeltaValue, lipOpenPoseDeltaValue, neutralPoseValue, faceMaskLowerValue,
    lower1, lower2,
    upper1, upper2,
    skinStrength, eyelidOpenOffset, blinkOffset, blinkStrength, lipOpenOffset,
    lowerFaceAlpha, upperFaceAlpha, lowerFaceStrength, upperFaceStrength,
    initialized
  );
}

} // Anonymous namespace.


using namespace nva2f;

std::error_code test::CalculateFaceMaskLower(
  float* faceMaskLower,
  const float* neutralPose,
  std::size_t poseSize,
  float faceMaskLevel,
  float faceMaskSoftness,
  cudaStream_t cudaStream
) {
    assert(poseSize % 3 == 0);
    A2F_CHECK_ERROR_WITH_MSG(faceMaskSoftness > 0, "Face Mask Softness should be greater than zero", ErrorCode::eOutOfRange);

    const size_t numVertices = poseSize / 3;

    dim3 numBlocks(IDIVUP(static_cast<unsigned int>(numVertices), 1024u));
    dim3 numThreads(1024u);

    AnimatorSkinGetYKernel<<<numBlocks, numThreads, 0, cudaStream>>>(faceMaskLower, neutralPose, numVertices);
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    float min, max;
    try
    {
        const thrust::device_ptr<float> faceMaskLowerPtr = thrust::device_pointer_cast<float>(faceMaskLower);
        thrust::pair<const thrust::device_ptr<float>, const thrust::device_ptr<float>> minmax =
            thrust::minmax_element(thrust::cuda::par.on(cudaStream), faceMaskLowerPtr, faceMaskLowerPtr + numVertices);
        A2F_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaThrustError);
        min = *(minmax.first);
        max = *(minmax.second);
    }
    catch (...)
    {
        LOG_ERROR("Unable to calculate minmax for neutral pose Y values");
        return nva2x::ErrorCode::eCudaThrustError;
    }

    CHECK_ERROR_WITH_MSG(max - min > 0, "Neutral pose: max Y value should be greater than min Y value", ErrorCode::eOutOfRange);

    AnimatorSkinGetFaceMaskLowerKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
        faceMaskLower, numVertices, min, max - min, faceMaskLevel, faceMaskSoftness);
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    return nva2x::ErrorCode::eSuccess;
}


std::error_code test::CalculateFaceMaskLowerPacked(
  float* faceMaskLower,
  const float* animatorData,
  std::size_t animatorDataStride,
  std::size_t poseSize,
  float faceMaskLevel,
  float faceMaskSoftness,
  cudaStream_t cudaStream
) {
    assert(poseSize % 3 == 0);
    A2F_CHECK_ERROR_WITH_MSG(faceMaskSoftness > 0, "Face Mask Softness should be greater than zero", ErrorCode::eOutOfRange);

    const size_t numVertices = poseSize / 3;

    dim3 numBlocks(IDIVUP(static_cast<unsigned int>(numVertices), 1024u));
    dim3 numThreads(1024u);

    AnimatorSkinGetYKernelPacked<<<numBlocks, numThreads, 0, cudaStream>>>(faceMaskLower, animatorData, animatorDataStride, numVertices);
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    float min, max;
    try
    {
        const thrust::device_ptr<float> faceMaskLowerPtr = thrust::device_pointer_cast<float>(faceMaskLower);
        thrust::pair<const thrust::device_ptr<float>, const thrust::device_ptr<float>> minmax =
            thrust::minmax_element(thrust::cuda::par.on(cudaStream), faceMaskLowerPtr, faceMaskLowerPtr + numVertices);
        A2F_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaThrustError);
        min = *(minmax.first);
        max = *(minmax.second);
    }
    catch (...)
    {
        LOG_ERROR("Unable to calculate minmax for neutral pose Y values");
        return nva2x::ErrorCode::eCudaThrustError;
    }

    CHECK_ERROR_WITH_MSG(max - min > 0, "Neutral pose: max Y value should be greater than min Y value", ErrorCode::eOutOfRange);

    AnimatorSkinGetFaceMaskLowerKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
        faceMaskLower, numVertices, min, max - min, faceMaskLevel, faceMaskSoftness);
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    return nva2x::ErrorCode::eSuccess;
}


std::error_code test::AnimateFusedKernel(
  float* result,
  const float* inputDeltas,
  const float* eyeClosePoseDelta, const float* lipOpenPoseDelta, const float* neutralPose, const float* faceMaskLower,
  float* interpLower, float* interpUpper,
  float skinStrength, float eyelidOpenOffset, float blinkOffset, float blinkStrength, float lipOpenOffset,
  float lowerFaceAlpha, float upperFaceAlpha,
  float lowerFaceStrength, float upperFaceStrength,
  std::size_t poseSize,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize), 1024u));
  dim3 numThreads(1024u);

  AnimatorSkinAnimateFusedKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
        result,
        inputDeltas,
        eyeClosePoseDelta, lipOpenPoseDelta, neutralPose, faceMaskLower,
        interpLower, interpUpper,
        skinStrength, eyelidOpenOffset, blinkOffset, blinkStrength, lipOpenOffset,
        lowerFaceAlpha, upperFaceAlpha,
        lowerFaceStrength, upperFaceStrength,
        poseSize
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::AnimateFusedKernelBatched(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* eyeClosePoseDelta, const float* lipOpenPoseDelta, const float* neutralPose, const float* faceMaskLower,
  float* interpLower, float* interpUpper,
  float skinStrength, float eyelidOpenOffset, float blinkOffset, float blinkStrength, float lipOpenOffset,
  float lowerFaceAlpha, float upperFaceAlpha,
  float lowerFaceStrength, float upperFaceStrength,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorSkinAnimateFusedKernelBatched<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        eyeClosePoseDelta, lipOpenPoseDelta, neutralPose, faceMaskLower,
        interpLower, interpUpper,
        skinStrength, eyelidOpenOffset, blinkOffset, blinkStrength, lipOpenOffset,
        lowerFaceAlpha, upperFaceAlpha,
        lowerFaceStrength, upperFaceStrength,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::AnimateFusedKernelBatchedParams(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* eyeClosePoseDelta, const float* lipOpenPoseDelta, const float* neutralPose, const float* faceMaskLower,
  float* interpLower, float* interpUpper,
  const float* skinStrengths, const float* eyelidOpenOffsets, const float* blinkOffsets, const float* blinkStrengths, const float* lipOpenOffsets,
  const float* lowerFaceAlphas, const float* upperFaceAlphas,
  const float* lowerFaceStrengths, const float* upperFaceStrengths,
  const bool* initializeds,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorSkinAnimateFusedKernelBatchedParams<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        eyeClosePoseDelta, lipOpenPoseDelta, neutralPose, faceMaskLower,
        interpLower, interpUpper,
        skinStrengths, eyelidOpenOffsets, blinkOffsets, blinkStrengths, lipOpenOffsets,
        lowerFaceAlphas, upperFaceAlphas,
        lowerFaceStrengths, upperFaceStrengths,
        initializeds,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::AnimateFusedKernelBatchedParamsPacked(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* eyeClosePoseDelta, const float* lipOpenPoseDelta, const float* neutralPose, const float* faceMaskLower,
  float* interpLower, float* interpUpper,
  const float* params, std::size_t paramsStride,
  const bool* initializeds,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorSkinAnimateFusedKernelBatchedParamsPacked<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        eyeClosePoseDelta, lipOpenPoseDelta, neutralPose, faceMaskLower,
        interpLower, interpUpper,
        params, paramsStride,
        initializeds,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::AnimatePacked(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const bool* initializeds,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorSkinAnimatePacked<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        animatorData, animatorDataStride,
        faceMaskLower,
        interpData, interpDataStride,
        params, paramsStride,
        initializeds,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::AnimatePackedInOut(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  const float* interpData, float* outInterpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const bool* initializeds,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorSkinAnimatePackedInOut<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        animatorData, animatorDataStride,
        faceMaskLower,
        interpData, outInterpData, interpDataStride,
        params, paramsStride,
        initializeds,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::AnimatePackedControl(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const std::uint64_t* initializedActives,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorSkinAnimatePackedControl<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        animatorData, animatorDataStride,
        faceMaskLower,
        interpData, interpDataStride,
        params, paramsStride,
        initializedActives,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code test::AnimatePackedControl_Set(
  std::uint64_t* initializedActives, std::size_t initializedActivesSize, std::size_t index, std::uint64_t value,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(initializedActivesSize), 32u));
  dim3 numThreads(32u);

  AnimatorSkinAnimatePackedControl_Set<<<numBlocks, numThreads, 0, cudaStream>>>(
        initializedActives, initializedActivesSize, index, value
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code test::AnimatePackedControl2(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const std::uint64_t* initializeds,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorSkinAnimatePackedControl2<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        animatorData, animatorDataStride,
        faceMaskLower,
        interpData, interpDataStride,
        params, paramsStride,
        initializeds,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}
