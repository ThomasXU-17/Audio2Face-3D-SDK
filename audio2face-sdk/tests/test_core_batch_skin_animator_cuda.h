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

#include "audio2x/cuda_fwd.h"

#include <cstdint>
#include <system_error>

namespace test {

std::error_code CalculateFaceMaskLower(
  float* faceMaskLower,
  const float* neutralPose,
  std::size_t poseSize,
  float faceMaskLevel,
  float faceMaskSoftness,
  cudaStream_t cudaStream
);

std::error_code CalculateFaceMaskLowerPacked(
  float* faceMaskLower,
  const float* animatorData,
  std::size_t animatorDataStride,
  std::size_t poseSize,
  float faceMaskLevel,
  float faceMaskSoftness,
  cudaStream_t cudaStream
);

std::error_code AnimateFusedKernel(
  float* result,
  const float* inputDeltas,
  const float* eyeClosePoseDelta, const float* lipOpenPoseDelta, const float* neutralPose, const float* faceMaskLower,
  float* interpLower, float* interpUpper,
  float skinStrength, float eyelidOpenOffset, float blinkOffset, float blinkStrength, float lipOpenOffset,
  float lowerFaceAlpha, float upperFaceAlpha,
  float lowerFaceStrength, float upperFaceStrength,
  std::size_t poseSize,
  cudaStream_t cudaStream
  );

std::error_code AnimateFusedKernelBatched(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* eyeClosePoseDelta, const float* lipOpenPoseDelta, const float* neutralPose, const float* faceMaskLower,
  float* interpLower, float* interpUpper,
  float skinStrength, float eyelidOpenOffset, float blinkOffset, float blinkStrength, float lipOpenOffset,
  float lowerFaceAlpha, float upperFaceAlpha,
  float lowerFaceStrength, float upperFaceStrength,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  );

std::error_code AnimateFusedKernelBatchedParams(
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
  );

std::error_code AnimateFusedKernelBatchedParamsPacked(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* eyeClosePoseDelta, const float* lipOpenPoseDelta, const float* neutralPose, const float* faceMaskLower,
  float* interpLower, float* interpUpper,
  const float* params, std::size_t paramsStride,
  const bool* initializeds,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  );

std::error_code AnimatePacked(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const bool* initializeds,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  );

std::error_code AnimatePackedInOut(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  const float* interpData, float* outInterpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const bool* initializeds,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  );

std::error_code AnimatePackedControl(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const std::uint64_t* initializedActives,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  );
std::error_code AnimatePackedControl_Set(
  std::uint64_t* initializedActives, std::size_t initializedActivesSize, std::size_t index, std::uint64_t value,
  cudaStream_t cudaStream
  );

std::error_code AnimatePackedControl2(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const std::uint64_t* initializeds,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  );

} // namespace test
