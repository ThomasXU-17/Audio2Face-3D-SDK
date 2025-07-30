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

std::error_code AnimateBatched(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  float tongueStrength, float tongueHeightOffset, float tongueDepthOffset,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  );

std::error_code AnimateBatchedParamsPacked(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  );

std::error_code AnimateBatchedParamsPackedVertices(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  );

std::error_code AnimateBatchedParamsPackedFull(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  );

} // namespace test
