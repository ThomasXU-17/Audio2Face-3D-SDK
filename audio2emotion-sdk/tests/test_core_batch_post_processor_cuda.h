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

std::error_code PostProcessGPU(
  float* outputEmotions, std::size_t outputEmotionsOffset, std::size_t outputEmotionsStride, std::size_t outputEmotionsSize,
  const float* inputEmotions, std::size_t inputEmotionsOffset, std::size_t inputEmotionsStride, std::size_t inputEmotionsSize,
  const std::int64_t* emotionCorrespondence,
  const float* postProcessParams, std::size_t postProcessParamsStride,
  const float* preferredEmotion,
  float* stateAndWorkBuffers, std::size_t stateAndWorkBuffersStride,
  std::size_t nbTracks,
  cudaStream_t cudaStream
  );

std::error_code PostProcessGPUShared(
  float* outputEmotions, std::size_t outputEmotionsOffset, std::size_t outputEmotionsStride, std::size_t outputEmotionsSize,
  const float* inputEmotions, std::size_t inputEmotionsOffset, std::size_t inputEmotionsStride, std::size_t inputEmotionsSize,
  const std::int64_t* emotionCorrespondence,
  const float* postProcessParams, std::size_t postProcessParamsStride,
  const float* preferredEmotion,
  float* stateAndWorkBuffers, std::size_t stateAndWorkBuffersStride,
  std::size_t nbTracks,
  cudaStream_t cudaStream
  );

std::error_code PostProcessGPULocal(
  float* outputEmotions, std::size_t outputEmotionsOffset, std::size_t outputEmotionsStride, std::size_t outputEmotionsSize,
  const float* inputEmotions, std::size_t inputEmotionsOffset, std::size_t inputEmotionsStride, std::size_t inputEmotionsSize,
  const std::int64_t* a2eEmotionCorrespondence, const std::int64_t* a2fEmotionCorrespondence,
  const float* postProcessParams, std::size_t postProcessParamsStride,
  const float* preferredEmotion,
  float* stateAndWorkBuffers, std::size_t stateAndWorkBuffersStride,
  std::size_t nbTracks,
  cudaStream_t cudaStream
  );

} // namespace test
