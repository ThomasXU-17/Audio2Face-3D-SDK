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

#include "audio2x/tensor.h"
#include "audio2x/float_accumulator.h"

namespace nva2x {

// This class is thread-safe, it is safe to call any functions concurrently,
// with the exception of Destroy().
class IAudioAccumulator : public IFloatAccumulator {
public:
  // Read samples from the accumulator.
  // The start sample is in number of samples since the start of the audio.
  // This function is GPU async.
  virtual std::error_code Read(
    DeviceTensorFloatView destination, timestamp_t absoluteStartSample, float inputStrength, cudaStream_t cudaStream
    ) const = 0;

  // Implement the IFloatAccumulator::Read() function as a wrapper around the more
  // specialized Read() function.
  std::error_code Read(
    DeviceTensorFloatView destination, timestamp_t absoluteStartSample, cudaStream_t cudaStream
    ) const final {
      return Read(destination, absoluteStartSample, 1.0f, cudaStream);
    }

protected:
  virtual ~IAudioAccumulator();
};

// Create an audio accumulator pre-allocated with tensorCount buffers holding tensorSize float values.
AUDIO2X_SDK_EXPORT IAudioAccumulator* CreateAudioAccumulator(std::size_t tensorSize, std::size_t tensorCount);

} // namespace nva2x
