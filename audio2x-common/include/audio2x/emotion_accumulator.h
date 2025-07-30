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

namespace nva2x {

// The IEmotionAccumulator class provides a thread-safe, GPU-accelerated time-values
// storage system for emotion data. It maintains a chronological sequence of emotion
// vectors, each associated with a timestamp, and provides efficient interpolation
// and memory management capabilities.
//
// This class is thread-safe, it is safe to call any functions concurrently,
// with the exception of Destroy().
//
// It supports pre-allocated buffers to minimize runtime allocations, as well
// as buffer reuse when dropping already processed emotions.
class IEmotionAccumulator {
public:
  using timestamp_t = std::int64_t;

  // Push emotion at the end of the accumuator.
  // Emotions must be added in a strictly monotonically inscreasing timestamp order way.
  // This function is GPU async, but might allocate memory.
  virtual std::error_code Accumulate(timestamp_t timestamp, DeviceTensorFloatConstView emotion, cudaStream_t cudaStream) = 0;
  virtual std::error_code Accumulate(timestamp_t timestamp, HostTensorFloatConstView emotion, cudaStream_t cudaStream) = 0;

  // Close the accumulator, so no more emotions will be added to it.
  virtual std::error_code Close() = 0;

  // Read emotion from the accumulator.
  // This function is GPU async.
  virtual std::error_code Read(DeviceTensorFloatView destination, timestamp_t timestamp, cudaStream_t cudaStream) const = 0;

  // Signal the accumulator that emotions before a given timestamp can be discared.
  // The timestamp is the first one that might be still be accessed.
  // This function might return some memory for reuse by the accumulator, or do nothing if
  // not enough emotions are ready to be dropped yet.
  virtual std::error_code DropEmotionsBefore(timestamp_t timestamp) = 0;

  // Return the size of emotions in this accumulator.
  virtual std::size_t GetEmotionSize() const = 0;
  // Return whether the accumulator is empty.
  virtual bool IsEmpty() const = 0;
  // Return the last timestamp accumulated so far.
  virtual timestamp_t LastAccumulatedTimestamp() const = 0;
  // Return the last timestamp dropped so far.
  virtual timestamp_t LastDroppedTimestamp() const = 0;
  // Return the number of dropped emotions so far.
  virtual std::size_t NbDroppedEmotions() const = 0;
  // Return whether the accumulator has been closed.
  virtual bool IsClosed() const = 0;

  // Reset the accumulator to empty emotions.
  virtual std::error_code Reset() = 0;

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IEmotionAccumulator();
};

// Create an emotion accumulator for emotionSize values per emotion, pre-allocated with preallocatedBufferCount buffers
// holding emotionCountPerBuffer emotions.
AUDIO2X_SDK_EXPORT IEmotionAccumulator* CreateEmotionAccumulator(
  std::size_t emotionSize, std::size_t emotionCountPerBuffer, std::size_t preallocatedBufferCount
  );

} // namespace nva2x
