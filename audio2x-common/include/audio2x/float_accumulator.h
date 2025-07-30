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

// This class provides a thread-safe, GPU-accelerated audio sample accumulator
// that efficiently manages memory using a tensor pool. It supports continuous streaming
// of audio data with efficient memory management and random access to accumulated samples.
//
// This class is thread-safe, it is safe to call any functions concurrently,
// with the exception of Destroy().
//
// It supports pre-allocated buffers to minimize runtime allocations, as well
// as buffer reuse when dropping already processed samples.
class IFloatAccumulator {
public:
  // Timestamp as a sample index, allowed to be negative.
  // Timestamps can be converted to time by being divided by the sampling rate.
  using timestamp_t = std::int64_t;

  // Accumulate samples at the end of the accumulator buffer.
  // This function is GPU async and may allocate additional memory if needed.
  // The samples are copied to GPU memory and stored in the internal buffer.
  virtual std::error_code Accumulate(HostTensorFloatConstView samples, cudaStream_t cudaStream) = 0;
  virtual std::error_code Accumulate(DeviceTensorFloatConstView samples, cudaStream_t cudaStream) = 0;

  // Close the accumulator, so no more data will be added to it.
  virtual std::error_code Close() = 0;

  // Read samples from the accumulator starting at the specified timestamp.
  // The start sample is specified as an absolute sample index since the beginning
  // of the accumulated data. Negative values are supported for lookback scenarios.
  // This function is GPU async and copies data to the destination tensor.
  virtual std::error_code Read(
    DeviceTensorFloatView destination, timestamp_t absoluteStartSample, cudaStream_t cudaStream) const = 0;

  // Signal the accumulator that samples before the specified sample can be discarded.
  // This allows the accumulator to free memory by returning buffers to the tensor pool.
  // The start sample is the first sample NOT to be dropped (inclusive).
  // This function may return memory for reuse or do nothing if not enough samples
  // are ready to be dropped yet.
  virtual std::error_code DropSamplesBefore(std::size_t absoluteStartSample) = 0;

  // Return the total number of samples accumulated so far, regardless of how many
  // samples have been dropped. This represents the total length of the data stream.
  virtual std::size_t NbAccumulatedSamples() const = 0;

  // Return the number of samples that have been dropped so far.
  virtual std::size_t NbDroppedSamples() const = 0;

  // Return whether the accumulator has been closed.
  // A closed accumulator cannot accumulate new samples but can still be read from.
  virtual bool IsClosed() const = 0;

  // Reset the accumulator to an empty state, clearing all accumulated data.
  // This frees all memory and resets internal counters, allowing the accumulator
  // to be reused for a new data stream.
  virtual std::error_code Reset() = 0;

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IFloatAccumulator();
};

// Create a float accumulator pre-allocated with the specified buffer configuration.
// The accumulator will be initialized with tensorCount buffers, each capable of
// holding tensorSize float values. This pre-allocation helps avoid memory
// allocation during streaming operations.
AUDIO2X_SDK_EXPORT IFloatAccumulator* CreateFloatAccumulator(std::size_t tensorSize, std::size_t tensorCount);

} // namespace nva2x
