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

#include "audio2x/audio_accumulator.h"
#include "audio2x/internal/tensor_pool.h"

#include <array>
#include <deque>
#include <mutex>

namespace nva2x {


class AudioAccumulator final : public IAudioAccumulator {
public:
  std::error_code Allocate(std::size_t tensorSize, std::size_t tensorCount);
  std::error_code Deallocate();

  std::error_code Accumulate(HostTensorFloatConstView samples, cudaStream_t cudaStream) override;
  std::error_code Accumulate(DeviceTensorFloatConstView samples, cudaStream_t cudaStream) override;
  std::error_code Close() override;
  std::error_code Read(
    DeviceTensorFloatView destination, timestamp_t absoluteStartSample, float inputStrength, cudaStream_t cudaStream
    ) const override;
  std::error_code DropSamplesBefore(std::size_t absoluteStartSample) override;
  std::size_t NbAccumulatedSamples() const override;
  std::size_t NbDroppedSamples() const override;
  bool IsClosed() const override;
  std::error_code Reset() override;
  void Destroy() override;

private:
  std::error_code Reset_NoLock();

  // Synchronization is very coarse.
  // We could allow more parallelism when reading / writing don't overlap and don't move stuff around,
  // but it doesn't seem worth it.
  mutable std::mutex _mutex;

  DeviceTensorPool _pool;
  std::deque<std::unique_ptr<DeviceTensorFloat>> _audio;
  std::size_t _absoluteNextSample{0};
  std::size_t _nbDroppedSamples{0};
  bool _isClosed{false};
};


// Progress parameters describing a sliding window which will be retrieved from
// the accumulator.
struct WindowProgressParameters {
  using timestamp_t = IAudioAccumulator::timestamp_t;

  // Size of the sliding window which will be retrieved.
  std::size_t windowSize{0};
  // Offset from which the window starts.
  timestamp_t startOffset{0};
  // Sample where the inference will be in the window.
  timestamp_t targetOffset{0};
  // Stride of the sliding window between each retrieval.
  // It is expressed as a fraction / rational number to be able to exactly represent
  // common frame rate (even thought they don't divide perfectly the sample rate)
  // as well as tricky ones like 29.97 (30000/1001).
  std::size_t strideNum{0};
  std::size_t strideDenom{0};
};

// State of progress as windows are read from the accumulator.
class WindowProgress {
public:
  using timestamp_t = WindowProgressParameters::timestamp_t;

  WindowProgress(const WindowProgressParameters& params);

  // Get, for the next window to read, the start, target and end (first after the window)
  // sample in absolute sample space.
  void GetCurrentWindow(timestamp_t& start, timestamp_t& target, timestamp_t& end, std::size_t readWindowOffset = 0) const;
  std::array<timestamp_t, 3> GetCurrentWindow(std::size_t readWindowOffset = 0) const;
  // Get, for the index of the given window, the start, target and end (first after the window)
  // sample in absolute sample space.
  void GetWindow(timestamp_t& start, timestamp_t& target, timestamp_t& end, std::size_t readWindowIndex) const;
  std::array<timestamp_t, 3> GetWindow(std::size_t readWindowIndex) const;
  // Return the number of windows that can be extracted with the current state,
  // where the timestamp is the one right after the last readable sample.
  std::size_t GetNbAvailableWindows(timestamp_t endTimestamp, bool isClosed) const;
  // Return the number of windows that can be extracted with the current state.
  // For convenience, this function taking a number of samples is provided, but simply
  // calls the other one with the end timestamp.
  std::size_t GetNbAvailableWindows(std::size_t nbAccumulatedSamples, bool isClosed) const;

  // Get the read window count.
  inline std::size_t GetReadWindowCount() const { return _readWindowCount; }
  // Increment the read window count.
  inline void IncrementReadWindowCount(std::size_t increment = 1) { _readWindowCount += increment; }
  // Reset the read window count.
  inline void ResetReadWindowCount(std::size_t readWindowCount = 0) { _readWindowCount = readWindowCount; }

  // Get window progress parameters.
  inline const WindowProgressParameters& GetParameters() const { return _params; }

 private:
  // Return the start of the next window to read, in absolute sample space.
  timestamp_t GetStartSample(std::size_t readWindowIndex) const;

  WindowProgressParameters _params;
  std::size_t _readWindowCount{0};
};


// Return an equivalent window progress object which advances per frame.
WindowProgress GetFrameProgress(const WindowProgress& progress, std::size_t nbFramesPerStride);
// Counts the number of frame in a given audio.  Returns 0 if the accumulator is not closed.
std::size_t GetTotalNbFrames(
  const IAudioAccumulator& audioAccumulator, const WindowProgress& progress, std::size_t nbFramesPerStride
  );


} // namespace nva2x
