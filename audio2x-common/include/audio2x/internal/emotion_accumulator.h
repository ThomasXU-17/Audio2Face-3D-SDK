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

#include "audio2x/emotion_accumulator.h"
#include "audio2x/internal/animated_tensor.h"

#include <limits>
#include <mutex>

namespace nva2x {


class EmotionAccumulator : public IEmotionAccumulator {
public:
  std::error_code Allocate(std::size_t emotionSize, std::size_t emotionCountPerBuffer, std::size_t preallocatedBufferCount);
  std::error_code Deallocate();

  std::error_code Accumulate(timestamp_t timestamp, DeviceTensorFloatConstView emotion, cudaStream_t cudaStream) override;
  std::error_code Accumulate(timestamp_t timestamp, HostTensorFloatConstView emotion, cudaStream_t cudaStream) override;
  std::error_code Close() override;
  std::error_code Read(DeviceTensorFloatView destination, timestamp_t timestamp, cudaStream_t cudaStream) const override;
  std::error_code DropEmotionsBefore(timestamp_t timestamp) override;
  std::size_t GetEmotionSize() const override;
  bool IsEmpty() const override;
  timestamp_t LastAccumulatedTimestamp() const override;
  timestamp_t LastDroppedTimestamp() const override;
  std::size_t NbDroppedEmotions() const override;
  bool IsClosed() const override;
  std::error_code Reset() override;
  void Destroy() override;

private:
  template <typename TensorViewType>
  std::error_code Accumulate_Helper(timestamp_t timestamp, TensorViewType emotion, cudaStream_t cudaStream);
  std::error_code Reset_NoLock();

  // Synchronization is very coarse.
  // We could allow more parallelism when reading / writing don't overlap and don't move stuff around,
  // but it doesn't seem worth it.
  mutable std::mutex _mutex;

  AnimatedDeviceTensor _emotions;
  mutable AnimatedDeviceTensor::CachedAccessor _accessor;
  timestamp_t _lastAccumulatedTimestamp{std::numeric_limits<timestamp_t>::min()};
  timestamp_t _lastDroppedTimestamp{std::numeric_limits<timestamp_t>::min()};
  std::size_t _nbDroppedEmotions{0};
  bool _isClosed{false};
};

} // namespace nva2x
