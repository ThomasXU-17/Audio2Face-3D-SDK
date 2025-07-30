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

#include "audio2x/internal/tensor_pool.h"

#include <deque>

namespace nva2x {


// This class is not thread-safe.  Any synchronization must be done externally.
class AnimatedDeviceTensor {
public:
  using timestamp_t = std::int64_t;

  std::error_code Allocate(std::size_t keySize, std::size_t keyCountPerBuffer, std::size_t preallocatedBufferCount);
  std::error_code Deallocate();

  std::error_code AddKey(timestamp_t timestamp, DeviceTensorFloatConstView key, cudaStream_t cudaStream);
  std::error_code AddKey(timestamp_t timestamp, HostTensorFloatConstView key, cudaStream_t cudaStream);

  size_t GetKeyCount() const;
  std::error_code GetKeyTimestamp(timestamp_t& output, std::size_t index) const;
  std::error_code GetKeyValue(DeviceTensorFloatView output, std::size_t index, cudaStream_t cudaStream) const;

  std::error_code FindSurroundingKeys(std::size_t& indexBefore, std::size_t& indexAfter, timestamp_t timestamp) const;
  std::error_code Sample(DeviceTensorFloatView output, timestamp_t timestamp, cudaStream_t cudaStream) const;

  std::error_code DropKeysBefore(timestamp_t timestamp);
  std::error_code Reset();

  inline std::size_t GetKeySize() const { return _keySize; }

  class CachedAccessor;
  CachedAccessor GetAccessor() const;

private:
  template <typename TensorViewType>
  std::error_code AddKey_Helper(timestamp_t timestamp, TensorViewType key, cudaStream_t cudaStream);
  std::error_code Sample(
    DeviceTensorFloatView output,
    timestamp_t timestamp,
    std::size_t indexBefore,
    std::size_t indexAfter,
    cudaStream_t cudaStream
    ) const;
  DeviceTensorFloatConstView GetKeyView(std::size_t index) const;

  DeviceTensorPool _pool;
  std::deque<std::unique_ptr<DeviceTensorFloat>> _keyBuffers;
  std::deque<timestamp_t> _timestamps;
  std::size_t _keySize{0};
  std::size_t _keyCountPerBuffer{0};
};


class AnimatedDeviceTensor::CachedAccessor
{
public:
  CachedAccessor() = default;
  std::error_code FindSurroundingKeys(std::size_t& indexBefore, std::size_t& indexAfter, timestamp_t timestamp);
  std::error_code Sample(DeviceTensorFloatView output, timestamp_t timestamp, cudaStream_t cudaStream);

private:
  friend class AnimatedDeviceTensor;
  CachedAccessor(const AnimatedDeviceTensor& animatedTensor);

  const AnimatedDeviceTensor* _animatedTensor{nullptr};
  std::size_t _lastIndexBefore{0};
};


} // namespace nva2x
