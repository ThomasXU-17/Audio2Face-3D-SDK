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
#include "audio2x/internal/float_accumulator.h"
#include "audio2x/internal/audio2x.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/error.h"

#include <cassert>
#include <numeric>

namespace nva2x {

IFloatAccumulator::~IFloatAccumulator() = default;

std::error_code FloatAccumulator::Allocate(std::size_t tensorSize, std::size_t tensorCount) {
  std::lock_guard lock(_mutex);

  A2X_CHECK_RESULT(Reset_NoLock());
  A2X_CHECK_RESULT_WITH_MSG(_pool.Allocate(tensorSize, tensorCount), "Error allocating tensor pool");
  return ErrorCode::eSuccess;
}

std::error_code FloatAccumulator::Deallocate() {
  std::lock_guard lock(_mutex);

  _absoluteNextSample = 0;
  _nbDroppedSamples = 0;
  // No need to return the buffers to the pool, things are being destroyed anyways.
  _buffers.clear();
  A2X_CHECK_RESULT_WITH_MSG(_pool.Deallocate(), "Unable to deallocate tensor pool");
  return ErrorCode::eSuccess;
}

std::error_code FloatAccumulator::DropSamplesBefore(const std::size_t absoluteStartSample) {
  std::lock_guard lock(_mutex);

  A2X_CHECK_ERROR_WITH_MSG(_pool.TensorSize() != 0,
      "FloatAccumulator is not initialized", ErrorCode::eNotInitialized);
  A2X_CHECK_ERROR_WITH_MSG(absoluteStartSample <= _absoluteNextSample,
      "Trying to drop more samples than accumulated", ErrorCode::eOutOfBounds);

  auto relativeStartSample = absoluteStartSample - _nbDroppedSamples;
  while (!_buffers.empty() && _buffers.front()->Size() <= relativeStartSample) {
      const auto samplesToRemove = _buffers.front()->Size();
      _nbDroppedSamples += samplesToRemove;
      relativeStartSample -= samplesToRemove;
      A2X_CHECK_RESULT_WITH_MSG(_pool.Return(std::move(_buffers.front())), "Error returning tensor to pool");
      _buffers.pop_front();
  }

  return ErrorCode::eSuccess;
}


namespace {

// Helper to traverse the accumulator to write and read.
template <typename Func>
std::error_code Traverse(
  const std::deque<std::unique_ptr<DeviceTensorFloat>>& buffers,
  std::size_t tensorSize,
  std::size_t startSample,
  std::size_t size,
  Func&& func) {
  const auto firstBufferIndex = startSample / tensorSize;
  const auto firstSampleIndex = startSample % tensorSize;
  const auto lastBufferIndex = (startSample + size - 1) / tensorSize;
  [[maybe_unused]] const auto lastSampleIndex = (startSample + size - 1) % tensorSize;

  if (firstBufferIndex == lastBufferIndex) {
    // All samples are in the same buffer.
    assert(firstSampleIndex <= lastSampleIndex);
    assert(lastSampleIndex + 1 - firstSampleIndex == size);
    const auto buffer = buffers[firstBufferIndex]->View(firstSampleIndex, size);
    A2X_CHECK_ERROR_WITH_MSG(buffer.Data(), "Unable to get region to access samples", ErrorCode::eInvalidValue);
    A2X_CHECK_RESULT(func(buffer));
  }
  else {
    assert(firstBufferIndex < lastBufferIndex);
    // Samples need to be accessed from the start sample to the end of the buffer,
    // Then all buffers in between,
    // Then from the beginning of the last buffer to the last sample.
    const auto firstBufferView = buffers[firstBufferIndex]->View(firstSampleIndex, tensorSize - firstSampleIndex);
    assert(firstBufferView.Data());
    A2X_CHECK_RESULT(func(firstBufferView));

    for (auto i = firstBufferIndex + 1; i < lastBufferIndex; ++i) {
      A2X_CHECK_RESULT(func(*buffers[i]));
    }

    const auto lastBufferView = buffers[lastBufferIndex]->View(0, lastSampleIndex + 1);
    assert(lastBufferView.Data());
    A2X_CHECK_RESULT(func(lastBufferView));
  }

  return ErrorCode::eSuccess;
}

} // namespace


template <typename TensorViewType>
std::error_code FloatAccumulator::Accumulate_Helper(TensorViewType source, cudaStream_t cudaStream) {
  std::lock_guard lock(_mutex);

  A2X_CHECK_ERROR_WITH_MSG(_pool.TensorSize() != 0, "FloatAccumulator is not initialized", ErrorCode::eNotInitialized);
  A2X_CHECK_ERROR_WITH_MSG(source.Data() != nullptr, "Empty samples received", ErrorCode::eInvalidValue);
  A2X_CHECK_ERROR_WITH_MSG(source.Size() > 0, "Empty samples received", ErrorCode::eInvalidValue);
  A2X_CHECK_ERROR_WITH_MSG(!_isClosed, "FloatAccumulator is already closed", ErrorCode::eInvalidValue);

  const auto tensorSize = _pool.TensorSize();
  const auto relativeNextSample = _absoluteNextSample - _nbDroppedSamples;
  const auto lastBufferIndex = (relativeNextSample + source.Size() - 1) / tensorSize;

  // Make sure there are enough GPU buffers.
  while (_buffers.size() <= lastBufferIndex) {
    auto tensor = _pool.Obtain();
    A2X_CHECK_ERROR_WITH_MSG(tensor, "Unable to get tensor from pool", ErrorCode::eNullPointer);
    _buffers.emplace_back(std::move(tensor));
  }

  // Upload the samples to the GPU.
  auto sourceToCopy = source;
  auto accumulate = [&sourceToCopy, cudaStream] (const DeviceTensorFloatView destination) -> std::error_code {
    const auto sourceView = sourceToCopy.View(0, destination.Size());
    assert(sourceView.Data());

    if constexpr (std::is_same_v<TensorViewType, HostTensorFloatConstView>) {
      A2X_CHECK_RESULT_WITH_MSG(CopyHostToDevice(destination, sourceView, cudaStream),
        "Unable to copy host tensor to acccumulator");
    }
    else {
      static_assert(std::is_same_v<TensorViewType, DeviceTensorFloatConstView>);
      A2X_CHECK_RESULT_WITH_MSG(CopyDeviceToDevice(destination, sourceView, cudaStream),
        "Unable to copy device tensor to acccumulator");
    }

    sourceToCopy = sourceToCopy.View(sourceView.Size(), sourceToCopy.Size() - sourceView.Size());

    return ErrorCode::eSuccess;
  };

  A2X_CHECK_RESULT(Traverse(_buffers, tensorSize, relativeNextSample, source.Size(), std::move(accumulate)));

  assert(sourceToCopy.Size() == 0);

  _absoluteNextSample += source.Size();

  return ErrorCode::eSuccess;
}

std::error_code FloatAccumulator::Accumulate(HostTensorFloatConstView samples, cudaStream_t cudaStream) {
  return Accumulate_Helper(samples, cudaStream);
}

std::error_code FloatAccumulator::Accumulate(DeviceTensorFloatConstView samples, cudaStream_t cudaStream) {
  return Accumulate_Helper(samples, cudaStream);
}

std::error_code FloatAccumulator::Close() {
  std::lock_guard lock(_mutex);

  A2X_CHECK_ERROR_WITH_MSG(!_isClosed, "FloatAccumulator is already closed", ErrorCode::eInvalidValue);
  _isClosed = true;
  return ErrorCode::eSuccess;
}

std::error_code FloatAccumulator::Read(
  DeviceTensorFloatView destination, timestamp_t absoluteStartSample, cudaStream_t cudaStream) const {
  std::lock_guard lock(_mutex);

  A2X_CHECK_ERROR_WITH_MSG(_pool.TensorSize() != 0, "FloatAccumulator is not initialized", ErrorCode::eNotInitialized);
  A2X_CHECK_ERROR_WITH_MSG(destination.Data() != nullptr, "Empty destination received", ErrorCode::eInvalidValue);
  A2X_CHECK_ERROR_WITH_MSG(destination.Size() > 0, "Empty destination received", ErrorCode::eInvalidValue);
  A2X_CHECK_ERROR_WITH_MSG(_nbDroppedSamples == 0 ||
    absoluteStartSample >= static_cast<timestamp_t>(_nbDroppedSamples),
    "Trying to read discarded samples", ErrorCode::eInvalidValue);
  A2X_CHECK_ERROR_WITH_MSG(_isClosed ||
    absoluteStartSample + static_cast<timestamp_t>(destination.Size()) <= static_cast<timestamp_t>(_absoluteNextSample),
    "Trying to read samples beyond accumulated samples", ErrorCode::eInvalidValue);

  // First, handle pre-pending zeros.
  if (0 > absoluteStartSample) {
    // We need to pre-pend with some zeros.
    const auto offsetBeforeBegin = static_cast<std::size_t>(-absoluteStartSample);
    if (destination.Size() > offsetBeforeBegin) {
      // Some samples will be read from the first buffer.
      const auto prependView = destination.View(0, offsetBeforeBegin);
      A2X_CHECK_RESULT_WITH_MSG(nva2x::FillOnDevice(prependView, 0.0f, cudaStream), "Unable to prepend with zeros");
      destination = destination.View(offsetBeforeBegin, destination.Size() - offsetBeforeBegin);
      absoluteStartSample = 0;
    }
    else {
      // All samples are read before the first buffer.
      const auto prependView = destination;
      A2X_CHECK_RESULT_WITH_MSG(nva2x::FillOnDevice(prependView, 0.0f, cudaStream), "Unable to prepend with zeros");
      return ErrorCode::eSuccess;
    }
  }

  // Second, handle post-pending zero.
  if (absoluteStartSample + destination.Size() > _absoluteNextSample) {
    // We need to post-pend with some zeros.
    const auto offsetAfterEnd = absoluteStartSample + destination.Size() - _absoluteNextSample;
    if (destination.Size() > offsetAfterEnd) {
      // Some samples will be read from the last buffer.
      const auto postpendView = destination.View(destination.Size() - offsetAfterEnd, offsetAfterEnd);
      A2X_CHECK_RESULT_WITH_MSG(nva2x::FillOnDevice(postpendView, 0.0f, cudaStream), "Unable to postpend with zeros");
      destination = destination.View(0, destination.Size() - offsetAfterEnd);
    }
    else {
      // All samples are read after the last buffer.
      const auto postpendView = destination;
      A2X_CHECK_RESULT_WITH_MSG(nva2x::FillOnDevice(postpendView, 0.0f, cudaStream), "Unable to postpend with zeros");
      return ErrorCode::eSuccess;
    }
  }

  // Third, copy the real data.
  assert(destination.Size() > 0);
  assert(absoluteStartSample >= static_cast<timestamp_t>(_nbDroppedSamples));
  const auto relativeStartSample = absoluteStartSample - _nbDroppedSamples;
  const auto tensorSize = _pool.TensorSize();

  auto destinationToCopy = destination;
  auto accumulate = [&destinationToCopy, cudaStream] (const DeviceTensorFloatConstView source) -> std::error_code {
    const auto destinationView = destinationToCopy.View(0, source.Size());
    assert(destinationView.Data());
    A2X_CHECK_RESULT_WITH_MSG(nva2x::CopyDeviceToDevice(destinationView, source, cudaStream),
      "Unable to copy from acccumulator");

    destinationToCopy = destinationToCopy.View(destinationView.Size(), destinationToCopy.Size() - destinationView.Size());

    return ErrorCode::eSuccess;
  };

  A2X_CHECK_RESULT(Traverse(_buffers, tensorSize, relativeStartSample, destination.Size(), std::move(accumulate)));

  assert(destinationToCopy.Size() == 0);

  return ErrorCode::eSuccess;
}

std::size_t FloatAccumulator::NbAccumulatedSamples() const {
  std::lock_guard lock(_mutex);

  return _absoluteNextSample;
}

std::size_t FloatAccumulator::NbDroppedSamples() const {
  std::lock_guard lock(_mutex);

  return _nbDroppedSamples;
}

bool FloatAccumulator::IsClosed() const {
  std::lock_guard lock(_mutex);

  return _isClosed;
}

std::error_code FloatAccumulator::Reset() {
  std::lock_guard lock(_mutex);

  return Reset_NoLock();
}

void FloatAccumulator::Destroy() {
  A2X_LOG_DEBUG("FloatAccumulator::Destroy()");
  delete this;
}

std::error_code FloatAccumulator::Reset_NoLock() {
  _absoluteNextSample = 0;
  _nbDroppedSamples = 0;
  _isClosed = false;
  for (auto& tensor : _buffers) {
      A2X_CHECK_RESULT_WITH_MSG(_pool.Return(std::move(tensor)), "Error returning tensor to pool");
  }
  _buffers.clear();
  return ErrorCode::eSuccess;
}

} // namespace nva2x


nva2x::IFloatAccumulator* nva2x::internal::CreateFloatAccumulator(std::size_t tensorSize, std::size_t tensorCount) {
  A2X_LOG_DEBUG("CreateFloatAccumulator()");
  auto accumulator = std::make_unique<FloatAccumulator>();
  if (accumulator->Allocate(tensorSize, tensorCount)) {
    A2X_LOG_ERROR("Unable to create float accumulator");
    return nullptr;
  }
  return accumulator.release();
}
