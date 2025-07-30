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
#include "audio2x/internal/emotion_accumulator.h"
#include "audio2x/internal/audio2x.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/error.h"

#include <cassert>

namespace nva2x {


IEmotionAccumulator::~IEmotionAccumulator() = default;

std::error_code EmotionAccumulator::Allocate(
    std::size_t emotionSize, std::size_t emotionCountPerBuffer, std::size_t preallocatedBufferCount
    ) {
    std::lock_guard lock(_mutex);

    A2X_CHECK_RESULT(Reset_NoLock());
    A2X_CHECK_RESULT_WITH_MSG(
        _emotions.Allocate(emotionSize, emotionCountPerBuffer, preallocatedBufferCount),
        "Error allocating tensor pool"
        );
    _accessor = _emotions.GetAccessor();

    return ErrorCode::eSuccess;
}

std::error_code EmotionAccumulator::Deallocate() {
    std::lock_guard lock(_mutex);

    return Reset_NoLock();
}

std::error_code EmotionAccumulator::Accumulate(
    timestamp_t timestamp, DeviceTensorFloatConstView emotion, cudaStream_t cudaStream
    ) {
    return Accumulate_Helper(timestamp, emotion, cudaStream);
}

std::error_code EmotionAccumulator::Accumulate(
    timestamp_t timestamp, HostTensorFloatConstView emotion, cudaStream_t cudaStream
    ) {
    return Accumulate_Helper(timestamp, emotion, cudaStream);
}

std::error_code EmotionAccumulator::Close() {
    std::lock_guard lock(_mutex);

    A2X_CHECK_ERROR_WITH_MSG(!_isClosed, "EmotionAccumulator is already closed", ErrorCode::eInvalidValue);
    A2X_CHECK_ERROR_WITH_MSG(
        _emotions.GetKeyCount() != 0,
        "EmotionAccumulator is empty, accumulate at least one emotion before closing",
        ErrorCode::eInvalidValue
        );
    _isClosed = true;
    return ErrorCode::eSuccess;
}

std::error_code EmotionAccumulator::Read(
    DeviceTensorFloatView destination, timestamp_t timestamp, cudaStream_t cudaStream
    ) const {
    std::lock_guard lock(_mutex);

    A2X_CHECK_ERROR_WITH_MSG(_emotions.GetKeySize() != 0, "EmotionAccumulator is not initialized", ErrorCode::eNotInitialized);
    A2X_CHECK_ERROR_WITH_MSG(destination.Data() != nullptr, "Empty destination received", ErrorCode::eInvalidValue);
    A2X_CHECK_ERROR_WITH_MSG(destination.Size() == _emotions.GetKeySize(), "Destination of wrong size received", ErrorCode::eMismatch);
    A2X_CHECK_ERROR_WITH_MSG(
        _lastDroppedTimestamp <= timestamp,
        "Trying to read discarded emotions", ErrorCode::eOutOfBounds
        );
    A2X_CHECK_ERROR_WITH_MSG(
        _isClosed || timestamp <= _lastAccumulatedTimestamp,
        "Trying to read emotions passed accumulated emotions", ErrorCode::eOutOfBounds
        );

    A2X_CHECK_RESULT_WITH_MSG(
        _accessor.Sample(destination, timestamp, cudaStream), "Unable to sample emotion"
    );

    return ErrorCode::eSuccess;
}

std::error_code EmotionAccumulator::DropEmotionsBefore(const timestamp_t timestamp) {
    std::lock_guard lock(_mutex);

    A2X_CHECK_ERROR_WITH_MSG(_emotions.GetKeySize() != 0, "EmotionAccumulator is not initialized", ErrorCode::eNotInitialized);
    A2X_CHECK_ERROR_WITH_MSG(_isClosed || timestamp <= _lastAccumulatedTimestamp, "Trying to drop more emotions than accumulated", ErrorCode::eOutOfBounds);

    const auto countBefore = _emotions.GetKeyCount();
    A2X_CHECK_RESULT_WITH_MSG(_emotions.DropKeysBefore(timestamp), "Unable to drop samples");
    const auto countAfter = _emotions.GetKeyCount();
    assert(countBefore >= countAfter);

    _lastDroppedTimestamp = timestamp;
    _nbDroppedEmotions += countBefore - countAfter;

    return ErrorCode::eSuccess;
}

std::size_t EmotionAccumulator::GetEmotionSize() const {
    std::lock_guard lock(_mutex);

    return _emotions.GetKeySize();
}

bool EmotionAccumulator::IsEmpty() const {
    std::lock_guard lock(_mutex);

    return _emotions.GetKeyCount() == 0;
}

EmotionAccumulator::timestamp_t EmotionAccumulator::LastAccumulatedTimestamp() const {
    std::lock_guard lock(_mutex);

    return _lastAccumulatedTimestamp;
}

EmotionAccumulator::timestamp_t EmotionAccumulator::LastDroppedTimestamp() const {
    std::lock_guard lock(_mutex);

    return _lastDroppedTimestamp;
}

std::size_t EmotionAccumulator::NbDroppedEmotions() const {
    std::lock_guard lock(_mutex);

    return _nbDroppedEmotions;
}

bool EmotionAccumulator::IsClosed() const {
    std::lock_guard lock(_mutex);

    return _isClosed;
}

std::error_code EmotionAccumulator::Reset() {
    std::lock_guard lock(_mutex);

    return Reset_NoLock();
}

void EmotionAccumulator::Destroy() {
    A2X_LOG_DEBUG("EmotionAccumulator::Destroy()");
    delete this;
}

template <typename TensorViewType>
std::error_code EmotionAccumulator::Accumulate_Helper(
    timestamp_t timestamp, TensorViewType emotion, cudaStream_t cudaStream
    ) {
    std::lock_guard lock(_mutex);

#ifndef NDEBUG
    if (_emotions.GetKeyCount() != 0) {
        timestamp_t lastTimestamp;
        assert(!_emotions.GetKeyTimestamp(lastTimestamp, _emotions.GetKeyCount() - 1));
        assert(lastTimestamp == _lastAccumulatedTimestamp);
    }
#endif

    A2X_CHECK_ERROR_WITH_MSG(_emotions.GetKeySize() != 0, "EmotionAccumulator is not initialized", ErrorCode::eNotInitialized);
    A2X_CHECK_ERROR_WITH_MSG(_lastAccumulatedTimestamp < timestamp, "Emotions must be accumulated in increasing timestamp order", ErrorCode::eInvalidValue);
    A2X_CHECK_ERROR_WITH_MSG(emotion.Data() != nullptr, "Empty emotion received", ErrorCode::eInvalidValue);
    A2X_CHECK_ERROR_WITH_MSG(emotion.Size() == _emotions.GetKeySize(), "Emotion of wrong size received", ErrorCode::eMismatch);
    A2X_CHECK_ERROR_WITH_MSG(!_isClosed, "EmotionAccumulator is already closed", ErrorCode::eInvalidValue);

    A2X_CHECK_RESULT_WITH_MSG(
        _emotions.AddKey(timestamp, emotion, cudaStream), "Unable to add emotion"
        )

    _lastAccumulatedTimestamp = timestamp;

    return ErrorCode::eSuccess;
}

std::error_code EmotionAccumulator::Reset_NoLock() {
    _lastAccumulatedTimestamp = std::numeric_limits<timestamp_t>::min();
    _lastDroppedTimestamp = std::numeric_limits<timestamp_t>::min();
    _nbDroppedEmotions = 0;
    _isClosed = false;

    A2X_CHECK_RESULT_WITH_MSG(_emotions.Reset(), "Unable to reset emotions");
    _accessor = _emotions.GetAccessor();

    return ErrorCode::eSuccess;
}


} // namespace nva2x


nva2x::IEmotionAccumulator* nva2x::internal::CreateEmotionAccumulator(
  std::size_t emotionSize, std::size_t emotionCountPerBuffer, std::size_t preallocatedBufferCount
  ) {
  A2X_LOG_DEBUG("CreateEmotionAccumulator()");
  auto accumulator = std::make_unique<EmotionAccumulator>();
  if (accumulator->Allocate(emotionSize, emotionCountPerBuffer, preallocatedBufferCount)) {
    A2X_LOG_ERROR("Unable to create emotion accumulator");
    return nullptr;
  }
  return accumulator.release();
}
