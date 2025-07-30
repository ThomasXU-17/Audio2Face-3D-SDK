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
#include "audio2x/internal/audio_accumulator.h"
#include "audio2x/internal/audio_accumulator_cuda.h"
#include "audio2x/internal/audio2x.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/error.h"

#include <cassert>
#include <numeric>

namespace nva2x {


IAudioAccumulator::~IAudioAccumulator() = default;

std::error_code AudioAccumulator::Allocate(std::size_t tensorSize, std::size_t tensorCount) {
    std::lock_guard lock(_mutex);

    A2X_CHECK_RESULT(Reset_NoLock());
    A2X_CHECK_RESULT_WITH_MSG(_pool.Allocate(tensorSize, tensorCount), "Error allocating tensor pool");
    return ErrorCode::eSuccess;
}

std::error_code AudioAccumulator::Deallocate() {
    std::lock_guard lock(_mutex);

    _absoluteNextSample = 0;
    _nbDroppedSamples = 0;
    // No need to return the buffers to the pool, things are being destroyed anyways.
    _audio.clear();
    A2X_CHECK_RESULT_WITH_MSG(_pool.Deallocate(), "Unable to deallocate tensor pool");
    return ErrorCode::eSuccess;
}

std::error_code AudioAccumulator::DropSamplesBefore(const std::size_t absoluteStartSample) {
    std::lock_guard lock(_mutex);

    A2X_CHECK_ERROR_WITH_MSG(_pool.TensorSize() != 0, "AudioAccumulator is not initialized", ErrorCode::eNotInitialized);
    A2X_CHECK_ERROR_WITH_MSG(absoluteStartSample <= _absoluteNextSample, "Trying to drop more samples than accumulated", ErrorCode::eOutOfBounds);

    auto relativeStartSample = absoluteStartSample - _nbDroppedSamples;
    while (!_audio.empty() && _audio.front()->Size() <= relativeStartSample) {
        const auto samplesToRemove = _audio.front()->Size();
        _nbDroppedSamples += samplesToRemove;
        relativeStartSample -= samplesToRemove;
        A2X_CHECK_RESULT_WITH_MSG(_pool.Return(std::move(_audio.front())), "Error returning tensor to pool");
        _audio.pop_front();
    }

    return ErrorCode::eSuccess;
}

namespace {

    // Helper to traverse the accumulator to write and read.
    template <typename Func>
    std::error_code Traverse(
        const std::deque<std::unique_ptr<DeviceTensorFloat>>& audio,
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
            const auto buffer = audio[firstBufferIndex]->View(firstSampleIndex, size);
            A2X_CHECK_ERROR_WITH_MSG(buffer.Data(), "Unable to get region to access samples", ErrorCode::eInvalidValue);
            A2X_CHECK_RESULT(func(buffer));
        }
        else {
            assert(firstBufferIndex < lastBufferIndex);
            // Samples need to be accessed from the start sample to the end of the buffer,
            // Then all buffers in between,
            // Then from the beginning of the last buffer to the last sample.
            const auto firstBufferView = audio[firstBufferIndex]->View(firstSampleIndex, tensorSize - firstSampleIndex);
            assert(firstBufferView.Data());
            A2X_CHECK_RESULT(func(firstBufferView));

            for (auto i = firstBufferIndex + 1; i < lastBufferIndex; ++i) {
                A2X_CHECK_RESULT(func(*audio[i]));
            }

            const auto lastBufferView = audio[lastBufferIndex]->View(0, lastSampleIndex + 1);
            assert(lastBufferView.Data());
            A2X_CHECK_RESULT(func(lastBufferView));
        }

        return ErrorCode::eSuccess;
    }
}

std::error_code AudioAccumulator::Accumulate(
    const HostTensorFloatConstView source, cudaStream_t cudaStream
    ) {
    std::lock_guard lock(_mutex);

    A2X_CHECK_ERROR_WITH_MSG(_pool.TensorSize() != 0, "AudioAccumulator is not initialized", ErrorCode::eNotInitialized);
    A2X_CHECK_ERROR_WITH_MSG(source.Data() != nullptr, "Empty samples received", ErrorCode::eInvalidValue);
    A2X_CHECK_ERROR_WITH_MSG(source.Size() > 0, "Empty samples received", ErrorCode::eInvalidValue);
    A2X_CHECK_ERROR_WITH_MSG(!_isClosed, "AudioAccumulator is already closed", ErrorCode::eInvalidValue);

    const auto tensorSize = _pool.TensorSize();
    const auto relativeNextSample = _absoluteNextSample - _nbDroppedSamples;
    const auto lastBufferIndex = (relativeNextSample + source.Size() - 1) / tensorSize;

    // Make sure there are enough GPU buffers.
    while (_audio.size() <= lastBufferIndex) {
        auto tensor = _pool.Obtain();
        A2X_CHECK_ERROR_WITH_MSG(tensor, "Unable to get tensor from pool", ErrorCode::eNullPointer);
        _audio.emplace_back(std::move(tensor));
    }

    // Upload and multiply by input strength.
    auto sourceToCopy = source;
    auto accumulate = [&sourceToCopy, cudaStream](
        const DeviceTensorFloatView destination
        ) -> std::error_code {
        const auto sourceView = sourceToCopy.View(0, destination.Size());
        assert(sourceView.Data());
        A2X_CHECK_RESULT_WITH_MSG(
            nva2x::CopyHostToDevice(destination, sourceView, cudaStream),
            "Unable to copy to acccumulator"
        );

        sourceToCopy = sourceToCopy.View(sourceView.Size(), sourceToCopy.Size() - sourceView.Size());

        return ErrorCode::eSuccess;
    };

    A2X_CHECK_RESULT(
        Traverse(_audio, tensorSize, relativeNextSample, source.Size(), std::move(accumulate))
    );

    assert(sourceToCopy.Size() == 0);

    _absoluteNextSample += source.Size();

    return ErrorCode::eSuccess;
}

std::error_code AudioAccumulator::Accumulate([[maybe_unused]] DeviceTensorFloatConstView samples, [[maybe_unused]] cudaStream_t cudaStream) {
    A2X_LOG_DEBUG("Call made to unimplemented AudioAccumulator::Accumulate(DeviceTensorFloatConstView samples, cudaStream_t cudaStream)");
    return ErrorCode::eSuccess;
}

std::error_code AudioAccumulator::Close() {
    std::lock_guard lock(_mutex);

    A2X_CHECK_ERROR_WITH_MSG(!_isClosed, "AudioAccumulator is already closed", ErrorCode::eInvalidValue);
    _isClosed = true;
    return ErrorCode::eSuccess;
}

std::error_code AudioAccumulator::Read(
    DeviceTensorFloatView destination, timestamp_t absoluteStartSample, float inputStrength, cudaStream_t cudaStream
    ) const {
    std::lock_guard lock(_mutex);

    A2X_CHECK_ERROR_WITH_MSG(_pool.TensorSize() != 0, "AudioAccumulator is not initialized", ErrorCode::eNotInitialized);
    A2X_CHECK_ERROR_WITH_MSG(destination.Data() != nullptr, "Empty destination received", ErrorCode::eInvalidValue);
    A2X_CHECK_ERROR_WITH_MSG(destination.Size() > 0, "Empty destination received", ErrorCode::eInvalidValue);
    A2X_CHECK_ERROR_WITH_MSG(
        _nbDroppedSamples == 0 || absoluteStartSample >= static_cast<timestamp_t>(_nbDroppedSamples),
        "Trying to read discarded samples", ErrorCode::eInvalidValue
        );
    A2X_CHECK_ERROR_WITH_MSG(
        _isClosed || absoluteStartSample + static_cast<timestamp_t>(destination.Size()) <= static_cast<timestamp_t>(_absoluteNextSample),
        "Trying to read samples beyond accumulated samples", ErrorCode::eInvalidValue
        );

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
    auto accumulate = [&destinationToCopy, inputStrength, cudaStream](
        const DeviceTensorFloatConstView source
        ) -> std::error_code {
        const auto destinationView = destinationToCopy.View(0, source.Size());
        assert(destinationView.Data());
        A2X_CHECK_RESULT_WITH_MSG(
            nva2x::CopyDeviceToDevice(destinationView, source, cudaStream),
            "Unable to copy from acccumulator"
        );

        destinationToCopy = destinationToCopy.View(destinationView.Size(), destinationToCopy.Size() - destinationView.Size());

        return ErrorCode::eSuccess;
    };

    A2X_CHECK_RESULT(
        Traverse(_audio, tensorSize, relativeStartSample, destination.Size(), std::move(accumulate))
    );

    assert(destinationToCopy.Size() == 0);

    if (inputStrength != 1.0f) {
        A2X_CHECK_RESULT_WITH_MSG(
            nva2x::cuda::MultiplyOnDevice(destination.Data(), destination.Size(), inputStrength, cudaStream),
            "Unable to apply input strength"
        );
    }

    return ErrorCode::eSuccess;
}

std::size_t AudioAccumulator::NbAccumulatedSamples() const {
    std::lock_guard lock(_mutex);

    return _absoluteNextSample;
}

std::size_t AudioAccumulator::NbDroppedSamples() const {
    std::lock_guard lock(_mutex);

    return _nbDroppedSamples;
}

bool AudioAccumulator::IsClosed() const {
    std::lock_guard lock(_mutex);

    return _isClosed;
}

std::error_code AudioAccumulator::Reset() {
    std::lock_guard lock(_mutex);

    return Reset_NoLock();
}

void AudioAccumulator::Destroy() {
    A2X_LOG_DEBUG("AudioAccumulator::Destroy()");
    delete this;
}

std::error_code AudioAccumulator::Reset_NoLock() {
    _absoluteNextSample = 0;
    _nbDroppedSamples = 0;
    _isClosed = false;
    for (auto& tensor : _audio) {
        A2X_CHECK_RESULT_WITH_MSG(_pool.Return(std::move(tensor)), "Error returning tensor to pool");
    }
    _audio.clear();
    return ErrorCode::eSuccess;
}


WindowProgress::WindowProgress(const WindowProgressParameters& params)
: _params{params} {
    assert(_params.windowSize > 0);
    assert(_params.strideNum > 0);
    assert(_params.strideDenom > 0);
    const auto divisor = std::gcd(_params.strideNum, _params.strideDenom);
    _params.strideNum /= divisor;
    _params.strideDenom /= divisor;
}

void WindowProgress::GetCurrentWindow(
    timestamp_t& start, timestamp_t& target, timestamp_t& end, std::size_t readWindowOffset
    ) const {
    GetWindow(start, target, end, _readWindowCount + readWindowOffset);
}

std::array<WindowProgress::timestamp_t, 3> WindowProgress::GetCurrentWindow(std::size_t readWindowOffset) const {
    std::array<timestamp_t, 3> window;
    GetCurrentWindow(window[0], window[1], window[2], readWindowOffset);
    return window;
}

void WindowProgress::GetWindow(
    timestamp_t& start, timestamp_t& target, timestamp_t& end, std::size_t readWindowIndex
    ) const {
    start = GetStartSample(readWindowIndex);
    target = start + _params.targetOffset;
    end = start + _params.windowSize;
}

std::array<WindowProgress::timestamp_t, 3> WindowProgress::GetWindow(std::size_t readWindowIndex) const {
    std::array<timestamp_t, 3> window;
    GetCurrentWindow(window[0], window[1], window[2], readWindowIndex);
    return window;
}

std::size_t WindowProgress::GetNbAvailableWindows(std::size_t nbAccumulatedSamples, bool isClosed) const {
    static_assert(0 <= std::numeric_limits<timestamp_t>::max());
    assert(nbAccumulatedSamples <= static_cast<std::size_t>(std::numeric_limits<timestamp_t>::max()));
    return GetNbAvailableWindows(
        static_cast<timestamp_t>(nbAccumulatedSamples), isClosed
    );
}

std::size_t WindowProgress::GetNbAvailableWindows(timestamp_t endTimestamp, bool isClosed) const {
    timestamp_t additionalOffset;
    timestamp_t limitOffset;
    if (isClosed) {
        // If no audio will be added, count how many inference to get the target sample
        // to after the last accumulated sample.
        // This will ensure all inferences generate a target within input audio will have been run,
        // without running any inference where the target is after the current audio.
        additionalOffset = _params.targetOffset;
        // Note that in this case, we DON'T count the inference which brings the target
        // sample equal to the number of accumulated samples.
        limitOffset = 0;
    }
    else {
        // If audio can still be added, count how many inference to get the last sample
        // to after the last accumulate sample.
        additionalOffset = static_cast<timestamp_t>(_params.windowSize);
        // Note that in this case we DO count the inference which brings the last sample
        // equal to the number of accumulated samples.
        limitOffset = -1;
    }

    const timestamp_t offset = _params.startOffset + additionalOffset;
    const timestamp_t strideDenom = static_cast<timestamp_t>(_params.strideDenom);
    const timestamp_t strideNum = static_cast<timestamp_t>(_params.strideNum);
    const timestamp_t size = endTimestamp;
    // Scale the data not to work with fractions anymore.
    const timestamp_t scaledSize = size * strideDenom;
    const timestamp_t scaledOffset =  offset * strideDenom;
    const timestamp_t scaledStride = strideNum;

    const timestamp_t scaledDistanceToCover = scaledSize - scaledOffset;
    timestamp_t nbReadWindowCount = (scaledDistanceToCover + scaledStride - 1) / scaledStride;
    if (offset + nbReadWindowCount * strideNum / strideDenom + limitOffset >= size) {
        --nbReadWindowCount;
    }

    return static_cast<std::size_t>(
        std::max<timestamp_t>(0, nbReadWindowCount - static_cast<int64_t>(_readWindowCount) + 1)
    );
}

WindowProgress::timestamp_t WindowProgress::GetStartSample(std::size_t readWindowIndex) const {
    return static_cast<timestamp_t>(
        readWindowIndex * _params.strideNum / _params.strideDenom
    ) + _params.startOffset;
}


WindowProgress GetFrameProgress(const WindowProgress& progress, std::size_t nbFramesPerStride) {
    assert(nbFramesPerStride > 0);
    auto frameProgressParams = progress.GetParameters();
    frameProgressParams.strideDenom *= nbFramesPerStride;
    nva2x::WindowProgress frameProgress(frameProgressParams);
    frameProgress.ResetReadWindowCount(progress.GetReadWindowCount() * nbFramesPerStride);
    return frameProgress;
}

std::size_t GetTotalNbFrames(const IAudioAccumulator& audioAccumulator, const WindowProgress& progress, std::size_t nbFramesPerStride) {
    assert(nbFramesPerStride > 0);
    if (!audioAccumulator.IsClosed()) {
        return 0;
    }

    auto totalProgress = nva2x::GetFrameProgress(progress, nbFramesPerStride);
    totalProgress.ResetReadWindowCount();

    const auto nbAccumulatedSamples = audioAccumulator.NbAccumulatedSamples();
    const auto nbTotalFrames = totalProgress.GetNbAvailableWindows(nbAccumulatedSamples, true);
    return nbTotalFrames;
}

} // namespace nva2x


nva2x::IAudioAccumulator* nva2x::internal::CreateAudioAccumulator(std::size_t tensorSize, std::size_t tensorCount) {
    A2X_LOG_DEBUG("CreateAudioAccumulator()");
    auto accumulator = std::make_unique<AudioAccumulator>();
    if (accumulator->Allocate(tensorSize, tensorCount)) {
        A2X_LOG_ERROR("Unable to create audio accumulator");
        return nullptr;
    }
    return accumulator.release();
}
