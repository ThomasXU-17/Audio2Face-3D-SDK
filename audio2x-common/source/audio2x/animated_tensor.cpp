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
#include "audio2x/internal/animated_tensor.h"
#include "audio2x/internal/animated_tensor_cuda.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/error.h"

#include <algorithm>
#include <cassert>

namespace {

    template <typename Container>
    inline void find_surrounding_indices(
        std::size_t& indexBefore,
        std::size_t& indexAfter,
        const Container& sortedVector,
        typename Container::value_type value
    ) {
        assert(!sortedVector.empty());

        if (value <= sortedVector.front()) {
            indexAfter = 0;
            indexBefore = indexAfter;
            return;
        }

        if (value >= sortedVector.back()) {
            indexAfter = sortedVector.size() - 1;
            indexBefore = indexAfter;
            return;
        }

        auto it = std::lower_bound(sortedVector.begin(), sortedVector.end(), value);
        indexAfter = std::distance(sortedVector.begin(), it);
        if (sortedVector[indexAfter] == value) {
            indexBefore = indexAfter;
            return;
        }

        indexBefore = indexAfter - 1;
        assert(sortedVector[indexBefore] < value);
        assert(value < sortedVector[indexAfter]);
    }

    template <typename Container>
    inline bool find_surrounding_indices_from_cache(
        std::size_t& indexBefore,
        std::size_t& indexAfter,
        const Container& sortedVector,
        typename Container::value_type value,
        std::size_t lastIndexBefore
    ) {
        if (lastIndexBefore >= sortedVector.size()) {
            // Last index is too large, cache can't be used (values were removed?)
            // OPTME: We could update the cache index when values are removed...?
            return false;
        }

        const auto valueLastBefore = sortedVector[lastIndexBefore];
        if (valueLastBefore > value) {
            // Requested value is behind last query, cache can't be used.
            // OPTME: We could start searching only up to here instead.
            return false;
        }
        else if (valueLastBefore == value) {
            indexBefore = lastIndexBefore;
            indexAfter = lastIndexBefore;
            return true;
        }

        assert(valueLastBefore < value);
        if (lastIndexBefore == sortedVector.size() - 1) {
            // At the last index, cache only helps if value is passed the last value, which it is in this case.
            indexBefore = lastIndexBefore;
            indexAfter = lastIndexBefore;
            return true;
        }

        const auto valueLastAfter = sortedVector[lastIndexBefore + 1];
        if (valueLastAfter < value) {
            // Requested value is after the next index, we give up using cache.
            // OPTME: We could start searching from here instead.
            return false;
        }

        // We can use cache!
        if (valueLastAfter > value) {
            indexBefore = lastIndexBefore;
            indexAfter = lastIndexBefore + 1;
        }
        else {
            assert(valueLastAfter == value);
            indexBefore = lastIndexBefore + 1;
            indexAfter = indexBefore;
        }
        return true;
    };

    template <typename Container>
    inline void find_surrounding_indices(
        std::size_t& indexBefore,
        std::size_t& indexAfter,
        const Container& sortedVector,
        typename Container::value_type value,
        std::size_t lastIndexBefore
    ) {
        const bool foundInCache = find_surrounding_indices_from_cache(
            indexBefore, indexAfter, sortedVector, value, lastIndexBefore
        );
        if (foundInCache) {
            return;
        }

        find_surrounding_indices(indexBefore, indexAfter, sortedVector, value);
    }
}

namespace nva2x {


std::error_code AnimatedDeviceTensor::Allocate(
    std::size_t keySize, std::size_t keyCountPerBuffer, std::size_t preallocatedBufferCount
    ) {
    A2X_CHECK_ERROR_WITH_MSG(keySize != 0, "Cannot allocate with empty key size", ErrorCode::eInvalidValue);
    A2X_CHECK_ERROR_WITH_MSG(keyCountPerBuffer != 0, "Cannot allocate with empty key count", ErrorCode::eInvalidValue);

    A2X_CHECK_RESULT(Reset());
    A2X_CHECK_RESULT_WITH_MSG(
        _pool.Allocate(keySize * keyCountPerBuffer, preallocatedBufferCount), "Error allocating tensor pool"
        );
    _keySize = keySize;
    _keyCountPerBuffer = keyCountPerBuffer;
    return ErrorCode::eSuccess;
}

std::error_code AnimatedDeviceTensor::Deallocate() {
    _keySize = 0;
    _keyCountPerBuffer = 0;
    // No need to return the buffers to the pool, things are being destroyed anyways.
    _keyBuffers.clear();
    _timestamps.clear();
    A2X_CHECK_RESULT_WITH_MSG(_pool.Deallocate(), "Unable to deallocate tensor pool");
    return ErrorCode::eSuccess;
}

std::error_code AnimatedDeviceTensor::AddKey(timestamp_t timestamp, DeviceTensorFloatConstView key, cudaStream_t cudaStream) {
    return AddKey_Helper(timestamp, key, cudaStream);
}

std::error_code AnimatedDeviceTensor::AddKey(timestamp_t timestamp, HostTensorFloatConstView key, cudaStream_t cudaStream) {
    return AddKey_Helper(timestamp, key, cudaStream);
}

std::size_t AnimatedDeviceTensor::GetKeyCount() const {
    return _timestamps.size();
}

std::error_code AnimatedDeviceTensor::GetKeyTimestamp(timestamp_t& output, std::size_t index) const {
    A2X_CHECK_ERROR_WITH_MSG(index < _timestamps.size(), "Accessing out-of-bounds key", ErrorCode::eOutOfBounds);

    output = _timestamps[index];

    return ErrorCode::eSuccess;
}

std::error_code AnimatedDeviceTensor::GetKeyValue(DeviceTensorFloatView output, std::size_t index, cudaStream_t cudaStream) const {
    A2X_CHECK_ERROR_WITH_MSG(index < _timestamps.size(), "Accessing out-of-bound key", ErrorCode::eOutOfBounds);

    A2X_CHECK_RESULT_WITH_MSG(
        CopyDeviceToDevice(output, GetKeyView(index), cudaStream),
        "Unable to read key"
    );

    return ErrorCode::eSuccess;
}

std::error_code AnimatedDeviceTensor::FindSurroundingKeys(
    std::size_t& indexBefore, std::size_t& indexAfter, timestamp_t timestamp
    ) const {
    A2X_CHECK_ERROR_WITH_MSG(!_timestamps.empty(), "AnimatedDeviceTensor is empty", ErrorCode::eNotInitialized);

    find_surrounding_indices(indexBefore, indexAfter, _timestamps, timestamp);

    return ErrorCode::eSuccess;
}

std::error_code AnimatedDeviceTensor::Sample(
    DeviceTensorFloatView output, timestamp_t timestamp, cudaStream_t cudaStream
    ) const {
    A2X_CHECK_ERROR_WITH_MSG(output.Size() == _keySize, "Wrong size for output in AnimatedDeviceTensor", ErrorCode::eMismatch);
    A2X_CHECK_ERROR_WITH_MSG(!_timestamps.empty(), "AnimatedDeviceTensor is empty", ErrorCode::eNotInitialized);

    std::size_t indexBefore = 0;
    std::size_t indexAfter = 0;
    find_surrounding_indices(indexBefore, indexAfter, _timestamps, timestamp);

    A2X_CHECK_RESULT_WITH_MSG(
        Sample(output, timestamp, indexBefore, indexAfter, cudaStream), "Unable to sample animated tensor"
    );

    return ErrorCode::eSuccess;
}

std::error_code AnimatedDeviceTensor::DropKeysBefore(timestamp_t timestamp) {
    A2X_CHECK_ERROR_WITH_MSG(_keySize != 0 && _keyCountPerBuffer != 0, "AnimatedDeviceTensor is not initialized", ErrorCode::eNotInitialized);

    // We can only drop _keyCountPerBuffer keys at a time.
    std::size_t keysToDrop = 0;
    while (true) {
        std::size_t newKeysToDrop = keysToDrop + _keyCountPerBuffer;
        if (newKeysToDrop >= _timestamps.size()) {
            // We checked everything, we are done.
            break;
        }

        const auto firstTimestamp = _timestamps[newKeysToDrop];
        if (firstTimestamp > timestamp) {
            // We found a greater timestamp, we can't drop anymore.
            break;
        }

        // Keep looking.
        keysToDrop = newKeysToDrop;
    }

    _timestamps.erase(_timestamps.begin(), _timestamps.begin() + keysToDrop);
    const std::size_t buffersToDrop = keysToDrop / _keyCountPerBuffer;
    for (std::size_t i = 0; i < buffersToDrop; ++i) {
        A2X_CHECK_RESULT_WITH_MSG(_pool.Return(std::move(_keyBuffers.front())), "Error returning tensor to pool");
        _keyBuffers.pop_front();
    }

    return ErrorCode::eSuccess;
}

std::error_code AnimatedDeviceTensor::Reset() {
    for (auto& tensor : _keyBuffers) {
        A2X_CHECK_RESULT_WITH_MSG(_pool.Return(std::move(tensor)), "Error returning tensor to pool");
    }
    _keyBuffers.clear();
    _timestamps.clear();
    return ErrorCode::eSuccess;
}

AnimatedDeviceTensor::CachedAccessor AnimatedDeviceTensor::GetAccessor() const {
    return CachedAccessor(*this);
}

template <typename TensorViewType>
std::error_code AnimatedDeviceTensor::AddKey_Helper(timestamp_t timestamp, TensorViewType key, cudaStream_t cudaStream) {
    A2X_CHECK_ERROR_WITH_MSG(_keySize != 0 && _keyCountPerBuffer != 0, "AnimatedDeviceTensor is not initialized", ErrorCode::eNotInitialized);
    A2X_CHECK_ERROR_WITH_MSG(key.Size() == _keySize, "Wrong size for key in AnimatedDeviceTensor", ErrorCode::eMismatch);

    // Keys have to be added in strictly increasing order.
    A2X_CHECK_ERROR_WITH_MSG(
        _timestamps.empty() || _timestamps.back() < timestamp,
        "Keys must be added in increasing order",
        ErrorCode::eInvalidValue
        );

    const std::size_t bufferIndex = _timestamps.size() / _keyCountPerBuffer;
    const std::size_t indexInBuffer = _timestamps.size() % _keyCountPerBuffer;

    // Make sure there are enough GPU buffers.  Only one should be necessary to add at a time.
    while (_keyBuffers.size() <= bufferIndex) {
        auto tensor = _pool.Obtain();
        A2X_CHECK_ERROR_WITH_MSG(tensor, "Unable to get tensor from pool", ErrorCode::eNullPointer);
        _keyBuffers.emplace_back(std::move(tensor));
    }
    assert(bufferIndex == _keyBuffers.size() - 1);

    // Small helper to deal with different functions to copy.
    auto copy = [](DeviceTensorFloatView destination, TensorViewType source, cudaStream_t cudaStream) {
        if constexpr (std::is_same_v<TensorViewType, HostTensorFloatConstView>) {
            return CopyHostToDevice(destination, source, cudaStream);
        }
        else {
            static_assert(std::is_same_v<TensorViewType, DeviceTensorFloatConstView>);
            return CopyDeviceToDevice(destination, source, cudaStream);
        }
    };
    // Copy at the right place.
    A2X_CHECK_RESULT_WITH_MSG(
        copy(_keyBuffers.back()->View(indexInBuffer * _keySize, key.Size()), key, cudaStream),
        "Unable to write key"
    );

    _timestamps.emplace_back(timestamp);

    return ErrorCode::eSuccess;
}

std::error_code AnimatedDeviceTensor::Sample(
    DeviceTensorFloatView output,
    timestamp_t timestamp,
    std::size_t indexBefore,
    std::size_t indexAfter,
    cudaStream_t cudaStream
    ) const {
    if (indexBefore == indexAfter) {
        A2X_CHECK_RESULT_WITH_MSG(
            CopyDeviceToDevice(output, GetKeyView(indexBefore), cudaStream), "Unable to copy key"
            );
    }
    else {
        const auto keyBefore = GetKeyView(indexBefore);
        const auto keyAfter = GetKeyView(indexAfter);
        assert(keyBefore.Size() == _keySize);
        assert(keyAfter.Size() == _keySize);

        assert(_timestamps[indexBefore] < timestamp);
        assert(timestamp < _timestamps[indexAfter]);
        const float t =
            (timestamp - _timestamps[indexBefore]) /
            static_cast<float>(_timestamps[indexAfter] - _timestamps[indexBefore]);
        assert(0 < t);
        assert(t < 1);

        A2X_CHECK_RESULT_WITH_MSG(
            cuda::Lerp(output.Data(), keyBefore.Data(), keyAfter.Data(), t, _keySize, cudaStream),
            "Unable to interpolate key"
            );
    }

    return ErrorCode::eSuccess;
}

DeviceTensorFloatConstView AnimatedDeviceTensor::GetKeyView(std::size_t index) const {
    const std::size_t bufferIndex = index / _keyCountPerBuffer;
    const std::size_t indexInBuffer = index % _keyCountPerBuffer;
    return _keyBuffers[bufferIndex]->View(indexInBuffer * _keySize, _keySize);
}


std::error_code AnimatedDeviceTensor::CachedAccessor::FindSurroundingKeys(
    std::size_t& indexBefore, std::size_t& indexAfter, timestamp_t timestamp
    ) {
    A2X_CHECK_ERROR_WITH_MSG(_animatedTensor, "Empty accessor", ErrorCode::eNullPointer);
    A2X_CHECK_ERROR_WITH_MSG(!_animatedTensor->_timestamps.empty(), "AnimatedDeviceTensor is empty", ErrorCode::eNotInitialized);

    find_surrounding_indices(indexBefore, indexAfter, _animatedTensor->_timestamps, timestamp, _lastIndexBefore);
    _lastIndexBefore = indexBefore;

    return ErrorCode::eSuccess;
}

std::error_code AnimatedDeviceTensor::CachedAccessor::Sample(
    DeviceTensorFloatView output, timestamp_t timestamp, cudaStream_t cudaStream
    ) {
    A2X_CHECK_ERROR_WITH_MSG(_animatedTensor, "Empty accessor", ErrorCode::eNullPointer);
    A2X_CHECK_ERROR_WITH_MSG(output.Size() == _animatedTensor->_keySize, "Wrong size for output in AnimatedDeviceTensor", ErrorCode::eMismatch);
    A2X_CHECK_ERROR_WITH_MSG(!_animatedTensor->_timestamps.empty(), "AnimatedDeviceTensor is empty", ErrorCode::eNotInitialized);

    std::size_t indexBefore = 0;
    std::size_t indexAfter = 0;
    find_surrounding_indices(indexBefore, indexAfter, _animatedTensor->_timestamps, timestamp, _lastIndexBefore);
    _lastIndexBefore = indexBefore;

    A2X_CHECK_RESULT_WITH_MSG(
        _animatedTensor->Sample(output, timestamp, indexBefore, indexAfter, cudaStream),
        "Unable to sample animated tensor"
    );

    return ErrorCode::eSuccess;
}

AnimatedDeviceTensor::CachedAccessor::CachedAccessor(const AnimatedDeviceTensor& animatedTensor)
: _animatedTensor{&animatedTensor}
{
}


} // namespace nva2x
