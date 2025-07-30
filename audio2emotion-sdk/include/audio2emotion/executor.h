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

#include "audio2x/executor.h"
#include "audio2x/audio_accumulator.h"

namespace nva2e {

// Interface for executors which provide emotions.
class IEmotionExecutor : public nva2x::IExecutor {
public:
    // Structure to receive results of a given execution.
    struct Results {
        std::size_t trackIndex{0};
        timestamp_t timeStampCurrentFrame{0};
        timestamp_t timeStampNextFrame{0};
        cudaStream_t cudaStream{nullptr};
        nva2x::DeviceTensorFloatConstView emotions;
    };
    // The results are given as a callback because a single execution can
    // provide 1 or multiple frames.
    // The user can return false to stop computation (in case of multi-frame execution).
    using results_callback_t = bool (*)(void* userdata, const Results& results);
    // Set the callback to use to collect results.  Setting this callback is mandatory before executing.
    // An error will be returned when running the execution if no callback is set.
    virtual std::error_code SetResultsCallback(results_callback_t callback, void* userdata) = 0;

    // Get the next audio sample to be read, so that all samples up to that
    // sample can be dropped.
    virtual std::size_t GetNextAudioSampleToRead(std::size_t trackIndex) const = 0;

    // Get the size of the emotions output.
    virtual std::size_t GetEmotionsSize() const = 0;
};

// Parameters for creating an emotion executor.
struct EmotionExecutorCreationParameters {
    // CUDA stream to use for the executor.
    cudaStream_t cudaStream{nullptr};
    // Number of audio tracks to process.
    std::size_t nbTracks{0};
    // Array of shared audio accumulators to sample from.
    // The number of accumulators is given by nbTracks.
    const nva2x::IAudioAccumulator* const* sharedAudioAccumulators{nullptr};
};

// Helper to bind the emotion executor to emotion accumulators.
// Sets the callback on the emotion executor to fill the emotion accumulators.
class IEmotionBinder {
public:
    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~IEmotionBinder();
};

} // namespace nva2e
