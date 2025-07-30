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

#include "audio2emotion/executor.h"
#include "audio2emotion/internal/executor_core.h"
#include "audio2x/emotion_accumulator.h"
#include "audio2x/internal/audio_accumulator.h"
#include "audio2x/internal/bit_vector.h"

#include <vector>

namespace nva2e {


// Interfaces used to determine whether or not an executor supports certain features.
class IEmotionExecutorAccessorInputStrength {
public:
    virtual std::error_code GetInputStrength(float& inputStrength) const = 0;
    virtual std::error_code SetInputStrength(float inputStrength) = 0;
};

class IEmotionExecutorAccessorPostProcessParameters {
public:
    using params_type = PostProcessParams;
    virtual std::error_code Get(std::size_t trackIndex, params_type& params) const = 0;
    virtual std::error_code Set(std::size_t trackIndex, const params_type& params) = 0;
};


std::error_code GetExecutorInputStrength_INTERNAL(const IEmotionExecutor& executor, float& inputStrength);
std::error_code SetExecutorInputStrength_INTERNAL(IEmotionExecutor& executor, float inputStrength);
std::error_code GetExecutorPostProcessParameters_INTERNAL(const IEmotionExecutor& executor, std::size_t trackIndex, PostProcessParams& params);
std::error_code SetExecutorPostProcessParameters_INTERNAL(IEmotionExecutor& executor, std::size_t trackIndex, const PostProcessParams& params);


// Base class common to concrete executor implementations.
class EmotionExecutorBase : public IEmotionExecutor
                          , public IEmotionExecutorAccessorInputStrength
                          , public IEmotionExecutorAccessorPostProcessParameters {
public:
    std::size_t GetNbTracks() const override;

    std::error_code Reset(std::size_t trackIndex) override;
    void Destroy() override;

    bool HasExecutionStarted(std::size_t trackIndex) const override;
    std::size_t GetNbAvailableExecutions(std::size_t trackIndex) const override;
    std::size_t GetTotalNbFrames(std::size_t trackIndex) const override;
    std::size_t GetSamplingRate() const override;
    void GetFrameRate(std::size_t& numerator, std::size_t& denominator) const override;

    timestamp_t GetFrameTimestamp(std::size_t frameIndex) const override;

    std::error_code SetResultsCallback(results_callback_t callback, void* userdata) override;

    std::size_t GetEmotionsSize() const override;

    std::error_code GetInputStrength(float& inputStrength) const override;
    std::error_code SetInputStrength(float inputStrength) override;
    std::error_code Get(std::size_t trackIndex, PostProcessParams& params) const override;
    std::error_code Set(std::size_t trackIndex, const PostProcessParams& params) override;

protected:
    std::error_code BaseExecute(std::size_t* pNbExecutedTracks);

    std::error_code BaseInit(
        const nva2e::EmotionExecutorCreationParameters& params,
        std::size_t emotionSize,
        const nva2x::WindowProgressParameters& progressParams,
        const nva2x::IEmotionAccumulator* const* sharedPreferredEmotionAccumulators
        );

    virtual std::error_code RunInference(std::size_t& outNbExecutedTracks) = 0;
    virtual EmotionExecutorCoreBase& GetCore() = 0;
    virtual const EmotionExecutorCoreBase& GetCore() const = 0;

    nva2x::WindowProgress GetSingleFrameProgress(std::size_t trackIndex) const;

    results_callback_t _resultsCallback{};
    void* _resultsUserdata{};

    struct TrackData {
        const nva2x::IAudioAccumulator* audioAccumulator{};
        const nva2x::IEmotionAccumulator* emotionAccumulator{};
        std::unique_ptr<nva2x::WindowProgress> progress;
    };
    std::vector<TrackData> _trackData;

    nva2x::bit_vector<std::uint64_t> _executedTracks;
    nva2x::bit_vector<std::uint64_t> _postProcessTracks;
    nva2x::bit_vector<std::uint64_t> _doneTracks;
};


class EmotionBinder : public IEmotionBinder {
public:
    void Destroy() override;

    std::error_code Init(
        IEmotionExecutor& executor,
        nva2x::IEmotionAccumulator* const* emotionAccumulators,
        std::size_t nbEmotionAccumulators
        );

private:
    static bool callbackForEmotion(void* userdata, const IEmotionExecutor::Results& results);

    std::vector<nva2x::IEmotionAccumulator*> _emotionAccumulators;
};

IEmotionBinder* CreateEmotionBinder_INTERNAL(
    IEmotionExecutor& executor,
    nva2x::IEmotionAccumulator* const* emotionAccumulators,
    std::size_t nbEmotionAccumulators
    );


} // namespace nva2e
