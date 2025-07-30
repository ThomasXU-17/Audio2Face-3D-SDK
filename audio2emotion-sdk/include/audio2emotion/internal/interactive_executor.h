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

#include "audio2emotion/interactive_executor.h"
#include "audio2emotion/internal/executor_core.h"

#include <atomic>

namespace nva2e {


// Interfaces used to determine whether or not an executor supports certain features.
class IEmotionInteractiveExecutorAccessorInferencesToSkip {
public:
    virtual std::error_code GetInferencesToSkip(std::size_t& inferencesToSkip) const = 0;
    virtual std::error_code SetInferencesToSkip(std::size_t inferencesToSkip) = 0;
};

class IEmotionInteractiveExecutorAccessorInputStrength {
public:
    virtual std::error_code GetInputStrength(float& inputStrength) const = 0;
    virtual std::error_code SetInputStrength(float inputStrength) = 0;
};

class IEmotionInteractiveExecutorAccessorPostProcessParameters {
public:
    using params_type = PostProcessParams;
    virtual std::error_code Get(params_type& params) const = 0;
    virtual std::error_code Set(const params_type& params) = 0;
};


std::error_code GetInteractiveExecutorInferencesToSkip_INTERNAL(const IEmotionInteractiveExecutor& executor, std::size_t& inferencesToSkip);
std::error_code SetInteractiveExecutorInferencesToSkip_INTERNAL(IEmotionInteractiveExecutor& executor, std::size_t inferencesToSkip);
std::error_code GetInteractiveExecutorInputStrength_INTERNAL(const IEmotionInteractiveExecutor& executor, float& inputStrength);
std::error_code SetInteractiveExecutorInputStrength_INTERNAL(IEmotionInteractiveExecutor& executor, float inputStrength);
std::error_code GetInteractiveExecutorPostProcessParameters_INTERNAL(const IEmotionInteractiveExecutor& executor, PostProcessParams& params);
std::error_code SetInteractiveExecutorPostProcessParameters_INTERNAL(IEmotionInteractiveExecutor& executor, const PostProcessParams& params);


// Base class common to concrete interactive executor implementations.
class EmotionInteractiveExecutorBase : public IEmotionInteractiveExecutor
                                     , public IEmotionInteractiveExecutorAccessorInferencesToSkip
                                     , public IEmotionInteractiveExecutorAccessorInputStrength
                                     , public IEmotionInteractiveExecutorAccessorPostProcessParameters {
public:
    std::error_code Invalidate(invalidation_layer_t layer) override;
    bool IsValid(invalidation_layer_t layer) const override;
    void Destroy() override;

    std::size_t GetTotalNbFrames() const override;
    std::size_t GetSamplingRate() const override;
    void GetFrameRate(std::size_t& numerator, std::size_t& denominator) const override;

    timestamp_t GetFrameTimestamp(std::size_t frameIndex) const override;

    std::error_code ComputeFrame(std::size_t frameIndex) override;
    std::error_code ComputeAllFrames() override;
    std::error_code Interrupt() override;

    std::error_code SetResultsCallback(results_callback_t callback, void* userdata) override;
    std::size_t GetEmotionsSize() const override;

    std::error_code GetInputStrength(float& inputStrength) const override;
    std::error_code SetInputStrength(float inputStrength) override;
    std::error_code Get(PostProcessParams& params) const override;
    std::error_code Set(const PostProcessParams& params) override;

protected:
    std::error_code BaseInit(
        const nva2e::EmotionExecutorCreationParameters& params,
        std::size_t emotionSize,
        const nva2x::WindowProgressParameters& progressParams,
        const nva2x::IEmotionAccumulator* const* sharedPreferredEmotionAccumulators
        );

    virtual EmotionExecutorCoreBase& GetCore() = 0;
    virtual const EmotionExecutorCoreBase& GetCore() const = 0;
    virtual std::error_code ComputeInference() = 0;

    nva2x::WindowProgress GetSingleFrameProgress() const;
    std::error_code CheckInputsState() const;

    enum class CallCallback {
        None,
        AllFrames,
        OnlyLastFrame
    };
    std::error_code ComputePostProcessing(std::size_t endFrameIndex, CallCallback callCallback);

    results_callback_t _resultsCallback{};
    void* _resultsUserdata{};

    const nva2x::IAudioAccumulator* _audioAccumulator{};
    const nva2x::IEmotionAccumulator* _emotionAccumulator{};
    std::unique_ptr<nva2x::WindowProgress> _progress;

    nva2x::DeviceTensorFloat _inferenceResults;
    bool _inferenceResultsValid{false};
    bool _postProcessResultsValid{false};

    std::atomic<bool> _isInterrupted{false};
};


} // namespace nva2e
