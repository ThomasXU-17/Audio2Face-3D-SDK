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

#include "audio2face/interactive_executor.h"
#include "audio2face/internal/executor_core.h"
#include "audio2x/internal/audio_accumulator.h"
#include "audio2x/internal/unique_ptr.h"

#include <atomic>

namespace nva2f {


// Interfaces used to determine whether or not an executor supports certain features.
class IFaceInteractiveExecutorAccessorInputStrength {
public:
    virtual std::error_code GetInputStrength(float& inputStrength) const = 0;
    virtual std::error_code SetInputStrength(float inputStrength) = 0;
};

class IFaceInteractiveExecutorAccessorSkinParameters {
public:
    using params_type = AnimatorSkinParams;
    virtual std::error_code Get(params_type& params) const = 0;
    virtual std::error_code Set(const params_type& params) = 0;
};

class IFaceInteractiveExecutorAccessorTongueParameters {
public:
    using params_type = AnimatorTongueParams;
    virtual std::error_code Get(params_type& params) const = 0;
    virtual std::error_code Set(const params_type& params) = 0;
};

class IFaceInteractiveExecutorAccessorTeethParameters {
public:
    using params_type = AnimatorTeethParams;
    virtual std::error_code Get(params_type& params) const = 0;
    virtual std::error_code Set(const params_type& params) = 0;
};

class IFaceInteractiveExecutorAccessorEyesParameters {
public:
    using params_type = AnimatorEyesParams;
    virtual std::error_code Get(params_type& params) const = 0;
    virtual std::error_code Set(const params_type& params) = 0;
};

class IFaceInteractiveExecutorAccessorGeometryInteractiveExecutor {
public:
    virtual std::error_code GetGeometryInteractiveExecutor(IGeometryInteractiveExecutor** geometryInteractiveExecutor) const = 0;
};

class IFaceInteractiveExecutorAccessorGeometryResultsCallback {
public:
    virtual std::error_code SetGeometryResultsCallback(
        IGeometryInteractiveExecutor::results_callback_t callback, void* userdata
        ) = 0;
};

class IFaceInteractiveExecutorAccessorBlendshapeParameters {
public:
    virtual std::error_code SetSkinConfig(const BlendshapeSolverConfig& config) = 0;
    virtual std::error_code GetSkinConfig(BlendshapeSolverConfig& config) const = 0;
    virtual std::error_code SetSkinParameters(const BlendshapeSolverParams& params) = 0;
    virtual std::error_code GetSkinParameters(BlendshapeSolverParams& params) const = 0;

    virtual std::error_code SetTongueConfig(const BlendshapeSolverConfig& config) = 0;
    virtual std::error_code GetTongueConfig(BlendshapeSolverConfig& config) const = 0;
    virtual std::error_code SetTongueParameters(const BlendshapeSolverParams& params) = 0;
    virtual std::error_code GetTongueParameters(BlendshapeSolverParams& params) const = 0;
};


std::error_code GetInteractiveExecutorInputStrength_INTERNAL(const IFaceInteractiveExecutor& executor, float& inputStrength);
std::error_code SetInteractiveExecutorInputStrength_INTERNAL(IFaceInteractiveExecutor& executor, float inputStrength);
std::error_code GetInteractiveExecutorSkinParameters_INTERNAL(const IFaceInteractiveExecutor& executor, AnimatorSkinParams& params);
std::error_code SetInteractiveExecutorSkinParameters_INTERNAL(IFaceInteractiveExecutor& executor, const AnimatorSkinParams& params);
std::error_code GetInteractiveExecutorTongueParameters_INTERNAL(const IFaceInteractiveExecutor& executor, AnimatorTongueParams& params);
std::error_code SetInteractiveExecutorTongueParameters_INTERNAL(IFaceInteractiveExecutor& executor, const AnimatorTongueParams& params);
std::error_code GetInteractiveExecutorTeethParameters_INTERNAL(const IFaceInteractiveExecutor& executor, AnimatorTeethParams& params);
std::error_code SetInteractiveExecutorTeethParameters_INTERNAL(IFaceInteractiveExecutor& executor, const AnimatorTeethParams& params);
std::error_code GetInteractiveExecutorEyesParameters_INTERNAL(const IFaceInteractiveExecutor& executor, AnimatorEyesParams& params);
std::error_code SetInteractiveExecutorEyesParameters_INTERNAL(IFaceInteractiveExecutor& executor, const AnimatorEyesParams& params);


std::error_code GetInteractiveExecutorGeometryExecutor_INTERNAL(const IFaceInteractiveExecutor& executor, IGeometryInteractiveExecutor** geometryExecutor);
std::error_code SetInteractiveExecutorGeometryResultsCallback_INTERNAL(IFaceInteractiveExecutor& executor, IGeometryInteractiveExecutor::results_callback_t callback, void* userdata);
std::error_code GetInteractiveExecutorBlendshapeSkinConfig_INTERNAL(const IFaceInteractiveExecutor& executor, BlendshapeSolverConfig& config);
std::error_code SetInteractiveExecutorBlendshapeSkinConfig_INTERNAL(IFaceInteractiveExecutor& executor, const BlendshapeSolverConfig& config);
std::error_code GetInteractiveExecutorBlendshapeSkinParameters_INTERNAL(const IFaceInteractiveExecutor& executor, BlendshapeSolverParams& params);
std::error_code SetInteractiveExecutorBlendshapeSkinParameters_INTERNAL(IFaceInteractiveExecutor& executor, const BlendshapeSolverParams& params);
std::error_code GetInteractiveExecutorBlendshapeTongueConfig_INTERNAL(const IFaceInteractiveExecutor& executor, BlendshapeSolverConfig& config);
std::error_code SetInteractiveExecutorBlendshapeTongueConfig_INTERNAL(IFaceInteractiveExecutor& executor, const BlendshapeSolverConfig& config);
std::error_code GetInteractiveExecutorBlendshapeTongueParameters_INTERNAL(const IFaceInteractiveExecutor& executor, BlendshapeSolverParams& params);
std::error_code SetInteractiveExecutorBlendshapeTongueParameters_INTERNAL(IFaceInteractiveExecutor& executor, const BlendshapeSolverParams& params);


// Base class common to concrete interactive executor implementations.
class GeometryInteractiveExecutorBase : public IGeometryInteractiveExecutor
                                      , public IFaceInteractiveExecutorAccessorInputStrength
                                      , public IFaceInteractiveExecutorAccessorSkinParameters
                                      , public IFaceInteractiveExecutorAccessorTongueParameters
                                      , public IFaceInteractiveExecutorAccessorTeethParameters
                                      , public IFaceInteractiveExecutorAccessorEyesParameters
                                      , public IFaceInteractiveExecutorAccessorGeometryResultsCallback {
public:
    std::error_code Invalidate(invalidation_layer_t layer) override;
    bool IsValid(invalidation_layer_t layer) const override;
    void Destroy() override;

    std::size_t GetTotalNbFrames() const override;
    std::size_t GetSamplingRate() const override;
    void GetFrameRate(std::size_t& numerator, std::size_t& denominator) const override;

    timestamp_t GetFrameTimestamp(std::size_t frameIndex) const override;

    std::error_code Interrupt() override;

    std::error_code SetResultsCallback(results_callback_t callback, void* userdata) override;
    std::size_t GetSkinGeometrySize() const override;
    std::size_t GetTongueGeometrySize() const override;
    std::size_t GetJawTransformSize() const override;
    std::size_t GetEyesRotationSize() const override;

    std::error_code GetInputStrength(float& inputStrength) const override;
    std::error_code SetInputStrength(float inputStrength) override;
    std::error_code Get(AnimatorSkinParams& params) const override;
    std::error_code Set(const AnimatorSkinParams& params) override;
    std::error_code Get(AnimatorTongueParams& params) const override;
    std::error_code Set(const AnimatorTongueParams& params) override;
    std::error_code Get(AnimatorTeethParams& params) const override;
    std::error_code Set(const AnimatorTeethParams& params) override;
    std::error_code Get(AnimatorEyesParams& params) const override;
    std::error_code Set(const AnimatorEyesParams& params) override;
    std::error_code SetGeometryResultsCallback(IGeometryInteractiveExecutor::results_callback_t callback, void* userdata) override;

protected:
    virtual GeometryExecutorCoreBase& GetCore() = 0;
    virtual const GeometryExecutorCoreBase& GetCore() const = 0;

    std::error_code BaseInit(
        const nva2f::GeometryExecutorCreationParameters& params,
        std::size_t emotionSize,
        const nva2x::WindowProgressParameters& progressParams,
        std::size_t nbFramesPerInference
        );

    nva2x::WindowProgress GetSingleFrameProgress() const;

    std::error_code CheckInputsState() const;

    std::size_t _nbFramesPerInference{0};

    results_callback_t _resultsCallback{};
    void *_resultsUserdata{};

    struct TrackData {
        const nva2x::IAudioAccumulator* audioAccumulator{};
        const nva2x::IEmotionAccumulator* emotionAccumulator{};
        std::unique_ptr<nva2x::WindowProgress> progress;
    };
    TrackData _track;

    std::size_t _nbFramesBeforeAudio{0};

    nva2x::DeviceTensorFloat _inferenceResults;
    bool _inferenceResultsValid{false};
    bool _skinResultsValid{false};
    bool _tongueResultsValid{false};
    bool _teethResultsValid{false};
    bool _eyesResultsValid{false};

    std::atomic<bool> _isInterrupted{false};


    // This class disables smoothing of the face so that all evaluations
    // don't require state from previous frames and therefore are only a
    // function of the current frame.
    class StatelessScope {
    public:
        StatelessScope(GeometryExecutorCoreBase& core) : _core(core) {
            const auto skinAnimator = _core.GetSkinAnimator();
            assert(skinAnimator);
            const auto skinParams = skinAnimator->GetParameters(0);
            assert(skinParams);
            _skinParams = *skinParams;

            auto statelessParams = _skinParams;
            statelessParams.lowerFaceSmoothing = 0.0f;
            statelessParams.upperFaceSmoothing = 0.0f;
            [[maybe_unused]] const auto result = skinAnimator->SetParameters(0, statelessParams);
            assert(!result);
        }

        ~StatelessScope() {
            const auto skinAnimator = _core.GetSkinAnimator();
            assert(skinAnimator);
            [[maybe_unused]] const auto result = skinAnimator->SetParameters(0, _skinParams);
            assert(!result);
        }

    private:
        GeometryExecutorCoreBase& _core;
        IAnimatorSkin::Params _skinParams;
    };
};


} // namespace nva2f
