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

#include "audio2face/internal/executor_core.h"
#include "audio2x/internal/audio_accumulator.h"
#include "audio2x/internal/unique_ptr.h"
#include "audio2x/internal/bit_vector.h"

namespace nva2f {


class GeometryExecutorCoreBase;


// Interfaces used to determine whether or not an executor supports certain features.
class IFaceExecutorAccessorInputStrength {
public:
    virtual std::error_code GetInputStrength(float& inputStrength) const = 0;
    virtual std::error_code SetInputStrength(float inputStrength) = 0;
};

class IFaceExecutorAccessorGeometryExecutor {
public:
    virtual std::error_code GetGeometryExecutor(IGeometryExecutor** geometryExecutor) const = 0;
};

class IFaceExecutorAccessorGeometryResultsCallback {
public:
    virtual std::error_code SetGeometryResultsCallback(
        IGeometryExecutor::results_callback_t callback, void* userdata
        ) = 0;
};

class IFaceExecutorAccessorBlendshapeSolvers {
public:
    virtual std::error_code GetSkinSolver(std::size_t trackIndex, IBlendshapeSolver** skinSolver) const = 0;
    virtual std::error_code GetTongueSolver(std::size_t trackIndex, IBlendshapeSolver** tongueSolver) const = 0;
};

class IFaceExecutorAccessorSkinParameters {
public:
    using params_type = AnimatorSkinParams;
    virtual std::error_code Get(std::size_t trackIndex, params_type& params) const = 0;
    virtual std::error_code Set(std::size_t trackIndex, const params_type& params) = 0;
};

class IFaceExecutorAccessorTongueParameters {
public:
    using params_type = AnimatorTongueParams;
    virtual std::error_code Get(std::size_t trackIndex, params_type& params) const = 0;
    virtual std::error_code Set(std::size_t trackIndex, const params_type& params) = 0;
};

class IFaceExecutorAccessorTeethParameters {
public:
    using params_type = AnimatorTeethParams;
    virtual std::error_code Get(std::size_t trackIndex, params_type& params) const = 0;
    virtual std::error_code Set(std::size_t trackIndex, const params_type& params) = 0;
};

class IFaceExecutorAccessorEyesParameters {
public:
    using params_type = AnimatorEyesParams;
    virtual std::error_code Get(std::size_t trackIndex, params_type& params) const = 0;
    virtual std::error_code Set(std::size_t trackIndex, const params_type& params) = 0;
};


std::error_code GetExecutorInputStrength_INTERNAL(const IFaceExecutor& executor, float& inputStrength);
std::error_code SetExecutorInputStrength_INTERNAL(IFaceExecutor& executor, float inputStrength);
std::error_code GetExecutorGeometryExecutor_INTERNAL(
    const IFaceExecutor& executor, IGeometryExecutor** geometryExecutor
    );
std::error_code SetExecutorGeometryResultsCallback_INTERNAL(
    IFaceExecutor& executor, IGeometryExecutor::results_callback_t callback, void* userdata
    );
std::error_code GetExecutorSkinSolver_INTERNAL(
    const IFaceExecutor& executor, std::size_t trackIndex, IBlendshapeSolver** skinSolver
    );
std::error_code GetExecutorTongueSolver_INTERNAL(
    const IFaceExecutor& executor, std::size_t trackIndex, IBlendshapeSolver** tongueSolver
    );
std::error_code GetExecutorSkinParameters_INTERNAL(
    const IFaceExecutor& executor, std::size_t trackIndex, AnimatorSkinParams& params
    );
std::error_code SetExecutorSkinParameters_INTERNAL(
    IFaceExecutor& executor, std::size_t trackIndex, const AnimatorSkinParams& params
    );
std::error_code GetExecutorTongueParameters_INTERNAL(
    const IFaceExecutor& executor, std::size_t trackIndex, AnimatorTongueParams& params
    );
std::error_code SetExecutorTongueParameters_INTERNAL(
    IFaceExecutor& executor, std::size_t trackIndex, const AnimatorTongueParams& params
    );
std::error_code GetExecutorTeethParameters_INTERNAL(
    const IFaceExecutor& executor, std::size_t trackIndex, AnimatorTeethParams& params
    );
std::error_code SetExecutorTeethParameters_INTERNAL(
    IFaceExecutor& executor, std::size_t trackIndex, const AnimatorTeethParams& params
    );
std::error_code GetExecutorEyesParameters_INTERNAL(
    const IFaceExecutor& executor, std::size_t trackIndex, AnimatorEyesParams& params
    );
std::error_code SetExecutorEyesParameters_INTERNAL(
    IFaceExecutor& executor, std::size_t trackIndex, const AnimatorEyesParams& params
    );


// Base class common to concrete executor implementations.
class GeometryExecutorBase : public IGeometryExecutor
                           , public IFaceExecutorAccessorInputStrength
                           , public IFaceExecutorAccessorGeometryResultsCallback
                           , public IFaceExecutorAccessorSkinParameters
                           , public IFaceExecutorAccessorTongueParameters
                           , public IFaceExecutorAccessorTeethParameters
                           , public IFaceExecutorAccessorEyesParameters {
public:
    std::size_t GetNbTracks() const override;

    void Destroy() override;

    bool HasExecutionStarted(std::size_t trackIndex) const override;
    std::size_t GetNbAvailableExecutions(std::size_t trackIndex) const override;
    std::size_t GetTotalNbFrames(std::size_t trackIndex) const override;
    void GetFrameRate(std::size_t& numerator, std::size_t& denominator) const override;
    timestamp_t GetFrameTimestamp(std::size_t frameIndex) const override;

    std::error_code SetExecutionOption(ExecutionOption executionOption) override;
    ExecutionOption GetExecutionOption() const override;

    std::size_t GetSamplingRate() const override;
    std::size_t GetSkinGeometrySize() const override;
    std::size_t GetTongueGeometrySize() const override;
    std::size_t GetJawTransformSize() const override;
    std::size_t GetEyesRotationSize() const override;

    std::error_code SetEmotionsCallback(emotions_callback_t callback, void* userdata) override;

    timestamp_t GetNextEmotionTimestampToRead(std::size_t trackIndex) const override;
    std::size_t GetNextAudioSampleToRead(std::size_t trackIndex) const override;

    std::error_code SetResultsCallback(results_callback_t callback, void* userdata) override;

    std::error_code GetInputStrength(float& inputStrength) const override;
    std::error_code SetInputStrength(float inputStrength) override;
    std::error_code SetGeometryResultsCallback(results_callback_t callback, void* userdata) override;
    std::error_code Get(std::size_t trackIndex, AnimatorSkinParams& params) const override;
    std::error_code Set(std::size_t trackIndex, const AnimatorSkinParams& params) override;
    std::error_code Get(std::size_t trackIndex, AnimatorTongueParams& params) const override;
    std::error_code Set(std::size_t trackIndex, const AnimatorTongueParams& params) override;
    std::error_code Get(std::size_t trackIndex, AnimatorTeethParams& params) const override;
    std::error_code Set(std::size_t trackIndex, const AnimatorTeethParams& params) override;
    std::error_code Get(std::size_t trackIndex, AnimatorEyesParams& params) const override;
    std::error_code Set(std::size_t trackIndex, const AnimatorEyesParams& params) override;

protected:
    virtual GeometryExecutorCoreBase& GetCore() = 0;
    virtual const GeometryExecutorCoreBase& GetCore() const = 0;

    std::error_code BaseInit(
        const nva2f::GeometryExecutorCreationParameters& params,
        std::size_t emotionSize,
        const nva2x::WindowProgressParameters& progressParams,
        std::size_t nbFramesPerInference
        );

    std::error_code BaseReset(std::size_t trackIndex);

    nva2x::WindowProgress GetSingleFrameProgress(std::size_t trackIndex) const;

    std::size_t _nbFramesPerInference{0};

    emotions_callback_t _emotionsCallback{};
    void* _emotionsUserdata{};
    results_callback_t _resultsCallback{};
    void *_resultsUserdata{};

    struct TrackData {
        const nva2x::IAudioAccumulator* audioAccumulator{};
        const nva2x::IEmotionAccumulator* emotionAccumulator{};
        std::unique_ptr<nva2x::WindowProgress> progress;
    };
    std::vector<TrackData> _trackData;

    std::size_t _nbFramesBeforeAudio{0};

    nva2x::bit_vector<std::uint64_t> _executedTracks;
};


namespace internal {

IGeometryExecutor::ExecutionOption operator|(IGeometryExecutor::ExecutionOption a, IGeometryExecutor::ExecutionOption b);
IGeometryExecutor::ExecutionOption& operator|=(IGeometryExecutor::ExecutionOption& a, IGeometryExecutor::ExecutionOption b);
IGeometryExecutor::ExecutionOption operator&(IGeometryExecutor::ExecutionOption a, IGeometryExecutor::ExecutionOption b);
IGeometryExecutor::ExecutionOption& operator&=(IGeometryExecutor::ExecutionOption& a, IGeometryExecutor::ExecutionOption b);
bool IsAnySet(IGeometryExecutor::ExecutionOption flags, IGeometryExecutor::ExecutionOption flagsToCheck);

}


} // namespace nva2f
