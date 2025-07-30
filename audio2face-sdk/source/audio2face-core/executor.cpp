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
#include "audio2face/internal/executor.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"
#include "audio2x/internal/executor.h"

#include <cassert>
#include <numeric>

namespace nva2f {

// For IGeometryExecutor::ExecutionOption operators.
using namespace ::nva2f::internal;


std::size_t GeometryExecutorBase::GetNbTracks() const {
    return _trackData.size();
}

void GeometryExecutorBase::Destroy() {
    delete this;
}

bool GeometryExecutorBase::HasExecutionStarted(std::size_t trackIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", false);
    const auto& trackData = _trackData[trackIndex];
    return trackData.progress->GetReadWindowCount() != 0;
}

std::size_t GeometryExecutorBase::GetNbAvailableExecutions(std::size_t trackIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", 0);
    const auto& trackData = _trackData[trackIndex];
    A2F_CHECK_ERROR_WITH_MSG(trackData.audioAccumulator, "Audio accumulator cannot be null", 0);
    A2F_CHECK_ERROR_WITH_MSG(trackData.emotionAccumulator, "Emotion accumulator cannot be null", 0);
    A2F_CHECK_ERROR_WITH_MSG(trackData.progress, "Progress cannot be null", 0);

    return nva2x::GetNbAvailableExecutions(
        *trackData.progress,
        *trackData.audioAccumulator,
        trackData.emotionAccumulator,
        _nbFramesPerInference
        );
}

std::size_t GeometryExecutorBase::GetTotalNbFrames(std::size_t trackIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", 0);
    const auto& trackData = _trackData[trackIndex];
    A2F_CHECK_ERROR_WITH_MSG(trackData.audioAccumulator, "Audio accumulator cannot be null", 0);
    A2F_CHECK_ERROR_WITH_MSG(trackData.progress, "Progress cannot be null", 0);

    if (!trackData.audioAccumulator->IsClosed()) {
        return 0;
    }

    auto singleFrameProgress = GetSingleFrameProgress(trackIndex);
    singleFrameProgress.ResetReadWindowCount();

    const auto nbAccumulatedSamples = trackData.audioAccumulator->NbAccumulatedSamples();
    const auto nbTotalFrames = singleFrameProgress.GetNbAvailableWindows(nbAccumulatedSamples, true);
    assert(nbTotalFrames >= _nbFramesBeforeAudio);

    return nbTotalFrames - _nbFramesBeforeAudio;
}

void GeometryExecutorBase::GetFrameRate(std::size_t& numerator, std::size_t& denominator) const {
    GetCore().GetFrameRate(numerator, denominator);
}

GeometryExecutorBase::timestamp_t GeometryExecutorBase::GetFrameTimestamp(std::size_t frameIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(0 < _trackData.size(), "Progress data not initialized yet", std::numeric_limits<timestamp_t>::min());
    // Use index 0 because they all have the same progress parameters.
    const auto& trackData = _trackData[0];
    A2F_CHECK_ERROR_WITH_MSG(trackData.progress, "Progress cannot be null", std::numeric_limits<timestamp_t>::min());
    const auto singleFrameProgress = GetSingleFrameProgress(0);
    frameIndex += _nbFramesBeforeAudio;

    timestamp_t start, target, end;
    singleFrameProgress.GetWindow(start, target, end, frameIndex);
    assert(target >= 0);
    return target;
}

std::error_code GeometryExecutorBase::SetExecutionOption(ExecutionOption executionOption) {
    A2F_CHECK_ERROR_WITH_MSG(
        !nva2x::HasExecutionStarted(*this),
        "Execution option can only be set before execution is started",
        nva2x::ErrorCode::eExecutionAlreadyStarted
        );

    A2F_CHECK_RESULT_WITH_MSG(
        GetCore().SetExecutionOption(executionOption),
        "Unable to set execution option"
        );

    return nva2x::ErrorCode::eSuccess;
}

IGeometryExecutor::ExecutionOption GeometryExecutorBase::GetExecutionOption() const {
    return GetCore().GetExecutionOption();
}

std::size_t GeometryExecutorBase::GetSamplingRate() const {
    return GetCore().GetSamplingRate();
}

std::size_t GeometryExecutorBase::GetSkinGeometrySize() const {
    return GetCore().GetSkinGeometrySize();
}

std::size_t GeometryExecutorBase::GetTongueGeometrySize() const {
    return GetCore().GetTongueGeometrySize();
}

std::size_t GeometryExecutorBase::GetJawTransformSize() const {
    return GetCore().GetJawTransformSize();
}

std::size_t GeometryExecutorBase::GetEyesRotationSize() const {
    return GetCore().GetEyesRotationSize();
}

std::error_code GeometryExecutorBase::SetEmotionsCallback(emotions_callback_t callback, void* userdata) {
    A2F_CHECK_ERROR_WITH_MSG(
        !nva2x::HasExecutionStarted(*this),
        "Emotion callback can only be set before execution is started",
        nva2x::ErrorCode::eExecutionAlreadyStarted
        );
    _emotionsCallback = callback;
    _emotionsUserdata = userdata;
    return nva2x::ErrorCode::eSuccess;
}

GeometryExecutorBase::timestamp_t GeometryExecutorBase::GetNextEmotionTimestampToRead(std::size_t trackIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", std::numeric_limits<timestamp_t>::min());
    timestamp_t start, target, end;
    _trackData[trackIndex].progress->GetCurrentWindow(start, target, end);
    return target;
}

std::size_t GeometryExecutorBase::GetNextAudioSampleToRead(std::size_t trackIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", std::numeric_limits<std::size_t>::min());
    timestamp_t start, target, end;
    _trackData[trackIndex].progress->GetCurrentWindow(start, target, end);
    if (start < 0) {
        return 0;
    }
    return static_cast<std::size_t>(start);
}

std::error_code GeometryExecutorBase::SetResultsCallback(results_callback_t callback, void* userdata) {
    A2F_CHECK_ERROR_WITH_MSG(
        !nva2x::HasExecutionStarted(*this),
        "Results callback can only be set before execution is started",
        nva2x::ErrorCode::eExecutionAlreadyStarted
        );
    A2F_CHECK_ERROR_WITH_MSG(callback || !userdata, "User data must be null if callback is null", nva2x::ErrorCode::eNullPointer);
    _resultsCallback = callback;
    _resultsUserdata = userdata;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorBase::GetInputStrength(float& inputStrength) const {
    inputStrength = GetCore().GetInputStrength();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorBase::SetInputStrength(float inputStrength) {
    assert(!nva2x::HasExecutionStarted(*this));
    GetCore().SetInputStrength(inputStrength);
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorBase::SetGeometryResultsCallback(results_callback_t callback, void* userdata) {
    return SetResultsCallback(callback, userdata);
}

std::error_code GeometryExecutorBase::Get(std::size_t trackIndex, AnimatorSkinParams& params) const {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
    const auto skinAnimator = GetCore().GetSkinAnimator();
    A2F_CHECK_ERROR_WITH_MSG(skinAnimator, "Skin animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    const auto returnParams = skinAnimator->GetParameters(trackIndex);
    assert(returnParams);
    params = *returnParams;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorBase::Set(std::size_t trackIndex, const AnimatorSkinParams& params) {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
    assert(!HasExecutionStarted(trackIndex));
    auto skinAnimator = GetCore().GetSkinAnimator();
    A2F_CHECK_ERROR_WITH_MSG(skinAnimator, "Skin animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    return skinAnimator->SetParameters(trackIndex, params);
}

std::error_code GeometryExecutorBase::Get(std::size_t trackIndex, AnimatorTongueParams& params) const {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
    const auto tongueAnimator = GetCore().GetTongueAnimator();
    A2F_CHECK_ERROR_WITH_MSG(tongueAnimator, "Tongue animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    const auto returnParams = tongueAnimator->GetParameters(trackIndex);
    assert(returnParams);
    params = *returnParams;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorBase::Set(std::size_t trackIndex, const AnimatorTongueParams& params) {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
    assert(!HasExecutionStarted(trackIndex));
    auto tongueAnimator = GetCore().GetTongueAnimator();
    A2F_CHECK_ERROR_WITH_MSG(tongueAnimator, "Tongue animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    return tongueAnimator->SetParameters(trackIndex, params);
}

std::error_code GeometryExecutorBase::Get(std::size_t trackIndex, AnimatorTeethParams& params) const {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
    const auto teethAnimator = GetCore().GetTeethAnimator();
    A2F_CHECK_ERROR_WITH_MSG(teethAnimator, "Teeth animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    const auto returnParams = teethAnimator->GetParameters(trackIndex);
    assert(returnParams);
    params = *returnParams;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorBase::Set(std::size_t trackIndex, const AnimatorTeethParams& params) {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
    assert(!HasExecutionStarted(trackIndex));
    auto teethAnimator = GetCore().GetTeethAnimator();
    A2F_CHECK_ERROR_WITH_MSG(teethAnimator, "Teeth animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    return teethAnimator->SetParameters(trackIndex, params);
}

std::error_code GeometryExecutorBase::Get(std::size_t trackIndex, AnimatorEyesParams& params) const {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
    const auto eyesAnimator = GetCore().GetEyesAnimator();
    A2F_CHECK_ERROR_WITH_MSG(eyesAnimator, "Eyes animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    const auto returnParams = eyesAnimator->GetParameters(trackIndex);
    assert(returnParams);
    params = *returnParams;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorBase::Set(std::size_t trackIndex, const AnimatorEyesParams& params) {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
    assert(!HasExecutionStarted(trackIndex));
    auto eyesAnimator = GetCore().GetEyesAnimator();
    A2F_CHECK_ERROR_WITH_MSG(eyesAnimator, "Eyes animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    return eyesAnimator->SetParameters(trackIndex, params);
}


std::error_code GeometryExecutorBase::BaseInit(
    const nva2f::GeometryExecutorCreationParameters& params,
    std::size_t emotionSize,
    const nva2x::WindowProgressParameters& progressParams,
    std::size_t nbFramesPerInference
    ) {
    A2F_CHECK_ERROR_WITH_MSG(params.sharedAudioAccumulators, "Audio accumulators cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_ERROR_WITH_MSG(params.sharedEmotionAccumulators, "Emotion accumulators cannot be null", nva2x::ErrorCode::eNullPointer);

    _nbFramesPerInference = nbFramesPerInference;

    // Initialize the track data.
    _trackData.resize(params.nbTracks);
    for (std::size_t i = 0; i < params.nbTracks; ++i) {
        A2F_CHECK_ERROR_WITH_MSG(params.sharedAudioAccumulators[i], "Audio accumulators cannot be null", nva2x::ErrorCode::eNullPointer);
        A2F_CHECK_ERROR_WITH_MSG(params.sharedEmotionAccumulators[i], "Emotion accumulators cannot be null", nva2x::ErrorCode::eNullPointer);

        A2F_CHECK_ERROR_WITH_MSG(
            params.sharedEmotionAccumulators[i]->GetEmotionSize() == emotionSize,
            "Emotion accumulator does not have the right size for emotions",
            nva2x::ErrorCode::eMismatch
        );

        auto& trackData = _trackData[i];
        trackData.audioAccumulator = params.sharedAudioAccumulators[i];
        trackData.emotionAccumulator = params.sharedEmotionAccumulators[i];

        trackData.progress = std::make_unique<nva2x::WindowProgress>(progressParams);
    }

    // Compute the number of frames which will be inferred before.
    // This is currently only used for the diffusion model, which starts before the audio.
    // The regression model should have 0 for this value.
    const auto singleFrameProgress = GetSingleFrameProgress(0);
    _nbFramesBeforeAudio = singleFrameProgress.GetNbAvailableWindows(std::size_t{0}, true);

    _executedTracks.resize(params.nbTracks);

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorBase::BaseReset(std::size_t trackIndex) {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
    auto& trackData = _trackData[trackIndex];
    // Don't reset the audio or emotion accumulator, we leave that to others.
    if (trackData.progress) {
        trackData.progress->ResetReadWindowCount();
    }

    return nva2x::ErrorCode::eSuccess;
}

nva2x::WindowProgress GeometryExecutorBase::GetSingleFrameProgress(std::size_t trackIndex) const {
    const auto& trackData = _trackData[trackIndex];
    assert(trackData.progress);
    const auto nbFramesPerInference = _nbFramesPerInference;
    assert(nbFramesPerInference > 0);
    return nva2x::GetFrameProgress(*trackData.progress, nbFramesPerInference);
}

} // namespace nva2f

namespace {

    // Just forward the request to the AccessorType interface.
    // NOTE: This dynamic_cast approach might fail if the user passes a IGeometryExecutor
    // derived class implemented in their binary, with their compiler, which might handle
    // dynamic_cast in a different way.
    template <typename AccessorType, typename... Args>
    std::error_code GetExecutorParameters(const nva2f::IFaceExecutor& executor, std::size_t trackIndex, Args&&... params) {
        const auto accessor = dynamic_cast<const AccessorType*>(&executor);
        if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
        return accessor->Get(trackIndex, std::forward<Args>(params)...);
    }

    template <typename AccessorType, typename... Args>
    std::error_code SetExecutorParameters(nva2f::IFaceExecutor& executor, std::size_t trackIndex, Args&&... params) {
        A2F_CHECK_ERROR_WITH_MSG(
            !executor.HasExecutionStarted(trackIndex),
            "Parameters can only be set before execution is started",
            nva2x::ErrorCode::eExecutionAlreadyStarted
            );
        const auto accessor = dynamic_cast<AccessorType*>(&executor);
        if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
        return accessor->Set(trackIndex, std::forward<Args>(params)...);
     }

}

std::error_code nva2f::GetExecutorInputStrength_INTERNAL(
    const IFaceExecutor& executor, float& inputStrength
    ) {
    const auto accessor = dynamic_cast<const IFaceExecutorAccessorInputStrength*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetInputStrength(inputStrength);
}

std::error_code nva2f::SetExecutorInputStrength_INTERNAL(
    IFaceExecutor& executor, float inputStrength
    ) {
    A2F_CHECK_ERROR_WITH_MSG(
        !nva2x::HasExecutionStarted(executor),
        "Input strength can only be set before execution is started",
        nva2x::ErrorCode::eExecutionAlreadyStarted
        );
    const auto accessor = dynamic_cast<IFaceExecutorAccessorInputStrength*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->SetInputStrength(inputStrength);
}

std::error_code nva2f::GetExecutorGeometryExecutor_INTERNAL(
    const IFaceExecutor& executor, IGeometryExecutor** geometryExecutor
    ) {
    const auto accessor = dynamic_cast<const IFaceExecutorAccessorGeometryExecutor*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetGeometryExecutor(geometryExecutor);
}

std::error_code nva2f::SetExecutorGeometryResultsCallback_INTERNAL(
    IFaceExecutor& executor, IGeometryExecutor::results_callback_t callback, void* userdata
    ) {
    A2F_CHECK_ERROR_WITH_MSG(
        !nva2x::HasExecutionStarted(executor),
        "Geometry results callback can only be set before execution is started",
        nva2x::ErrorCode::eExecutionAlreadyStarted
        );
    const auto accessor = dynamic_cast<IFaceExecutorAccessorGeometryResultsCallback*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->SetGeometryResultsCallback(callback, userdata);
}

std::error_code nva2f::GetExecutorSkinSolver_INTERNAL(
    const IFaceExecutor& executor, std::size_t trackIndex, IBlendshapeSolver** skinSolver
    ) {
    const auto accessor = dynamic_cast<const IFaceExecutorAccessorBlendshapeSolvers*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetSkinSolver(trackIndex, skinSolver);
}

std::error_code nva2f::GetExecutorTongueSolver_INTERNAL(
    const IFaceExecutor& executor, std::size_t trackIndex, IBlendshapeSolver** tongueSolver
    ) {
    const auto accessor = dynamic_cast<const IFaceExecutorAccessorBlendshapeSolvers*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetTongueSolver(trackIndex, tongueSolver);
}

std::error_code nva2f::GetExecutorSkinParameters_INTERNAL(
    const IFaceExecutor& executor, std::size_t trackIndex, AnimatorSkinParams& params
    ) {
    return GetExecutorParameters<IFaceExecutorAccessorSkinParameters>(executor, trackIndex, params);
}

std::error_code nva2f::SetExecutorSkinParameters_INTERNAL(
    IFaceExecutor& executor, std::size_t trackIndex, const AnimatorSkinParams& params
    ) {
    return SetExecutorParameters<IFaceExecutorAccessorSkinParameters>(executor, trackIndex, params);
}

std::error_code nva2f::GetExecutorTongueParameters_INTERNAL(
    const IFaceExecutor& executor, std::size_t trackIndex, AnimatorTongueParams& params
    ) {
    return GetExecutorParameters<IFaceExecutorAccessorTongueParameters>(executor, trackIndex, params);
}

std::error_code nva2f::SetExecutorTongueParameters_INTERNAL(
    IFaceExecutor& executor, std::size_t trackIndex, const AnimatorTongueParams& params
    ) {
    return SetExecutorParameters<IFaceExecutorAccessorTongueParameters>(executor, trackIndex, params);
}

std::error_code nva2f::GetExecutorTeethParameters_INTERNAL(
    const IFaceExecutor& executor, std::size_t trackIndex, AnimatorTeethParams& params
    ) {
    return GetExecutorParameters<IFaceExecutorAccessorTeethParameters>(executor, trackIndex, params);
}

std::error_code nva2f::SetExecutorTeethParameters_INTERNAL(
    IFaceExecutor& executor, std::size_t trackIndex, const AnimatorTeethParams& params
    ) {
    return SetExecutorParameters<IFaceExecutorAccessorTeethParameters>(executor, trackIndex, params);
}

std::error_code nva2f::GetExecutorEyesParameters_INTERNAL(
    const IFaceExecutor& executor, std::size_t trackIndex, AnimatorEyesParams& params
    ) {
    return GetExecutorParameters<IFaceExecutorAccessorEyesParameters>(executor, trackIndex, params);
}

std::error_code nva2f::SetExecutorEyesParameters_INTERNAL(
    IFaceExecutor& executor, std::size_t trackIndex, const AnimatorEyesParams& params
    ) {
    return SetExecutorParameters<IFaceExecutorAccessorEyesParameters>(executor, trackIndex, params);
}

nva2f::IGeometryExecutor::ExecutionOption nva2f::internal::operator|(IGeometryExecutor::ExecutionOption a, IGeometryExecutor::ExecutionOption b) {
    return static_cast<IGeometryExecutor::ExecutionOption>(
        static_cast<std::underlying_type_t<IGeometryExecutor::ExecutionOption>>(a) |
        static_cast<std::underlying_type_t<IGeometryExecutor::ExecutionOption>>(b)
    );
}

nva2f::IGeometryExecutor::ExecutionOption& nva2f::internal::operator|=(IGeometryExecutor::ExecutionOption& a, IGeometryExecutor::ExecutionOption b) {
    a = a | b;
    return a;
}

nva2f::IGeometryExecutor::ExecutionOption nva2f::internal::operator&(IGeometryExecutor::ExecutionOption a, IGeometryExecutor::ExecutionOption b) {
    return static_cast<IGeometryExecutor::ExecutionOption>(
        static_cast<std::underlying_type_t<IGeometryExecutor::ExecutionOption>>(a) &
        static_cast<std::underlying_type_t<IGeometryExecutor::ExecutionOption>>(b)
    );
}

nva2f::IGeometryExecutor::ExecutionOption& nva2f::internal::operator&=(IGeometryExecutor::ExecutionOption& a, IGeometryExecutor::ExecutionOption b) {
    a = a & b;
    return a;
}

bool nva2f::internal::IsAnySet(IGeometryExecutor::ExecutionOption flags, IGeometryExecutor::ExecutionOption flagsToCheck) {
    return (flags & flagsToCheck) != IGeometryExecutor::ExecutionOption::None;
}
