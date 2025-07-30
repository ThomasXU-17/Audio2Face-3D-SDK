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
#include "audio2face/internal/interactive_executor.h"
#include "audio2face/internal/macros.h"
#include "audio2face/internal/logger.h"
#include "audio2x/error.h"

#include <cassert>
#include <memory>
#include <numeric>

namespace nva2f {

std::error_code GeometryInteractiveExecutorBase::Invalidate(invalidation_layer_t layer) {
    switch (layer) {
        case kLayerNone:
            break;
        case kLayerAll:
            [[fallthrough]];
        case kLayerInference:
            _inferenceResultsValid = false;
            _skinResultsValid = false;
            _tongueResultsValid = false;
            _teethResultsValid = false;
            _eyesResultsValid = false;
            break;
        case kLayerSkin:
            _skinResultsValid = false;
            break;
        case kLayerTongue:
            _tongueResultsValid = false;
            break;
        case kLayerTeeth:
            _teethResultsValid = false;
            break;
        case kLayerEyes:
            _eyesResultsValid = false;
            break;
        default:
            return nva2x::ErrorCode::eInvalidValue;
    }
    return nva2x::ErrorCode::eSuccess;
}

bool GeometryInteractiveExecutorBase::IsValid(invalidation_layer_t layer) const {
    switch (layer) {
        case kLayerNone:
            return true;
        case kLayerAll:
            return _inferenceResultsValid
                && _skinResultsValid
                && _tongueResultsValid
                && _teethResultsValid
                && _eyesResultsValid;
        case kLayerInference:
            return _inferenceResultsValid;
        case kLayerSkin:
            return _skinResultsValid;
        case kLayerTongue:
            return _tongueResultsValid;
        case kLayerTeeth:
            return _teethResultsValid;
        case kLayerEyes:
            return _eyesResultsValid;
        default:
            return false;
    }
}

void GeometryInteractiveExecutorBase::Destroy() {
    delete this;
}

std::size_t GeometryInteractiveExecutorBase::GetTotalNbFrames() const {
    A2F_CHECK_ERROR_WITH_MSG(!CheckInputsState(), "Invalid inputs state", 0);

    auto singleFrameProgress = GetSingleFrameProgress();
    singleFrameProgress.ResetReadWindowCount();

    const auto nbAccumulatedSamples = _track.audioAccumulator->NbAccumulatedSamples();
    const auto nbTotalFrames = singleFrameProgress.GetNbAvailableWindows(nbAccumulatedSamples, true);
    assert(nbTotalFrames >= _nbFramesBeforeAudio);

    return nbTotalFrames - _nbFramesBeforeAudio;
}

std::size_t GeometryInteractiveExecutorBase::GetSamplingRate() const {
    return GetCore().GetSamplingRate();
}

void GeometryInteractiveExecutorBase::GetFrameRate(std::size_t& numerator, std::size_t& denominator) const {
    return GetCore().GetFrameRate(numerator, denominator);
}

GeometryInteractiveExecutorBase::timestamp_t GeometryInteractiveExecutorBase::GetFrameTimestamp(std::size_t frameIndex) const {
    const auto singleFrameProgress = GetSingleFrameProgress();

    timestamp_t start, target, end;
    singleFrameProgress.GetWindow(start, target, end, frameIndex);
    assert(target >= 0);
    return target;
}

std::error_code GeometryInteractiveExecutorBase::Interrupt() {
    A2F_CHECK_RESULT(CheckInputsState());
    _isInterrupted = true;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryInteractiveExecutorBase::SetResultsCallback(results_callback_t callback, void* userdata) {
    A2F_CHECK_ERROR_WITH_MSG(callback || !userdata, "User data must be null if callback is null", nva2x::ErrorCode::eNullPointer);
    _resultsCallback = callback;
    _resultsUserdata = userdata;
    return nva2x::ErrorCode::eSuccess;
}

std::size_t GeometryInteractiveExecutorBase::GetSkinGeometrySize() const {
    return GetCore().GetSkinGeometrySize();
}

std::size_t GeometryInteractiveExecutorBase::GetTongueGeometrySize() const {
    return GetCore().GetTongueGeometrySize();
}

std::size_t GeometryInteractiveExecutorBase::GetJawTransformSize() const {
    return GetCore().GetJawTransformSize();
}

std::size_t GeometryInteractiveExecutorBase::GetEyesRotationSize() const {
    return GetCore().GetEyesRotationSize();
}

std::error_code GeometryInteractiveExecutorBase::GetInputStrength(float& inputStrength) const {
    inputStrength = GetCore().GetInputStrength();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryInteractiveExecutorBase::SetInputStrength(float inputStrength) {
    if (GetCore().GetInputStrength() != inputStrength) {
        A2F_CHECK_RESULT_WITH_MSG(Invalidate(kLayerInference), "Unable to invalidate inference layer for input strength");
    }
    GetCore().SetInputStrength(inputStrength);
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryInteractiveExecutorBase::Get(AnimatorSkinParams& params) const {
    const auto skinAnimator = GetCore().GetSkinAnimator();
    A2F_CHECK_ERROR_WITH_MSG(skinAnimator, "Skin animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    const auto returnParams = skinAnimator->GetParameters(0);
    A2F_CHECK_ERROR_WITH_MSG(returnParams, "Skin parameters could not be read", nva2x::ErrorCode::eNullPointer);
    params = *returnParams;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryInteractiveExecutorBase::Set(const AnimatorSkinParams& params) {
    const auto skinAnimator = GetCore().GetSkinAnimator();
    A2F_CHECK_ERROR_WITH_MSG(skinAnimator, "Skin animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    const auto* skinParams = skinAnimator->GetParameters(0);
    A2F_CHECK_ERROR_WITH_MSG(skinParams, "Skin parameters could not be read", nva2x::ErrorCode::eNullPointer);
    if (!AreEqual_INTERNAL(*skinParams, params)) {
        A2F_CHECK_RESULT_WITH_MSG(Invalidate(kLayerSkin), "Unable to invalidate skin layer for parameters");
    }
    return skinAnimator->SetParameters(0, params);
}

std::error_code GeometryInteractiveExecutorBase::Get(AnimatorTongueParams& params) const {
    const auto tongueAnimator = GetCore().GetTongueAnimator();
    A2F_CHECK_ERROR_WITH_MSG(tongueAnimator, "Tongue animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    const auto returnParams = tongueAnimator->GetParameters(0);
    A2F_CHECK_ERROR_WITH_MSG(returnParams, "Tongue parameters could not be read", nva2x::ErrorCode::eNullPointer);
    params = *returnParams;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryInteractiveExecutorBase::Set(const AnimatorTongueParams& params) {
    const auto tongueAnimator = GetCore().GetTongueAnimator();
    A2F_CHECK_ERROR_WITH_MSG(tongueAnimator, "Tongue animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    const auto* tongueParams = tongueAnimator->GetParameters(0);
    A2F_CHECK_ERROR_WITH_MSG(tongueParams, "Tongue parameters could not be read", nva2x::ErrorCode::eNullPointer);
    if (!AreEqual_INTERNAL(*tongueParams, params)) {
        A2F_CHECK_RESULT_WITH_MSG(Invalidate(kLayerTongue), "Unable to invalidate tongue layer for parameters");
    }
    return tongueAnimator->SetParameters(0, params);
}

std::error_code GeometryInteractiveExecutorBase::Get(AnimatorTeethParams& params) const {
    const auto teethAnimator = GetCore().GetTeethAnimator();
    A2F_CHECK_ERROR_WITH_MSG(teethAnimator, "Teeth animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    const auto returnParams = teethAnimator->GetParameters(0);
    A2F_CHECK_ERROR_WITH_MSG(returnParams, "Teeth parameters could not be read", nva2x::ErrorCode::eNullPointer);
    params = *returnParams;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryInteractiveExecutorBase::Set(const AnimatorTeethParams& params) {
    const auto teethAnimator = GetCore().GetTeethAnimator();
    A2F_CHECK_ERROR_WITH_MSG(teethAnimator, "Teeth animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    const auto* teethParams = teethAnimator->GetParameters(0);
    A2F_CHECK_ERROR_WITH_MSG(teethParams, "Teeth parameters could not be read", nva2x::ErrorCode::eNullPointer);
    if (!AreEqual_INTERNAL(*teethParams, params)) {
        A2F_CHECK_RESULT_WITH_MSG(Invalidate(kLayerTeeth), "Unable to invalidate teeth layer for parameters");
    }
    return teethAnimator->SetParameters(0, params);
}

std::error_code GeometryInteractiveExecutorBase::Get(AnimatorEyesParams& params) const {
    const auto eyesAnimator = GetCore().GetEyesAnimator();
    A2F_CHECK_ERROR_WITH_MSG(eyesAnimator, "Eyes animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    const auto returnParams = eyesAnimator->GetParameters(0);
    A2F_CHECK_ERROR_WITH_MSG(returnParams, "Eyes parameters could not be read", nva2x::ErrorCode::eNullPointer);
    params = *returnParams;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryInteractiveExecutorBase::Set(const AnimatorEyesParams& params) {
    const auto eyesAnimator = GetCore().GetEyesAnimator();
    A2F_CHECK_ERROR_WITH_MSG(eyesAnimator, "Eyes animator cannot be null", nva2x::ErrorCode::eNotInitialized);
    const auto* eyesParams = eyesAnimator->GetParameters(0);
    A2F_CHECK_ERROR_WITH_MSG(eyesParams, "Eyes parameters could not be read", nva2x::ErrorCode::eNullPointer);
    if (!AreEqual_INTERNAL(*eyesParams, params)) {
        A2F_CHECK_RESULT_WITH_MSG(Invalidate(kLayerEyes), "Unable to invalidate eyes layer for parameters");
    }
    return eyesAnimator->SetParameters(0, params);
}

std::error_code GeometryInteractiveExecutorBase::SetGeometryResultsCallback(
    IGeometryInteractiveExecutor::results_callback_t callback, void* userdata
    ) {
    return SetResultsCallback(callback, userdata);
}

std::error_code GeometryInteractiveExecutorBase::BaseInit(
    const nva2f::GeometryExecutorCreationParameters& params,
    std::size_t emotionSize,
    const nva2x::WindowProgressParameters& progressParams,
    std::size_t nbFramesPerInference
    ) {
    A2F_CHECK_ERROR_WITH_MSG(params.nbTracks == 1, "Number of tracks must be 1", nva2x::ErrorCode::eInvalidValue);
    A2F_CHECK_ERROR_WITH_MSG(params.sharedAudioAccumulators, "Audio accumulators cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_ERROR_WITH_MSG(params.sharedEmotionAccumulators, "Emotion accumulators cannot be null", nva2x::ErrorCode::eNullPointer);

    _nbFramesPerInference = nbFramesPerInference;

    // Initialize the track data.
    A2F_CHECK_ERROR_WITH_MSG(params.sharedAudioAccumulators[0], "Audio accumulators cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_ERROR_WITH_MSG(params.sharedEmotionAccumulators[0], "Emotion accumulators cannot be null", nva2x::ErrorCode::eNullPointer);

    A2F_CHECK_ERROR_WITH_MSG(
        params.sharedEmotionAccumulators[0]->GetEmotionSize() == emotionSize,
        "Emotion accumulator does not have the right size for emotions",
        nva2x::ErrorCode::eMismatch
    );

    _track.audioAccumulator = params.sharedAudioAccumulators[0];
    _track.emotionAccumulator = params.sharedEmotionAccumulators[0];

    _track.progress = std::make_unique<nva2x::WindowProgress>(progressParams);

    // Compute the number of frames which will be inferred before.
    // This is currently only used for the diffusion model, which starts before the audio.
    // The regression model should have 0 for this value.
    const auto singleFrameProgress = GetSingleFrameProgress();
    _nbFramesBeforeAudio = singleFrameProgress.GetNbAvailableWindows(std::size_t{0}, true);

    return nva2x::ErrorCode::eSuccess;
}

nva2x::WindowProgress GeometryInteractiveExecutorBase::GetSingleFrameProgress() const {
    assert(_track.progress);
    const auto nbFramesPerInference = _nbFramesPerInference;
    assert(nbFramesPerInference > 0);
    return nva2x::GetFrameProgress(*_track.progress, nbFramesPerInference);
}

std::error_code GeometryInteractiveExecutorBase::CheckInputsState() const {
    const auto audioAccumulator = _track.audioAccumulator;
    A2F_CHECK_ERROR_WITH_MSG(audioAccumulator, "Audio accumulator cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_ERROR_WITH_MSG(audioAccumulator->IsClosed(), "Audio accumulator is not closed", nva2x::ErrorCode::eInvalidValue);
    A2F_CHECK_ERROR_WITH_MSG(audioAccumulator->NbDroppedSamples() == 0, "Audio accumulator has dropped samples", nva2x::ErrorCode::eInvalidValue);

    const auto emotionAccumulator = _track.emotionAccumulator;
    if (emotionAccumulator) {
        A2F_CHECK_ERROR_WITH_MSG(emotionAccumulator->IsClosed(), "Emotion accumulator is not closed", nva2x::ErrorCode::eInvalidValue);
        A2F_CHECK_ERROR_WITH_MSG(emotionAccumulator->NbDroppedEmotions() == 0, "Emotion accumulator has dropped emotions", nva2x::ErrorCode::eInvalidValue);
        A2F_CHECK_ERROR_WITH_MSG(emotionAccumulator->LastDroppedTimestamp() == std::numeric_limits<timestamp_t>::min(), "Emotion accumulator has dropped samples", nva2x::ErrorCode::eInvalidValue);
    }

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f


namespace {

    // Just forward the request to the AccessorType interface.
    // NOTE: This dynamic_cast approach might fail if the user passes a IFaceInteractiveExecutor
    // derived class implemented in their binary, with their compiler, which might handle
    // dynamic_cast in a different way.
    template <typename AccessorType, typename... Args>
    std::error_code GetInteractiveExecutorParameters(const nva2f::IFaceInteractiveExecutor& executor, Args&&... params) {
        const auto accessor = dynamic_cast<const AccessorType*>(&executor);
        if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
        return accessor->Get(std::forward<Args>(params)...);
    }

    template <typename AccessorType, typename... Args>
    std::error_code SetInteractiveExecutorParameters(nva2f::IFaceInteractiveExecutor& executor, Args&&... params) {
        const auto accessor = dynamic_cast<AccessorType*>(&executor);
        if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
        return accessor->Set(std::forward<Args>(params)...);
     }

}

std::error_code nva2f::GetInteractiveExecutorInputStrength_INTERNAL(
    const IFaceInteractiveExecutor& executor, float& inputStrength
    ) {
    const auto accessor = dynamic_cast<const IFaceInteractiveExecutorAccessorInputStrength*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetInputStrength(inputStrength);
}

std::error_code nva2f::SetInteractiveExecutorInputStrength_INTERNAL(
    IFaceInteractiveExecutor& executor, float inputStrength
    ) {
    const auto accessor = dynamic_cast<IFaceInteractiveExecutorAccessorInputStrength*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->SetInputStrength(inputStrength);
}

std::error_code nva2f::GetInteractiveExecutorSkinParameters_INTERNAL(
    const IFaceInteractiveExecutor& executor, AnimatorSkinParams& params
    ) {
    return GetInteractiveExecutorParameters<IFaceInteractiveExecutorAccessorSkinParameters>(executor, params);
}

std::error_code nva2f::SetInteractiveExecutorSkinParameters_INTERNAL(
    IFaceInteractiveExecutor& executor, const AnimatorSkinParams& params
    ) {
    return SetInteractiveExecutorParameters<IFaceInteractiveExecutorAccessorSkinParameters>(executor, params);
}

std::error_code nva2f::GetInteractiveExecutorTongueParameters_INTERNAL(
    const IFaceInteractiveExecutor& executor, AnimatorTongueParams& params
    ) {
    return GetInteractiveExecutorParameters<IFaceInteractiveExecutorAccessorTongueParameters>(executor, params);
}

std::error_code nva2f::SetInteractiveExecutorTongueParameters_INTERNAL(
    IFaceInteractiveExecutor& executor, const AnimatorTongueParams& params
    ) {
    return SetInteractiveExecutorParameters<IFaceInteractiveExecutorAccessorTongueParameters>(executor, params);
}

std::error_code nva2f::GetInteractiveExecutorTeethParameters_INTERNAL(
    const IFaceInteractiveExecutor& executor, AnimatorTeethParams& params
    ) {
    return GetInteractiveExecutorParameters<IFaceInteractiveExecutorAccessorTeethParameters>(executor, params);
}

std::error_code nva2f::SetInteractiveExecutorTeethParameters_INTERNAL(
    IFaceInteractiveExecutor& executor, const AnimatorTeethParams& params
    ) {
    return SetInteractiveExecutorParameters<IFaceInteractiveExecutorAccessorTeethParameters>(executor, params);
}

std::error_code nva2f::GetInteractiveExecutorEyesParameters_INTERNAL(
    const IFaceInteractiveExecutor& executor, AnimatorEyesParams& params
    ) {
    return GetInteractiveExecutorParameters<IFaceInteractiveExecutorAccessorEyesParameters>(executor, params);
}

std::error_code nva2f::SetInteractiveExecutorEyesParameters_INTERNAL(
    IFaceInteractiveExecutor& executor, const AnimatorEyesParams& params
    ) {
    return SetInteractiveExecutorParameters<IFaceInteractiveExecutorAccessorEyesParameters>(executor, params);
}

std::error_code nva2f::GetInteractiveExecutorGeometryExecutor_INTERNAL(
    const IFaceInteractiveExecutor& executor, IGeometryInteractiveExecutor** geometryExecutor
    ) {
    const auto accessor = dynamic_cast<const IFaceInteractiveExecutorAccessorGeometryInteractiveExecutor*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetGeometryInteractiveExecutor(geometryExecutor);
}

std::error_code nva2f::SetInteractiveExecutorGeometryResultsCallback_INTERNAL(
    IFaceInteractiveExecutor& executor, IGeometryInteractiveExecutor::results_callback_t callback, void* userdata
    ) {
    const auto accessor = dynamic_cast<IFaceInteractiveExecutorAccessorGeometryResultsCallback*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->SetGeometryResultsCallback(callback, userdata);
}

std::error_code nva2f::GetInteractiveExecutorBlendshapeSkinConfig_INTERNAL(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverConfig& config
    ) {
    const auto accessor = dynamic_cast<const IFaceInteractiveExecutorAccessorBlendshapeParameters*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetSkinConfig(config);
}

std::error_code nva2f::SetInteractiveExecutorBlendshapeSkinConfig_INTERNAL(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverConfig& config
    ) {
    const auto accessor = dynamic_cast<IFaceInteractiveExecutorAccessorBlendshapeParameters*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->SetSkinConfig(config);
}

std::error_code nva2f::GetInteractiveExecutorBlendshapeSkinParameters_INTERNAL(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverParams& params
    ) {
    const auto accessor = dynamic_cast<const IFaceInteractiveExecutorAccessorBlendshapeParameters*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetSkinParameters(params);
}

std::error_code nva2f::SetInteractiveExecutorBlendshapeSkinParameters_INTERNAL(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverParams& params
    ) {
    const auto accessor = dynamic_cast<IFaceInteractiveExecutorAccessorBlendshapeParameters*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->SetSkinParameters(params);
}

std::error_code nva2f::GetInteractiveExecutorBlendshapeTongueConfig_INTERNAL(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverConfig& config
    ) {
    const auto accessor = dynamic_cast<const IFaceInteractiveExecutorAccessorBlendshapeParameters*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetTongueConfig(config);
}

std::error_code nva2f::SetInteractiveExecutorBlendshapeTongueConfig_INTERNAL(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverConfig& config
    ) {
    const auto accessor = dynamic_cast<IFaceInteractiveExecutorAccessorBlendshapeParameters*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->SetTongueConfig(config);
}

std::error_code nva2f::GetInteractiveExecutorBlendshapeTongueParameters_INTERNAL(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverParams& params
    ) {
    const auto accessor = dynamic_cast<const IFaceInteractiveExecutorAccessorBlendshapeParameters*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetTongueParameters(params);
}

std::error_code nva2f::SetInteractiveExecutorBlendshapeTongueParameters_INTERNAL(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverParams& params
    ) {
    const auto accessor = dynamic_cast<IFaceInteractiveExecutorAccessorBlendshapeParameters*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->SetTongueParameters(params);
}
