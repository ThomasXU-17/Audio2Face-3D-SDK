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
#include "audio2emotion/internal/interactive_executor.h"
#include "audio2emotion/internal/macros.h"
#include "audio2emotion/internal/logger.h"
#include "audio2x/error.h"
#include "audio2x/internal/nvtx_trace.h"

#include <cassert>
#include <memory>
#include <numeric>

namespace nva2e {

std::error_code EmotionInteractiveExecutorBase::Invalidate(invalidation_layer_t layer) {
    switch (layer) {
        case kLayerNone:
            break;
        case kLayerAll:
            [[fallthrough]];
        case kLayerInference:
            _inferenceResultsValid = false;
            _postProcessResultsValid = false;
            break;
        case kLayerPostProcessing:
            _postProcessResultsValid = false;
            break;
        default:
            return nva2x::ErrorCode::eInvalidValue;
    }
    return nva2x::ErrorCode::eSuccess;
}

bool EmotionInteractiveExecutorBase::IsValid(invalidation_layer_t layer) const {
    switch (layer) {
        case kLayerNone:
            return true;
        case kLayerAll:
            return _inferenceResultsValid && _postProcessResultsValid;
        case kLayerInference:
            return _inferenceResultsValid;
        case kLayerPostProcessing:
            return _postProcessResultsValid;
        default:
            return false;
    }
}

void EmotionInteractiveExecutorBase::Destroy() {
    delete this;
}

std::size_t EmotionInteractiveExecutorBase::GetTotalNbFrames() const {
    A2E_CHECK_ERROR_WITH_MSG(!CheckInputsState(), "Invalid inputs state", 0);

    return nva2x::GetTotalNbFrames(
        *_audioAccumulator,
        GetSingleFrameProgress(),
        1
        );
}

std::size_t EmotionInteractiveExecutorBase::GetSamplingRate() const {
    return GetCore().GetSamplingRate();
}

void EmotionInteractiveExecutorBase::GetFrameRate(std::size_t& numerator, std::size_t& denominator) const {
    GetCore().GetFrameRate(numerator, denominator);
}

EmotionInteractiveExecutorBase::timestamp_t EmotionInteractiveExecutorBase::GetFrameTimestamp(std::size_t frameIndex) const {
    const auto singleFrameProgress = GetSingleFrameProgress();

    timestamp_t start, target, end;
    singleFrameProgress.GetWindow(start, target, end, frameIndex);
    assert(target >= 0);
    return target;
}

std::error_code EmotionInteractiveExecutorBase::ComputeFrame(std::size_t frameIndex) {
    A2E_CHECK_RESULT(CheckInputsState());
    A2E_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);

    NVTX_TRACE("ComputeFrame");

    _isInterrupted = false;

    const std::size_t endFrameIndex = frameIndex + 1;

    // First, compute inferences (if necessary)
    A2E_CHECK_RESULT(ComputeInference());

    // Then, compute post-processing
    A2E_CHECK_RESULT(ComputePostProcessing(endFrameIndex, CallCallback::OnlyLastFrame));

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionInteractiveExecutorBase::ComputeAllFrames() {
    A2E_CHECK_RESULT(CheckInputsState());
    A2E_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);

    NVTX_TRACE("ComputeAllFrames");

    _isInterrupted = false;

    // First, compute inferences (if necessary)
    A2E_CHECK_RESULT(ComputeInference());
    assert(_inferenceResultsValid);

    // Then, compute post-processing
    const std::size_t endFrameIndex = GetTotalNbFrames();
    A2E_CHECK_RESULT(ComputePostProcessing(endFrameIndex, CallCallback::AllFrames));

    // We don't cache the post-processing results.  But after computing all frames,
    // we mark them as valid.
    _postProcessResultsValid = true;

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionInteractiveExecutorBase::Interrupt() {
    A2E_CHECK_RESULT(CheckInputsState());
    _isInterrupted = true;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionInteractiveExecutorBase::SetResultsCallback(results_callback_t callback, void* userdata) {
    A2E_CHECK_ERROR_WITH_MSG(callback || !userdata, "User data must be null if callback is null", nva2x::ErrorCode::eNullPointer);
    _resultsCallback = callback;
    _resultsUserdata = userdata;
    return nva2x::ErrorCode::eSuccess;
}

std::size_t EmotionInteractiveExecutorBase::GetEmotionsSize() const {
    return GetCore().GetPostProcessor().GetOutputEmotionsSize();
}

std::error_code EmotionInteractiveExecutorBase::GetInputStrength(float& inputStrength) const {
    inputStrength = GetCore().GetInputStrength();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionInteractiveExecutorBase::SetInputStrength(float inputStrength) {
    if (GetCore().GetInputStrength() != inputStrength) {
        A2E_CHECK_RESULT_WITH_MSG(Invalidate(kLayerInference), "Unable to invalidate inference layer for input strength");
    }
    GetCore().SetInputStrength(inputStrength);
    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionInteractiveExecutorBase::Get(PostProcessParams& params) const {
    const auto* postProcessParams = GetCore().GetPostProcessor().GetParameters(0);
    A2E_CHECK_ERROR_WITH_MSG(postProcessParams, "Post-process parameters could not be read", nva2x::ErrorCode::eNullPointer);
    params = *postProcessParams;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionInteractiveExecutorBase::Set(const PostProcessParams& params) {
    const auto* postProcessParams = GetCore().GetPostProcessor().GetParameters(0);
    A2E_CHECK_ERROR_WITH_MSG(postProcessParams, "Post-process parameters could not be read", nva2x::ErrorCode::eNullPointer);
    if (!AreEqual_INTERNAL(*postProcessParams, params)) {
        A2E_CHECK_RESULT_WITH_MSG(Invalidate(kLayerPostProcessing), "Unable to invalidate post-process layer for parameters");
    }
    return GetCore().GetPostProcessor().SetParameters(0, params);
}

std::error_code EmotionInteractiveExecutorBase::BaseInit(
    const nva2e::EmotionExecutorCreationParameters& params,
    std::size_t emotionSize,
    const nva2x::WindowProgressParameters& progressParams,
    const nva2x::IEmotionAccumulator* const* sharedPreferredEmotionAccumulators
    ) {
    A2E_CHECK_ERROR_WITH_MSG(params.nbTracks == 1, "Number of tracks must be 1", nva2x::ErrorCode::eInvalidValue);
    A2E_CHECK_ERROR_WITH_MSG(params.sharedAudioAccumulators, "Audio accumulators cannot be null", nva2x::ErrorCode::eNullPointer);

    const nva2x::IAudioAccumulator* audioAccumulator = params.sharedAudioAccumulators[0];
    A2E_CHECK_ERROR_WITH_MSG(audioAccumulator, "Audio accumulator cannot be null", nva2x::ErrorCode::eNullPointer);

    const nva2x::IEmotionAccumulator* emotionAccumulator = nullptr;
    if (sharedPreferredEmotionAccumulators) {
        emotionAccumulator = sharedPreferredEmotionAccumulators[0];
        if (emotionAccumulator) {
            A2E_CHECK_ERROR_WITH_MSG(
                emotionAccumulator->GetEmotionSize() == emotionSize,
                "Emotion accumulator does not have the right size for emotions",
                nva2x::ErrorCode::eMismatch
            );
        }
    }

    _audioAccumulator = audioAccumulator;
    _emotionAccumulator = emotionAccumulator;
    _progress = std::make_unique<nva2x::WindowProgress>(progressParams);

    return nva2x::ErrorCode::eSuccess;
}

nva2x::WindowProgress EmotionInteractiveExecutorBase::GetSingleFrameProgress() const {
    return GetCore().GetSingleFrameProgress(*_progress);
}

std::error_code EmotionInteractiveExecutorBase::CheckInputsState() const {
    A2E_CHECK_ERROR_WITH_MSG(_audioAccumulator, "Audio accumulator cannot be null", nva2x::ErrorCode::eNullPointer);
    A2E_CHECK_ERROR_WITH_MSG(_audioAccumulator->IsClosed(), "Audio accumulator is not closed", nva2x::ErrorCode::eInvalidValue);
    A2E_CHECK_ERROR_WITH_MSG(_audioAccumulator->NbDroppedSamples() == 0, "Audio accumulator has dropped samples", nva2x::ErrorCode::eInvalidValue);

    if (_emotionAccumulator) {
        A2E_CHECK_ERROR_WITH_MSG(_emotionAccumulator->IsClosed(), "Emotion accumulator is not closed", nva2x::ErrorCode::eInvalidValue);
        A2E_CHECK_ERROR_WITH_MSG(_emotionAccumulator->NbDroppedEmotions() == 0, "Emotion accumulator has dropped emotions", nva2x::ErrorCode::eInvalidValue);
        A2E_CHECK_ERROR_WITH_MSG(_emotionAccumulator->LastDroppedTimestamp() == std::numeric_limits<timestamp_t>::min(), "Emotion accumulator has dropped samples", nva2x::ErrorCode::eInvalidValue);
    }

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionInteractiveExecutorBase::ComputePostProcessing(std::size_t endFrameIndex, CallCallback callCallback) {
    assert(!CheckInputsState());
    assert(_inferenceResultsValid);

    NVTX_TRACE("ComputePostProcessing");

    const auto nbFramesPerExecution = GetCore().GetNbFramesPerExecution();
    const auto emotionSize = GetCore().GetPostProcessor().GetInputEmotionsSize();
    A2E_CHECK_ERROR_WITH_MSG(
        endFrameIndex / nbFramesPerExecution <= _inferenceResults.Size() / emotionSize,
        "End frame index is too large",
        nva2x::ErrorCode::eOutOfBounds
        );

    // Only post-processing has state.
    A2E_CHECK_RESULT_WITH_MSG(GetCore().GetPostProcessor().Reset(0), "Unable to reset post-processor");

    auto frameProgress = GetSingleFrameProgress();
    frameProgress.ResetReadWindowCount();
    for (std::size_t frameIndex = 0; frameIndex < endFrameIndex; ++frameIndex) {
        if (_isInterrupted) {
            return nva2x::ErrorCode::eInterrupted;
        }

        NVTX_TRACE("Frame");

        // Restore the cached inference results.
        if (frameIndex % nbFramesPerExecution == 0) {
            const auto inferenceIndex = frameIndex / nbFramesPerExecution;
            const auto cachedInferenceResults = _inferenceResults.View(
                inferenceIndex * emotionSize, emotionSize
                );
            auto inferenceResultsToPostProcess = GetCore().GetInferenceOutputBuffer(0, 1);
            A2E_CHECK_RESULT_WITH_MSG(
                nva2x::CopyDeviceToDevice(
                    inferenceResultsToPostProcess,
                    cachedInferenceResults,
                    GetCore().GetCudaStream()
                    ),
                "Unable to copy inference results to post-process buffer"
                );
        }

        timestamp_t start, target, end;
        frameProgress.GetCurrentWindow(start, target, end, frameIndex);
        {
            NVTX_TRACE("ReadPreferredEmotion");
            A2E_CHECK_RESULT_WITH_MSG(
                GetCore().ReadPreferredEmotion(_emotionAccumulator, 0, target),
                "Unable to read preferred emotion from emotion accumulator, has enough emotion data been provided?"
            );
        }

        {
            NVTX_TRACE("RunPostProcess");
            A2E_CHECK_RESULT_WITH_MSG(GetCore().RunPostProcess(), "Unable to run post-process");
        }

        const bool runCallback = _resultsCallback && (
            (callCallback == CallCallback::AllFrames) ||
            (callCallback == CallCallback::OnlyLastFrame && frameIndex == endFrameIndex - 1)
        );
        if (runCallback) {
            NVTX_TRACE("ResultsCallback");

            timestamp_t startNext, targetNext, endNext;
            frameProgress.GetCurrentWindow(startNext, targetNext, endNext, frameIndex + 1);

            Results results;
            results.trackIndex = 0;
            results.timeStampCurrentFrame = target;
            results.timeStampNextFrame = targetNext;
            results.cudaStream = GetCore().GetCudaStream();
            results.emotions = GetCore().GetOutputEmotions(0, 1);

            const bool keepGoing = _resultsCallback(_resultsUserdata, results);
            if (!keepGoing) {
                return nva2x::ErrorCode::eInterrupted;
            }
        }
    }

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2e

namespace {

    // Just forward the request to the AccessorType interface.
    // NOTE: This dynamic_cast approach might fail if the user passes a IEmotionInteractiveExecutor
    // derived class implemented in their binary, with their compiler, which might handle
    // dynamic_cast in a different way.
    template <typename AccessorType, typename... Args>
    std::error_code GetInteractiveExecutorParameters(const nva2e::IEmotionInteractiveExecutor& executor, Args&&... params) {
        const auto accessor = dynamic_cast<const AccessorType*>(&executor);
        if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
        return accessor->Get(std::forward<Args>(params)...);
    }

    template <typename AccessorType, typename... Args>
    std::error_code SetInteractiveExecutorParameters(nva2e::IEmotionInteractiveExecutor& executor, Args&&... params) {
        const auto accessor = dynamic_cast<AccessorType*>(&executor);
        if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
        return accessor->Set(std::forward<Args>(params)...);
     }

}

std::error_code nva2e::GetInteractiveExecutorInferencesToSkip_INTERNAL(
    const IEmotionInteractiveExecutor& executor, std::size_t& inferencesToSkip
    ) {
    const auto accessor = dynamic_cast<const IEmotionInteractiveExecutorAccessorInferencesToSkip*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetInferencesToSkip(inferencesToSkip);
}

std::error_code nva2e::SetInteractiveExecutorInferencesToSkip_INTERNAL(
    IEmotionInteractiveExecutor& executor, std::size_t inferencesToSkip
    ) {
    const auto accessor = dynamic_cast<IEmotionInteractiveExecutorAccessorInferencesToSkip*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->SetInferencesToSkip(inferencesToSkip);
}

std::error_code nva2e::GetInteractiveExecutorInputStrength_INTERNAL(
    const IEmotionInteractiveExecutor& executor, float& inputStrength
    ) {
    const auto accessor = dynamic_cast<const IEmotionInteractiveExecutorAccessorInputStrength*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetInputStrength(inputStrength);
}

std::error_code nva2e::SetInteractiveExecutorInputStrength_INTERNAL(
    IEmotionInteractiveExecutor& executor, float inputStrength
    ) {
    const auto accessor = dynamic_cast<IEmotionInteractiveExecutorAccessorInputStrength*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->SetInputStrength(inputStrength);
}

std::error_code nva2e::GetInteractiveExecutorPostProcessParameters_INTERNAL(
    const IEmotionInteractiveExecutor& executor, PostProcessParams& params
    ) {
    return GetInteractiveExecutorParameters<IEmotionInteractiveExecutorAccessorPostProcessParameters>(executor, params);
}

std::error_code nva2e::SetInteractiveExecutorPostProcessParameters_INTERNAL(
    IEmotionInteractiveExecutor& executor, const PostProcessParams& params
    ) {
    return SetInteractiveExecutorParameters<IEmotionInteractiveExecutorAccessorPostProcessParameters>(executor, params);
}
