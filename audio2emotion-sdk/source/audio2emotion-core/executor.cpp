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
#include "audio2emotion/internal/executor.h"
#include "audio2emotion/internal/macros.h"
#include "audio2emotion/internal/logger.h"
#include "audio2x/error.h"
#include "audio2x/internal/executor.h"
#include "audio2x/internal/nvtx_trace.h"

#include <cassert>
#include <memory>
#include <numeric>


namespace nva2e {

std::size_t EmotionExecutorBase::GetNbTracks() const {
    return _trackData.size();
}

std::error_code EmotionExecutorBase::Reset(std::size_t trackIndex) {
    A2E_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
    auto& trackData = _trackData[trackIndex];
    // Don't reset the audio or emotion accumulator, we leave that to others.
    if (trackData.progress) {
        trackData.progress->ResetReadWindowCount();
    }

    A2E_CHECK_RESULT_WITH_MSG(GetCore().Reset(trackIndex), "Unable to reset post-processor");

    return nva2x::ErrorCode::eSuccess;
}

void EmotionExecutorBase::Destroy() {
    delete this;
}

bool EmotionExecutorBase::HasExecutionStarted(std::size_t trackIndex) const {
    A2E_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", false);
    const auto& trackData = _trackData[trackIndex];
    return trackData.progress->GetReadWindowCount() != 0;
}

std::size_t EmotionExecutorBase::GetNbAvailableExecutions(std::size_t trackIndex) const {
    A2E_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", 0);
    const auto& trackData = _trackData[trackIndex];
    const auto audioAccumulator = trackData.audioAccumulator;
    const auto emotionAccumulator = trackData.emotionAccumulator;
    const auto& progress = *trackData.progress;
    A2E_CHECK_ERROR_WITH_MSG(audioAccumulator, "Audio accumulator cannot be null", 0);

    const bool checkEmotions = GetCore().GetPostProcessor().GetParameters(trackIndex)->enablePreferredEmotion;
    return nva2x::GetNbAvailableExecutions(
        progress,
        *audioAccumulator,
        checkEmotions ? emotionAccumulator : nullptr,
        GetCore().GetNbFramesPerExecution()
        );
}

std::size_t EmotionExecutorBase::GetTotalNbFrames(std::size_t trackIndex) const {
    A2E_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", 0);
    const auto& trackData = _trackData[trackIndex];
    A2E_CHECK_ERROR_WITH_MSG(trackData.audioAccumulator, "Audio accumulator cannot be null", 0);

    return nva2x::GetTotalNbFrames(
        *trackData.audioAccumulator,
        GetSingleFrameProgress(trackIndex),
        1
        );
}

std::size_t EmotionExecutorBase::GetSamplingRate() const {
    return GetCore().GetSamplingRate();
}

void EmotionExecutorBase::GetFrameRate(std::size_t& numerator, std::size_t& denominator) const {
    GetCore().GetFrameRate(numerator, denominator);
}

EmotionExecutorBase::timestamp_t EmotionExecutorBase::GetFrameTimestamp(std::size_t frameIndex) const {
    A2E_CHECK_ERROR_WITH_MSG(0 < _trackData.size(), "Progress data not initialized yet", std::numeric_limits<timestamp_t>::min());
    // Use index 0 because they all have the same progress parameters.
    const auto singleFrameProgress = GetSingleFrameProgress(0);

    timestamp_t start, target, end;
    singleFrameProgress.GetWindow(start, target, end, frameIndex);
    assert(target >= 0);
    return target;
}

std::error_code EmotionExecutorBase::SetResultsCallback(results_callback_t callback, void* userdata) {
    A2E_CHECK_ERROR_WITH_MSG(
        !nva2x::HasExecutionStarted(*this),
        "Results callback can only be set before execution is started",
        nva2x::ErrorCode::eExecutionAlreadyStarted
        );
    A2E_CHECK_ERROR_WITH_MSG(callback || !userdata, "User data must be null if callback is null", nva2x::ErrorCode::eNullPointer);
    _resultsCallback = callback;
    _resultsUserdata = userdata;
    return nva2x::ErrorCode::eSuccess;
}

std::size_t EmotionExecutorBase::GetEmotionsSize() const {
    return GetCore().GetPostProcessor().GetOutputEmotionsSize();
}

std::error_code EmotionExecutorBase::GetInputStrength(float& inputStrength) const {
    inputStrength = GetCore().GetInputStrength();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutorBase::SetInputStrength(float inputStrength) {
    assert(!nva2x::HasExecutionStarted(*this));
    GetCore().SetInputStrength(inputStrength);
    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutorBase::Get(std::size_t trackIndex, PostProcessParams& params) const {
    const auto* postProcessParams = GetCore().GetPostProcessor().GetParameters(trackIndex);
    A2E_CHECK_ERROR_WITH_MSG(postProcessParams, "Post-process parameters coult not be read", nva2x::ErrorCode::eNullPointer);
    params = *postProcessParams;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutorBase::Set(std::size_t trackIndex, const PostProcessParams& params) {
    return GetCore().GetPostProcessor().SetParameters(trackIndex, params);
}

std::error_code EmotionExecutorBase::BaseExecute(std::size_t* pNbExecutedTracks) {
    if (pNbExecutedTracks) {
        *pNbExecutedTracks = 0;
    }

    A2E_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);

    // Forward handling of the the inference execution to the derived class.
    // Accumulate the inference input to be able to run the inference.
    // Data is not tightly packed to avoid moving data around if indices are skipped.
    _executedTracks.reset_all();
    std::size_t nbExecutedTracks = 0;
    A2E_CHECK_RESULT(RunInference(nbExecutedTracks));
    A2E_CHECK_ERROR_WITH_MSG(nbExecutedTracks > 0, "No tracks to execute", nva2x::ErrorCode::eNoTracksToExecute);

    // Generate 1 frame + the number of skipped frames since they are cheap to generate.
    _doneTracks.reset_all();
    for (std::size_t frameIndex = 0; frameIndex < GetCore().GetNbFramesPerExecution(); ++frameIndex) {
        NVTX_TRACE("Frame");

        bool atLeastOneTrack = false;
        _postProcessTracks.reset_all();
        for (std::size_t trackIndex = 0; trackIndex < _trackData.size(); ++trackIndex) {
            const bool executed = _executedTracks[trackIndex];
            if (!executed) {
                continue;
            }
            const bool done = _doneTracks[trackIndex];
            if (done) {
                continue;
            }

            const auto frameProgress = GetSingleFrameProgress(trackIndex);

            timestamp_t start, target, end;
            frameProgress.GetCurrentWindow(start, target, end, frameIndex);

            auto& trackData = _trackData[trackIndex];

            if (target >= static_cast<timestamp_t>(trackData.audioAccumulator->NbAccumulatedSamples())) {
                assert(trackData.audioAccumulator->IsClosed());
                // Don't report the frames after the audio, and stop processing.
                _doneTracks.set(trackIndex);
                continue;
            }

            A2E_CHECK_RESULT_WITH_MSG(
                GetCore().ReadPreferredEmotion(trackData.emotionAccumulator, trackIndex, target),
                "Unable to read preferred emotion from emotion accumulator, has enough emotion data been provided?"
            );

            atLeastOneTrack = true;
            _postProcessTracks.set(trackIndex);
        }

        // Do batch post-processing.
        if (atLeastOneTrack) {
            NVTX_TRACE("PostProcess");
            // We run post-processing on all executed tracks, even the ones which are done.
            // This is because the active tracks are used to know where in the packed array
            // the track data is located.
            A2E_CHECK_RESULT_WITH_MSG(
                GetCore().GetPostProcessor().SetActiveTracks(_executedTracks.block_data(), _executedTracks.block_size()),
                "Unable to set active tracks for post-processing"
                );
            A2E_CHECK_RESULT_WITH_MSG(GetCore().RunPostProcess(), "Unable to run post-processing");
        }

        for (std::size_t trackIndex = 0; trackIndex < _trackData.size(); ++trackIndex) {
            const bool postProcess = _postProcessTracks[trackIndex];
            if (!postProcess) {
                continue;
            }

            const auto frameProgress = GetSingleFrameProgress(trackIndex);

            timestamp_t start, target, end;
            frameProgress.GetCurrentWindow(start, target, end, frameIndex);

            timestamp_t startNext, targetNext, endNext;
            frameProgress.GetCurrentWindow(startNext, targetNext, endNext, frameIndex + 1);

            Results results;
            results.trackIndex = trackIndex;
            results.timeStampCurrentFrame = target;
            results.timeStampNextFrame = targetNext;
            results.cudaStream = GetCore().GetCudaStream();
            results.emotions = GetCore().GetOutputEmotions(trackIndex, 1);

            // Many frames are computed per inference, so check if user wants to stop.
            bool keepGoing;
            {
                NVTX_TRACE("ResultsCallback");
                keepGoing = _resultsCallback(_resultsUserdata, results);
            }

            if (!keepGoing) {
                _doneTracks.set(trackIndex);
            }
        }
    }

    for (std::size_t trackIndex = 0; trackIndex < _trackData.size(); ++trackIndex) {
        if (_executedTracks[trackIndex]) {
            _trackData[trackIndex].progress->IncrementReadWindowCount();
        }
    }

    if (pNbExecutedTracks) {
        *pNbExecutedTracks = nbExecutedTracks;
    }

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutorBase::BaseInit(
    const nva2e::EmotionExecutorCreationParameters& params,
    std::size_t emotionSize,
    const nva2x::WindowProgressParameters& progressParams,
    const nva2x::IEmotionAccumulator* const* sharedPreferredEmotionAccumulators
    ) {
    A2E_CHECK_ERROR_WITH_MSG(params.sharedAudioAccumulators, "Audio accumulators cannot be null", nva2x::ErrorCode::eNullPointer);

    _trackData.resize(params.nbTracks);
    for (std::size_t i = 0; i < params.nbTracks; ++i) {
        const nva2x::IAudioAccumulator* audioAccumulator = params.sharedAudioAccumulators[i];
        A2E_CHECK_ERROR_WITH_MSG(audioAccumulator, "Audio accumulator cannot be null", nva2x::ErrorCode::eNullPointer);

        const nva2x::IEmotionAccumulator* emotionAccumulator = nullptr;
        if (sharedPreferredEmotionAccumulators) {
            emotionAccumulator = sharedPreferredEmotionAccumulators[i];
            if (emotionAccumulator) {
                A2E_CHECK_ERROR_WITH_MSG(
                    emotionAccumulator->GetEmotionSize() == emotionSize,
                    "Emotion accumulator does not have the right size for emotions",
                    nva2x::ErrorCode::eMismatch
                );
            }
        }

        auto& trackData = _trackData[i];
        trackData.audioAccumulator = audioAccumulator;
        trackData.emotionAccumulator = emotionAccumulator;
        trackData.progress = std::make_unique<nva2x::WindowProgress>(progressParams);
    }

    _executedTracks.resize(params.nbTracks);
    _postProcessTracks.resize(_executedTracks.size());
    _doneTracks.resize(_executedTracks.size());

    return nva2x::ErrorCode::eSuccess;
}

nva2x::WindowProgress EmotionExecutorBase::GetSingleFrameProgress(std::size_t trackIndex) const {
    return GetCore().GetSingleFrameProgress(*_trackData[trackIndex].progress);
}

} // namespace nva2e


namespace {

    // Just forward the request to the AccessorType interface.
    // NOTE: This dynamic_cast approach might fail if the user passes a IEmotionExecutor
    // derived class implemented in their binary, with their compiler, which might handle
    // dynamic_cast in a different way.
    template <typename AccessorType, typename... Args>
    std::error_code GetExecutorParameters(const nva2e::IEmotionExecutor& executor, std::size_t trackIndex, Args&&... params) {
        const auto accessor = dynamic_cast<const AccessorType*>(&executor);
        if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
        return accessor->Get(trackIndex, std::forward<Args>(params)...);
    }

    template <typename AccessorType, typename... Args>
    std::error_code SetExecutorParameters(nva2e::IEmotionExecutor& executor, std::size_t trackIndex, Args&&... params) {
        A2E_CHECK_ERROR_WITH_MSG(
            !executor.HasExecutionStarted(trackIndex),
            "Parameters can only be set before execution is started",
            nva2x::ErrorCode::eExecutionAlreadyStarted
            );
        const auto accessor = dynamic_cast<AccessorType*>(&executor);
        if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
        return accessor->Set(trackIndex, std::forward<Args>(params)...);
     }

}

std::error_code nva2e::GetExecutorInputStrength_INTERNAL(
    const IEmotionExecutor& executor, float& inputStrength
    ) {
    const auto accessor = dynamic_cast<const IEmotionExecutorAccessorInputStrength*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->GetInputStrength(inputStrength);
}

std::error_code nva2e::SetExecutorInputStrength_INTERNAL(
    IEmotionExecutor& executor, float inputStrength
    ) {
    A2E_CHECK_ERROR_WITH_MSG(
        !nva2x::HasExecutionStarted(executor),
        "Input strength can only be set before execution is started",
        nva2x::ErrorCode::eExecutionAlreadyStarted
        );
    const auto accessor = dynamic_cast<IEmotionExecutorAccessorInputStrength*>(&executor);
    if (!accessor) { return nva2x::ErrorCode::eUnsupported; }
    return accessor->SetInputStrength(inputStrength);
}

std::error_code nva2e::GetExecutorPostProcessParameters_INTERNAL(
    const IEmotionExecutor& executor, std::size_t trackIndex, PostProcessParams& params
    ) {
    return GetExecutorParameters<IEmotionExecutorAccessorPostProcessParameters>(executor, trackIndex, params);
}

std::error_code nva2e::SetExecutorPostProcessParameters_INTERNAL(
    IEmotionExecutor& executor, std::size_t trackIndex, const PostProcessParams& params
    ) {
    return SetExecutorParameters<IEmotionExecutorAccessorPostProcessParameters>(executor, trackIndex, params);
}


namespace nva2e {

IEmotionBinder::~IEmotionBinder() = default;

void EmotionBinder::Destroy() {
    delete this;
}

std::error_code EmotionBinder::Init(
    IEmotionExecutor& executor,
    nva2x::IEmotionAccumulator* const* emotionAccumulators,
    std::size_t nbEmotionAccumulators
    ) {
    A2E_CHECK_ERROR_WITH_MSG(emotionAccumulators, "Emotion accumulators cannot be null", nva2x::ErrorCode::eNullPointer);
    A2E_CHECK_ERROR_WITH_MSG(
        executor.GetNbTracks() == nbEmotionAccumulators,
        "Executor must have the same number of tracks as the number of emotion accumulators",
        nva2x::ErrorCode::eMismatch
        );

    std::vector<nva2x::IEmotionAccumulator*> localEmotionAccumulators(nbEmotionAccumulators);
    for (std::size_t i = 0; i < nbEmotionAccumulators; ++i) {
        A2E_CHECK_ERROR_WITH_MSG(
            emotionAccumulators[i],
            "Emotion accumulator cannot be null",
            nva2x::ErrorCode::eNullPointer
            );
        localEmotionAccumulators[i] = emotionAccumulators[i];
    }

    A2E_CHECK_RESULT_WITH_MSG(
        executor.SetResultsCallback(callbackForEmotion, this),
        "Failed to set results callback"
        );

    _emotionAccumulators = std::move(localEmotionAccumulators);
    return nva2x::ErrorCode::eSuccess;
}

bool EmotionBinder::callbackForEmotion(void* userdata, const IEmotionExecutor::Results& results) {
    auto* binder = static_cast<EmotionBinder*>(userdata);
    const auto success = binder->_emotionAccumulators[results.trackIndex]->Accumulate(
        results.timeStampCurrentFrame, results.emotions, results.cudaStream
    );
    (void) success;
    assert(!success);
    return true;
}

}


nva2e::IEmotionBinder* nva2e::CreateEmotionBinder_INTERNAL(
    IEmotionExecutor& executor,
    nva2x::IEmotionAccumulator* const* emotionAccumulators,
    std::size_t nbEmotionAccumulators
    ) {
    LOG_DEBUG("CreateEmotionBinder()");
    auto binder = std::make_unique<EmotionBinder>();
    if (nva2x::ErrorCode::eSuccess != binder->Init(executor, emotionAccumulators, nbEmotionAccumulators)) {
        LOG_ERROR("Unable to create emotion binder");
        return nullptr;
    }
    return binder.release();
}
