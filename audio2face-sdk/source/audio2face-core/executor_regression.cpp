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
#include "audio2face/internal/executor_regression.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2face/error.h"
#include "audio2x/error.h"
#include "audio2x/internal/nvtx_trace.h"

#include <cassert>

#include <cuda_runtime_api.h>

namespace nva2f::IRegressionModel {

// For IGeometryExecutor::ExecutionOption operators.
using namespace ::nva2f::internal;

std::error_code GeometryExecutor::Reset(std::size_t trackIndex) {
    A2F_CHECK_RESULT(BaseReset(trackIndex));

    A2F_CHECK_RESULT_WITH_MSG(_core.Reset(trackIndex), "Unable to reset core");

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutor::Execute(std::size_t* pNbExecutedTracks) {
    NVTX_TRACE("IRegressionModel::GeometryExecutor::Execute");

    if (pNbExecutedTracks) {
        *pNbExecutedTracks = 0;
    }

    A2F_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);

    const auto cudaStream = _core.GetCudaStream();
    auto& inferenceInputBuffers = _core.GetInferenceInputBuffers();
    auto& inferenceOutputBuffers = _core.GetInferenceOutputBuffers();
    auto& resultBuffers = _core.GetResultBuffers();

    // Accumulate the inference input to be able to run the inference.
    // Data is not tightly packed to avoid moving data around if indices are skipped.
    _executedTracks.reset_all();
    std::size_t nbExecutedTracks = 0;
    for (std::size_t trackIndex = 0; trackIndex < _trackData.size(); ++trackIndex) {
        // OPTME: This does a bit of redundant computation to check if we can actually execute.
        const auto nbAvailableExecutions = GetNbAvailableExecutions(trackIndex);
        if (nbAvailableExecutions == 0) {
            continue;
        }

        auto& trackData = _trackData[trackIndex];
        assert(trackData.audioAccumulator);
        assert(trackData.emotionAccumulator);
        assert(trackData.progress);

        timestamp_t start, target, end;
        trackData.progress->GetCurrentWindow(start, target, end);
        assert(static_cast<std::size_t>(end - start) == inferenceInputBuffers.GetInput(trackIndex).Size());

        // Check that we are not executing a frame beyond the accumulated samples.
        A2F_CHECK_ERROR_WITH_MSG(
            target < static_cast<timestamp_t>(trackData.audioAccumulator->NbAccumulatedSamples()),
            "Trying to execute a frame beyond accumulated samples",
            nva2x::ErrorCode::eInvalidValue
            );

        A2F_CHECK_RESULT_WITH_MSG(
            _core.ReadAudioBuffer(*trackData.audioAccumulator, trackIndex, start),
            "Unable to read inference window"
        );

        A2F_CHECK_ERROR_WITH_MSG(
            !_core.ReadExplicitEmotion(trackData.emotionAccumulator, trackIndex, target),
            "Unable to read emotions from emotion accumulator, has enough emotion data been provided?",
            ErrorCode::eEmotionNotAvailable
        );

        _executedTracks.set(trackIndex);
        ++nbExecutedTracks;
    }

    A2F_CHECK_ERROR_WITH_MSG(nbExecutedTracks > 0, "No tracks to execute", nva2x::ErrorCode::eNoTracksToExecute);

    {
        NVTX_TRACE("Inference");
        A2F_CHECK_RESULT_WITH_MSG(_core.RunInference(), "Unable to run inference");
    }

    std::size_t frameRateNumerator{0}, frameRateDenominator{0};
    _core.GetFrameRate(frameRateNumerator, frameRateDenominator);
    const float dt = static_cast<float>(frameRateDenominator) / frameRateNumerator;
    const auto executionOption = _core.GetExecutionOption();
    const bool doSkin = IsAnySet(executionOption, ExecutionOption::Skin);
    const bool doTongue = IsAnySet(executionOption, ExecutionOption::Tongue);
    const bool doJaw = IsAnySet(executionOption, ExecutionOption::Jaw);
    const bool doEyes = IsAnySet(executionOption, ExecutionOption::Eyes);

    // Do the batch post-processing.
    const bool allTracks = (nbExecutedTracks == _trackData.size());
    const auto activeTracks = allTracks ? nullptr : _executedTracks.block_data();
    const auto activeTracksSize = allTracks ? 0 : _executedTracks.block_size();
    const auto inferenceOutput = inferenceOutputBuffers.GetResultTensor();
    const auto result = resultBuffers.GetResultTensor();
    if (doSkin) {
        NVTX_TRACE("SkinAnimation");
        const auto skinInferenceOutputInfo = inferenceOutputBuffers.GetSkinBatchInfo();
        const auto skinResultInfo = resultBuffers.GetSkinBatchInfo();

        auto skinPcaAnimator = _core.GetSkinPcaAnimator();
        assert(skinPcaAnimator);
        A2F_CHECK_RESULT_WITH_MSG(
            skinPcaAnimator->SetActiveTracks(activeTracks, activeTracksSize),
            "Unable to set active tracks for skin PCA reconstruction"
        );
        A2F_CHECK_RESULT_WITH_MSG(
            skinPcaAnimator->Animate(inferenceOutput, skinInferenceOutputInfo, result, skinResultInfo),
            "Unable to run skin PCA reconstruction"
            );

        auto skinAnimator = _core.GetSkinAnimator();
        assert(skinAnimator);
        A2F_CHECK_RESULT_WITH_MSG(
            skinAnimator->SetActiveTracks(activeTracks, activeTracksSize),
            "Unable to set active tracks for skin animation"
        );
        A2F_CHECK_RESULT_WITH_MSG(
            skinAnimator->Animate(result, skinResultInfo, result, skinResultInfo),
            "Unable to run skin animation"
        );
    }

    if (doTongue) {
        NVTX_TRACE("TongueAnimation");
        const auto tongueInferenceOutputInfo = inferenceOutputBuffers.GetTongueBatchInfo();
        const auto tongueResultInfo = resultBuffers.GetTongueBatchInfo();

        auto tonguePcaAnimator = _core.GetTonguePcaAnimator();
        assert(tonguePcaAnimator);
        A2F_CHECK_RESULT_WITH_MSG(
            tonguePcaAnimator->SetActiveTracks(activeTracks, activeTracksSize),
            "Unable to set active tracks for tongue PCA reconstruction"
        );
        A2F_CHECK_RESULT_WITH_MSG(
            tonguePcaAnimator->Animate(inferenceOutput, tongueInferenceOutputInfo, result, tongueResultInfo),
            "Unable to run tongue PCA reconstruction"
            );

        auto tongueAnimator = _core.GetTongueAnimator();
        assert(tongueAnimator);
        A2F_CHECK_RESULT_WITH_MSG(
            tongueAnimator->SetActiveTracks(activeTracks, activeTracksSize),
            "Unable to set active tracks for tongue animation"
        );
        A2F_CHECK_RESULT_WITH_MSG(
            tongueAnimator->Animate(result, tongueResultInfo, result, tongueResultInfo),
            "Unable to run tongue animation"
        );
    }

    if (doJaw) {
        NVTX_TRACE("TeethAnimation");
        const auto jawInferenceOutputInfo = inferenceOutputBuffers.GetJawBatchInfo();
        const auto jawResultInfo = resultBuffers.GetJawBatchInfo();

        auto teethAnimator = _core.GetTeethAnimator();
        assert(teethAnimator);
        A2F_CHECK_RESULT_WITH_MSG(
            teethAnimator->SetActiveTracks(activeTracks, activeTracksSize),
            "Unable to set active tracks for teeth animation"
        );
        A2F_CHECK_RESULT_WITH_MSG(
            teethAnimator->ComputeJawTransform(inferenceOutput, jawInferenceOutputInfo, result, jawResultInfo),
            "Unable to run teeth animation"
        );
    }

    if (doEyes) {
        NVTX_TRACE("EyesAnimation");
        const auto eyesInferenceOutputInfo = inferenceOutputBuffers.GetEyesBatchInfo();
        const auto eyesResultInfo = resultBuffers.GetEyesBatchInfo();

        auto eyesAnimator = _core.GetEyesAnimator();
        assert(eyesAnimator);
        A2F_CHECK_RESULT_WITH_MSG(
            eyesAnimator->SetActiveTracks(activeTracks, activeTracksSize),
            "Unable to set active tracks for eyes animation"
        );
        A2F_CHECK_RESULT_WITH_MSG(
            eyesAnimator->ComputeEyesRotation(inferenceOutput, eyesInferenceOutputInfo, result, eyesResultInfo),
            "Unable to run eyes animation"
        );
    }

    // Then, do the per-track results reporting.
    for (std::size_t trackIndex = 0; trackIndex < _trackData.size(); ++trackIndex) {
        const bool executed = _executedTracks[trackIndex];
        if (!executed) {
            continue;
        }

        auto& trackData = _trackData[trackIndex];

        timestamp_t start, target, end;
        trackData.progress->GetCurrentWindow(start, target, end);

        timestamp_t startNext, targetNext, endNext;
        trackData.progress->GetCurrentWindow(startNext, targetNext, endNext, 1);

        Results results;
        results.trackIndex = trackIndex;
        results.timeStampCurrentFrame = target;
        results.timeStampNextFrame = targetNext;

        if (doSkin) {
            results.skinGeometry = resultBuffers.GetResultSkinGeometry(trackIndex);
            results.skinCudaStream = cudaStream;
        }

        if (doTongue) {
            results.tongueGeometry = resultBuffers.GetResultTongueGeometry(trackIndex);
            results.tongueCudaStream = cudaStream;
        }

        if (doJaw) {
            results.jawTransform = resultBuffers.GetResultJawTransform(trackIndex);
            results.jawCudaStream = cudaStream;
        }

        if (doEyes) {
            results.eyesRotation = resultBuffers.GetResultEyesRotation(trackIndex);
            results.eyesCudaStream = cudaStream;
        }

        // Results are ready, call the callbacks
        if (_emotionsCallback) {
            Emotions emotions;
            emotions.trackIndex = trackIndex;
            emotions.timeStampCurrentFrame = target;
            emotions.timeStampNextFrame = targetNext;
            emotions.cudaStream = cudaStream;
            emotions.emotions = inferenceInputBuffers.GetExplicitEmotions(trackIndex);

            {
                NVTX_TRACE("EmotionsCallback");
                _emotionsCallback(_emotionsUserdata, emotions);
            }
        }

        // Only one frame is computed, so ignore the callback return value.
        {
            NVTX_TRACE("ResultsCallback");
            _resultsCallback(_resultsUserdata, results);
        }

        trackData.progress->IncrementReadWindowCount();
    }

    if (pNbExecutedTracks) {
        *pNbExecutedTracks = nbExecutedTracks;
    }

    return nva2x::ErrorCode::eSuccess;
}


std::error_code GeometryExecutor::Init(
    const nva2f::GeometryExecutorCreationParameters& params,
    const nva2f::IRegressionModel::GeometryExecutorCreationParameters& regressionParams
    ) {
    // Initialize the core first, since it does a bunch of validation.
    A2F_CHECK_RESULT(
        _core.Init(params.nbTracks, params.cudaStream, regressionParams, params.nbTracks)
        );

    nva2x::WindowProgressParameters progressParams;
    A2F_CHECK_RESULT_WITH_MSG(
        GeometryExecutorCore::GetProgressParameters(
            progressParams,
            regressionParams.networkInfo,
            regressionParams.frameRateNumerator,
            regressionParams.frameRateDenominator
        ),
        "Unable to get progress parameters"
    );

    const auto nbFramesPerInference = 1;
    A2F_CHECK_RESULT(
        BaseInit(params, regressionParams.networkInfo.explicitEmotionLength, progressParams, nbFramesPerInference)
        );
    assert(_nbFramesBeforeAudio == 0);

    A2F_CHECK_ERROR_WITH_MSG(regressionParams.emotionDatabase, "Emotion database cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_ERROR_WITH_MSG(regressionParams.sourceShot, "Source shot cannot be null", nva2x::ErrorCode::eNullPointer);
    for (std::size_t trackIndex = 0; trackIndex < params.nbTracks; ++trackIndex) {
        A2F_CHECK_RESULT_WITH_MSG(
            regressionParams.emotionDatabase->GetEmotion(
                regressionParams.sourceShot,
                regressionParams.sourceFrame,
                _core.GetInferenceInputBuffers().GetImplicitEmotions(trackIndex)
                ),
            "Unable to get implicit emotion"
            );
    }

    return nva2x::ErrorCode::eSuccess;
}

GeometryExecutorCore& GeometryExecutor::GetCore() {
    return _core;
}

const GeometryExecutorCore& GeometryExecutor::GetCore() const {
    return _core;
}

} // namespace nva2f::IRegressionModel

nva2f::IGeometryExecutor* nva2f::CreateRegressionGeometryExecutor_INTERNAL(
    const nva2f::GeometryExecutorCreationParameters& params,
    const nva2f::IRegressionModel::GeometryExecutorCreationParameters& regressionParams
    ) {
  LOG_DEBUG("CreateRegressionGeometryExecutor()");
  auto executor = std::make_unique<nva2f::IRegressionModel::GeometryExecutor>();
  if (executor->Init(params, regressionParams)) {
    LOG_ERROR("Unable to create regression geometry executor");
    return nullptr;
  }
  return executor.release();
}
