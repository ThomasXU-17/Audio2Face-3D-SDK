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
#include "audio2face/internal/executor_diffusion.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2face/error.h"
#include "audio2x/error.h"
#include "audio2x/internal/io.h"
#include "audio2x/internal/nvtx_trace.h"

#include <cassert>

#include <cuda_runtime_api.h>

namespace nva2f::IDiffusionModel {

// For IGeometryExecutor::ExecutionOption operators.
using namespace ::nva2f::internal;

std::error_code GeometryExecutor::Reset(std::size_t trackIndex) {
    A2F_CHECK_RESULT(BaseReset(trackIndex));

    A2F_CHECK_RESULT_WITH_MSG(_core.Reset(trackIndex), "Unable to reset core");

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutor::Execute(std::size_t* pNbExecutedTracks) {
    NVTX_TRACE("IDiffusionModel::GeometryExecutor::Execute");

    if (pNbExecutedTracks) {
        *pNbExecutedTracks = 0;
    }

    A2F_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);

    const auto cudaStream = _core.GetCudaStream();
    auto& inferenceInputBuffers = _core.GetInferenceInputBuffers();
    auto& inferenceStateBuffers = _core.GetInferenceStateBuffers();
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

        // Read emotions for the target frames.
        const auto frameProgress = GetSingleFrameProgress(trackIndex);
        for (std::size_t frameIndex = 0; frameIndex < _nbFramesPerInference; ++frameIndex) {
            frameProgress.GetCurrentWindow(start, target, end, frameIndex);
            assert(static_cast<std::size_t>(end - start) == inferenceInputBuffers.GetInput(trackIndex).Size());
            A2F_CHECK_ERROR_WITH_MSG(
                !_core.ReadEmotion(trackData.emotionAccumulator, trackIndex, target, frameIndex),
                "Unable to read emotions from emotion accumulator, has enough emotion data been provided?",
                ErrorCode::eEmotionNotAvailable
                );
        }

        // Generate noise for the track, if needed.
        auto noiseGenerator = _core.GetNoiseGenerator();
        if (noiseGenerator) {
            NVTX_TRACE("GenerateNoise");
            A2F_CHECK_RESULT_WITH_MSG(
                noiseGenerator->Generate(trackIndex, inferenceInputBuffers.GetNoise(trackIndex)),
                "Unable to generate noise"
                );
        }

        _executedTracks.set(trackIndex);
        ++nbExecutedTracks;
    }

    A2F_CHECK_ERROR_WITH_MSG(nbExecutedTracks > 0, "No tracks to execute", nva2x::ErrorCode::eNoTracksToExecute);

    A2F_CHECK_RESULT_WITH_MSG(_core.BindBuffers(), "Unable to bind buffers");

    {
        NVTX_TRACE("Inference");
        A2F_CHECK_RESULT_WITH_MSG(_core.RunInference(), "Unable to run inference");
    }
    A2F_CHECK_RESULT_WITH_MSG(inferenceStateBuffers.Swap(), "Unable to swap state buffers");
    if (nbExecutedTracks != _trackData.size()) {
        // Not all track were executed, make sure we use the old state data.
        // This is optimized to suppose that most tracks get executed, so not much data
        // is moved around when tracks are not executed.
        // Note that we could optimize the small case as well by not swapping and just copying
        // the output to the input for executed tracks when there are less than half of the tracks
        // being executed.
        for (std::size_t trackIndex = 0; trackIndex < _trackData.size(); ++trackIndex) {
            const bool executed = _executedTracks[trackIndex];
            if (!executed) {
                A2F_CHECK_RESULT_WITH_MSG(
                    inferenceStateBuffers.CopyOutputToInputGRUState(cudaStream, trackIndex),
                    "Unable to restore GRU state"
                    );
            }
        }
    }

    std::size_t frameRateNumerator{0};
    std::size_t frameRateDenominator{0};
    _core.GetFrameRate(frameRateNumerator, frameRateDenominator);
    const float dt = static_cast<float>(frameRateDenominator) / frameRateNumerator;
    const auto executionOption = _core.GetExecutionOption();
    const bool doSkin = IsAnySet(executionOption, ExecutionOption::Skin);
    const bool doTongue = IsAnySet(executionOption, ExecutionOption::Tongue);
    const bool doJaw = IsAnySet(executionOption, ExecutionOption::Jaw);
    const bool doEyes = IsAnySet(executionOption, ExecutionOption::Eyes);

    // Frames are processed in order: for each frame, we process all tracks, to return frames
    // for each track as soon as possible.
    _doneTracks.reset_all();
    for (std::size_t frameIndex = 0; frameIndex < _nbFramesPerInference; ++frameIndex) {
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

            if (target < 0) {
                // Don't report the frames before the audio, and skip processing for them.
                continue;
            }

            auto& trackData = _trackData[trackIndex];

            if (target >= static_cast<timestamp_t>(trackData.audioAccumulator->NbAccumulatedSamples())) {
                assert(trackData.audioAccumulator->IsClosed());
                // Don't report the frames after the audio, and stop processing.
                _doneTracks.set(trackIndex);
                continue;
            }

            atLeastOneTrack = true;
            _postProcessTracks.set(trackIndex);
        }

        // Do the batch post-processing.
        if (atLeastOneTrack) {
            const auto activeTracks = _postProcessTracks.block_data();
            const auto activeTracksSize = _postProcessTracks.block_size();

            if (doSkin) {
                NVTX_TRACE("SkinAnimation");
                auto skinAnimator = _core.GetSkinAnimator();
                assert(skinAnimator);
                const auto skinInput = inferenceOutputBuffers.GetResultTensor();
                const auto skinInputInfo = inferenceOutputBuffers.GetSkinBatchInfo(frameIndex);
                const auto skinOutput = resultBuffers.GetResultTensor();
                const auto skinOutputInfo = resultBuffers.GetSkinBatchInfo();

                A2F_CHECK_RESULT_WITH_MSG(
                    skinAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                    "Unable to set active tracks for skin animation"
                );
                A2F_CHECK_RESULT_WITH_MSG(
                    skinAnimator->Animate(skinInput, skinInputInfo, skinOutput, skinOutputInfo),
                    "Unable to run skin animation"
                );
            }

            if (doTongue) {
                NVTX_TRACE("TongueAnimation");
                auto tongueAnimator = _core.GetTongueAnimator();
                assert(tongueAnimator);
                const auto tongueInput = inferenceOutputBuffers.GetResultTensor();
                const auto tongueInputInfo = inferenceOutputBuffers.GetTongueBatchInfo(frameIndex);
                const auto tongueOutput = resultBuffers.GetResultTensor();
                const auto tongueOutputInfo = resultBuffers.GetTongueBatchInfo();

                A2F_CHECK_RESULT_WITH_MSG(
                    tongueAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                    "Unable to set active tracks for tongue animation"
                );
                A2F_CHECK_RESULT_WITH_MSG(
                    tongueAnimator->Animate(tongueInput, tongueInputInfo, tongueOutput, tongueOutputInfo),
                    "Unable to run tongue animation"
                );
            }

            if (doJaw) {
                NVTX_TRACE("TeethAnimation");
                auto teethAnimator = _core.GetTeethAnimator();
                assert(teethAnimator);
                const auto jawInput = inferenceOutputBuffers.GetResultTensor();
                const auto jawInputInfo = inferenceOutputBuffers.GetJawBatchInfo(frameIndex);
                const auto jawOutput = resultBuffers.GetResultTensor();
                const auto jawOutputInfo = resultBuffers.GetJawBatchInfo();

                A2F_CHECK_RESULT_WITH_MSG(
                    teethAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                    "Unable to set active tracks for teeth animation"
                );
                A2F_CHECK_RESULT_WITH_MSG(
                    teethAnimator->ComputeJawTransform(jawInput, jawInputInfo, jawOutput, jawOutputInfo),
                    "Unable to run teeth animation"
                );
            }

            if (doEyes) {
                NVTX_TRACE("EyesAnimation");
                auto eyesAnimator = _core.GetEyesAnimator();
                assert(eyesAnimator);
                const auto eyesInput = inferenceOutputBuffers.GetResultTensor();
                const auto eyesInputInfo = inferenceOutputBuffers.GetEyesBatchInfo(frameIndex);
                const auto eyesOutput = resultBuffers.GetResultTensor();
                const auto eyesOutputInfo = resultBuffers.GetEyesBatchInfo();

                A2F_CHECK_RESULT_WITH_MSG(
                    eyesAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                    "Unable to set active tracks for eyes animation"
                );
                A2F_CHECK_RESULT_WITH_MSG(
                    eyesAnimator->ComputeEyesRotation(eyesInput, eyesInputInfo, eyesOutput, eyesOutputInfo),
                    "Unable to run eyes animation"
                );
            }
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

            assert(results.timeStampCurrentFrame >= 0);

            if (doSkin) {
                results.skinCudaStream = cudaStream;
                results.skinGeometry = resultBuffers.GetResultSkinGeometry(trackIndex);
            }

            if (doTongue) {
                results.tongueCudaStream = cudaStream;
                results.tongueGeometry = resultBuffers.GetResultTongueGeometry(trackIndex);
            }

            if (doJaw) {
                results.jawCudaStream = cudaStream;
                results.jawTransform = resultBuffers.GetResultJawTransform(trackIndex);
            }

            if (doEyes) {
                results.eyesCudaStream = cudaStream;
                results.eyesRotation = resultBuffers.GetResultEyesRotation(trackIndex);
            }

            // Results are ready, call the callbacks
            if (_emotionsCallback) {
                Emotions emotions;
                emotions.trackIndex = trackIndex;
                emotions.timeStampCurrentFrame = results.timeStampCurrentFrame;
                emotions.timeStampNextFrame = results.timeStampNextFrame;
                emotions.cudaStream = cudaStream;
                emotions.emotions = inferenceInputBuffers.GetEmotions(frameIndex, trackIndex);

                {
                    NVTX_TRACE("EmotionsCallback");
                    _emotionsCallback(_emotionsUserdata, emotions);
                }
            }

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


std::error_code GeometryExecutor::Init(
    const nva2f::GeometryExecutorCreationParameters& params,
    const nva2f::IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams
    ) {
    // Initialize the core first, since it does a bunch of validation.
    A2F_CHECK_RESULT(
        _core.Init(params.nbTracks, params.cudaStream, diffusionParams, params.nbTracks)
        );

    nva2x::WindowProgressParameters progressParams;
    A2F_CHECK_RESULT_WITH_MSG(
        GeometryExecutorCore::GetProgressParameters(progressParams, diffusionParams.networkInfo),
        "Unable to get progress parameters"
        );

    const auto nbFramesPerInference = diffusionParams.networkInfo.numFramesCenter;
    A2F_CHECK_RESULT(
        BaseInit(params, diffusionParams.networkInfo.emotionLength, progressParams, nbFramesPerInference)
        );
    assert(_nbFramesBeforeAudio > 0);

    _postProcessTracks.resize(_executedTracks.size());
    _doneTracks.resize(_executedTracks.size());

    return nva2x::ErrorCode::eSuccess;
}

GeometryExecutorCore& GeometryExecutor::GetCore() {
    return _core;
}

const GeometryExecutorCore& GeometryExecutor::GetCore() const {
    return _core;
}

} // namespace nva2f::IDiffusionModel

nva2f::IGeometryExecutor* nva2f::CreateDiffusionGeometryExecutor_INTERNAL(
    const nva2f::GeometryExecutorCreationParameters& params,
    const nva2f::IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams
    ) {
  LOG_DEBUG("CreateDiffusionGeometryExecutor()");
  auto executor = std::make_unique<nva2f::IDiffusionModel::GeometryExecutor>();
  if (executor->Init(params, diffusionParams)) {
    LOG_ERROR("Unable to create diffusion geometry executor");
    return nullptr;
  }
  return executor.release();
}
