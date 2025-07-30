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
#include "audio2face/internal/interactive_executor_regression.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"
#include "audio2x/internal/nvtx_trace.h"

namespace nva2f::IRegressionModel {

std::error_code GeometryInteractiveExecutor::ComputeFrame(std::size_t frameIndex) {
    A2F_CHECK_RESULT(CheckInputsState());
    A2F_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_ERROR_WITH_MSG(frameIndex < GetTotalNbFrames(), "Frame index out of range", nva2x::ErrorCode::eOutOfBounds);

    NVTX_TRACE("ComputeFrame");

    _isInterrupted = false;

    const std::size_t endFrameIndex = frameIndex + 1;

    StatelessScope statelessScope(_core);
    A2F_CHECK_RESULT(Compute(frameIndex, endFrameIndex));

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryInteractiveExecutor::ComputeAllFrames() {
    A2F_CHECK_RESULT(CheckInputsState());
    A2F_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);

    NVTX_TRACE("ComputeAllFrames");

    _isInterrupted = false;

    const std::size_t endFrameIndex = GetTotalNbFrames();

    A2F_CHECK_RESULT(Compute(0, endFrameIndex));

    _inferenceResultsValid = true;
    // We don't cache the post-processing results.  But after computing all frames,
    // we mark them as valid.
    _skinResultsValid = true;
    _tongueResultsValid = true;
    _teethResultsValid = true;
    _eyesResultsValid = true;

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryInteractiveExecutor::Init(
    const nva2f::GeometryExecutorCreationParameters& params,
    const nva2f::IRegressionModel::GeometryExecutorCreationParameters& regressionParams,
    std::size_t batchSize
    ) {
    A2F_CHECK_ERROR_WITH_MSG(params.nbTracks == 1, "Number of tracks must be 1", nva2x::ErrorCode::eInvalidValue);
    A2F_CHECK_ERROR_WITH_MSG(params.sharedAudioAccumulators, "Audio accumulators cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_RESULT_WITH_MSG(
        _core.Init(params.nbTracks, params.cudaStream, regressionParams, batchSize),
        "Unable to initialize core"
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

    return nva2x::ErrorCode::eSuccess;
}

GeometryExecutorCoreBase& GeometryInteractiveExecutor::GetCore() {
    return _core;
}

const GeometryExecutorCoreBase& GeometryInteractiveExecutor::GetCore() const {
    return _core;
}

std::error_code GeometryInteractiveExecutor::Compute(std::size_t beginFrameIndex, std::size_t endFrameIndex) {
    assert(!CheckInputsState());

    const auto nbFrames = GetTotalNbFrames();
    assert(beginFrameIndex < nbFrames);
    assert(endFrameIndex <= nbFrames);

    assert(_track.audioAccumulator);
    const auto& audioAccumulator = *_track.audioAccumulator;
    assert(audioAccumulator.IsClosed());
    assert(_track.emotionAccumulator);
    const auto& emotionAccumulator = *_track.emotionAccumulator;
    assert(emotionAccumulator.IsClosed());
    assert(_track.progress);
    auto& progress = *_track.progress;
    progress.ResetReadWindowCount(beginFrameIndex);

    // Allocate the cache.
    // We allocate the whole cache even if we are not using all of it right away.
    A2F_CHECK_RESULT_WITH_MSG(
        _inferenceResults.Init(_core.GetNetworkInfo(), nbFrames),
        "Unable to allocate inference results"
        );
    A2F_CHECK_RESULT_WITH_MSG(
        _resultsBuffers.Init(_core.GetNetworkInfo(), 1),
        "Unable to allocate results buffers"
        );

    std::size_t frameRateNumerator{0}, frameRateDenominator{0};
    _core.GetFrameRate(frameRateNumerator, frameRateDenominator);
    const float dt = static_cast<float>(frameRateDenominator) / frameRateNumerator;
    const bool doSkin = true;
    const bool doTongue = true;
    const bool doJaw = true;
    const bool doEyes = true;

    A2F_CHECK_RESULT_WITH_MSG(_core.Reset(0), "Unable to reset core");
    if (doEyes) {
        // This might not give exactly the same results as adding the dt repeatedly,
        // but it's close enough for our purposes.
        A2F_CHECK_RESULT_WITH_MSG(
            _core.GetEyesAnimator()->SetLiveTime(0, dt * beginFrameIndex),
            "Unable to set eyes live time"
            );
    }

    const auto cudaStream = _core.GetCudaStream();

    // Run inferences and post-processing by batching them.
    const auto batchSize = _core.GetBatchSize();
    for (std::size_t inferenceIndex = beginFrameIndex; inferenceIndex < endFrameIndex; inferenceIndex += batchSize) {
        if (_isInterrupted) {
            return nva2x::ErrorCode::eInterrupted;
        }

        const std::size_t nbFramesToProcess = std::min(endFrameIndex - inferenceIndex, batchSize);
        if (!_inferenceResultsValid) {
            NVTX_TRACE("Inference");

            for (std::size_t frameIndex = 0; frameIndex < nbFramesToProcess; ++frameIndex) {
                timestamp_t start, target, end;
                progress.GetCurrentWindow(start, target, end, frameIndex);
                assert(static_cast<std::size_t>(end - start) == _core.GetNetworkInfo().bufferLength);
                assert(target < static_cast<timestamp_t>(audioAccumulator.NbAccumulatedSamples()));
                A2F_CHECK_RESULT_WITH_MSG(
                    _core.ReadAudioBuffer(audioAccumulator, frameIndex, start),
                    "Unable to read inference window"
                    );

                A2F_CHECK_ERROR_WITH_MSG(
                    !_core.ReadExplicitEmotion(&emotionAccumulator, frameIndex, target),
                    "Unable to read emotions from emotion accumulator, has enough emotion data been provided?",
                    ErrorCode::eEmotionNotAvailable
                    );
            }

            {
                NVTX_TRACE("RunInference");
                A2F_CHECK_RESULT_WITH_MSG(_core.BindBuffers(nbFramesToProcess), "Unable to bind buffers");
                A2F_CHECK_RESULT_WITH_MSG(_core.RunInference(), "Unable to run inference");
            }

            {
                // Copy the inference results to the cache.
                const auto source = _core.GetInferenceOutputBuffers().GetInferenceResult(0, nbFramesToProcess);
                const auto destination = _inferenceResults.GetInferenceResult(
                    inferenceIndex, nbFramesToProcess
                    );
                A2F_CHECK_RESULT_WITH_MSG(
                    nva2x::CopyDeviceToDevice(destination, source, cudaStream),
                    "Unable to copy inference results"
                    );
            }
        }

        static constexpr const auto activeTracks = nullptr;
        static constexpr const auto activeTracksSize = 0;

        const bool canReusePreviousCompute = _inferenceResultsValid
            && (beginFrameIndex == _pcaFrameBeginIndex && endFrameIndex == _pcaFrameEndIndex);
        if (!canReusePreviousCompute)
        {
            if (doSkin) {
                NVTX_TRACE("SkinPcaReconstruction");

                const auto pcaInput = _inferenceResults.GetInferenceResult(inferenceIndex, nbFramesToProcess);
                const nva2x::TensorBatchInfo pcaInputInfo = _inferenceResults.GetSkinBatchInfo();

                const auto pcaOutput = _core.GetResultBuffers().GetResultTensor(nbFramesToProcess);
                const auto pcaOutputInfo = _core.GetResultBuffers().GetSkinBatchInfo();

                auto skinPcaAnimator = _core.GetSkinPcaAnimator();
                assert(skinPcaAnimator);
                A2F_CHECK_RESULT_WITH_MSG(
                    skinPcaAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                    "Unable to set active tracks for skin PCA reconstruction"
                    );
                A2F_CHECK_RESULT_WITH_MSG(
                    skinPcaAnimator->Animate(pcaInput, pcaInputInfo, pcaOutput, pcaOutputInfo, nbFramesToProcess),
                    "Unable to run skin PCA reconstruction"
                    );
            }

            if (doTongue) {
                NVTX_TRACE("TonguePcaReconstruction");

                const auto pcaInput = _inferenceResults.GetInferenceResult(inferenceIndex, nbFramesToProcess);
                const nva2x::TensorBatchInfo pcaInputInfo = _inferenceResults.GetTongueBatchInfo();

                const auto pcaOutput = _core.GetResultBuffers().GetResultTensor(nbFramesToProcess);
                const auto pcaOutputInfo = _core.GetResultBuffers().GetTongueBatchInfo();

                auto tonguePcaAnimator = _core.GetTonguePcaAnimator();
                assert(tonguePcaAnimator);
                A2F_CHECK_RESULT_WITH_MSG(
                    tonguePcaAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                    "Unable to set active tracks for tongue PCA reconstruction"
                    );
                A2F_CHECK_RESULT_WITH_MSG(
                    tonguePcaAnimator->Animate(pcaInput, pcaInputInfo, pcaOutput, pcaOutputInfo, nbFramesToProcess),
                    "Unable to run tongue PCA reconstruction"
                    );
            }

            _pcaFrameBeginIndex = inferenceIndex;
            _pcaFrameEndIndex = inferenceIndex + nbFramesToProcess;
        }

        {
            NVTX_TRACE("PostProcessing");
            for (std::size_t localFrameIndex = 0; localFrameIndex < nbFramesToProcess; ++localFrameIndex) {
                NVTX_TRACE("Frame");

                const auto frameIndex = inferenceIndex + localFrameIndex;

                const bool runSkin = doSkin && (!_skinResultsValid || !canReusePreviousCompute);
                if (runSkin) {
                    NVTX_TRACE("SkinAnimation");

                    // Get the results directly inside the result buffers.
                    const auto skinInput = _core.GetResultBuffers().GetResultSkinGeometry(localFrameIndex);
                    const nva2x::TensorBatchInfo skinInputInfo = {0, skinInput.Size(), skinInput.Size()};

                    const auto skinResult = _resultsBuffers.GetResultSkinGeometry(0);
                    const nva2x::TensorBatchInfo skinResultInfo = {0, skinResult.Size(), skinResult.Size()};

                    auto skinAnimator = _core.GetSkinAnimator();
                    assert(skinAnimator);
                    A2F_CHECK_RESULT_WITH_MSG(
                        skinAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                        "Unable to set active tracks for skin animation"
                        );
                    A2F_CHECK_RESULT_WITH_MSG(
                        skinAnimator->Animate(skinInput, skinInputInfo, skinResult, skinResultInfo),
                        "Unable to run skin animation"
                        );
                }

                const bool runTongue = doTongue && (!_tongueResultsValid || !canReusePreviousCompute);
                if (runTongue) {
                    NVTX_TRACE("TongueAnimation");

                    // Get the results directly inside the result buffers.
                    const auto tongueInput = _core.GetResultBuffers().GetResultTongueGeometry(localFrameIndex);
                    const nva2x::TensorBatchInfo tongueInputInfo = {0, tongueInput.Size(), tongueInput.Size()};

                    const auto tongueResult = _resultsBuffers.GetResultTongueGeometry(0);
                    const nva2x::TensorBatchInfo tongueResultInfo = {0, tongueResult.Size(), tongueResult.Size()};

                    auto tongueAnimator = _core.GetTongueAnimator();
                    assert(tongueAnimator);
                    A2F_CHECK_RESULT_WITH_MSG(
                        tongueAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                        "Unable to set active tracks for tongue animation"
                        );
                    A2F_CHECK_RESULT_WITH_MSG(
                        tongueAnimator->Animate(tongueInput, tongueInputInfo, tongueResult, tongueResultInfo),
                        "Unable to run tongue animation"
                        );
                }

                const bool runJaw = doJaw && (!_teethResultsValid || !canReusePreviousCompute);
                if (runJaw) {
                    NVTX_TRACE("TeethAnimation");

                    // Get the results directly from the inference results.
                    const auto teethInput = _inferenceResults.GetInferenceResult(frameIndex, 1);
                    nva2x::TensorBatchInfo teethInputInfo = _inferenceResults.GetJawBatchInfo();

                    const auto teethResult = _resultsBuffers.GetResultJawTransform(0);
                    const nva2x::TensorBatchInfo teethResultInfo = {0, teethResult.Size(), teethResult.Size()};

                    auto teethAnimator = _core.GetTeethAnimator();
                    assert(teethAnimator);
                    A2F_CHECK_RESULT_WITH_MSG(
                        teethAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                        "Unable to set active tracks for teeth animation"
                        );
                    A2F_CHECK_RESULT_WITH_MSG(
                        teethAnimator->ComputeJawTransform(teethInput, teethInputInfo, teethResult, teethResultInfo),
                        "Unable to compute jaw transform"
                        );
                }

                const bool runEyes = doEyes && (!_eyesResultsValid || !canReusePreviousCompute);
                if (runEyes) {
                    NVTX_TRACE("EyesAnimation");

                    // Get the results directly from the inference results.
                    const auto eyesInput = _inferenceResults.GetInferenceResult(frameIndex, 1);
                    nva2x::TensorBatchInfo eyesInputInfo = _inferenceResults.GetEyesBatchInfo();

                    const auto eyesResult = _resultsBuffers.GetResultEyesRotation(0);
                    const nva2x::TensorBatchInfo eyesResultInfo = {0, eyesResult.Size(), eyesResult.Size()};

                    auto eyesAnimator = _core.GetEyesAnimator();
                    assert(eyesAnimator);
                    A2F_CHECK_RESULT_WITH_MSG(
                        eyesAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                        "Unable to set active tracks for eyes animation"
                        );
                    A2F_CHECK_RESULT_WITH_MSG(
                        eyesAnimator->ComputeEyesRotation(eyesInput, eyesInputInfo, eyesResult, eyesResultInfo),
                        "Unable to compute jaw transform"
                        );
                }

                timestamp_t start, target, end;
                progress.GetCurrentWindow(start, target, end, localFrameIndex);

                timestamp_t startNext, targetNext, endNext;
                progress.GetCurrentWindow(startNext, targetNext, endNext, localFrameIndex + 1);

                Results results;
                results.trackIndex = 0;
                results.timeStampCurrentFrame = target;
                results.timeStampNextFrame = targetNext;
                results.skinGeometry = _resultsBuffers.GetResultSkinGeometry(0);
                results.skinCudaStream = cudaStream;
                results.tongueGeometry = _resultsBuffers.GetResultTongueGeometry(0);
                results.tongueCudaStream = cudaStream;
                results.jawTransform = _resultsBuffers.GetResultJawTransform(0);
                results.jawCudaStream = cudaStream;
                results.eyesRotation = _resultsBuffers.GetResultEyesRotation(0);
                results.eyesCudaStream = cudaStream;

                // Results are ready, call the callback
                assert(_resultsCallback);
                {
                    NVTX_TRACE("ResultsCallback");
                    const bool keepGoing = _resultsCallback(_resultsUserdata, results);
                    if (!keepGoing) {
                        return nva2x::ErrorCode::eInterrupted;
                    }
                }
            }
        }

        progress.IncrementReadWindowCount(nbFramesToProcess);
    }

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f::IRegressionModel

nva2f::IGeometryInteractiveExecutor* nva2f::CreateRegressionGeometryInteractiveExecutor_INTERNAL(
    const nva2f::GeometryExecutorCreationParameters& params,
    const nva2f::IRegressionModel::GeometryExecutorCreationParameters& regressionParams,
    std::size_t batchSize
    ) {
  A2X_LOG_DEBUG("CreateRegressionGeometryInteractiveExecutor()");
  auto executor = std::make_unique<IRegressionModel::GeometryInteractiveExecutor>();
  if (nva2x::ErrorCode::eSuccess != executor->Init(params, regressionParams, batchSize)) {
    A2X_LOG_ERROR("Unable to create regression geometry interactive executor");
    return nullptr;
  }
  return executor.release();
}
