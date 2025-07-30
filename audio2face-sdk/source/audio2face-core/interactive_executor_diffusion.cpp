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
#include "audio2face/internal/interactive_executor_diffusion.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"
#include "audio2x/internal/nvtx_trace.h"

namespace {

    // This class saves and restores the state of a window progress object.
    class ProgressScope {
    public:
        ProgressScope(nva2x::WindowProgress& progress) : _progress(progress), _copy(progress) {
        }

        ~ProgressScope() {
            _progress = _copy;
        }

    private:
        nva2x::WindowProgress& _progress;
        nva2x::WindowProgress _copy;
    };

}

namespace nva2f::IDiffusionModel {

std::error_code GeometryInteractiveExecutor::ComputeFrame(std::size_t frameIndex) {
    A2F_CHECK_RESULT(CheckInputsState());
    A2F_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_ERROR_WITH_MSG(frameIndex < GetTotalNbFrames(), "Frame index out of range", nva2x::ErrorCode::eOutOfBounds);

    NVTX_TRACE("ComputeFrame");

    _isInterrupted = false;

    assert(_track.audioAccumulator);
    const auto& audioAccumulator = *_track.audioAccumulator;
    assert(audioAccumulator.IsClosed());
    assert(_track.emotionAccumulator);
    const auto& emotionAccumulator = *_track.emotionAccumulator;
    assert(emotionAccumulator.IsClosed());
    assert(_track.progress);
    auto& progress = *_track.progress;

    const std::size_t targetInference = (frameIndex + _nbFramesBeforeAudio) / _nbFramesPerInference;
    const std::size_t targetFrameIndex = (frameIndex + _nbFramesBeforeAudio) % _nbFramesPerInference;
    const std::size_t endInference = targetInference + 1;
    std::size_t startInference;

    if (_inferenceResultsValid) {
        // We might be able to re-use the previous inference results
        if (progress.GetReadWindowCount() == targetInference + 1) {
            // We already have the right inference results loaded in the buffer.
            // Do nothing.
            startInference = endInference;
        }
        else {
            // We need to run a single inference to get the results in the buffer.
            // Note: we could cache it instead, but this will happen only once when scrubbing.
            // Caching would take a lot of memory, but could also speed up recomputing all frames
            // when inference results can be reused.
            startInference = targetInference;

            // Restore the GRU state.
            const auto destination = _core.GetInferenceStateBuffers().GetInputGRUStateTensor();
            const auto source = _inferenceResults.View(startInference * destination.Size(), destination.Size());
            A2F_CHECK_RESULT_WITH_MSG(
                nva2x::CopyDeviceToDevice(destination, source, _core.GetCudaStream()),
                "Unable to copy GRU state"
                );
        }
    }
    else {
        if (_nbInferencesForPreview == 0) {
            // We want to always start from the beginning to have exact results.
            startInference = 0;
        } else {
            // We want to go a back a bit, but not necessarily up to the the start.
            startInference = endInference - std::min(endInference, _nbInferencesForPreview);
        }
        A2F_CHECK_RESULT_WITH_MSG(
            _core.GetInferenceStateBuffers().Reset(0),
            "Unable to reset GRU state"
            );
    }

    if (startInference < endInference) {
        _currentFrameIndex = kInvalidIndex;

        progress.ResetReadWindowCount(startInference);
        if (_core.GetNoiseGenerator()) {
            NVTX_TRACE("ResetNoise");
            A2F_CHECK_RESULT_WITH_MSG(
                _core.GetNoiseGenerator()->Reset(0, startInference),
                "Unable to reset noise"
                );
        }
        for (std::size_t inferenceIndex = startInference; inferenceIndex < endInference; ++inferenceIndex) {
            if (_isInterrupted) {
                return nva2x::ErrorCode::eInterrupted;
            }

            {
                NVTX_TRACE("Inference");

                timestamp_t start, target, end;
                progress.GetCurrentWindow(start, target, end);
                assert(static_cast<std::size_t>(end - start) == _core.GetInferenceInputBuffers().GetInput(0).Size());

                A2F_CHECK_RESULT_WITH_MSG(
                    _core.ReadAudioBuffer(audioAccumulator, 0, start),
                    "Unable to read inference window"
                );

                // Read emotions for the target frames.
                const auto frameProgress = GetSingleFrameProgress();
                for (std::size_t frameIndex = 0; frameIndex < _nbFramesPerInference; ++frameIndex) {
                    frameProgress.GetCurrentWindow(start, target, end, frameIndex);
                    assert(static_cast<std::size_t>(end - start) == _core.GetInferenceInputBuffers().GetInput(0).Size());
                    A2F_CHECK_ERROR_WITH_MSG(
                        !_core.ReadEmotion(&emotionAccumulator, 0, target, frameIndex),
                        "Unable to read emotions from emotion accumulator, has enough emotion data been provided?",
                        ErrorCode::eEmotionNotAvailable
                        );
                }

                // Generate noise for the track, if needed.
                auto noiseGenerator = _core.GetNoiseGenerator();
                if (noiseGenerator) {
                    NVTX_TRACE("GenerateNoise");
                    A2F_CHECK_RESULT_WITH_MSG(
                        noiseGenerator->Generate(0, _core.GetInferenceInputBuffers().GetNoise(0)),
                        "Unable to generate noise"
                        );
                }

                {
                    NVTX_TRACE("RunInference");
                    A2F_CHECK_RESULT_WITH_MSG(_core.BindBuffers(), "Unable to bind buffers");
                    A2F_CHECK_RESULT_WITH_MSG(_core.RunInference(), "Unable to run inference");
                }

                A2F_CHECK_RESULT_WITH_MSG(_core.GetInferenceStateBuffers().Swap(), "Unable to swap state buffers");
            }

            progress.IncrementReadWindowCount();
        }
    }

    // Set the eye state (live time), if needed.
    std::size_t frameRateNumerator{0}, frameRateDenominator{0};
    _core.GetFrameRate(frameRateNumerator, frameRateDenominator);
    const float dt = static_cast<float>(frameRateDenominator) / frameRateNumerator;
    // This might not give exactly the same results as adding the dt repeatedly,
    // but it's close enough for our purposes.
    A2F_CHECK_RESULT_WITH_MSG(
        _core.GetEyesAnimator()->SetLiveTime(0, dt * frameIndex),
        "Unable to set eyes live time"
        );

    // The callbacks expect the progress not to have been incremented yet.
    ProgressScope progressScope(progress);
    assert(progress.GetReadWindowCount() == targetInference + 1);
    progress.ResetReadWindowCount(progress.GetReadWindowCount() - 1);

    StatelessScope statelessScope(_core);
    const bool canReusePreviousCompute = _currentFrameIndex == frameIndex;
    A2F_CHECK_RESULT(ComputePostProcessing(targetFrameIndex, targetFrameIndex + 1, canReusePreviousCompute));

    _currentFrameIndex = frameIndex;

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryInteractiveExecutor::ComputeAllFrames() {
    A2F_CHECK_RESULT(CheckInputsState());
    A2F_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);

    NVTX_TRACE("ComputeAllFrames");

    _isInterrupted = false;

    _currentFrameIndex = kInvalidIndex;

    assert(_track.audioAccumulator);
    const auto& audioAccumulator = *_track.audioAccumulator;
    assert(audioAccumulator.IsClosed());
    assert(_track.emotionAccumulator);
    const auto& emotionAccumulator = *_track.emotionAccumulator;
    assert(emotionAccumulator.IsClosed());
    assert(_track.progress);
    auto& progress = *_track.progress;
    progress.ResetReadWindowCount(0);

    const auto nbAccumulatedSamples = _track.audioAccumulator->NbAccumulatedSamples();
    const auto nbTotalInferences = progress.GetNbAvailableWindows(nbAccumulatedSamples, true);

    // Allocate the cache.
    const auto gruSize = _core.GetInferenceStateBuffers().GetInputGRUStateTensor().Size();
    A2F_CHECK_RESULT_WITH_MSG(
        _inferenceResults.Allocate(nbTotalInferences * gruSize),
        "Unable to allocate GRU state"
        );

    A2F_CHECK_RESULT_WITH_MSG(_core.Reset(0), "Unable to reset core");

    for (std::size_t inferenceIndex = 0; inferenceIndex < nbTotalInferences; ++inferenceIndex) {
        if (_isInterrupted) {
            return nva2x::ErrorCode::eInterrupted;
        }

        {
            NVTX_TRACE("Inference");

            timestamp_t start, target, end;
            progress.GetCurrentWindow(start, target, end);
            assert(static_cast<std::size_t>(end - start) == _core.GetInferenceInputBuffers().GetInput(0).Size());

            A2F_CHECK_RESULT_WITH_MSG(
                _core.ReadAudioBuffer(audioAccumulator, 0, start),
                "Unable to read inference window"
            );

            // Read emotions for the target frames.
            const auto frameProgress = GetSingleFrameProgress();
            for (std::size_t frameIndex = 0; frameIndex < _nbFramesPerInference; ++frameIndex) {
                frameProgress.GetCurrentWindow(start, target, end, frameIndex);
                assert(static_cast<std::size_t>(end - start) == _core.GetInferenceInputBuffers().GetInput(0).Size());
                A2F_CHECK_ERROR_WITH_MSG(
                    !_core.ReadEmotion(&emotionAccumulator, 0, target, frameIndex),
                    "Unable to read emotions from emotion accumulator, has enough emotion data been provided?",
                    ErrorCode::eEmotionNotAvailable
                    );
            }

            // Generate noise for the track, if needed.
            auto noiseGenerator = _core.GetNoiseGenerator();
            if (noiseGenerator) {
                NVTX_TRACE("GenerateNoise");
                A2F_CHECK_RESULT_WITH_MSG(
                    noiseGenerator->Generate(0, _core.GetInferenceInputBuffers().GetNoise(0)),
                    "Unable to generate noise"
                    );
            }

            {
                NVTX_TRACE("RunInference");
                A2F_CHECK_RESULT_WITH_MSG(_core.BindBuffers(), "Unable to bind buffers");
                A2F_CHECK_RESULT_WITH_MSG(_core.RunInference(), "Unable to run inference");
            }

            {
                // Copy the GRU state to the cache.
                const auto source = _core.GetInferenceStateBuffers().GetInputGRUStateTensor();
                const auto destination = _inferenceResults.View(inferenceIndex * source.Size(), source.Size());
                A2F_CHECK_RESULT_WITH_MSG(
                    nva2x::CopyDeviceToDevice(destination, source, _core.GetCudaStream()),
                    "Unable to copy GRU state"
                    );
            }

            A2F_CHECK_RESULT_WITH_MSG(_core.GetInferenceStateBuffers().Swap(), "Unable to swap state buffers");
        }

        {
            A2F_CHECK_RESULT_WITH_MSG(
                ComputePostProcessing(0, _nbFramesPerInference, false),
                "Unable to compute post-processing"
                );
        }

        progress.IncrementReadWindowCount();
    }

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
    const nva2f::IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams,
    std::size_t nbInferencesForPreview
    ) {
    A2F_CHECK_ERROR_WITH_MSG(params.nbTracks == 1, "Number of tracks must be 1", nva2x::ErrorCode::eInvalidValue);
    A2F_CHECK_ERROR_WITH_MSG(params.sharedAudioAccumulators, "Audio accumulators cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_RESULT_WITH_MSG(
        _core.Init(params.nbTracks, params.cudaStream, diffusionParams, 1),
        "Unable to initialize core"
        );

    nva2x::WindowProgressParameters progressParams;
    A2F_CHECK_RESULT_WITH_MSG(
        GeometryExecutorCore::GetProgressParameters(
            progressParams,
            diffusionParams.networkInfo
            ),
        "Unable to get progress parameters"
        );

    const auto nbFramesPerInference = diffusionParams.networkInfo.numFramesCenter;
    A2F_CHECK_RESULT(
        BaseInit(params, diffusionParams.networkInfo.emotionLength, progressParams, nbFramesPerInference)
        );
    assert(_nbFramesBeforeAudio > 0);

    _nbInferencesForPreview = nbInferencesForPreview;

    return nva2x::ErrorCode::eSuccess;
}

GeometryExecutorCoreBase& GeometryInteractiveExecutor::GetCore() {
    return _core;
}

const GeometryExecutorCoreBase& GeometryInteractiveExecutor::GetCore() const {
    return _core;
}

std::error_code GeometryInteractiveExecutor::ComputePostProcessing(
    std::size_t beginFrameLocalIndex, std::size_t endFrameLocalIndex, bool canReusePreviousCompute
    ) {
    std::size_t frameRateNumerator{0}, frameRateDenominator{0};
    _core.GetFrameRate(frameRateNumerator, frameRateDenominator);
    const float dt = static_cast<float>(frameRateDenominator) / frameRateNumerator;
    const bool doSkin = true;
    const bool doTongue = true;
    const bool doJaw = true;
    const bool doEyes = true;

    static constexpr const auto activeTracks = nullptr;
    static constexpr const auto activeTracksSize = 0;

    const auto frameProgress = GetSingleFrameProgress();
    for (std::size_t localFrameIndex = beginFrameLocalIndex; localFrameIndex < endFrameLocalIndex; ++localFrameIndex) {
        timestamp_t start, target, end;
        frameProgress.GetCurrentWindow(start, target, end, localFrameIndex);
        assert(static_cast<std::size_t>(end - start) == _core.GetInferenceInputBuffers().GetInput(0).Size());

        if (target < 0) {
            // Don't report the frames before the audio, and skip processing for them.
            continue;
        }

        if (target >= static_cast<timestamp_t>(_track.audioAccumulator->NbAccumulatedSamples())) {
            assert(_track.audioAccumulator->IsClosed());
            // Don't report the frames after the audio, and stop processing.
            break;
        }

        {
            NVTX_TRACE("PostProcessing");

            const bool runSkin = doSkin && (!_skinResultsValid || !canReusePreviousCompute);
            if (runSkin) {
                NVTX_TRACE("SkinAnimation");

                // Get the results directly from the inference results.
                const auto skinInput = _core.GetInferenceOutputBuffers().GetInferenceResultSkin(localFrameIndex);
                const nva2x::TensorBatchInfo skinInputInfo = {0, skinInput.Size(), skinInput.Size()};

                const auto skinOutput = _core.GetResultBuffers().GetResultSkinGeometry(0);
                const nva2x::TensorBatchInfo skinOutputInfo = {0, skinOutput.Size(), skinOutput.Size()};

                auto skinAnimator = _core.GetSkinAnimator();
                assert(skinAnimator);
                A2F_CHECK_RESULT_WITH_MSG(
                    skinAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                    "Unable to set active tracks for skin animation"
                    );
                A2F_CHECK_RESULT_WITH_MSG(
                    skinAnimator->Animate(skinInput, skinInputInfo, skinOutput, skinOutputInfo),
                    "Unable to run skin animation"
                    );
            }

            const bool runTongue = doTongue && (!_tongueResultsValid || !canReusePreviousCompute);
            if (runTongue) {
                NVTX_TRACE("TongueAnimation");

                // Get the results directly from the inference results.
                const auto tongueInput = _core.GetInferenceOutputBuffers().GetInferenceResultTongue(localFrameIndex);
                const nva2x::TensorBatchInfo tongueInputInfo = {0, tongueInput.Size(), tongueInput.Size()};

                const auto tongueOutput = _core.GetResultBuffers().GetResultTongueGeometry(0);
                const nva2x::TensorBatchInfo tongueOutputInfo = {0, tongueOutput.Size(), tongueOutput.Size()};

                auto tongueAnimator = _core.GetTongueAnimator();
                assert(tongueAnimator);
                A2F_CHECK_RESULT_WITH_MSG(
                    tongueAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                    "Unable to set active tracks for tongue animation"
                    );
                A2F_CHECK_RESULT_WITH_MSG(
                    tongueAnimator->Animate(tongueInput, tongueInputInfo, tongueOutput, tongueOutputInfo),
                    "Unable to run tongue animation"
                    );
            }

            const bool runJaw = doJaw && (!_teethResultsValid || !canReusePreviousCompute);
            if (runJaw) {
                NVTX_TRACE("TeethAnimation");

                // Get the results directly from the inference results.
                const auto teethInput = _core.GetInferenceOutputBuffers().GetInferenceResultJaw(localFrameIndex);
                const nva2x::TensorBatchInfo teethInputInfo = {0, teethInput.Size(), teethInput.Size()};

                const auto teethOutput = _core.GetResultBuffers().GetResultJawTransform(0);
                const nva2x::TensorBatchInfo teethOutputInfo = {0, teethOutput.Size(), teethOutput.Size()};

                auto teethAnimator = _core.GetTeethAnimator();
                assert(teethAnimator);
                A2F_CHECK_RESULT_WITH_MSG(
                    teethAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                    "Unable to set active tracks for teeth animation"
                    );
                A2F_CHECK_RESULT_WITH_MSG(
                    teethAnimator->ComputeJawTransform(teethInput, teethInputInfo, teethOutput, teethOutputInfo),
                    "Unable to compute jaw transform"
                    );
            }

            const bool runEyes = doEyes && (!_eyesResultsValid || !canReusePreviousCompute);
            if (runEyes) {
                NVTX_TRACE("EyesAnimation");

                // Get the results directly from the inference results.
                const auto eyesInput = _core.GetInferenceOutputBuffers().GetInferenceResultEyes(localFrameIndex);
                const nva2x::TensorBatchInfo eyesInputInfo = {0, eyesInput.Size(), eyesInput.Size()};

                const auto eyesOutput = _core.GetResultBuffers().GetResultEyesRotation(0);
                const nva2x::TensorBatchInfo eyesOutputInfo = {0, eyesOutput.Size(), eyesOutput.Size()};

                auto eyesAnimator = _core.GetEyesAnimator();
                assert(eyesAnimator);
                A2F_CHECK_RESULT_WITH_MSG(
                    eyesAnimator->SetActiveTracks(activeTracks, activeTracksSize),
                    "Unable to set active tracks for eyes animation"
                    );
                A2F_CHECK_RESULT_WITH_MSG(
                    eyesAnimator->ComputeEyesRotation(eyesInput, eyesInputInfo, eyesOutput, eyesOutputInfo),
                    "Unable to compute eyes rotation"
                );
            }
        }

        timestamp_t startNext, targetNext, endNext;
        frameProgress.GetCurrentWindow(startNext, targetNext, endNext, localFrameIndex + 1);

        Results results;
        results.trackIndex = 0;
        results.timeStampCurrentFrame = target;
        results.timeStampNextFrame = targetNext;
        results.skinGeometry = _core.GetResultBuffers().GetResultSkinGeometry(0);
        results.skinCudaStream = _core.GetCudaStream();
        results.tongueGeometry = _core.GetResultBuffers().GetResultTongueGeometry(0);
        results.tongueCudaStream = _core.GetCudaStream();
        results.jawTransform = _core.GetResultBuffers().GetResultJawTransform(0);
        results.jawCudaStream = _core.GetCudaStream();
        results.eyesRotation = _core.GetResultBuffers().GetResultEyesRotation(0);
        results.eyesCudaStream = _core.GetCudaStream();

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

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f::IDiffusionModel

nva2f::IGeometryInteractiveExecutor* nva2f::CreateDiffusionGeometryInteractiveExecutor_INTERNAL(
    const nva2f::GeometryExecutorCreationParameters& params,
    const nva2f::IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams,
    std::size_t nbInferencesForPreview
    ) {
  A2X_LOG_DEBUG("CreateDiffusionGeometryInteractiveExecutor()");
  auto executor = std::make_unique<IDiffusionModel::GeometryInteractiveExecutor>();
  if (nva2x::ErrorCode::eSuccess != executor->Init(params, diffusionParams, nbInferencesForPreview)) {
    A2X_LOG_ERROR("Unable to create diffusion geometry interactive executor");
    return nullptr;
  }
  return executor.release();
}
