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
#include "audio2emotion/internal/interactive_executor_classifier.h"
#include "audio2emotion/internal/macros.h"
#include "audio2emotion/internal/model.h"
#include "audio2x/error.h"
#include "audio2x/internal/nvtx_trace.h"

#include <cassert>

namespace nva2e::IClassifierModel {

std::error_code EmotionInteractiveExecutor::GetInferencesToSkip(std::size_t& inferencesToSkip) const {
    inferencesToSkip = _core.GetInferencesToSkip();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionInteractiveExecutor::SetInferencesToSkip(std::size_t inferencesToSkip) {
    if (_core.GetInferencesToSkip() != inferencesToSkip) {
        A2E_CHECK_RESULT_WITH_MSG(Invalidate(kLayerInference), "Unable to invalidate inference layer for inferences to skip");
    }

    // Update the progress parameters.
    nva2x::WindowProgressParameters progressParams;
    std::size_t frameRateNumerator, frameRateDenominator;
    _core.GetFrameRate(frameRateNumerator, frameRateDenominator);
    A2E_CHECK_RESULT_WITH_MSG(
        EmotionExecutorCore::GetProgressParameters(
            progressParams,
            _core.GetNetworkInfo(),
            inferencesToSkip,
            frameRateNumerator,
            frameRateDenominator
            ),
        "Unable to get progress parameters"
        );

    _core.SetInferencesToSkip(inferencesToSkip);
    *_progress = progressParams;

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionInteractiveExecutor::Init(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    std::size_t batchSize
    ) {
    A2E_CHECK_ERROR_WITH_MSG(params.nbTracks == 1, "Number of tracks must be 1", nva2x::ErrorCode::eInvalidValue);
    A2E_CHECK_RESULT_WITH_MSG(
        _core.Init(params.nbTracks, params.cudaStream, classifierParams, batchSize),
        "Unable to initialize core"
        );

    nva2x::WindowProgressParameters progressParams;
    A2E_CHECK_RESULT_WITH_MSG(
        EmotionExecutorCore::GetProgressParameters(
            progressParams,
            classifierParams.networkInfo,
            classifierParams.inferencesToSkip,
            classifierParams.frameRateNumerator,
            classifierParams.frameRateDenominator
            ),
        "Unable to get progress parameters"
        );

    A2E_CHECK_RESULT(
        BaseInit(
            params,
            _core.GetPostProcessor().GetOutputEmotionsSize(),
            progressParams,
            classifierParams.sharedPreferredEmotionAccumulators
            )
        );

    return nva2x::ErrorCode::eSuccess;
}

EmotionExecutorCoreBase& EmotionInteractiveExecutor::GetCore() {
    return _core;
}

const EmotionExecutorCoreBase& EmotionInteractiveExecutor::GetCore() const {
    return _core;
}

std::error_code EmotionInteractiveExecutor::ComputeInference() {
    assert(!CheckInputsState());
    if (_inferenceResultsValid) {
        return nva2x::ErrorCode::eSuccess;
    }

    NVTX_TRACE("ComputeInferences");

    assert(_audioAccumulator);
    assert(_audioAccumulator->IsClosed());
    _progress->ResetReadWindowCount();
    const std::size_t nbInferences = _progress->GetNbAvailableWindows(
        _audioAccumulator->NbAccumulatedSamples(), true
        );
    const std::size_t emotionSize = _core.GetNetworkInfo().emotionLength;

    // Allocate the cache.
    A2E_CHECK_RESULT_WITH_MSG(
        _inferenceResults.Allocate(nbInferences * emotionSize),
        "Unable to allocate inference results"
        );

    // Run inferences by batching them.
    const auto batchSize = _core.GetBatchSize();
    for (std::size_t inferenceIndex = 0; inferenceIndex < nbInferences; inferenceIndex += batchSize) {
        if (_isInterrupted) {
            return nva2x::ErrorCode::eInterrupted;
        }

        NVTX_TRACE("Inference");
        const std::size_t nbInferencesToProcess = std::min(nbInferences - inferenceIndex, batchSize);

        for (std::size_t frameIndex = 0; frameIndex < nbInferencesToProcess; ++frameIndex) {
            timestamp_t start, target, end;
            _progress->GetCurrentWindow(start, target, end, frameIndex);
            assert(static_cast<std::size_t>(end - start) == _core.GetNetworkInfo().bufferLength);
            assert(target < static_cast<timestamp_t>(_audioAccumulator->NbAccumulatedSamples()));
            A2E_CHECK_RESULT_WITH_MSG(
                _core.ReadAudioBuffer(*_audioAccumulator, frameIndex, start),
                "Unable to read inference window"
                );
        }

        {
            NVTX_TRACE("RunInference");
            A2E_CHECK_RESULT_WITH_MSG(_core.BindBuffers(nbInferencesToProcess), "Unable to bind buffers");
            A2E_CHECK_RESULT_WITH_MSG(_core.RunInference(), "Unable to run inference");
        }

        {
            // Copy the inference results to the cache.
            const auto source = _core.GetInferenceOutputBuffer(0, nbInferencesToProcess);
            const auto destination = _inferenceResults.View(
                inferenceIndex * emotionSize, nbInferencesToProcess * emotionSize
                );
            A2E_CHECK_RESULT_WITH_MSG(
                nva2x::CopyDeviceToDevice(destination, source, _core.GetCudaStream()),
                "Unable to copy inference results"
                );
        }

        _progress->IncrementReadWindowCount(nbInferencesToProcess);
    }

    _inferenceResultsValid = true;

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2e::IClassiferModel

nva2e::IEmotionInteractiveExecutor* nva2e::CreateClassifierEmotionInteractiveExecutor_INTERNAL(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    std::size_t batchSize
    ) {
  A2X_LOG_DEBUG("CreateClassifierEmotionInteractiveExecutor()");
  auto executor = std::make_unique<IClassifierModel::EmotionInteractiveExecutor>();
  if (nva2x::ErrorCode::eSuccess != executor->Init(params, classifierParams, batchSize)) {
    A2X_LOG_ERROR("Unable to create classifier emotion interactive executor");
    return nullptr;
  }
  return executor.release();
}
