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
#include "audio2emotion/internal/interactive_executor_postprocess.h"
#include "audio2emotion/internal/executor_classifier_core.h"
#include "audio2emotion/internal/macros.h"
#include "audio2emotion/internal/model.h"
#include "audio2x/error.h"
#include "audio2x/internal/nvtx_trace.h"

#include <cassert>

namespace nva2e::IPostProcessModel {

std::error_code EmotionInteractiveExecutor::GetInferencesToSkip(std::size_t& inferencesToSkip) const {
    inferencesToSkip = 0;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionInteractiveExecutor::SetInferencesToSkip(std::size_t) {
    // Always invalidate the inference layer, event if it's meaningless.
    A2E_CHECK_RESULT_WITH_MSG(Invalidate(kLayerInference), "Unable to invalidate inference layer for inferences to skip");

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionInteractiveExecutor::Init(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    std::size_t batchSize
    ) {
    const auto postProcessParams = IClassifierModel::EmotionExecutorCore::GetPostProcessParameters(classifierParams);
    return Init(params, postProcessParams);
}

std::error_code EmotionInteractiveExecutor::Init(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
    ) {
    A2E_CHECK_ERROR_WITH_MSG(params.nbTracks == 1, "Number of tracks must be 1", nva2x::ErrorCode::eInvalidValue);
    A2E_CHECK_RESULT_WITH_MSG(
        _core.Init(params.nbTracks, params.cudaStream, postProcessParams),
        "Unable to initialize core"
        );

    nva2x::WindowProgressParameters progressParams;
    A2E_CHECK_RESULT_WITH_MSG(
        EmotionExecutorCore::GetProgressParameters(
            progressParams,
            postProcessParams.samplingRate,
            postProcessParams.frameRateNumerator,
            postProcessParams.frameRateDenominator
            ),
        "Unable to get progress parameters"
        );

    A2E_CHECK_RESULT(
        BaseInit(
            params,
            _core.GetPostProcessor().GetOutputEmotionsSize(),
            progressParams,
            postProcessParams.sharedPreferredEmotionAccumulators
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
    const std::size_t emotionSize = _core.GetPostProcessor().GetInputEmotionsSize();

    // Allocate the cache.
    A2E_CHECK_RESULT_WITH_MSG(
        _inferenceResults.Allocate(nbInferences * emotionSize),
        "Unable to allocate inference results"
        );

    // Simply zero the results.
    A2E_CHECK_RESULT_WITH_MSG(
        nva2x::FillOnDevice(_inferenceResults, 0.0f, _core.GetCudaStream()),
        "Unable to zero inference results"
        );

    _inferenceResultsValid = true;

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2e::IPostProcessModel

nva2e::IEmotionInteractiveExecutor* nva2e::CreatePostProcessEmotionInteractiveExecutor_INTERNAL(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    std::size_t batchSize
    ) {
  A2X_LOG_DEBUG("CreatePostProcessEmotionInteractiveExecutor()");
  auto executor = std::make_unique<IPostProcessModel::EmotionInteractiveExecutor>();
  if (nva2x::ErrorCode::eSuccess != executor->Init(params, classifierParams, batchSize)) {
    A2X_LOG_ERROR("Unable to create post-process emotion interactive executor");
    return nullptr;
  }
  return executor.release();
}

nva2e::IEmotionInteractiveExecutor* nva2e::CreatePostProcessEmotionInteractiveExecutor_INTERNAL(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
    ) {
  A2X_LOG_DEBUG("CreatePostProcessEmotionInteractiveExecutor()");
  auto executor = std::make_unique<IPostProcessModel::EmotionInteractiveExecutor>();
  if (nva2x::ErrorCode::eSuccess != executor->Init(params, postProcessParams)) {
    A2X_LOG_ERROR("Unable to create post-process emotion interactive executor");
    return nullptr;
  }
  return executor.release();
}
