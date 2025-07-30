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
#include "audio2emotion/internal/executor_postprocess.h"
#include "audio2emotion/internal/executor_classifier_core.h"
#include "audio2emotion/internal/macros.h"
#include "audio2x/error.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/internal/nvtx_trace.h"

#include <cassert>
#include <numeric>

namespace nva2e::IPostProcessModel {

std::size_t EmotionExecutor::GetNextAudioSampleToRead(std::size_t trackIndex) const {
    A2E_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", std::numeric_limits<std::size_t>::min());

    // This executor does not actually read audio, so it can drop everything.
    return _trackData[trackIndex].audioAccumulator->NbAccumulatedSamples();
}

std::error_code EmotionExecutor::Execute(std::size_t* pNbExecutedTracks) {
    NVTX_TRACE("IPostProcessModel::EmotionExecutor::Execute");

    A2E_CHECK_RESULT(BaseExecute(pNbExecutedTracks));

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutor::Init(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams
    ) {
    const auto postProcessParams = IClassifierModel::EmotionExecutorCore::GetPostProcessParameters(classifierParams);
    return Init(params, postProcessParams);
}

std::error_code EmotionExecutor::Init(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
    ) {
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

std::error_code EmotionExecutor::RunInference(std::size_t& outNbExecutedTracks) {
    // We don't actually run inference, but we still need to identify the tracks that can be executed.
    std::size_t nbExecutedTracks = 0;
    for (std::size_t trackIndex = 0; trackIndex < _trackData.size(); ++trackIndex) {
        // OPTME: This does a bit of redundant computation to check if we can actually execute.
        const auto nbAvailableExecutions = GetNbAvailableExecutions(trackIndex);
        if (nbAvailableExecutions == 0) {
            continue;
        }

        _executedTracks.set(trackIndex);
        ++nbExecutedTracks;
    }

    A2E_CHECK_ERROR_WITH_MSG(nbExecutedTracks > 0, "No tracks to execute", nva2x::ErrorCode::eNoTracksToExecute);

    outNbExecutedTracks = nbExecutedTracks;

    return nva2x::ErrorCode::eSuccess;
}

EmotionExecutorCore& EmotionExecutor::GetCore() {
    return _core;
}

const EmotionExecutorCore& EmotionExecutor::GetCore() const {
    return _core;
}

} // namespace nva2e::IClassiferModel

nva2e::IEmotionExecutor* nva2e::CreatePostProcessEmotionExecutor_INTERNAL(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams
    ) {
  A2X_LOG_DEBUG("CreatePostProcessEmotionExecutor()");
  auto executor = std::make_unique<IPostProcessModel::EmotionExecutor>();
  if (nva2x::ErrorCode::eSuccess != executor->Init(params, classifierParams)) {
    A2X_LOG_ERROR("Unable to create post-process emotion executor");
    return nullptr;
  }
  return executor.release();
}

nva2e::IEmotionExecutor* nva2e::CreatePostProcessEmotionExecutor_INTERNAL(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
    ) {
  A2X_LOG_DEBUG("CreatePostProcessEmotionExecutor()");
  auto executor = std::make_unique<IPostProcessModel::EmotionExecutor>();
  if (nva2x::ErrorCode::eSuccess != executor->Init(params, postProcessParams)) {
    A2X_LOG_ERROR("Unable to create post-process emotion executor");
    return nullptr;
  }
  return executor.release();
}
