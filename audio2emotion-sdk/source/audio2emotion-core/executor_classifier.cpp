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
#include "audio2emotion/internal/executor_classifier.h"
#include "audio2emotion/internal/macros.h"
#include "audio2emotion/internal/model.h"
#include "audio2x/error.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/internal/nvtx_trace.h"

#include <cassert>
#include <numeric>

namespace nva2e::IClassifierModel {

std::size_t EmotionExecutor::GetNextAudioSampleToRead(std::size_t trackIndex) const {
    A2E_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", std::numeric_limits<std::size_t>::min());
    timestamp_t start, target, end;
    _trackData[trackIndex].progress->GetCurrentWindow(start, target, end);
    if (start < 0) {
        return 0;
    }
    return static_cast<std::size_t>(start);
}

std::error_code EmotionExecutor::Execute(std::size_t* pNbExecutedTracks) {
    NVTX_TRACE("IClassifierModel::EmotionExecutor::Execute");

    A2E_CHECK_RESULT(BaseExecute(pNbExecutedTracks));

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutor::Init(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams
    ) {
    A2E_CHECK_RESULT_WITH_MSG(
        _core.Init(params.nbTracks, params.cudaStream, classifierParams, params.nbTracks),
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

std::error_code EmotionExecutor::RunInference(std::size_t& outNbExecutedTracks) {
    std::size_t nbExecutedTracks = 0;
    for (std::size_t trackIndex = 0; trackIndex < _trackData.size(); ++trackIndex) {
        // OPTME: This does a bit of redundant computation to check if we can actually execute.
        const auto nbAvailableExecutions = GetNbAvailableExecutions(trackIndex);
        if (nbAvailableExecutions == 0) {
            continue;
        }

        auto& trackData = _trackData[trackIndex];
        assert(trackData.audioAccumulator);

        timestamp_t start, target, end;
        trackData.progress->GetCurrentWindow(start, target, end);
        assert(static_cast<std::size_t>(end - start) == _core.GetNetworkInfo().bufferLength);

        // Check that we are not executing a frame beyond the accumulated samples.
        A2E_CHECK_ERROR_WITH_MSG(
            target < static_cast<timestamp_t>(trackData.audioAccumulator->NbAccumulatedSamples()),
            "Trying to execute a frame beyond accumulated samples",
            nva2x::ErrorCode::eInvalidValue
            );

        A2E_CHECK_RESULT_WITH_MSG(
            _core.ReadAudioBuffer(*trackData.audioAccumulator, nbExecutedTracks, start),
            "Unable to read inference window"
        );

        _executedTracks.set(trackIndex);
        ++nbExecutedTracks;
    }

    A2E_CHECK_ERROR_WITH_MSG(nbExecutedTracks > 0, "No tracks to execute", nva2x::ErrorCode::eNoTracksToExecute);

    A2E_CHECK_RESULT_WITH_MSG(_core.BindBuffers(nbExecutedTracks), "Unable to bind buffers");

    {
        NVTX_TRACE("Inference");
        A2E_CHECK_RESULT_WITH_MSG(_core.RunInference(), "Unable to run inference");
    }

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

nva2e::IEmotionExecutor* nva2e::CreateClassifierEmotionExecutor_INTERNAL(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams
    ) {
  A2X_LOG_DEBUG("CreateClassifierEmotionExecutor()");
  auto executor = std::make_unique<IClassifierModel::EmotionExecutor>();
  if (nva2x::ErrorCode::eSuccess != executor->Init(params, classifierParams)) {
    A2X_LOG_ERROR("Unable to create classifier emotion executor");
    return nullptr;
  }
  return executor.release();
}
