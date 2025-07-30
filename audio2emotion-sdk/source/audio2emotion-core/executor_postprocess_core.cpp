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
#include "audio2emotion/internal/executor_postprocess_core.h"
#include "audio2emotion/internal/macros.h"
#include "audio2emotion/internal/postprocess.h"
#include "audio2x/error.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"

#include <cassert>
#include <numeric>

namespace nva2e::IPostProcessModel {

std::error_code EmotionExecutorCore::Init(
    std::size_t nbTracks, cudaStream_t cudaStream,
    const nva2e::IPostProcessModel::EmotionExecutorCreationParameters& params
    ) {
    A2E_CHECK_RESULT(EmotionExecutorCoreBase::Init(nbTracks, cudaStream, params, 1, nbTracks));

    A2E_CHECK_RESULT_WITH_MSG(
        nva2x::FillOnDevice(_inferenceOutputBuffer, 0.0f, _cudaStream),
        "Unable to fill inference output buffer"
    );

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutorCore::GetProgressParameters(
    nva2x::WindowProgressParameters& outProgressParams,
    std::size_t samplingRate,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator
    ) {
    A2E_CHECK_ERROR_WITH_MSG(
        samplingRate > 0,
        "Sampling rate must be greater than 0",
        nva2x::ErrorCode::eInvalidValue
    );
    A2E_CHECK_ERROR_WITH_MSG(
        frameRateNumerator > 0,
        "Frame rate numerator must be greater than 0",
        nva2x::ErrorCode::eInvalidValue
    );
    A2E_CHECK_ERROR_WITH_MSG(
        frameRateDenominator > 0,
        "Frame rate denominator must be greater than 0",
        nva2x::ErrorCode::eInvalidValue
    );

    nva2x::WindowProgressParameters progressParams;
    progressParams.windowSize = 1;
    progressParams.targetOffset = 0;
    progressParams.startOffset = 0;
    progressParams.strideNum = samplingRate * frameRateDenominator;
    progressParams.strideDenom = frameRateNumerator;

    outProgressParams = progressParams;
    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2e::IPostProcessModel
