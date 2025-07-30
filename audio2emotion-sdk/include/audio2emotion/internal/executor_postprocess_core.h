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
#pragma once

#include "audio2emotion/executor_postprocess.h"
#include "audio2emotion/internal/executor_core.h"

namespace nva2e {

namespace IPostProcessModel {

class EmotionExecutorCore : public EmotionExecutorCoreBase {
public:
    std::error_code Init(
        std::size_t nbTracks, cudaStream_t cudaStream,
        const nva2e::IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
        );

    static std::error_code GetProgressParameters(
        nva2x::WindowProgressParameters& outProgressParams,
        std::size_t samplingRate,
        std::size_t frameRateNumerator,
        std::size_t frameRateDenominator
        );
};

} // namespace IPostProcessModel

} // namespace nva2e
