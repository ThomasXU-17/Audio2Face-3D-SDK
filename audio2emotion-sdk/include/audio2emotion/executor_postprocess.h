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

#include "audio2emotion/model.h"
#include "audio2emotion/postprocess.h"
#include "audio2x/emotion_accumulator.h"

namespace nva2e {

namespace IPostProcessModel {

// Parameters for creating an emotion executor with post-processing capabilities.
// This executor does not run inference, it only runs post-processing.
struct EmotionExecutorCreationParameters {
    // Audio input sampling rate in Hz.
    std::size_t samplingRate;
    // Input strength parameter for consistency with other executors.
    // Not actually used by the post-process model but can be set and queried.
    float inputStrength;

    // This model supports any FPS.
    // Numerator of the frame rate fraction.
    std::size_t frameRateNumerator{0};
    // Denominator of the frame rate fraction.
    std::size_t frameRateDenominator{0};

    // Required data for post-processing.
    PostProcessData postProcessData;
    // Parameters controlling post-processing behavior.
    PostProcessParams postProcessParams;
    // Array of preferred emotion accumulators to sample from.
    // The number of accumulators is given by the number of tracks in
    // nva2e::EmotionExecutorCreationParameters::nbTracks.
    const nva2x::IEmotionAccumulator* const* sharedPreferredEmotionAccumulators{nullptr};
};

} // namespace IPostProcessModel

} // namespace nva2e
