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

namespace IClassifierModel {

// Parameters for creating an emotion classifier executor
// Contains network configuration, input settings, and post-processing options
struct EmotionExecutorCreationParameters {
    // Network information details.
    NetworkInfo networkInfo{0};
    // Pointer to the TensorRT network model data.
    const void* networkData{nullptr};
    // Size of the TensorRT network model data in bytes.
    std::size_t networkDataSize{0};
    // Strength multiplier for input audio.
    float inputStrength{1.0f};

    // This model supports any FPS.
    // Numerator of the frame rate fraction.
    std::size_t frameRateNumerator{0};
    // Denominator of the frame rate fraction.
    std::size_t frameRateDenominator{0};

    // Number of inference to skip between actual inference runs.
    std::size_t inferencesToSkip{0};
    // Required data for post-processing.
    PostProcessData postProcessData;
    // Parameters controlling post-processing behavior.
    PostProcessParams postProcessParams;
    // Array of preferred emotion accumulators to sample from.
    // The number of accumulators is given by the number of tracks in
    // nva2e::EmotionExecutorCreationParameters::nbTracks.
    const nva2x::IEmotionAccumulator* const* sharedPreferredEmotionAccumulators{nullptr};
};

} // namespace IClassifierModel

} // namespace nva2e
