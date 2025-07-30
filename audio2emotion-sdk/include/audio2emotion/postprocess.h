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

#include "audio2x/tensor.h"

namespace nva2e {

// Data structure containing post-processing configuration and input data.
// It holds the data needed to perform post-processing and cannot be tweaked
// after initialization like parameters.
struct PostProcessData {
    // Length of the inference emotion output.
    std::size_t inferenceEmotionLength{0};
    // Length of the post-processed emotion output.
    std::size_t outputEmotionLength{0};
    // Mapping array for emotion correspondence.
    // It is used to map the inference emotion output to the post-processed emotion output.
    const int* emotionCorrespondence{nullptr};
    // Size of the emotion correspondence array.
    // It is the length of the emotion correspondence array.
    std::size_t emotionCorrespondenceSize{0};
};

// Parameters controlling post-processing behavior and emotion blending.
struct PostProcessParams {
    // Contrast factor applied to emotion values.
    float emotionContrast{1.0f};
    // Maximum number of emotions to keep.
    std::size_t maxEmotions{0};
    // Initial emotion state.
    nva2x::HostTensorFloatConstView beginningEmotion{};
    // Preferred emotion state for blending.
    nva2x::HostTensorFloatConstView preferredEmotion{};
    // Coefficient for live emotion blending.
    float liveBlendCoef{0.7f};
    // Flag to enable preferred emotion blending.
    bool enablePreferredEmotion{false};
    // Strength of the preferred emotion influence.
    float preferredEmotionStrength{0.5f};
    // Time duration for live emotion transitions.
    float liveTransitionTime{0.5f};
    // Fixed time delta for processing and time-based blending.
    float fixedDt{0.033f};
    // Overall strength multiplier for emotions.
    float emotionStrength{0.6f};
};

// Interface for post-processing emotion inference output.
class IPostProcessor {
public:
    // Initialize the post-processor with data and parameters.
    virtual std::error_code Init(const PostProcessData& data, const PostProcessParams& params) = 0;

    // Get the current post-processing data configuration.
    virtual const PostProcessData& GetData() const = 0;

    // Set the post-processing parameters.
    virtual std::error_code SetParameters(const PostProcessParams& params) = 0;
    // Get the current post-processing parameters.
    virtual const PostProcessParams& GetParameters() const = 0;

    // Reset the post-processor to initial state.
    virtual std::error_code Reset() = 0;
    // Process input emotions and generate output emotions.
    virtual std::error_code PostProcess(nva2x::HostTensorFloatView outputEmotions,
                                nva2x::HostTensorFloatConstView inputEmotions) = 0;

    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~IPostProcessor();
};

} // namespace nva2e
