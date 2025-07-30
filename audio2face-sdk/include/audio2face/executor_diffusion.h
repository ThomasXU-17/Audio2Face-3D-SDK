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

#include "audio2face/model_diffusion.h"
#include "audio2face/animator.h"

namespace nva2f {

namespace IDiffusionModel {

// Parameters for creating a geometry executor for diffusion-based face animation.
struct GeometryExecutorCreationParameters {
    // Network information details.
    NetworkInfo networkInfo{0};
    // Pointer to the TensorRT network model data.
    const void* networkData{nullptr};
    // Size of the TensorRT network model data in bytes.
    std::size_t networkDataSize{0};
    // Strength multiplier for input audio.
    float inputStrength{1.0f};

    using SkinParameters = IAnimatorSkin::InitData;
    using TongueParameters = IAnimatorTongue::InitData;
    using TeethParameters = IAnimatorTeeth::InitData;
    using EyesParameters = IAnimatorEyes::InitData;

    // Initialization parameters for skin animation.
    // If null, the skin will not be initialized nor computed.
    const SkinParameters* initializationSkinParams{nullptr};
    // Initialization parameters for tongue animation.
    // If null, the tongue will not be initialized nor computed.
    const TongueParameters* initializationTongueParams{nullptr};
    // Initialization parameters for teeth animation.
    // If null, the teeth will not be initialized nor computed.
    const TeethParameters* initializationTeethParams{nullptr};
    // Initialization parameters for eyes animation.
    // If null, the eyes will not be initialized nor computed.
    const EyesParameters* initializationEyesParams{nullptr};

    // Identity index for multi-identity models.
    std::size_t identityIndex{0};
    // Whether to use constant noise during inference.
    // If true, the noise will be constant during inference.
    // If false, the noise will be different for each inference.
    bool constantNoise{true};
};

} // namespace IDiffusionModel

} // namespace nva2f
