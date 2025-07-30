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

#include "audio2face/executor.h"
#include "audio2x/interactive_executor.h"

namespace nva2f {

using IFaceInteractiveExecutor = nva2x::IInteractiveExecutor;

// Interface for interactive executors which provide geometry.
class IGeometryInteractiveExecutor : public IFaceInteractiveExecutor {
public:
    // Layer identifier for operations affecting inference.
    static constexpr const invalidation_layer_t kLayerInference = 2;
    // Layer identifier for operations affecting skin output.
    static constexpr const invalidation_layer_t kLayerSkin = 3;
    // Layer identifier for operations affecting tongue output.
    static constexpr const invalidation_layer_t kLayerTongue = 4;
    // Layer identifier for operations affecting teeth output.
    static constexpr const invalidation_layer_t kLayerTeeth = 5;
    // Layer identifier for operations affecting eyes output.
    static constexpr const invalidation_layer_t kLayerEyes = 6;
    // Layer identifier for audio accumulator, which affects inference.
    static constexpr const invalidation_layer_t kLayerAudioAccumulator = kLayerInference;
    // Layer identifier for emotion accumulator, which affects inference.
    static constexpr const invalidation_layer_t kLayerEmotionAccumulator = kLayerInference;

    // Results structure for interactive emotion executor.
    // It is the same as the one for the non-interactive executor.
    using Results = IGeometryExecutor::Results;
    // Callback function type for results.
    // It is the same as the one for the non-interactive executor.
    using results_callback_t = IGeometryExecutor::results_callback_t;

    // Set the callback to use to collect results.
    virtual std::error_code SetResultsCallback(results_callback_t callback, void* userdata) = 0;

    // Get the size of the skin geometry output.
    virtual std::size_t GetSkinGeometrySize() const = 0;
    // Get the size of the tongue geometry output.
    virtual std::size_t GetTongueGeometrySize() const = 0;
    // Get the size of the jaw transform output.
    virtual std::size_t GetJawTransformSize() const = 0;
    // Get the size of the eyes rotation output.
    virtual std::size_t GetEyesRotationSize() const = 0;
};

// Interface for interactive executors which provide blendshape weights.
class IBlendshapeInteractiveExecutor : public IFaceInteractiveExecutor {
public:
    // Layer identifier for operations affecting skin solver preparation.
    static constexpr const invalidation_layer_t kLayerSkinSolverPrepare = 101;
    // Layer identifier for operations affecting tongue solver preparation.
    static constexpr const invalidation_layer_t kLayerTongueSolverPrepare = 102;
    // Layer identifier for operations affecting blendshape weights output.
    static constexpr const invalidation_layer_t kLayerBlendshapeWeights = 103;

    // Get the number of weights for output.
    virtual std::size_t GetWeightCount() const = 0;

    // Type of results returned by the executor.
    // It is the same as the one for the non-interactive executor.
    using ResultsType = IBlendshapeExecutor::ResultsType;

    // Return whether this executor returns GPU or CPU results.
    virtual ResultsType GetResultType() const = 0;

    // Set the callback to use to collect results.
    using HostResults = IBlendshapeExecutor::HostResults;
    using host_results_callback_t = IBlendshapeExecutor::host_results_callback_t;
    virtual std::error_code SetResultsCallback(host_results_callback_t callback, void* userdata) = 0;

    // Set the callback to use to collect results.
    using DeviceResults = IBlendshapeExecutor::DeviceResults;
    using device_results_callback_t = IBlendshapeExecutor::device_results_callback_t;
    virtual std::error_code SetResultsCallback(device_results_callback_t callback, void* userdata) = 0;
};

} // namespace nva2f
