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

#include "audio2emotion/executor.h"
#include "audio2x/interactive_executor.h"

namespace nva2e {

// Interface for interactive executors which provide emotions.
class IEmotionInteractiveExecutor : public nva2x::IInteractiveExecutor {
public:
    // Layer identifier for operations affecting inference.
    static constexpr const invalidation_layer_t kLayerInference = 2;
    // Layer identifier for operations affecting post-processing.
    static constexpr const invalidation_layer_t kLayerPostProcessing = 3;
    // Layer identifier for audio accumulator, which affects inference.
    static constexpr const invalidation_layer_t kLayerAudioAccumulator = kLayerInference;
    // Layer identifier for preferred emotion accumulator, which affects post-processing.
    static constexpr const invalidation_layer_t kLayerPreferredEmotionAccumulator = kLayerPostProcessing;

    // Results structure for interactive emotion executor.
    // It is the same as the one for the non-interactive executor.
    using Results = IEmotionExecutor::Results;
    // Callback function type for results.
    // It is the same as the one for the non-interactive executor.
    using results_callback_t = IEmotionExecutor::results_callback_t;

    // Set the callback to use to collect results.
    virtual std::error_code SetResultsCallback(results_callback_t callback, void* userdata) = 0;

    // Get the size of the emotions output.
    virtual std::size_t GetEmotionsSize() const = 0;
};

} // namespace nva2e
