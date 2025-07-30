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

#include "audio2x/export.h"

#include <cstdint>
#include <system_error>

namespace nva2x {

// Base interface for interactive executors.
//
// It abstracts the execution of some computation in interactive workflows.
class IInteractiveExecutor {
public:
    using invalidation_layer_t = std::size_t;
    static constexpr const invalidation_layer_t kLayerNone = 0;
    static constexpr const invalidation_layer_t kLayerAll = 1;

    // Invalidate part of the state held by the executor.
    virtual std::error_code Invalidate(invalidation_layer_t layer) = 0;
    // Check if a layer is valid.
    virtual bool IsValid(invalidation_layer_t layer) const = 0;

    // Destroy the executor.
    virtual void Destroy() = 0;

    // Based on accumulated audio, check how many frames can be generated.
    virtual std::size_t GetTotalNbFrames() const = 0;
    // Get expected sampling rate for this executor.
    virtual std::size_t GetSamplingRate() const = 0;
    // Get generation frame rate for this executor.
    virtual void GetFrameRate(std::size_t& numerator, std::size_t& denominator) const = 0;

    // Timestamp as a sample index, allowed to be negative.
    // Timestamps can be converted to time by being divided by the sampling rate.
    using timestamp_t = std::int64_t;

    // Return the timestamp associated with the frameIndex-th generated frame.
    virtual timestamp_t GetFrameTimestamp(std::size_t frameIndex) const = 0;

    // Execute the minimum amount of work to produce the requested frame.
    virtual std::error_code ComputeFrame(std::size_t frameIndex) = 0;
    // Execute all the frames.
    virtual std::error_code ComputeAllFrames() = 0;
    // Interrupt the execution of ComputeAllFrames() if possible.
    // This must be called from a different thread than the one in the
    // ComputeAllFrames() call.
    virtual std::error_code Interrupt() = 0;

protected:
    virtual ~IInteractiveExecutor();
};

} // namespace nva2x
