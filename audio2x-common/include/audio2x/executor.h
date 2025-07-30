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

// Base interface for executors.
//
// It abstracts the execution of some computation.
class IExecutor {
public:

    // Number of tracks supported by this executor.
    virtual std::size_t GetNbTracks() const = 0;

    // Reset the state of an executor so it can start processing new data.
    virtual std::error_code Reset(std::size_t trackIndex) = 0;
    // Destroy the executor.
    virtual void Destroy() = 0;

    //
    // "Producer" side is provided by the audio accumulator part.
    // It is safe to push audio while execution is being run.
    //
    // "Consumer" side is provided by this interface.
    // It is safe to call this stuff while audio is being pushed.
    //

    // Return whether execution has started or not.
    virtual bool HasExecutionStarted(std::size_t trackIndex) const = 0;
    // Based on accumulated audio, check how many executions (inferences) can
    // be run.  This is a minimum number, as more might become available while
    // the call is being made, for example more audio is added or the audio is
    // closed.
    virtual std::size_t GetNbAvailableExecutions(std::size_t trackIndex) const = 0;
    // Based on accumulated audio, check how many frames can be generated.
    // Return 0 if the audio accumulator is not closed yet.
    virtual std::size_t GetTotalNbFrames(std::size_t trackIndex) const = 0;
    // Get expected sampling rate for this executor.
    virtual std::size_t GetSamplingRate() const = 0;
    // Get generation frame rate for this executor.
    virtual void GetFrameRate(std::size_t& numerator, std::size_t& denominator) const = 0;

    // Timestamp as a sample index, allowed to be negative.
    // Timestamps can be converted to time by being divided by the sampling rate.
    using timestamp_t = std::int64_t;

    // Return the timestamp associated with the frameIndex-th generated frame.
    virtual timestamp_t GetFrameTimestamp(std::size_t frameIndex) const = 0;

    // Run the actual inference.
    virtual std::error_code Execute(std::size_t* pNbExecutedTracks) = 0;

protected:
    virtual ~IExecutor();
};


// Check if execution has started for any track.
AUDIO2X_SDK_EXPORT bool HasExecutionStarted(const IExecutor& executor);

// Get the number of tracks that are ready to be executed.
// Just like with GetNbAvailableExecutions(), this is a minimum number, as more
// might become available while the call is being made, for example more audio is
// added or the audio is closed.
AUDIO2X_SDK_EXPORT std::size_t GetNbReadyTracks(const IExecutor& executor);

} // namespace nva2x
