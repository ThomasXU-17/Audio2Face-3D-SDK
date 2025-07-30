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
#include "audio2x/executor.h"
#include "audio2x/internal/executor.h"
#include "audio2x/internal/audio2x.h"

namespace nva2x {

IExecutor::~IExecutor() = default;

std::size_t GetNbAvailableExecutions(
  const WindowProgress& progress,
  const IAudioAccumulator& audioAccumulator,
  const IEmotionAccumulator* emotionAccumulator,
  std::size_t nbFramesPerWindow
) {
    // We need to get isClosed first, in case it gets close while we are querying this.
    const auto isClosed = audioAccumulator.IsClosed();
    const auto nbAccumulatedSamples = audioAccumulator.NbAccumulatedSamples();
    auto nbAvailableWindows = progress.GetNbAvailableWindows(nbAccumulatedSamples, isClosed);

    // Number of available executions might be limited by the number of emotions available
    // if emotions are still being accumulated.
    if (emotionAccumulator && !emotionAccumulator->IsClosed()) {
        if (emotionAccumulator->IsEmpty()) {
            // No emotions available yet.
            // We do this test for empty even though the following math should work
            // to be sure not to have overflow when doing math on very small values,
            // which will the value of last accumulated timestamp if no emotions are available.
            nbAvailableWindows = 0;
        }
        else {
            // Get the number of available windows from the last accumulated timestamp.
            // 1 is added to the last time frame because it is legal to have a frame at
            // that timestamp, but not one after
            const auto& singleFrameProgress = GetFrameProgress(progress, nbFramesPerWindow);
            const auto nbAvailableFramesFromEmotions = singleFrameProgress.GetNbAvailableWindows(
                emotionAccumulator->LastAccumulatedTimestamp() + 1, true
                );
            // Get the number of available windows from the number of available frames.
            const auto nbAvailableWindowsFromEmotions = nbAvailableFramesFromEmotions / nbFramesPerWindow;
            nbAvailableWindows = std::min(nbAvailableWindows, nbAvailableWindowsFromEmotions);
        }
    }

    return nbAvailableWindows;
}

} // namespace nva2x


bool nva2x::internal::HasExecutionStarted(const IExecutor& executor) {
    const auto nbTracks = executor.GetNbTracks();
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        if (executor.HasExecutionStarted(trackIndex)) {
            return true;
        }
    }
    return false;
}

std::size_t nva2x::internal::GetNbReadyTracks(const IExecutor& executor) {
    std::size_t nbReadyTracks = 0;
    const auto nbTracks = executor.GetNbTracks();
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        if (executor.GetNbAvailableExecutions(trackIndex) > 0) {
            ++nbReadyTracks;
        }
    }
    return nbReadyTracks;
}
