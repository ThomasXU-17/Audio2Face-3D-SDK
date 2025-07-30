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

#include "audio2emotion/executor_classifier.h"
#include "audio2emotion/internal/executor.h"
#include "audio2emotion/internal/executor_classifier_core.h"

namespace nva2e {

namespace IClassifierModel {

class EmotionExecutor : public EmotionExecutorBase {
public:
    std::size_t GetNextAudioSampleToRead(std::size_t trackIndex) const override;

    std::error_code Execute(std::size_t* pNbExecutedTracks) override;

    std::error_code Init(
        const nva2e::EmotionExecutorCreationParameters& params,
        const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams
        );

protected:
    std::error_code RunInference(std::size_t& outNbExecutedTracks) override;
    EmotionExecutorCore& GetCore() override;
    const EmotionExecutorCore& GetCore() const override;

private:
    EmotionExecutorCore _core;
};

} // namespace IClassifierModel

IEmotionExecutor* CreateClassifierEmotionExecutor_INTERNAL(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams
    );

} // namespace nva2e
