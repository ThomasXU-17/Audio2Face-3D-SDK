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
#include "audio2emotion/internal/interactive_executor.h"
#include "audio2emotion/internal/executor_postprocess_core.h"

namespace nva2e {

namespace IPostProcessModel {

class EmotionInteractiveExecutor : public EmotionInteractiveExecutorBase {
public:
    std::error_code GetInferencesToSkip(std::size_t& inferencesToSkip) const override;
    std::error_code SetInferencesToSkip(std::size_t inferencesToSkip) override;

    std::error_code Init(
        const nva2e::EmotionExecutorCreationParameters& params,
        const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
        std::size_t batchSize
        );
    std::error_code Init(
        const nva2e::EmotionExecutorCreationParameters& params,
        const nva2e::IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
        );

private:
    EmotionExecutorCoreBase& GetCore() override;
    const EmotionExecutorCoreBase& GetCore() const override;
    std::error_code ComputeInference() override;

    EmotionExecutorCore _core;
};

} // namespace IClassifierModel

IEmotionInteractiveExecutor* CreatePostProcessEmotionInteractiveExecutor_INTERNAL(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    std::size_t batchSize
    );
IEmotionInteractiveExecutor* CreatePostProcessEmotionInteractiveExecutor_INTERNAL(
    const nva2e::EmotionExecutorCreationParameters& params,
    const nva2e::IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
    );

} // namespace nva2e
