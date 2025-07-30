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

#include "audio2face/internal/interactive_executor.h"
#include "audio2face/internal/executor_regression_core.h"

namespace nva2f {

namespace IRegressionModel {

class GeometryInteractiveExecutor : public GeometryInteractiveExecutorBase {
public:
    std::error_code ComputeFrame(std::size_t frameIndex) override;
    std::error_code ComputeAllFrames() override;

    std::error_code Init(
        const nva2f::GeometryExecutorCreationParameters& params,
        const nva2f::IRegressionModel::GeometryExecutorCreationParameters& regressionParams,
        std::size_t batchSize
        );

protected:
    GeometryExecutorCoreBase& GetCore() override;
    const GeometryExecutorCoreBase& GetCore() const override;

private:
    std::error_code Compute(std::size_t beginFrameIndex, std::size_t endFrameIndex);

    GeometryExecutorCore _core;

    nva2f::IRegressionModel::InferenceOutputBuffers _inferenceResults;
    nva2f::IRegressionModel::ResultBuffers _resultsBuffers;
    std::size_t _pcaFrameBeginIndex{0};
    std::size_t _pcaFrameEndIndex{0};
};

} // namespace IRegressionModel

IGeometryInteractiveExecutor* CreateRegressionGeometryInteractiveExecutor_INTERNAL(
    const GeometryExecutorCreationParameters& params,
    const IRegressionModel::GeometryExecutorCreationParameters& regressionParams,
    std::size_t batchSize
    );

} // namespace nva2f
