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
#include "audio2face/internal/executor_diffusion_core.h"

#include <limits>

namespace nva2f {

namespace IDiffusionModel {

class GeometryInteractiveExecutor : public GeometryInteractiveExecutorBase {
public:
    std::error_code ComputeFrame(std::size_t frameIndex) override;
    std::error_code ComputeAllFrames() override;

    std::error_code Init(
        const nva2f::GeometryExecutorCreationParameters& params,
        const nva2f::IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams,
        std::size_t nbInferencesForPreview
        );

protected:
    GeometryExecutorCoreBase& GetCore() override;
    const GeometryExecutorCoreBase& GetCore() const override;

private:
    std::error_code ComputePostProcessing(
        std::size_t beginFrameLocalIndex, std::size_t endFrameLocalIndex, bool canReusePreviousCompute
        );

    GeometryExecutorCore _core;

    std::size_t _nbInferencesForPreview{0};
    static constexpr const std::size_t kInvalidIndex = std::numeric_limits<std::size_t>::max();
    std::size_t _currentFrameIndex{kInvalidIndex};
};

} // namespace IDiffusionModel

IGeometryInteractiveExecutor* CreateDiffusionGeometryInteractiveExecutor_INTERNAL(
    const GeometryExecutorCreationParameters& params,
    const IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams,
    std::size_t nbInferencesForPreview
    );

} // namespace nva2f
