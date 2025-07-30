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

#include "audio2face/internal/executor_diffusion_core.h"
#include "audio2face/internal/executor.h"
#include "audio2face/internal/model_diffusion.h"
#include "audio2x/internal/bit_vector.h"

namespace nva2f {

namespace IDiffusionModel {

class GeometryExecutor : public GeometryExecutorBase {
public:
    std::error_code Reset(std::size_t trackIndex) override;

    std::error_code Execute(std::size_t* pNbExecutedTracks) override;

    std::error_code Init(
        const nva2f::GeometryExecutorCreationParameters& params,
        const nva2f::IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams
    );

protected:
    GeometryExecutorCore& GetCore() override;
    const GeometryExecutorCore& GetCore() const override;

private:
    GeometryExecutorCore _core;

    nva2x::bit_vector<std::uint64_t> _postProcessTracks;
    nva2x::bit_vector<std::uint64_t> _doneTracks;
};

} // namespace IDiffusionModel

IGeometryExecutor* CreateDiffusionGeometryExecutor_INTERNAL(
    const GeometryExecutorCreationParameters& params,
    const IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams
    );

} // namespace nva2f
