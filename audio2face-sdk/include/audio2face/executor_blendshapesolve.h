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

#include "audio2face/job_runner.h"
#include "audio2face/blendshape_solver.h"

namespace nva2f {

// Parameters for creating blendshape solve executors.
// Contains configuration for different blendshape components like skin and tongue.
struct BlendshapeSolveExecutorCreationParameters {
    // Configuration for a single blendshape component.
    // Contains solver parameters, configuration, and data view.
    struct BlendshapeParams {
        BlendshapeSolverParams params;
        BlendshapeSolverConfig config;
        BlendshapeSolverDataView data;
    };

    // Initialization parameters for skin blendshape solving.
    // If null, the skin will not be initialized nor computed.
    const BlendshapeParams* initializationSkinParams{nullptr};
    // Initialization parameters for tongue blendshape solving.
    // If null, the skin will not be initialized nor computed.
    const BlendshapeParams* initializationTongueParams{nullptr};
};

// Parameters for creating host-based blendshape solve executors.
// Extends base parameters with optional shared job runner for parallel processing.
struct HostBlendshapeSolveExecutorCreationParameters : public BlendshapeSolveExecutorCreationParameters {
    // Optional shared job runner for parallel execution.
    // If null, a new job runner will be created internally.
    IJobRunner* sharedJobRunner{nullptr};
};

// Parameters for creating device-based blendshape solve executors.
// Used for GPU-accelerated blendshape solving operations.
struct DeviceBlendshapeSolveExecutorCreationParameters : public BlendshapeSolveExecutorCreationParameters {
};

} // namespace nva2f
