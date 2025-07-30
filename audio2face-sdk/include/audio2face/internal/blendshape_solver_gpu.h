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

#include "audio2face/blendshape_solver.h"
#include "audio2face/internal/blendshape_solver_base.h"
#include "audio2face/internal/validator.h"
#include "audio2x/internal/tensor.h"

#include <memory>

namespace nva2f {

class BlendshapeSolverGPU : public BlendshapeSolverBase {
public:
  BlendshapeSolverGPU();
  ~BlendshapeSolverGPU() override;

  std::error_code SetMultipliers(nva2x::HostTensorFloatConstView multipliers) override;
  std::error_code SetMultiplier(const char* poseName, const float val) override;
  std::error_code SetOffsets(nva2x::HostTensorFloatConstView offsets) override;
  std::error_code SetOffset(const char* poseName, const float val) override;

  std::error_code Solve(nva2x::DeviceTensorFloatConstView targetPoseDevice, nva2x::HostTensorFloatView outWeights) override;
  std::error_code SolveAsync(nva2x::DeviceTensorFloatConstView targetPoseDevice, nva2x::DeviceTensorFloatView outWeights) override;
  std::error_code SolveAsync(nva2x::DeviceTensorFloatConstView targetPoseDevice, nva2x::HostTensorFloatView outWeights, BlendshapeSolverCallback callback, void* data) override;
  std::error_code Wait() override;
  std::error_code Reset() override;
private:
  struct BlendshapeSolverGPUCache;
  struct WorkingBufferGPU;

  std::error_code Cache(PrepareData& data) override;
  std::unique_ptr<BlendshapeSolverGPUCache> mCache;
  std::unique_ptr<WorkingBufferGPU> mWorkingBufferGPU;

  // full solved weights buffer on device, only used for Solve() synchronous variant.
  nva2x::DeviceTensorFloat mFullSolvedWeightsDevice;
  nva2x::DeviceTensorFloat d_Multipliers;
  nva2x::DeviceTensorFloat d_Offsets;
  // temporal smoothing
  nva2x::DeviceTensorFloat mPrevWeights;
};

} // namespace nva2f
