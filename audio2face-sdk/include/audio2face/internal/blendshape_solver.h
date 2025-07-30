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

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <memory>

#include <Eigen/Dense>

namespace nva2f {

class BlendshapeSolver : public BlendshapeSolverBase {
public:
  BlendshapeSolver();
  ~BlendshapeSolver() override;

  std::error_code Solve(nva2x::DeviceTensorFloatConstView targetPoseDevice, nva2x::HostTensorFloatView outWeights) override;
  std::error_code SolveAsync(nva2x::DeviceTensorFloatConstView targetPoseDevice, nva2x::DeviceTensorFloatView outWeights) override;
  std::error_code SolveAsync(nva2x::DeviceTensorFloatConstView targetPoseDevice, nva2x::HostTensorFloatView outWeights, BlendshapeSolverCallback callback, void* data) override;
  std::error_code Wait() override;
  std::error_code Reset() override;
private:
  struct BlendshapeSolverCache;
  struct WorkingBuffer;

  std::error_code Cache(PrepareData& data) override;
  std::unique_ptr<BlendshapeSolverCache> mCache;
  std::unique_ptr<WorkingBuffer> mWorkingBuffer;

  // temporal smoothing
  Eigen::VectorXf mPrevWeights;

  std::mutex mCPUSolveMtx; // ensure only one cpuSolve routine is execute at a time.
  std::mutex mCallbackOrderMtx; // ensure callbacks are called in solve order.

  cudaEvent_t preBlendshapeSolveCompleted; // For recording an event after the GPU works in SolveAsync is enqueued.
  std::mutex mBHostPinnedMtx;
  std::condition_variable mBHostPinnedCV;
  bool mBHostPinnedWritable{true};
  nva2x::HostPinnedTensorFloat mBHostPinned;

  std::mutex mAsyncTaskCountMtx;
  std::condition_variable mAsyncTaskCountCV;
  std::atomic<int> mAsyncTaskCount{0};
};

} // namespace nva2f
