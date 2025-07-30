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

#include "audio2face/animator.h"
#include "audio2face/job_runner.h"
#include "audio2x/tensor.h"

namespace nva2f {

// Default regularization parameters for blendshape solver.
constexpr float kDefaultL1Reg = 1.0f;
constexpr float kDefaultL2Reg = 3.5f;
constexpr float kDefaultSymmetryReg = 100.0f;
constexpr float kDefaultTemporalReg = 0.0f;
constexpr float kDefaultTemplateBBSize = 54.7f;
constexpr float kDefaultTolerance = 1e-10f;

// Parameters for configuring the blendshape solver behavior.
struct BlendshapeSolverParams {
  // L1 regularization strength for sparsity.
  float L1Reg{kDefaultL1Reg};
  // L2 regularization strength for smoothness.
  float L2Reg{kDefaultL2Reg};
  // Symmetry regularization strength for paired blendshapes.
  float SymmetryReg{kDefaultSymmetryReg};
  // Temporal regularization strength for frame-to-frame smoothness.
  float TemporalReg{kDefaultTemporalReg};

  // Template bounding box size for scaling regularization.
  float templateBBSize{kDefaultTemplateBBSize};

  // Tolerance for solver convergence.
  float tolerance{kDefaultTolerance};
};

// Data structure containing blendshape information for solver initialization.
struct BlendshapeSolverDataView {
  // Neutral pose vertex positions.
  nva2x::HostTensorFloatConstView neutralPose;
  // Delta poses for each blendshape.
  nva2x::HostTensorFloatConstView deltaPoses;
  // Optional mask for active vertex indices.
  const int* poseMask{nullptr};
  // Size of the pose mask array.
  std::size_t poseMaskSize{0};
  // Array of blendshape pose names.
  const char* const* poseNames{nullptr};
  // Number of pose names.
  std::size_t poseNamesSize{0};
};

// Configuration structure for blendshape solver settings.
struct BlendshapeSolverConfig {
  // Number of blendshape poses.
  std::size_t numBlendshapes{0};
  // Array indicating which poses are active.
  const int* activePoses{nullptr};
  // Array defining poses which cancel each other.
  const int* cancelPoses{nullptr};
  // Array defining symmetry pairs between poses.
  const int* symmetryPoses{nullptr};
  // Multipliers for each blendshape weight.
  nva2x::HostTensorFloatConstView multipliers;
  // Offsets for each blendshape weight.
  nva2x::HostTensorFloatConstView offsets;
};

// Callback function type for asynchronous solve operations.
using BlendshapeSolverCallback = void(*)(void*, std::error_code);

// Interface for blendshape solvers that compute optimal blendshape weights from target poses.
class IBlendshapeSolver {
public:

  // Set the CUDA stream for GPU operations.
  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;
  // Get the current CUDA stream.
  virtual cudaStream_t GetCudaStream() = 0;
  // Set the job runner for asynchronous CPU operations.
  virtual std::error_code SetJobRunner(nva2f::IJobRunner* jobRunner) = 0;
  // Get the current job runner.
  virtual nva2f::IJobRunner* GetJobRunner() = 0;

  // Set the blendshape data including neutral pose and delta poses.
  virtual std::error_code SetBlendshapeData(const BlendshapeSolverDataView& blendshapeData) = 0;
  // Set the blendshape configuration including active poses and constraints.
  virtual std::error_code SetBlendshapeConfig(const BlendshapeSolverConfig& config) = 0;
  // Get the current blendshape configuration.
  virtual BlendshapeSolverConfig GetBlendshapeConfig() const = 0;

  // Evaluate a pose from blendshape weights.
  virtual std::error_code EvaluatePose(
    nva2x::HostTensorFloatConstView inWeights,
    nva2x::HostTensorFloatView outPose) = 0;
  // Get the size of pose data in vertices.
  virtual int PoseSize() = 0;
  // Get the number of blendshape poses.
  virtual int NumBlendshapePoses() = 0;
  // Get the name of a blendshape pose by index.
  virtual const char* GetPoseName(size_t index) const = 0;

  // Get the range config for a field of the BlendshapeSolverParams struct.
  static const RangeConfig<float> GetRangeConfig(float BlendshapeSolverParams::* field);

  // The active poses are used to determine which blendshapes are used in the solver.
  // The active poses are stored in a buffer of size NumBlendshapePoses().
  // The buffer holds int indicating if a blendshape is used or not.
  // Non-zero values are treated as `true`, zero values are treated as `false`.

  // Set the active poses.
  virtual std::error_code SetActivePoses(const int* activePoses, size_t activePosesSize) = 0;
  // Get the active poses.
  virtual std::error_code GetActivePoses(int* outPoses, size_t outSize) = 0;
  // Set the active pose by name.
  virtual std::error_code SetActivePose(const char* poseName, const int val) = 0;
  // Get the active pose by name.
  virtual std::error_code GetActivePose(const char* poseName, int& outVal) = 0;

  // The cancel poses are used to determine which blendshapes cancel each other.
  // The cancel poses are stored in a buffer of size NumBlendshapePoses().
  // A cancel pair of blendshapes involves two blendshapes that are designed to counteract each other.
  //
  // This array represents pair relationships using indices and value.
  //
  // For example, given the array:
  //  [0, 1, 0, 2, 2, 1, -1, -1]
  //
  // Indices 0 and 2 have the value 0, so they form a pair.
  // Indices 1 and 5 have the value 1, so they form a pair.
  // Indices 3 and 4 have the value 2, so they form a pair.
  // Indices 6 and 7 have the value -1, so they are not part of any pair.

  // Set the cancel poses.
  virtual std::error_code SetCancelPoses(const int* cancelPoses, size_t cancelPosesSize) = 0;
  // Get the cancel poses.
  virtual std::error_code GetCancelPoses(int* outPoses, size_t outSize) = 0;
  // Set the cancel pose by name.
  virtual std::error_code SetCancelPose(const char* poseName, const int val) = 0;
  // Get the cancel pose by name.
  virtual std::error_code GetCancelPose(const char* poseName, int& outVal) = 0;

  // The symmetry poses are used to determine which blendshapes are mirrored.
  // The symmetry poses are stored in a buffer of size NumBlendshapePoses().
  // A symmetry pair of blendshapes involves two blendshapes that are designed to mirror each other (same strength).
  //
  // The memory layout is the same as the cancel blendshapes.

  // Set the symmetry poses.
  virtual std::error_code SetSymmetryPoses(const int* symmetryPoses, size_t symmetryPosesSize) = 0;
  // Get the symmetry poses.
  virtual std::error_code GetSymmetryPoses(int* outPoses, size_t outSize) = 0;
  // Set the symmetry pose by name.
  virtual std::error_code SetSymmetryPose(const char* poseName, const int val) = 0;
  // Get the symmetry pose by name.
  virtual std::error_code GetSymmetryPose(const char* poseName, int& outVal) = 0;

  // The multipliers are used to scale the blendshape weights.
  // The multipliers are stored in a buffer of size NumBlendshapePoses().

  // Set multipliers for all blendshape weights.
  virtual std::error_code SetMultipliers(nva2x::HostTensorFloatConstView multipliers) = 0;
  // Get multipliers for all blendshape weights.
  virtual std::error_code GetMultipliers(nva2x::HostTensorFloatView outMultipliers) = 0;
  // Set multiplier for a specific blendshape weight by name.
  virtual std::error_code SetMultiplier(const char* poseName, const float val) = 0;
  // Get multiplier for a specific blendshape weight by name.
  virtual std::error_code GetMultiplier(const char* poseName, float& outVal) = 0;

  // The offsets are used to offset the blendshape weights.
  // The offsets are stored in a buffer of size NumBlendshapePoses().

  // Set offsets for all blendshape weights.
  virtual std::error_code SetOffsets(nva2x::HostTensorFloatConstView offsets) = 0;
  // Get offsets for all blendshape weights.
  virtual std::error_code GetOffsets(nva2x::HostTensorFloatView outOffsets) = 0;
  // Set offset for a specific blendshape weight by name.
  virtual std::error_code SetOffset(const char* poseName, const float val) = 0;
  // Get offset for a specific blendshape weight by name.
  virtual std::error_code GetOffset(const char* poseName, float& outVal) = 0;

  // Set solver parameters for regularization and tolerance.
  virtual std::error_code SetParameters(const BlendshapeSolverParams& params) = 0;
  // Get current solver parameters.
  virtual const BlendshapeSolverParams& GetParameters() const = 0;

  // Setup internal buffers for efficent solve and checks for solver states.
  virtual std::error_code Prepare() = 0; // call this after updating any params

  // Solve for blendshape weights synchronously.
  virtual std::error_code Solve(nva2x::DeviceTensorFloatConstView targetPoseDevice, nva2x::HostTensorFloatView outWeights) = 0;
  // Solve for blendshape weights asynchronously with device output.
  // After this call, the GPU work is scheduled on the CUDA stream but not necessarily completed.
  // This can only be called on a GPU blendshape solver.
  virtual std::error_code SolveAsync(nva2x::DeviceTensorFloatConstView targetPoseDevice, nva2x::DeviceTensorFloatView outWeights) = 0;
  // Solve for blendshape weights asynchronously with host output and callback.
  // After this call, the CPU work is scheduled on the job runner but not necessarily completed.
  // Callback is called when the CPU work is completed.
  // This can only be called on a CPU blendshape solver.
  virtual std::error_code SolveAsync(nva2x::DeviceTensorFloatConstView targetPoseDevice, nva2x::HostTensorFloatView outWeights, BlendshapeSolverCallback callback, void* data) = 0;

  // Wait for all pending asynchronous operations to complete.
  virtual std::error_code Wait() = 0;

  // Reset solver state.
  virtual std::error_code Reset() = 0;

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IBlendshapeSolver();
};


} // namespace nva2f
