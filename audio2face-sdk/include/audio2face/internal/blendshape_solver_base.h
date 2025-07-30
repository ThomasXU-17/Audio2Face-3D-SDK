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

#include <cassert>
#include <mutex>
#include <memory>

#include <Eigen/Dense>

#include <cublas_v2.h>

#include "audio2face/blendshape_solver.h"
#include "audio2face/internal/validator.h"
#include "audio2x/internal/tensor.h"

REFL_AUTO(
  type(nva2f::BlendshapeSolverParams),
  field(L1Reg,          nva2f::Validator<float>(nva2f::kDefaultL1Reg, 0)),
  field(L2Reg,          nva2f::Validator<float>(nva2f::kDefaultL2Reg, 0)),
  field(SymmetryReg,    nva2f::Validator<float>(nva2f::kDefaultSymmetryReg, 0)),
  field(TemporalReg,    nva2f::Validator<float>(nva2f::kDefaultTemporalReg, 0)),
  field(templateBBSize, nva2f::Validator<float>(nva2f::kDefaultTemplateBBSize, 1.f))
)

namespace nva2f {

class BlendshapeSolverBase : public IBlendshapeSolver {
public:
  BlendshapeSolverBase();
  ~BlendshapeSolverBase() override;

  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  cudaStream_t GetCudaStream() override;
  std::error_code SetJobRunner(nva2f::IJobRunner* jobRunner) override;
  nva2f::IJobRunner* GetJobRunner() override;
  std::error_code SetBlendshapeData(const BlendshapeSolverDataView& blendshapeData) override;
  std::error_code SetBlendshapeConfig(const BlendshapeSolverConfig& config) override;
  BlendshapeSolverConfig GetBlendshapeConfig() const override;
  std::error_code EvaluatePose(
    nva2x::HostTensorFloatConstView inWeights,
    nva2x::HostTensorFloatView outPose) override;
  int PoseSize() override;
  int NumBlendshapePoses() override;
  const char* GetPoseName(size_t index) const override;
  std::error_code SetActivePoses(const int* activePoses, size_t activePosesSize) override;
  std::error_code GetActivePoses(int* outPoses, size_t outSize) override;
  std::error_code SetActivePose(const char* poseName, const int val) override;
  std::error_code GetActivePose(const char* poseName, int& outVal) override;
  std::error_code SetCancelPoses(const int* cancelPoses, size_t cancelPosesSize) override;
  std::error_code GetCancelPoses(int* outPoses, size_t outSize) override;
  std::error_code SetCancelPose(const char* poseName, const int val) override;
  std::error_code GetCancelPose(const char* poseName, int& outVal) override;
  std::error_code SetSymmetryPoses(const int* symmetryPoses, size_t symmetryPosesSize) override;
  std::error_code GetSymmetryPoses(int* outPoses, size_t outSize) override;
  std::error_code SetSymmetryPose(const char* poseName, const int val) override;
  std::error_code GetSymmetryPose(const char* poseName, int& outVal) override;
  std::error_code SetMultipliers(nva2x::HostTensorFloatConstView multipliers) override;
  std::error_code GetMultipliers(nva2x::HostTensorFloatView outMultipliers) override;
  std::error_code SetMultiplier(const char* poseName, const float val) override;
  std::error_code GetMultiplier(const char* poseName, float& outVal) override;
  std::error_code SetOffsets(nva2x::HostTensorFloatConstView offsets) override;
  std::error_code GetOffsets(nva2x::HostTensorFloatView outOffsets) override;
  std::error_code SetOffset(const char* poseName, const float val) override;
  std::error_code GetOffset(const char* poseName, float& outVal) override;

  std::error_code SetParameters(const BlendshapeSolverParams& params);
  const BlendshapeSolverParams& GetParameters() const;

  std::error_code Prepare() override;

  void Destroy() override;
protected:
  // resources needed to run the job
  cublasHandle_t mCublasHandle{nullptr};
  cudaStream_t mCudaStream{nullptr};
  IJobRunner *mJobRunner{nullptr};

  // params exposed by setters
  struct {
    size_t numVertexPositions;
    size_t numBlendshapePoses;

    std::vector<float> neutralPose;
    std::vector<float> deltaPoses;
    std::vector<int> poseMask;
    std::vector<std::string> poseNames;
    std::unordered_map<std::string, int> poseNamesToIdx;
  } mRawData;
  struct {
    std::vector<int> activePoses;
    std::vector<int> cancelPoses;
    std::vector<int> symmetryPoses;
    std::vector<float> multipliers;
    std::vector<float> offsets;

    BlendshapeSolverParams params; // for fine tuning the solver
  } mConfig;

  bool mPrepared{false}; // flag to inform the user Perpare must be called first
  bool mATADirty{true}; // only recompute A^TA when necessary because it's time consuming
  Eigen::MatrixXf mATA;

  struct PrepareData {
    size_t numVertexPositions;
    size_t numBlendshapes;
    std::vector<int> activeVertexPositionIndices;
    std::vector<int> activeBlendshapeIndices;
    Eigen::MatrixXf blendshapeDeltas;
    Eigen::VectorXf neutralVertexPositions;
    std::vector<std::pair<int, int>> cancelPairs;
    float scaleFactor;
    Eigen::MatrixXf AMat;
  };

  ValidatorProxy<BlendshapeSolverParams> mValidator;

  virtual std::error_code Cache(PrepareData& data) = 0;
};

IBlendshapeSolver *CreateBlendshapeSolver_INTERNAL(bool gpuSolver);

bool AreEqual_INTERNAL(const BlendshapeSolverParams& a, const BlendshapeSolverParams& b);
bool AreEqual_INTERNAL(const BlendshapeSolverConfig& a, const BlendshapeSolverConfig& b);

} // namespace nva2f
