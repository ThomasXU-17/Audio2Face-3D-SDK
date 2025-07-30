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
#include "audio2face/internal/blendshape_solver_base.h"
#include "audio2face/internal/blendshape_solver.h"
#include "audio2face/internal/blendshape_solver_gpu.h"

#include <future>
#include <thread>
#include <cuda_runtime_api.h>

#include "audio2face/internal/job_runner.h"
#include "audio2face/internal/mask_extraction.h"
#include "audio2face/internal/bvls.h"
#include "audio2face/internal/macros.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/eigen_utils.h"
#include "audio2x/error.h"

namespace nva2f {

namespace {

/**
 * The input array represents pair relationships using indices and value.
 * For example, given the array:
 *  [0, 1, 0, 2, 2, 1]
 *
 * Indices 0 and 2 have the value 0, so they form a pair.
 * Indices 1 and 5 have the value 1, so they form a pair.
 * Indices 3 and 4 have the value 2, so they form a pair.
 */
std::error_code ValidatePairs(std::vector<int>& pairArr) {
  std::unordered_map<int, std::vector<int>> indexMap;
  for(int i=0;i<pairArr.size();++i) {
    if (pairArr[i] != -1) { // ignore -1
      indexMap[pairArr[i]].push_back(i);
    }
  }
  for(auto& pair : indexMap) {
    CHECK_ERROR_WITH_MSG( 2 == pair.second.size() , //pair must be 2 elements
                         "Found invalid pair with "<<pair.second.size()<<
                         " elements. A pair should have exactly 2 elements",
                         nva2x::ErrorCode::eInvalidValue);
  }
  return nva2x::ErrorCode::eSuccess;
}

} // Anonymous

constexpr auto kBSParamsType = refl::reflect<BlendshapeSolverParams>();
constexpr auto kBSParamsMembers = filter(kBSParamsType.members, [](auto member) {
  return refl::descriptor::has_attribute<Validator>(member);
});
MAKE_ERROR_CATEGORY_NAME_TRAITS(BlendshapeSolverParams, "BlendshapeSolverParams", kBSParamsMembers);

const RangeConfig<float> IBlendshapeSolver::GetRangeConfig(float BlendshapeSolverParams::* field) {
  return GetRangeConfigImpl<float, BlendshapeSolverParams>(field);
}

BlendshapeSolverBase::BlendshapeSolverBase() : mValidator(mConfig.params) {
  cublasCreate(&mCublasHandle);
}

BlendshapeSolverBase::~BlendshapeSolverBase() {
  cublasDestroy(mCublasHandle);
}

void BlendshapeSolverBase::Destroy() {
  delete this;
}

std::error_code BlendshapeSolverBase::SetCudaStream(cudaStream_t cudaStream) {
  mCudaStream = cudaStream;
  cublasSetStream(mCublasHandle, mCudaStream);
  return nva2x::ErrorCode::eSuccess;
}

cudaStream_t BlendshapeSolverBase::GetCudaStream() { return mCudaStream; }

std::error_code BlendshapeSolverBase::SetJobRunner(nva2f::IJobRunner* jobRunner) {
  if (jobRunner == nullptr) {
    return nva2x::ErrorCode::eNullPointer;
  }
  mJobRunner = jobRunner;
  return nva2x::ErrorCode::eSuccess;
}

nva2f::IJobRunner* BlendshapeSolverBase::GetJobRunner() {
  return mJobRunner;
}

std::error_code BlendshapeSolverBase::SetBlendshapeData(const BlendshapeSolverDataView& blendshapeData) {
  CHECK_ERROR_WITH_MSG(blendshapeData.neutralPose.Data() != nullptr,
                       "SetBlendshapeData: neutralPose cannot be nullptr.",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(blendshapeData.neutralPose.Size() != 0,
                       "SetBlendshapeData: neutralPose size cannot be 0.",
                       nva2x::ErrorCode::eInvalidValue);
  CHECK_ERROR_WITH_MSG(blendshapeData.deltaPoses.Data() != nullptr,
                       "SetBlendshapeData: deltaPoses cannot be nullptr.",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(blendshapeData.deltaPoses.Size() != 0,
                       "SetBlendshapeData: deltaPoses size cannot be 0.",
                       nva2x::ErrorCode::eInvalidValue);
  CHECK_ERROR_WITH_MSG(blendshapeData.deltaPoses.Size() % blendshapeData.neutralPose.Size() == 0,
                       "SetBlendshapeData: deltaPoses size should be divisible by neutralPose size.",
                       nva2x::ErrorCode::eMismatch);
  CHECK_ERROR_WITH_MSG(blendshapeData.deltaPoses.Size() / blendshapeData.neutralPose.Size() == blendshapeData.poseNamesSize,
                       "SetBlendshapeData: numPoses(deltaPoses size / neutralPose size) should equal to the poseNamesSize.",
                       nva2x::ErrorCode::eMismatch);

  mRawData.numVertexPositions = blendshapeData.neutralPose.Size();
  mRawData.numBlendshapePoses = blendshapeData.deltaPoses.Size() / mRawData.numVertexPositions;
  mRawData.neutralPose.assign(blendshapeData.neutralPose.Data(), blendshapeData.neutralPose.Data() + mRawData.numVertexPositions);
  mRawData.deltaPoses.assign(blendshapeData.deltaPoses.Data(), blendshapeData.deltaPoses.Data() + mRawData.numVertexPositions * mRawData.numBlendshapePoses);
  if (blendshapeData.poseMask == nullptr || blendshapeData.poseMaskSize == 0) {
    mRawData.poseMask.clear();
  } else {
    mRawData.poseMask.assign(blendshapeData.poseMask, blendshapeData.poseMask + blendshapeData.poseMaskSize);
  }
  mRawData.poseNames.clear();
  mRawData.poseNamesToIdx.clear();
  mRawData.poseNames.reserve(blendshapeData.poseNamesSize);
  for(int i=0;i<blendshapeData.poseNamesSize;++i) {
    mRawData.poseNames.emplace_back(blendshapeData.poseNames[i]);
    mRawData.poseNamesToIdx[blendshapeData.poseNames[i]] = i;
  }
  assert(mRawData.numBlendshapePoses == mRawData.poseNames.size());

  if (mConfig.activePoses.empty()) {
    // Assuming the blendshape configuration has not been set,
    // this part sets a default configuration to allow the solver to function and be used for exploration.
    mConfig.activePoses.resize(mRawData.numBlendshapePoses, 1);
    mConfig.cancelPoses.resize(mRawData.numBlendshapePoses, -1);
    mConfig.symmetryPoses.resize(mRawData.numBlendshapePoses, -1);
    mConfig.multipliers.resize(mRawData.numBlendshapePoses, 1.0f);
    mConfig.offsets.resize(mRawData.numBlendshapePoses, 0.0f);
    mConfig.params = BlendshapeSolverParams(); // apply default values
  }
  mATADirty = true;
  mPrepared = false;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::SetBlendshapeConfig(const BlendshapeSolverConfig& config) {
  CHECK_ERROR_WITH_MSG(config.activePoses != nullptr,
                       "SetBlendshapeConfig: activePoses cannot be nullptr.",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(config.cancelPoses != nullptr,
                       "SetBlendshapeConfig: cancelPoses cannot be nullptr.",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(config.symmetryPoses != nullptr,
                       "SetBlendshapeConfig: symmetryPoses cannot be nullptr.",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(config.multipliers.Data() != nullptr,
                       "SetBlendshapeConfig: multipliers cannot be nullptr.",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(config.multipliers.Size() == mRawData.numBlendshapePoses,
                       "SetBlendshapeConfig: multipliers size should be equal to numBlendshapePoses.",
                       nva2x::ErrorCode::eMismatch);
  CHECK_ERROR_WITH_MSG(config.offsets.Data() != nullptr,
                       "SetBlendshapeConfig: offsets cannot be nullptr.",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(config.offsets.Size() == mRawData.numBlendshapePoses,
                       "SetBlendshapeConfig: offsets size should be equal to numBlendshapePoses.",
                       nva2x::ErrorCode::eMismatch);

  A2F_CHECK_RESULT(SetActivePoses(config.activePoses, config.numBlendshapes));
  A2F_CHECK_RESULT(SetCancelPoses(config.cancelPoses, config.numBlendshapes));
  A2F_CHECK_RESULT(SetSymmetryPoses(config.symmetryPoses, config.numBlendshapes));
  A2F_CHECK_RESULT(SetMultipliers(config.multipliers));
  A2F_CHECK_RESULT(SetOffsets(config.offsets));

  mPrepared = false;
  return nva2x::ErrorCode::eSuccess;
}

BlendshapeSolverConfig BlendshapeSolverBase::GetBlendshapeConfig() const {
  BlendshapeSolverConfig config;
  config.numBlendshapes = mRawData.numBlendshapePoses;
  config.activePoses = mConfig.activePoses.data();
  config.cancelPoses = mConfig.cancelPoses.data();
  config.symmetryPoses = mConfig.symmetryPoses.data();
  config.multipliers = nva2x::ToConstView(mConfig.multipliers);
  config.offsets = nva2x::ToConstView(mConfig.offsets);
  return config;
}

const char* BlendshapeSolverBase::GetPoseName(size_t index) const {
  if (index < mRawData.poseNames.size()) {
    return mRawData.poseNames[index].data();
  }
  return nullptr;
}

std::error_code BlendshapeSolverBase::SetActivePoses(const int* activePoses, size_t activePosesSize) {
  CHECK_ERROR_WITH_MSG(activePoses != nullptr,
                       "SetActivePoses: null pointer",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(activePosesSize == mRawData.numBlendshapePoses,
                       "SetActivePoses: expecting size = " << mRawData.numBlendshapePoses << ", but got " << activePosesSize,
                       nva2x::ErrorCode::eMismatch);
  int activeCnt = 0;
  for(size_t i=0;i<activePosesSize;++i) {
    if (activePoses[i]) {
      activeCnt += 1;
    }
  }
  if (activeCnt == 0) {
    LOG_ERROR("SetActivePoses: at least one pose must be set to active.");
    return nva2x::ErrorCode::eInvalidValue;
  }
  mConfig.activePoses.assign(activePoses, activePoses + activePosesSize);
  mATADirty = true;
  mPrepared = false;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::GetActivePoses(int* outPoses, size_t outSize) {
  CHECK_ERROR_WITH_MSG(outPoses != nullptr,
                       "GetActivePoses: null pointer",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(outSize == mConfig.activePoses.size(),
                        "GetActivePoses: expecting outSize =" << mConfig.activePoses.size() << ", but got " << outSize,
                        nva2x::ErrorCode::eMismatch);
  std::copy(mConfig.activePoses.begin(), mConfig.activePoses.end(), outPoses);
  outSize = mConfig.activePoses.size();
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::SetActivePose(const char* poseName, const int val) {
  CHECK_ERROR_WITH_MSG(poseName != nullptr,
                       "SetActivePose: poseName can't be null",
                       nva2x::ErrorCode::eNullPointer);
  auto it = mRawData.poseNamesToIdx.find(poseName);
  if (it == mRawData.poseNamesToIdx.end()) {
    LOG_ERROR("SetActivePose: poseName '" << poseName << "' not found");
    return nva2x::ErrorCode::eMismatch;
  }
  mConfig.activePoses[it->second] = val;
  mATADirty = true;
  mPrepared = false;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::GetActivePose(const char* poseName, int& outVal) {
  CHECK_ERROR_WITH_MSG(poseName != nullptr,
                       "SetActivePose: poseName can't be null",
                       nva2x::ErrorCode::eNullPointer);
  auto it = mRawData.poseNamesToIdx.find(poseName);
  if (it == mRawData.poseNamesToIdx.end()) {
    LOG_ERROR("GetActivePose: poseName '" << poseName << "' not found");
    return nva2x::ErrorCode::eMismatch;
  }
  outVal = mConfig.activePoses[it->second];
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::SetCancelPoses(const int* cancelPoses, size_t cancelPosesSize) {
  CHECK_ERROR_WITH_MSG(cancelPoses != nullptr,
                       "SetCancelPoses: null pointer",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(cancelPosesSize == mRawData.numBlendshapePoses,
                       "SetCancelPoses: expecting size = " << mRawData.numBlendshapePoses << ", but got " << cancelPosesSize,
                       nva2x::ErrorCode::eMismatch);
  std::vector<int> input(cancelPoses, cancelPoses + cancelPosesSize);
  CHECK_RESULT_WITH_MSG(ValidatePairs(input), "SetCancelPoses invalid input pairs");
  mConfig.cancelPoses = std::move(input);
  mPrepared = false;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::GetCancelPoses(int* outPoses, size_t outSize) {
  CHECK_ERROR_WITH_MSG(outPoses != nullptr,
                       "GetCancelPoses: null pointer",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG( outSize == mConfig.cancelPoses.size(),
                        "GetCancelPoses: expecting outSize =" << mConfig.cancelPoses.size() << ", but got " << outSize,
                        nva2x::ErrorCode::eMismatch);
  std::copy(mConfig.cancelPoses.begin(), mConfig.cancelPoses.end(), outPoses);
  outSize = mConfig.cancelPoses.size();
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::SetCancelPose(const char* poseName, const int val) {
  CHECK_ERROR_WITH_MSG(poseName != nullptr,
                       "SetCancelPose: poseName can't be null",
                       nva2x::ErrorCode::eNullPointer);
  auto it = mRawData.poseNamesToIdx.find(poseName);
  if (it == mRawData.poseNamesToIdx.end()) {
    LOG_ERROR("SetCancelPose: poseName '" << poseName << "' not found");
    return nva2x::ErrorCode::eMismatch;
  }
  mConfig.cancelPoses[it->second] = val;
  mPrepared = false;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::GetCancelPose(const char* poseName, int& outVal) {
  CHECK_ERROR_WITH_MSG(poseName != nullptr,
                       "GetCancelPose: poseName can't be null",
                       nva2x::ErrorCode::eNullPointer);
  auto it = mRawData.poseNamesToIdx.find(poseName);
  if (it == mRawData.poseNamesToIdx.end()) {
    LOG_ERROR("GetCancelPose: poseName '" << poseName << "' not found");
    return nva2x::ErrorCode::eMismatch;
  }
  outVal = mConfig.cancelPoses[it->second];
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::SetSymmetryPoses(const int* symmetryPoses, size_t symmetryPosesSize) {
  CHECK_ERROR_WITH_MSG(symmetryPoses != nullptr,
                       "SetSymmetryPoses: null pointer",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(symmetryPosesSize == mRawData.numBlendshapePoses,
                       "SetSymmetryPoses: expecting size = " << mRawData.numBlendshapePoses << ", but got " << symmetryPosesSize,
                       nva2x::ErrorCode::eMismatch);
  std::vector<int> input(symmetryPoses, symmetryPoses + symmetryPosesSize);
  CHECK_RESULT_WITH_MSG(ValidatePairs(input), "SetSymmetryPoses invalid input pairs");
  mConfig.symmetryPoses = std::move(input);
  mPrepared = false;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::GetSymmetryPoses(int* outPoses, size_t outSize) {
  CHECK_ERROR_WITH_MSG(outPoses != nullptr,
                       "GetSymmetryPoses: null pointer",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG( outSize == mConfig.symmetryPoses.size(),
                        "GetSymmetryPoses: expecting outSize =" << mConfig.symmetryPoses.size() << ", but got " << outSize,
                        nva2x::ErrorCode::eMismatch);
  std::copy(mConfig.symmetryPoses.begin(), mConfig.symmetryPoses.end(), outPoses);
  outSize = mConfig.symmetryPoses.size();
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::SetSymmetryPose(const char* poseName, const int val) {
  CHECK_ERROR_WITH_MSG(poseName != nullptr,
                       "SetSymmetryPose: poseName can't be null",
                       nva2x::ErrorCode::eNullPointer);
  auto it = mRawData.poseNamesToIdx.find(poseName);
  if (it == mRawData.poseNamesToIdx.end()) {
    LOG_ERROR("SetSymmetryPose: poseName '" << poseName << "' not found");
    return nva2x::ErrorCode::eMismatch;
  }
  mConfig.symmetryPoses[it->second] = val;
  mPrepared = false;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::GetSymmetryPose(const char* poseName, int& outVal) {
  CHECK_ERROR_WITH_MSG(poseName != nullptr,
                       "GetSymmetryPose: poseName can't be null",
                       nva2x::ErrorCode::eNullPointer);
  auto it = mRawData.poseNamesToIdx.find(poseName);
  if (it == mRawData.poseNamesToIdx.end()) {
    LOG_ERROR("GetSymmetryPose: poseName '" << poseName << "' not found");
    return nva2x::ErrorCode::eMismatch;
  }
  outVal = mConfig.symmetryPoses[it->second];
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::SetMultipliers(nva2x::HostTensorFloatConstView multipliers) {
  CHECK_ERROR_WITH_MSG(multipliers.Data() != nullptr, "SetMultipliers: null pointer", nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(multipliers.Size() == mRawData.numBlendshapePoses,
                       "SetMultipliers: expecting size = " << mRawData.numBlendshapePoses << " but got " << multipliers.Size(),
                       nva2x::ErrorCode::eMismatch);
  mConfig.multipliers.assign(multipliers.Data(), multipliers.Data() + multipliers.Size());
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::GetMultipliers(nva2x::HostTensorFloatView outMultipliers) {
  CHECK_ERROR_WITH_MSG(outMultipliers.Data() != nullptr,
                       "GetMultipliers: null pointer",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(outMultipliers.Size() == mConfig.multipliers.size(),
                       "GetMultipliers: expecting size = " << mConfig.multipliers.size() << " but got " << outMultipliers.Size(),
                       nva2x::ErrorCode::eMismatch);
  return nva2x::CopyHostToHost(outMultipliers, nva2x::ToConstView(mConfig.multipliers));
}

std::error_code BlendshapeSolverBase::SetMultiplier(const char* poseName, const float val) {
  CHECK_ERROR_WITH_MSG(poseName != nullptr,
                       "SetMultiplier: poseName can't be null",
                       nva2x::ErrorCode::eNullPointer);
  auto it = mRawData.poseNamesToIdx.find(poseName);
  if (it == mRawData.poseNamesToIdx.end()) {
    LOG_ERROR("SetMultiplier: poseName '" << poseName << "' not found");
    return nva2x::ErrorCode::eMismatch;
  }
  mConfig.multipliers[it->second] = val;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::GetMultiplier(const char* poseName, float& outVal) {
  CHECK_ERROR_WITH_MSG(poseName != nullptr,
                       "GetMultiplier: poseName can't be null",
                       nva2x::ErrorCode::eNullPointer);
  auto it = mRawData.poseNamesToIdx.find(poseName);
  if (it == mRawData.poseNamesToIdx.end()) {
    LOG_ERROR("GetMultiplier: poseName '" << poseName << "' not found");
    return nva2x::ErrorCode::eMismatch;
  }
  outVal = mConfig.multipliers[it->second];
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::SetOffsets(nva2x::HostTensorFloatConstView offsets) {
  CHECK_ERROR_WITH_MSG(offsets.Data() != nullptr, "SetOffsets: null pointer", nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(offsets.Size() == mRawData.numBlendshapePoses,
                       "SetOffsets: expecting size = " << mRawData.numBlendshapePoses << " but got " << offsets.Size(),
                       nva2x::ErrorCode::eMismatch);
  mConfig.offsets.assign(offsets.Data(), offsets.Data() + offsets.Size());
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::GetOffsets(nva2x::HostTensorFloatView outOffsets) {
  CHECK_ERROR_WITH_MSG(outOffsets.Data() != nullptr,
                       "GetOffsets: null pointer",
                       nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(outOffsets.Size() == mConfig.offsets.size(),
                       "GetOffsets: expecting size = " << mConfig.offsets.size() << " but got " << outOffsets.Size(),
                       nva2x::ErrorCode::eMismatch);
  return nva2x::CopyHostToHost(outOffsets, nva2x::ToConstView(mConfig.offsets));
}

std::error_code BlendshapeSolverBase::SetOffset(const char* poseName, const float val) {
  CHECK_ERROR_WITH_MSG(poseName != nullptr,
                       "SetOffset: poseName can't be null",
                       nva2x::ErrorCode::eNullPointer);
  auto it = mRawData.poseNamesToIdx.find(poseName);
  if (it == mRawData.poseNamesToIdx.end()) {
    LOG_ERROR("SetOffset: poseName '" << poseName << "' not found");
    return nva2x::ErrorCode::eMismatch;
  }
  mConfig.offsets[it->second] = val;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::GetOffset(const char* poseName, float& outVal) {
  CHECK_ERROR_WITH_MSG(poseName != nullptr,
                       "GetOffset: poseName can't be null",
                       nva2x::ErrorCode::eNullPointer);
  auto it = mRawData.poseNamesToIdx.find(poseName);
  if (it == mRawData.poseNamesToIdx.end()) {
    LOG_ERROR("GetOffset: poseName '" << poseName << "' not found");
    return nva2x::ErrorCode::eMismatch;
  }
  outVal = mConfig.offsets[it->second];
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverBase::SetParameters(const BlendshapeSolverParams& params) {
  CHECK_RESULT_WITH_MSG(mValidator.L1Reg(params.L1Reg), "SetParameters");
  CHECK_RESULT_WITH_MSG(mValidator.L2Reg(params.L2Reg), "SetParameters");
  CHECK_RESULT_WITH_MSG(mValidator.SymmetryReg(params.SymmetryReg), "SetParameters");
  CHECK_RESULT_WITH_MSG(mValidator.TemporalReg(params.TemporalReg), "SetParameters");
  CHECK_RESULT_WITH_MSG(mValidator.templateBBSize(params.templateBBSize), "SetParameters");
  mConfig.params = params;
  mPrepared = false;
  return nva2x::ErrorCode::eSuccess;
}

const BlendshapeSolverParams& BlendshapeSolverBase::GetParameters() const {
  return mConfig.params;
}

std::error_code BlendshapeSolverBase::EvaluatePose(
    nva2x::HostTensorFloatConstView inWeights, nva2x::HostTensorFloatView outPose
) {
  Eigen::Map<const Eigen::VectorXf> weights(inWeights.Data(), inWeights.Size());
  Eigen::Map<Eigen::VectorXf> pose(outPose.Data(), outPose.Size());

  Eigen::VectorXf neutralPose = Eigen::Map<const Eigen::VectorXf>(mRawData.neutralPose.data(), mRawData.neutralPose.size());
  Eigen::MatrixXf deltaPoses = Eigen::Map<const Eigen::MatrixXf>(
        mRawData.deltaPoses.data(),
        mRawData.numVertexPositions,
        mRawData.numBlendshapePoses);
  pose = neutralPose + deltaPoses * weights;
  return nva2x::ErrorCode::eSuccess;
}

int BlendshapeSolverBase::PoseSize() {
  return mRawData.numVertexPositions;
}

int BlendshapeSolverBase::NumBlendshapePoses() {
  return mRawData.numBlendshapePoses;
}

std::error_code BlendshapeSolverBase::Prepare() {
  PrepareData data;
  auto ComputeAMat = [
    &data,
    &ATA = this->mATA,
    &ATADirty = this->mATADirty,
    &rawData = this->mRawData,
    &config = this->mConfig,
    cudaStream = this->mCudaStream]() -> std::error_code {

    const auto params = config.params;

    // get the actual vertices and blendshapes that is used.
    std::vector<int> activeVertexPositionIndices = [
      poseMask = rawData.poseMask]() {
      std::vector<int> ret;
      if (!poseMask.empty()) {
        ret.resize(poseMask.size()*3);
        for(int i=0;i<poseMask.size();++i) {
          auto idx = poseMask[i];
          ret[3*i+0] = 3*idx+0;
          ret[3*i+1] = 3*idx+1;
          ret[3*i+2] = 3*idx+2;
        }
      }
      return ret;
    }();
    const size_t numVertexPositions = (activeVertexPositionIndices.empty() ? rawData.numVertexPositions : activeVertexPositionIndices.size());
    data.numVertexPositions = numVertexPositions;

    std::vector<int> activeBlendshapeIndices = [&activePoses = config.activePoses]() {
      std::vector<int> ret;
      for(int i=0;i<activePoses.size();++i) {
        if (activePoses[i]) {
          ret.push_back(i);
        }
      }
      return ret;
    }();
    if (activeBlendshapeIndices.empty()) {
      LOG_ERROR("There's no active blendshape");
      return nva2x::ErrorCode::eMismatch;
    }
    const size_t numBlendshapes = activeBlendshapeIndices.size();
    data.numBlendshapes = numBlendshapes;


    data.activeVertexPositionIndices = activeVertexPositionIndices;
    data.activeBlendshapeIndices = activeBlendshapeIndices;

    // filter neutralVertexPositions and blendshapeDeltas and cache on gpu
    Eigen::MatrixXf blendshapeDeltas;
    if (activeVertexPositionIndices.empty()) {
      blendshapeDeltas = Eigen::Map<const Eigen::MatrixXf>(
        rawData.deltaPoses.data(),
        rawData.numVertexPositions,
        rawData.numBlendshapePoses)(Eigen::all, activeBlendshapeIndices);
    } else {
      blendshapeDeltas = Eigen::Map<const Eigen::MatrixXf>(
        rawData.deltaPoses.data(),
        rawData.numVertexPositions,
        rawData.numBlendshapePoses)(activeVertexPositionIndices, activeBlendshapeIndices);
    }
    data.blendshapeDeltas = blendshapeDeltas;

    Eigen::VectorXf neutralVertexPositions;
    if (activeVertexPositionIndices.empty()) {
      neutralVertexPositions = Eigen::Map<const Eigen::VectorXf>(rawData.neutralPose.data(), rawData.neutralPose.size());
    } else {
      neutralVertexPositions = Eigen::Map<const Eigen::VectorXf>(rawData.neutralPose.data(), rawData.neutralPose.size())(activeVertexPositionIndices);
    }
    data.neutralVertexPositions = neutralVertexPositions;

    // helper for converting cancelPoses and symmetryPoses into std::vector<std::pair<int, int>>
    auto getPairs = [&activePoses = config.activePoses](int* pairBlendshapes) -> std::vector<std::pair<int, int>> {
      std::vector<std::pair<int, int>> ret;
      std::vector<int> activePairBlendshapes;
      for(int i=0;i<activePoses.size();++i) {
        if (activePoses[i]) {
          activePairBlendshapes.push_back(pairBlendshapes[i]);
        }
      }

      std::unordered_map<int, std::vector<int>> pairIdToIndices;
      for(int i=0;i<activePairBlendshapes.size();++i) {
        pairIdToIndices[activePairBlendshapes[i]].push_back(i);
      }

      for (const auto& [pairId, indices] : pairIdToIndices) {
        if (indices.size() != 2) {
          continue;
        }
        ret.emplace_back(indices[0], indices[1]);
      }
      return ret;
    };
    data.cancelPairs = getPairs(config.cancelPoses.data());
    Eigen::MatrixXf symMat = [getPairs, symmetryPoses = config.symmetryPoses.data(), numBlendshapes = activeBlendshapeIndices.size()]() {
      auto symmetryPairs = getPairs(symmetryPoses);
      Eigen::MatrixXf ret = Eigen::MatrixXf::Zero(symmetryPairs.size(), numBlendshapes);
      for (int i=0;i<symmetryPairs.size();++i) {
        auto sp = symmetryPairs[i];
        ret(i, sp.first) = 1.0f;
        ret(i, sp.second) = -1.0f;
      }
      return ret;
    }();

    // compute reg scale based on the actual bounding box size.
    const float scaleFactor = [&rawData, templateBBSize = params.templateBBSize]() {
      /**
       * Convert into
       * x0, y0, z0
       * x1, y1, z1
       * ...
       */
      const size_t numVertex = rawData.numVertexPositions / 3;
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> neutralMat(rawData.neutralPose.data(), numVertex, 3);
      Eigen::Vector3f maxXYZ = neutralMat.colwise().maxCoeff();
      Eigen::Vector3f minXYZ = neutralMat.colwise().minCoeff();
      float targetBBSize = (maxXYZ - minXYZ).norm();

      return (float)std::pow(targetBBSize / templateBBSize, 2);
    }();
    const float L1RegScale = 0.25f * scaleFactor;
    const float L2RegScale = 10.0f * scaleFactor;
    const float TemporalRegScale = 100.0f * scaleFactor;
    const float SymmetryRegScale = 10.0f * scaleFactor;
    data.scaleFactor = scaleFactor;

    // Compute AMat
    if (ATADirty) {
      // only recompute when cache is invalid. Because A^TA is slow on CPU.
      ATA = blendshapeDeltas.transpose() * blendshapeDeltas;
      ATADirty = false;
    }
    data.AMat = ATA +
      params.L1Reg * params.L1Reg * L1RegScale * Eigen::MatrixXf::Ones(numBlendshapes, numBlendshapes) +
      params.L2Reg * L2RegScale * Eigen::MatrixXf::Identity(numBlendshapes, numBlendshapes) +
      params.TemporalReg * TemporalRegScale * Eigen::MatrixXf::Identity(numBlendshapes, numBlendshapes) +
      params.SymmetryReg * SymmetryRegScale * symMat.transpose() * symMat;
    return nva2x::ErrorCode::eSuccess;
  };
  CHECK_NO_ERROR(ComputeAMat());
  CHECK_NO_ERROR(Cache(data));
  CHECK_NO_ERROR(Reset());
  mPrepared = true;
  return nva2x::ErrorCode::eSuccess;
}


IBlendshapeSolver *CreateBlendshapeSolver_INTERNAL(bool gpuSolver) {
  if (gpuSolver) {
    return new BlendshapeSolverGPU();
  }
  else {
    return new BlendshapeSolver();
  }
}

bool AreEqual_INTERNAL(const BlendshapeSolverParams& a, const BlendshapeSolverParams& b) {
  return a.L1Reg == b.L1Reg &&
         a.L2Reg == b.L2Reg &&
         a.SymmetryReg == b.SymmetryReg &&
         a.TemporalReg == b.TemporalReg &&
         a.templateBBSize == b.templateBBSize &&
         a.tolerance == b.tolerance;
}

bool AreEqual_INTERNAL(const BlendshapeSolverConfig& a, const BlendshapeSolverConfig& b) {
  return a.numBlendshapes == b.numBlendshapes &&
         std::equal(a.activePoses, a.activePoses + a.numBlendshapes, b.activePoses) &&
         std::equal(a.cancelPoses, a.cancelPoses + a.numBlendshapes, b.cancelPoses) &&
         std::equal(a.symmetryPoses, a.symmetryPoses + a.numBlendshapes, b.symmetryPoses) &&
         a.multipliers.Size() == b.multipliers.Size() &&
         std::equal(a.multipliers.Data(), a.multipliers.Data() + a.multipliers.Size(), b.multipliers.Data()) &&
         a.offsets.Size() == b.offsets.Size() &&
         std::equal(a.offsets.Data(), a.offsets.Data() + a.offsets.Size(), b.offsets.Data());
}

} // namespace nva2f
