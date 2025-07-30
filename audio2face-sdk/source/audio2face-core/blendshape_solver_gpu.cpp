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
#include "audio2face/internal/blendshape_solver_gpu.h"

#include "audio2face/internal/admm.h"
#include "audio2face/internal/job_runner.h"
#include "audio2face/internal/mask_extraction.h"
#include "audio2face/internal/bvls.h"
#include "audio2face/internal/macros.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/eigen_utils.h"
#include "audio2x/cuda_utils.h"
#include "audio2x/error.h"

#include "audio2x/internal/nvtx_trace.h"

#include <future>
#include <thread>
#include <cuda_runtime_api.h>

namespace nva2f {

std::error_code SetUpperByCancelPairs(float* upper, float* weights, int* first, int* second, int num, cudaStream_t stream);
std::error_code UnmapActiveBlendshapes(float* fullWeights, float* solvedWeights, int* activeShapeMap, int num, cudaStream_t stream);
std::error_code ApplyBlendshapeMultiplersAndOffsets(float* weights, const float* multipliers, const float* offsets, int numWeights, cudaStream_t stream);

/**
 * Holds data precomputed by the Prepare method for reuse in the Solve method.
 */
struct BlendshapeSolverGPU::BlendshapeSolverGPUCache {
  size_t numVertexPositions;
  size_t numBlendshapes;

  float scaleFactor{1.0f};

  nva2x::DeviceTensorFloat neutralPoseDevice;
  nva2x::DeviceTensorFloat deltaPosesDevice;
  int* maskIndicesDevice{nullptr};

  int* d_activeBlendshapeIndices;
  size_t numActiveShapes;
  int* d_cancel_pair_first;
  int* d_cancel_pair_second;
  size_t numCancelPairs;

  // ADMM
  nva2x::DeviceTensorFloat d_AMat;
  nva2x::DeviceTensorFloat d_AmatInv;
  nva2x::DeviceTensorFloat d_AdmmWeights;
  nva2x::DeviceTensorFloat d_AdmmMatInv;

  BlendshapeSolverGPUCache(size_t numVertexPositions, size_t numBlendshapes, size_t numActiveShapes, size_t numCancelPairs);
  ~BlendshapeSolverGPUCache();
};

/**
 * Temporary storage buffers for intermediate results in the Solve method.
 */
struct BlendshapeSolverGPU::WorkingBufferGPU {
  nva2x::DeviceTensorFloat targetDeltaDevice;
  nva2x::DeviceTensorFloat BDevice;

  nva2x::DeviceTensorFloat d_lower;
  nva2x::DeviceTensorFloat d_upper;

  nva2x::DeviceTensorFloat d_solvedWeights;

  // intermediate for ADMM
  nva2x::DeviceTensorFloat d_ATb;
  nva2x::DeviceTensorFloat d_u1;
  nva2x::DeviceTensorFloat d_z2;
  nva2x::DeviceTensorFloat d_u2;

  WorkingBufferGPU(size_t numVertexPositions, size_t numBlendshapes, cudaStream_t cudaStream);
};

BlendshapeSolverGPU::WorkingBufferGPU::WorkingBufferGPU(size_t numVertexPositions, size_t numBlendshapes, cudaStream_t cudaStream) {
  targetDeltaDevice.Allocate(numVertexPositions);
  BDevice.Allocate(numBlendshapes);
  d_lower.Allocate(numBlendshapes);
  nva2x::FillOnDevice(d_lower, 0.0f, cudaStream);
  d_upper.Allocate(numBlendshapes);
  d_solvedWeights.Allocate(numBlendshapes);

  d_ATb.Allocate(numBlendshapes);
  d_u1.Allocate(numBlendshapes);
  d_z2.Allocate(numBlendshapes);
  d_u2.Allocate(numBlendshapes);
}

BlendshapeSolverGPU::BlendshapeSolverGPUCache::BlendshapeSolverGPUCache(size_t numVertexPositions, size_t numBlendshapes, size_t numActiveShapes, size_t numCancelPairs) :
    numVertexPositions(numVertexPositions), numBlendshapes(numBlendshapes), numActiveShapes(numActiveShapes), numCancelPairs(numCancelPairs)
{
  neutralPoseDevice.Allocate(numVertexPositions);
  deltaPosesDevice.Allocate(numVertexPositions * numBlendshapes);

  d_AMat.Allocate(numBlendshapes * numBlendshapes);
  d_AmatInv.Allocate(numBlendshapes * numBlendshapes);
  d_AdmmWeights.Allocate(numBlendshapes);
  d_AdmmMatInv.Allocate(numBlendshapes * numBlendshapes);
  cudaMalloc((void**)&maskIndicesDevice, numVertexPositions * sizeof(int));
  if (numCancelPairs > 0) {
    cudaMalloc((void**)&d_cancel_pair_first, numCancelPairs * sizeof(int));
    cudaMalloc((void**)&d_cancel_pair_second, numCancelPairs * sizeof(int));
    cudaMemset(d_cancel_pair_first, -1, sizeof(int) * numBlendshapes);
    cudaMemset(d_cancel_pair_second, -1, sizeof(int) * numBlendshapes);
  }
  cudaMalloc((void**)&d_activeBlendshapeIndices, numActiveShapes * sizeof(int));
}

BlendshapeSolverGPU::BlendshapeSolverGPUCache::~BlendshapeSolverGPUCache() {
  cudaFree(maskIndicesDevice);
  if (numCancelPairs > 0) {
    cudaFree(d_cancel_pair_first);
    cudaFree(d_cancel_pair_second);
  }
  cudaFree(d_activeBlendshapeIndices);
}

BlendshapeSolverGPU::BlendshapeSolverGPU() : BlendshapeSolverBase() {
}

BlendshapeSolverGPU::~BlendshapeSolverGPU() {
}

std::error_code BlendshapeSolverGPU::SetMultipliers(nva2x::HostTensorFloatConstView multipliers) {
  CHECK_NO_ERROR(BlendshapeSolverBase::SetMultipliers(multipliers));
  CHECK_RESULT(d_Multipliers.Init(nva2x::ToConstView(mConfig.multipliers), mCudaStream));
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverGPU::SetMultiplier(const char* poseName, const float val) {
  CHECK_NO_ERROR(BlendshapeSolverBase::SetMultiplier(poseName, val));
  CHECK_RESULT(d_Multipliers.Init(nva2x::ToConstView(mConfig.multipliers), mCudaStream));
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverGPU::SetOffsets(nva2x::HostTensorFloatConstView offsets) {
  CHECK_NO_ERROR(BlendshapeSolverBase::SetOffsets(offsets));
  CHECK_RESULT(d_Offsets.Init(nva2x::ToConstView(mConfig.offsets), mCudaStream));
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverGPU::SetOffset(const char* poseName, const float val) {
  CHECK_NO_ERROR(BlendshapeSolverBase::SetOffset(poseName, val));
  CHECK_RESULT(d_Offsets.Init(nva2x::ToConstView(mConfig.offsets), mCudaStream));
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverGPU::Cache(PrepareData& data) {
  mCache = std::make_unique<BlendshapeSolverGPUCache>(data.numVertexPositions, data.numBlendshapes, data.activeBlendshapeIndices.size(), data.cancelPairs.size());
  if (!data.activeBlendshapeIndices.empty()) {
    CUDA_CHECK_ERROR(cudaMemcpyAsync(mCache->d_activeBlendshapeIndices, data.activeBlendshapeIndices.data(),
      data.activeBlendshapeIndices.size() * sizeof(int), cudaMemcpyHostToDevice, mCudaStream), nva2x::ErrorCode::eCudaMemcpyHostToDeviceError);
  }
  if (!data.activeVertexPositionIndices.empty()) {
    CUDA_CHECK_ERROR(cudaMemcpyAsync(mCache->maskIndicesDevice, data.activeVertexPositionIndices.data(),
      data.activeVertexPositionIndices.size() * sizeof(int), cudaMemcpyHostToDevice, mCudaStream), nva2x::ErrorCode::eCudaMemcpyHostToDeviceError);
  }
  CHECK_RESULT(nva2x::CopyHostToDevice(mCache->deltaPosesDevice, ToConstView(data.blendshapeDeltas), mCudaStream));
  CHECK_RESULT(nva2x::CopyHostToDevice(mCache->neutralPoseDevice, ToConstView(data.neutralVertexPositions), mCudaStream));
  mCache->scaleFactor = data.scaleFactor;

  mCache->numCancelPairs = data.cancelPairs.size();

  // TODO: set d_cancel_pair_first and d_cancel_pair_second
  std::vector<int> cancelPairFirst;
  std::vector<int> cancelPairSecond;
  for(auto& pair : data.cancelPairs) {
    cancelPairFirst.push_back(pair.first);
    cancelPairSecond.push_back(pair.second);
  }
  CUDA_CHECK_ERROR(cudaMemcpyAsync(mCache->d_cancel_pair_first, cancelPairFirst.data(), cancelPairFirst.size() * sizeof(int), cudaMemcpyHostToDevice, mCudaStream), nva2x::ErrorCode::eCudaMemcpyHostToDeviceError);
  CUDA_CHECK_ERROR(cudaMemcpyAsync(mCache->d_cancel_pair_second, cancelPairSecond.data(), cancelPairSecond.size() * sizeof(int), cudaMemcpyHostToDevice, mCudaStream), nva2x::ErrorCode::eCudaMemcpyHostToDeviceError);

  auto AMat = data.AMat;
  // Preallocate buffer for ADMM
  CHECK_RESULT(nva2x::CopyHostToDevice(mCache->d_AMat, ToConstView(AMat), mCudaStream));
  Eigen::MatrixXf aMatInv = AMat.inverse();
  CHECK_RESULT(nva2x::CopyHostToDevice(mCache->d_AmatInv, ToConstView(aMatInv), mCudaStream));

  // preconditioning weights
  Eigen::MatrixXf admm_sysmat = AMat.transpose() * AMat;
  Eigen::VectorXf admm_weights = admm_sysmat.diagonal();
  admm_weights = 0.25f * admm_weights.array().sqrt();
  CHECK_RESULT(nva2x::CopyHostToDevice(mCache->d_AdmmWeights, ToConstView(admm_weights), mCudaStream));

  // This is the inverse of (A^T * A + w**2), which we also need, so copy this to the device too:
  admm_sysmat += admm_weights.array().square().matrix().asDiagonal();
  Eigen::MatrixXf admmMatInv = admm_sysmat.inverse();
  CHECK_RESULT(nva2x::CopyHostToDevice(mCache->d_AdmmMatInv, ToConstView(admmMatInv), mCudaStream));

  CHECK_RESULT(d_Multipliers.Init(nva2x::ToConstView(mConfig.multipliers), mCudaStream));
  CHECK_RESULT(d_Offsets.Init(nva2x::ToConstView(mConfig.offsets), mCudaStream));
  mWorkingBufferGPU = std::make_unique<WorkingBufferGPU>(data.numVertexPositions, data.numBlendshapes, mCudaStream);
  CHECK_RESULT(mPrevWeights.Allocate(mCache->numBlendshapes));
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverGPU::Solve(nva2x::DeviceTensorFloatConstView targetPoseDevice, nva2x::HostTensorFloatView outWeights) {
  NVTX_TRACE("BlendshapeSolverGPU::Solve");
  CHECK_RESULT(mFullSolvedWeightsDevice.Allocate(mRawData.numBlendshapePoses));
  CHECK_RESULT(SolveAsync(targetPoseDevice, mFullSolvedWeightsDevice));
  CHECK_RESULT(nva2x::CopyDeviceToHost(outWeights, mFullSolvedWeightsDevice, mCudaStream));
  return Wait();
}

std::error_code BlendshapeSolverGPU::SolveAsync(
  nva2x::DeviceTensorFloatConstView targetPoseDevice,
  nva2x::HostTensorFloatView outWeights,
  BlendshapeSolverCallback callback, void* data) {
    return nva2x::ErrorCode::eUnsupported;
}

std::error_code BlendshapeSolverGPU::SolveAsync(nva2x::DeviceTensorFloatConstView targetPoseDevice, nva2x::DeviceTensorFloatView outWeights) {
  NVTX_TRACE("BlendshapeSolverGPU::SolveAsync");
  if (!mPrepared) {
    LOG_ERROR("BlendshapeSolverGPU::Prepare() must be called before calling Solve() or SolveAsync()"
      " or after any setter that invalidates the prepared state");
    return nva2x::ErrorCode::eNotInitialized;
  }
  // Extract variables for easier reference
  const size_t numVertexPositions = mCache->numVertexPositions;
  const size_t numBlendshapes = mCache->numBlendshapes;

  CHECK_ERROR_WITH_MSG(targetPoseDevice.Size() == mRawData.numVertexPositions, "Mismatched size for pose in BlendshapeSolverGPU", nva2x::ErrorCode::eMismatch);
  CHECK_ERROR_WITH_MSG(outWeights.Size() == mRawData.numBlendshapePoses, "Mismatched size for output weights in BlendshapeSolverGPU", nva2x::ErrorCode::eMismatch);

  // Compute b on the GPU
  {
    // targetDeltaDevice = masked(targetPoseDevice) - neutralPoseDevice
    {
      if (!mRawData.poseMask.empty()) {
        CHECK_RESULT(nva2f::CopyIndices(mWorkingBufferGPU->targetDeltaDevice.Data(), targetPoseDevice.Data(),
                 mCache->maskIndicesDevice, numVertexPositions, mCudaStream));
      } else {
        CHECK_RESULT(nva2x::CopyDeviceToDevice(mWorkingBufferGPU->targetDeltaDevice, targetPoseDevice, mCudaStream));
      }
      const static float alpha = -1.0f;
      cublasSaxpy(mCublasHandle,
                  numVertexPositions,
                  &alpha,
                  mCache->neutralPoseDevice.Data(),
                  1,
                  mWorkingBufferGPU->targetDeltaDevice.Data(),
                  1);
    }
    // BDevice = d_blendsahepDeltas^T @ targetDeltaDevice + params.TemporalReg * mCache->scaleFactor * mPrevWeights
    {
      // set BDevice = mPrevWeights
      CHECK_RESULT(nva2x::CopyDeviceToDevice(mWorkingBufferGPU->BDevice, mPrevWeights, mCudaStream));
      const static float alpha = 1.0f;
      const float beta = mConfig.params.TemporalReg * mCache->scaleFactor;
      cublasSgemv(mCublasHandle, /*trans=*/CUBLAS_OP_T,
              /*m=*/numVertexPositions, /*n=*/numBlendshapes, &alpha,
              /*A=*/mCache->deltaPosesDevice.Data(), /*lda=*/numVertexPositions,
              /*x=*/mWorkingBufferGPU->targetDeltaDevice.Data(), /*incx=*/1, &beta,
              /*y=*/mWorkingBufferGPU->BDevice.Data(), /*incy=*/1);
    }
  }

  auto admmSolveAsync = [this, numBlendshapes, outWeights] {
    float* z1 = mWorkingBufferGPU->d_solvedWeights.Data();
    float* u1 = mWorkingBufferGPU->d_u1.Data();
    float* z2 = mWorkingBufferGPU->d_z2.Data();
    float* u2 = mWorkingBufferGPU->d_u2.Data();
    float* ATb = mWorkingBufferGPU->d_ATb.Data();
    float* b = mWorkingBufferGPU->BDevice.Data();
    float* l = mWorkingBufferGPU->d_lower.Data();
    float* u = mWorkingBufferGPU->d_upper.Data();
    float* A = mCache->d_AMat.Data();
    float* AInv = mCache->d_AmatInv.Data();

    float* AdmmWeights = mCache->d_AdmmWeights.Data();
    float* AdmmMatInv = mCache->d_AdmmMatInv.Data();

    CHECK_NO_ERROR(admm_init(z1, u1, ATb, b, A, AInv, l, u, 1, numBlendshapes, mCublasHandle, mCudaStream));
    for(int i=0;i<2;++i) {
      CHECK_NO_ERROR(admm_update(z1, u1, z2, u2,
              AdmmWeights, ATb, AdmmMatInv,
              l, u,
              1, numBlendshapes, mCudaStream));
      CHECK_NO_ERROR(admm_update(z2, u2, z1, u1,
              AdmmWeights, ATb, AdmmMatInv,
              l, u,
              1, numBlendshapes, mCudaStream));
    }
    // set u = 1

    return make_error_code(nva2x::ErrorCode::eSuccess);
  };
  // set u = 1
  CHECK_NO_ERROR(nva2x::FillOnDevice(mWorkingBufferGPU->d_upper, 1.0f, mCudaStream));
  CHECK_NO_ERROR(admmSolveAsync());
  // set u based on cancel shapes
  if (mCache->numCancelPairs > 0) {
    CHECK_NO_ERROR(SetUpperByCancelPairs(mWorkingBufferGPU->d_upper.Data(), mWorkingBufferGPU->d_solvedWeights.Data(),
      mCache->d_cancel_pair_first, mCache->d_cancel_pair_second, mCache->numCancelPairs, mCudaStream));
    CHECK_NO_ERROR(admmSolveAsync());
  }
  // save previous weights
  CHECK_NO_ERROR(nva2x::CopyDeviceToDevice(mPrevWeights, mWorkingBufferGPU->d_solvedWeights, mCudaStream));

  // unmap active weights to full blendshape weights
  CHECK_NO_ERROR(nva2x::FillOnDevice(outWeights, 0.0f, mCudaStream));
  CHECK_NO_ERROR(UnmapActiveBlendshapes(outWeights.Data(), mWorkingBufferGPU->d_solvedWeights.Data(),
    mCache->d_activeBlendshapeIndices, mCache->numActiveShapes, mCudaStream));

  // multiplier and offsets
  CHECK_NO_ERROR(ApplyBlendshapeMultiplersAndOffsets(outWeights.Data(), d_Multipliers.Data(), d_Offsets.Data(),
    mRawData.numBlendshapePoses, mCudaStream));

  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverGPU::Wait() {
  CUDA_CHECK_ERROR(cudaStreamSynchronize(mCudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError);
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolverGPU::Reset() {
  return nva2x::FillOnDevice(mPrevWeights, 0.0f, mCudaStream);
}

} // namespace nva2f
