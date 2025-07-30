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
#include "audio2face/internal/blendshape_solver.h"

#include "audio2face/internal/job_runner.h"
#include "audio2face/internal/mask_extraction.h"
#include "audio2face/internal/bvls.h"
#include "audio2face/internal/macros.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/eigen_utils.h"
#include "audio2x/cuda_utils.h"

#include "audio2x/internal/nvtx_trace.h"

#include <future>
#include <thread>
#include <cuda_runtime_api.h>

namespace nva2f {

IBlendshapeSolver::~IBlendshapeSolver() = default;

/**
 * Holds data precomputed by the Prepare method for reuse in the Solve method.
 */
struct BlendshapeSolver::BlendshapeSolverCache {
  size_t numVertexPositions; // masked
  size_t numBlendshapes; // only active poses

  float scaleFactor{1.0f};

  Eigen::MatrixXf AMat;
  nva2x::DeviceTensorFloat neutralPoseDevice;
  nva2x::DeviceTensorFloat deltaPosesDevice;
  int* maskIndicesDevice{nullptr};

  std::vector<int> activeBlendshapeIndices;
  std::vector<std::pair<int, int>> cancelPairs;

  BlendshapeSolverCache(size_t numVertexPositions, size_t numBlendshapes);
  ~BlendshapeSolverCache();
};

/**
 * Temporary storage buffers for intermediate results in the Solve method.
 */
struct BlendshapeSolver::WorkingBuffer {
  nva2x::DeviceTensorFloat targetDeltaDevice;
  nva2x::DeviceTensorFloat BDevice;
  nva2x::HostPinnedTensorFloat B;

  Eigen::VectorXf l;
  Eigen::VectorXf u;
  WorkingBuffer(size_t numVertexPositions, size_t numBlendshapes);
};

BlendshapeSolver::BlendshapeSolverCache::BlendshapeSolverCache(size_t numVertexPositions, size_t numBlendshapes) :
  numVertexPositions(numVertexPositions), numBlendshapes(numBlendshapes) {
  neutralPoseDevice.Allocate(numVertexPositions);
  deltaPosesDevice.Allocate(numVertexPositions * numBlendshapes);
  cudaMalloc((void**)&maskIndicesDevice, numVertexPositions * sizeof(int));
}

BlendshapeSolver::BlendshapeSolverCache::~BlendshapeSolverCache() {
  cudaFree(maskIndicesDevice);
}

BlendshapeSolver::WorkingBuffer::WorkingBuffer(size_t numVertexPositions, size_t numBlendshapes) {
  targetDeltaDevice.Allocate(numVertexPositions);
  BDevice.Allocate(numBlendshapes);
  B.Allocate(numBlendshapes);
  l = Eigen::VectorXf::Zero(numBlendshapes);
  u = Eigen::VectorXf::Ones(numBlendshapes);
}

BlendshapeSolver::BlendshapeSolver() : BlendshapeSolverBase() {
  cudaEventCreateWithFlags(&preBlendshapeSolveCompleted, cudaEventDisableTiming);
}

BlendshapeSolver::~BlendshapeSolver() {
  Wait();
  assert(cudaEventQuery(preBlendshapeSolveCompleted) == cudaSuccess);
  cudaEventDestroy(preBlendshapeSolveCompleted);
};

std::error_code BlendshapeSolver::Cache(PrepareData& data) {
  mCache = std::make_unique<BlendshapeSolverCache>(data.numVertexPositions, data.numBlendshapes);
  mCache->activeBlendshapeIndices = data.activeBlendshapeIndices;
  if (!data.activeVertexPositionIndices.empty()) {
    CUDA_CHECK_ERROR(cudaMemcpyAsync(mCache->maskIndicesDevice, data.activeVertexPositionIndices.data(),
      data.activeVertexPositionIndices.size() * sizeof(int), cudaMemcpyHostToDevice, mCudaStream), nva2x::ErrorCode::eCudaMemcpyHostToDeviceError);
  }
  CHECK_RESULT(nva2x::CopyHostToDevice(mCache->deltaPosesDevice, ToConstView(data.blendshapeDeltas), mCudaStream));
  CHECK_RESULT(nva2x::CopyHostToDevice(mCache->neutralPoseDevice, ToConstView(data.neutralVertexPositions), mCudaStream));
  mCache->cancelPairs = data.cancelPairs;
  mCache->scaleFactor = data.scaleFactor;
  mCache->AMat = data.AMat;
  mWorkingBuffer = std::make_unique<WorkingBuffer>(data.numVertexPositions, data.numBlendshapes);
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolver::Solve(nva2x::DeviceTensorFloatConstView targetPoseDevice, nva2x::HostTensorFloatView outWeights) {
  NVTX_TRACE("BlendshapeSolver::Solve");
  std::promise<std::error_code> promise;
  std::future<std::error_code> future = promise.get_future();
  std::error_code status = SolveAsync(targetPoseDevice, outWeights, [](void* p, std::error_code error) {
    reinterpret_cast<std::promise<std::error_code>*>(p)->set_value(error);
  }, &promise);
  CHECK_NO_ERROR(status);
  return future.get();
}

std::error_code BlendshapeSolver::SolveAsync(
  nva2x::DeviceTensorFloatConstView targetPoseDevice,
  nva2x::DeviceTensorFloatView outWeights) {
  return nva2x::ErrorCode::eUnsupported;
}

std::error_code BlendshapeSolver::SolveAsync(
  nva2x::DeviceTensorFloatConstView targetPoseDevice,
  nva2x::HostTensorFloatView outWeights,
  BlendshapeSolverCallback callback, void* callbackArg) {
  NVTX_TRACE("BlendshapeSolver::SolveAsync");
  if (mJobRunner == nullptr) {
    LOG_ERROR("Prepare: no job runner is set. Please make sure to set it using the SetJobRunner() method.");
    return nva2x::ErrorCode::eNullPointer;
  }
  if (!mPrepared) {
    LOG_ERROR("BlendshapeSolver::Prepare() must be called before calling Solve() or SolveAsync()"
      " or after any setter that invalidates the prepared state");
    return nva2x::ErrorCode::eNotInitialized;
  }

  // Extract variables for easier reference
  const size_t numVertexPositions = mCache->numVertexPositions;
  const size_t numBlendshapes = mCache->numBlendshapes;

  CHECK_ERROR_WITH_MSG(targetPoseDevice.Size() == mRawData.numVertexPositions, "Mismatched size for pose in BlendshapeSolver", nva2x::ErrorCode::eMismatch);
  CHECK_ERROR_WITH_MSG(outWeights.Size() == mRawData.numBlendshapePoses, "Mismatched size for weights in BlendshapeSolver", nva2x::ErrorCode::eMismatch);

  // Compute b on the GPU
  {
    // targetDeltaDevice = masked(targetPoseDevice) - neutralPoseDevice
    {
      if (!mRawData.poseMask.empty()) {
        CHECK_RESULT(nva2f::CopyIndices(mWorkingBuffer->targetDeltaDevice.Data(), targetPoseDevice.Data(),
                 mCache->maskIndicesDevice, numVertexPositions, mCudaStream));
      } else {
        CHECK_RESULT(nva2x::CopyDeviceToDevice(mWorkingBuffer->targetDeltaDevice, targetPoseDevice, mCudaStream));
      }
      const static float alpha = -1.0f;
      cublasSaxpy(mCublasHandle,
                 numVertexPositions,
                 &alpha,
                 mCache->neutralPoseDevice.Data(),
                 1,
                 mWorkingBuffer->targetDeltaDevice.Data(),
                 1);
    }
    // BDevice = d_blendsahepDeltas^T @ targetDeltaDevice
    {
      const static float alpha = 1.0f;
      const static float beta = 0.0f;
      cublasSgemv(mCublasHandle, /*trans=*/CUBLAS_OP_T,
              /*m=*/numVertexPositions, /*n=*/numBlendshapes, &alpha,
              /*A=*/mCache->deltaPosesDevice.Data(), /*lda=*/numVertexPositions,
              /*x=*/mWorkingBuffer->targetDeltaDevice.Data(), /*incx=*/1, &beta,
              /*y=*/mWorkingBuffer->BDevice.Data(), /*incy=*/1);
    }
  }

  {
    std::unique_lock<std::mutex> lock(mBHostPinnedMtx);
    mBHostPinnedCV.wait(lock, [this]() { return this->mBHostPinnedWritable; });
    // this could happen when Prepare() is called and the number of active poses changed.
    CHECK_RESULT(mBHostPinned.Allocate(numBlendshapes));
    mBHostPinnedWritable = false; // prevent another write until the callback finished reading
    CHECK_RESULT(nva2x::CopyDeviceToHost(mBHostPinned, mWorkingBuffer->BDevice, mCudaStream));
    CUDA_CHECK_ERROR(cudaEventRecord(preBlendshapeSolveCompleted, mCudaStream), ErrorCode::eCudaEventRecordError);
  }

  static auto cpuSolve = [](
    BlendshapeSolver* solver, nva2x::HostTensorFloatView outWeightsHost, std::unique_lock<std::mutex>& callbackLock
    ) -> std::error_code {
    // aliases to simplify the code below
    auto  outWeights = outWeightsHost.Data();
    auto& mRawData = solver->mRawData;
    auto& mConfig = solver->mConfig;
    auto& mWorkingBuffer = solver->mWorkingBuffer;
    auto& mCache = solver->mCache;
    auto& mPrevWeights = solver->mPrevWeights;
    const size_t numBlendshapes = mCache->numBlendshapes;
    const auto& params = mConfig.params;

    // synchronize with the cuda stream
    CUDA_CHECK_ERROR(cudaEventSynchronize(solver->preBlendshapeSolveCompleted), nva2x::ErrorCode::eCudaStreamSynchronizeError);

    // Ensure that the previous cpuSolve has completed before proceeding.
    // This is necessary to ensure that the `prevWeights` has been updated.
    std::unique_lock<std::mutex> cpuSolveLock = std::unique_lock<std::mutex>(solver->mCPUSolveMtx);

    std::unique_lock<std::mutex> readHostPinnedTensorLock(solver->mBHostPinnedMtx);
    // reading from the pinned buffer
    Eigen::VectorXf ATb = Eigen::Map<const Eigen::VectorXf>(
      solver->mBHostPinned.Data(), solver->mBHostPinned.Size()
      );
    Eigen::VectorXf b = ATb + params.TemporalReg * mCache->scaleFactor * mPrevWeights;
    solver->mBHostPinnedWritable = true;
    readHostPinnedTensorLock.unlock(); // pinned buffer is read. release for the next solveAsync call to reuse.
    solver->mBHostPinnedCV.notify_one();

    Eigen::VectorXf x(numBlendshapes);
    mWorkingBuffer->u.setConstant(1.0f);
    bvls::solveSystem(x, mCache->AMat, b, mWorkingBuffer->l, mWorkingBuffer->u, params.tolerance);
    // update u based on cancel pairs
    if (mCache->cancelPairs.size() > 0) {
      for(const auto& [i, j] : mCache->cancelPairs) {
        mWorkingBuffer->u[x[i] >= x[j] ? j : i] = 1e-10f;
      }
      bvls::solveSystem(x, mCache->AMat, b, mWorkingBuffer->l, mWorkingBuffer->u, params.tolerance);
    }
    // save prev weights
    mPrevWeights = x;

    // We can actually unlock here since the mPrevWeights is ready for the next iteration
    // But before we do, we lock the callback mutex to make sure the callback is called in order.
    callbackLock.lock();
    cpuSolveLock.unlock();

    // unmap active blendshape weights to full blendshape weights
    int cnt = 0;
    for(int i=0;i<mRawData.numBlendshapePoses;++i) {
      if (mConfig.activePoses[i]) {
        outWeights[i] = x[cnt++];
      } else {
        outWeights[i] = 0.0f;
      }
    }

    if (!mConfig.multipliers.empty()) {
      for(int i=0;i<mRawData.numBlendshapePoses;++i) {
        outWeights[i] *= mConfig.multipliers[i];
      }
    }
    if (!mConfig.offsets.empty()) {
      for(int i=0;i<mRawData.numBlendshapePoses;++i) {
        outWeights[i] += mConfig.offsets[i];
      }
    }
    return nva2x::ErrorCode::eSuccess;
  };

  struct WrapperArg {
    BlendshapeSolver* solver;
    nva2x::HostTensorFloatView outWeights;
    BlendshapeSolverCallback callback;
    void* callbackArg;
  };

  auto cpuSolveWrapper = [](void* arg) {
    std::unique_ptr<WrapperArg> wrapperArg(reinterpret_cast<WrapperArg*>(arg));

    {
      // The callback lock is locked in the cpuSolve function.
      std::unique_lock<std::mutex> callbackLock(wrapperArg->solver->mCallbackOrderMtx, std::defer_lock);
      // Captures any potential error that might be thrown from cpuSolve
      const auto error = cpuSolve(wrapperArg->solver, wrapperArg->outWeights, callbackLock);
      // Invoke the callback to notify the caller regardless of the outcome
      wrapperArg->callback(wrapperArg->callbackArg, error);
    }

    // notify the async task is done by decrement the counter
    auto solver = wrapperArg->solver;
    if (--solver->mAsyncTaskCount == 0) {
      solver->mAsyncTaskCountCV.notify_all();
    }
  };

  WrapperArg* cpuSolveWrapperArg = new WrapperArg{
    this,
    outWeights,
    callback,
    callbackArg
  };

  ++mAsyncTaskCount;
  mJobRunner->Enqueue(cpuSolveWrapper, cpuSolveWrapperArg);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolver::Wait() {
  std::unique_lock<std::mutex> lock(mAsyncTaskCountMtx);
  mAsyncTaskCountCV.wait(lock, [this]() { return this->mAsyncTaskCount == 0; });
  return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolver::Reset() {
  mPrevWeights = Eigen::VectorXf::Zero(mCache->numBlendshapes);
  return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f
