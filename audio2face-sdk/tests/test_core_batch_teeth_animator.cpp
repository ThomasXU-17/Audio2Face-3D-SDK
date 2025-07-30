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
#include "audio2face/internal/multitrack_animator.h"
#include "audio2face/internal/parse_helper.h"
#include "audio2face/internal/model_regression.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"
#include "utils.h"

#include "test_core_batch_teeth_animator_cuda.h"

#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Dense>
#include <random>


namespace test {

using namespace nva2f;


// Interface for benchmarking: data starts on the GPU and ends up on the CPU.
class IMultiTrackAnimatorTeethHost {
public:
  using HostData = IMultiTrackAnimatorTeeth::HostData;
  using Params = IMultiTrackAnimatorTeeth::Params;

  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;
  virtual std::error_code Init(const HostData& data, const Params& params, std::size_t nbTracks) = 0;

  virtual std::error_code ComputeJawTransform(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::HostTensorFloatView outputTransforms
    ) = 0;

  virtual void Destroy() = 0;
};



//
// Reference implementation, uses the single track CPU implementation.
//
class MultiTrackAnimatorTeethHostReference : public IMultiTrackAnimatorTeethHost {
public:
  MultiTrackAnimatorTeethHostReference(bool useDeviceCopy);

  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const HostData& data, const Params& params, std::size_t nbTracks) override; // GPU Async

  std::error_code ComputeJawTransform(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::HostTensorFloatView outputTransforms
    ) override; // GPU Async

  void Destroy() override;

protected:
  cudaStream_t _cudaStream{nullptr};
  std::vector<AnimatorTeeth> _animators;
  std::vector<float> _inputDeltas;
  std::size_t _jawSize{0};

  bool _useDeviceCopy{false};
  nva2x::DeviceTensorFloat _inputDeltasDevice;
};

MultiTrackAnimatorTeethHostReference::MultiTrackAnimatorTeethHostReference(bool useDeviceCopy)
  : _useDeviceCopy(useDeviceCopy) {
}

std::error_code MultiTrackAnimatorTeethHostReference::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeethHostReference::Init(const HostData& data, const Params& params, std::size_t nbTracks) {
  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  _animators.resize(nbTracks);
  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.Init(params), "Unable to initialize animator");
    A2F_CHECK_RESULT_WITH_MSG(animator.SetAnimatorData(data), "Unable to set animator data");
  }
  _inputDeltas.resize(data.neutralJaw.Size() * nbTracks);
  _jawSize = data.neutralJaw.Size();

  if (_useDeviceCopy) {
    A2F_CHECK_RESULT_WITH_MSG(
      _inputDeltasDevice.Allocate(_inputDeltas.size()),
      "Unable to allocate input deltas device"
      );
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeethHostReference::ComputeJawTransform(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::HostTensorFloatView outputTransforms
  ) {
  if (inputDeltasInfo.stride != inputDeltasInfo.size) {
    if (_useDeviceCopy) {
      for (std::size_t trackIndex = 0; trackIndex < _animators.size(); ++trackIndex) {
        const auto outputDevice = _inputDeltasDevice.View(
          inputDeltasInfo.size * trackIndex, inputDeltasInfo.size
          );
        const auto inputDevice = inputDeltas.View(
          inputDeltasInfo.offset +  trackIndex * inputDeltasInfo.stride + inputDeltasInfo.offset, inputDeltasInfo.size
          );
        A2F_CHECK_RESULT_WITH_MSG(
          nva2x::CopyDeviceToDevice(outputDevice, inputDevice, _cudaStream),
          "Unable to copy input to host"
          );
      }
      A2F_CHECK_RESULT_WITH_MSG(
        nva2x::CopyDeviceToHost(nva2x::ToView(_inputDeltas), _inputDeltasDevice, _cudaStream),
        "Unable to copy input to host"
        );
    }
    else {
      for (std::size_t trackIndex = 0; trackIndex < _animators.size(); ++trackIndex) {
        const auto inputHost = nva2x::ToView(_inputDeltas).View(
          inputDeltasInfo.size * trackIndex, inputDeltasInfo.size
          );
        const auto inputDevice = inputDeltas.View(
          inputDeltasInfo.offset + trackIndex * inputDeltasInfo.stride + inputDeltasInfo.offset, inputDeltasInfo.size
          );
        A2F_CHECK_RESULT_WITH_MSG(
          nva2x::CopyDeviceToHost(inputHost, inputDevice, _cudaStream),
          "Unable to copy input to host"
          );
      }
    }
  }
  else {
    // To do a single read, we must not have a stride.
    A2F_CHECK_ERROR_WITH_MSG(
      inputDeltasInfo.stride == inputDeltasInfo.size, "Stride must be equal to size", nva2x::ErrorCode::eInvalidValue
      );
    A2F_CHECK_RESULT_WITH_MSG(
      nva2x::CopyDeviceToHost(nva2x::ToView(_inputDeltas), inputDeltas, _cudaStream),
      "Unable to copy input to host"
      );
  }

  for (std::size_t trackIndex = 0; trackIndex < _animators.size(); ++trackIndex) {
    const auto inputDevice = inputDeltas.View(
      trackIndex * inputDeltasInfo.stride + inputDeltasInfo.offset, inputDeltasInfo.size
      );
    const auto inputHost = nva2x::ToView(_inputDeltas).View(trackIndex * _jawSize, _jawSize);
    const auto outputHost = outputTransforms.View(trackIndex * 16, 16);
    A2F_CHECK_RESULT_WITH_MSG(
      _animators[trackIndex].ComputeJawTransform(outputHost, inputHost), "Unable to compute jaw transform"
      );
  }
  return nva2x::ErrorCode::eSuccess;
}

void MultiTrackAnimatorTeethHostReference::Destroy() {
  delete this;
}




//
// Wrapper implementation, uses a GPU implementation.
//
class MultiTrackAnimatorTeethHostWrapper : public IMultiTrackAnimatorTeethHost {
public:
  MultiTrackAnimatorTeethHostWrapper(nva2x::UniquePtr<test::IMultiTrackAnimatorTeeth> impl);

  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const HostData& data, const Params& params, std::size_t nbTracks) override; // GPU Async

  std::error_code ComputeJawTransform(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::HostTensorFloatView outputTransforms
    ) override; // GPU Async

  void Destroy() override;

protected:
  nva2x::UniquePtr<test::IMultiTrackAnimatorTeeth> _impl;
  cudaStream_t _cudaStream{nullptr};
  nva2x::DeviceTensorFloat _results;
  nva2x::TensorBatchInfo _resultsInfo;
};

MultiTrackAnimatorTeethHostWrapper::MultiTrackAnimatorTeethHostWrapper(
  nva2x::UniquePtr<test::IMultiTrackAnimatorTeeth> impl
  ) : _impl(std::move(impl)) {
}

std::error_code MultiTrackAnimatorTeethHostWrapper::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return _impl->SetCudaStream(cudaStream);
}

std::error_code MultiTrackAnimatorTeethHostWrapper::Init(const HostData& data, const Params& params, std::size_t nbTracks) {
  A2F_CHECK_RESULT_WITH_MSG(
    _results.Allocate(16 * nbTracks),
    "Unable to allocate results"
    );
  _resultsInfo.offset = 0;
  _resultsInfo.size = 16;
  _resultsInfo.stride = 16;
  A2F_CHECK_RESULT_WITH_MSG(_impl->Init(params, nbTracks), "Unable to initialize animator");
  A2F_CHECK_RESULT_WITH_MSG(_impl->SetAnimatorData(data), "Unable to set animator data");
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeethHostWrapper::ComputeJawTransform(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::HostTensorFloatView outputTransforms
  ) {
  A2F_CHECK_RESULT_WITH_MSG(
    _impl->ComputeJawTransform(inputDeltas, inputDeltasInfo, _results, _resultsInfo),
    "Unable to compute jaw transform"
    );
  A2F_CHECK_RESULT_WITH_MSG(
    nva2x::CopyDeviceToHost(outputTransforms, _results, _cudaStream),
    "Unable to copy results to host"
    );
  return nva2x::ErrorCode::eSuccess;
}

void MultiTrackAnimatorTeethHostWrapper::Destroy() {
  delete this;
}


//
// Base implementation, used by the other implementations.
//
class MultiTrackAnimatorTeethBase : public IMultiTrackAnimatorTeeth {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;

  std::error_code SetParameters(std::size_t trackIndex, const Params& params) override;
  const Params* GetParameters(std::size_t trackIndex) const override;

  std::error_code Reset(std::size_t trackIndex) override;
  std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) override;

  void Destroy() override;

protected:
  cudaStream_t _cudaStream{nullptr};
};

std::error_code MultiTrackAnimatorTeethBase::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeethBase::SetParameters(std::size_t trackIndex, const Params& params) {
  return nva2x::ErrorCode::eUnsupported;
}

const MultiTrackAnimatorTeethBase::Params* MultiTrackAnimatorTeethBase::GetParameters(std::size_t trackIndex) const {
  return nullptr;
}

std::error_code MultiTrackAnimatorTeethBase::Reset(std::size_t trackIndex) {
  return nva2x::ErrorCode::eUnsupported;
}

std::error_code MultiTrackAnimatorTeethBase::SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) {
  return nva2x::ErrorCode::eUnsupported;
}

void MultiTrackAnimatorTeethBase::Destroy() {
  delete this;
}




//
// GPU implementation, first attempt.
//
class MultiTrackAnimatorTeethGPU : public MultiTrackAnimatorTeethBase {
public:
  std::error_code Init(const Params& params, std::size_t nbTracks) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

  std::error_code ComputeJawTransform(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputTransforms, const nva2x::TensorBatchInfo& outputTransformsInfo
    ) override; // GPU Async

protected:
  nva2x::DeviceTensorFloat _teethParams;
  std::size_t _teethParamsStride{0};
  nva2x::DeviceTensorFloat _neutralJaw;
  std::size_t _nbPoints{0};
};

std::error_code MultiTrackAnimatorTeethGPU::Init(const Params& params, std::size_t nbTracks) {
  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);

  std::vector<float> paramsHost;
  _teethParamsStride = 3;
  paramsHost.resize(nbTracks * _teethParamsStride);
  for (std::size_t i = 0; i < nbTracks; ++i) {
    paramsHost[i * _teethParamsStride + 0] = params.lowerTeethStrength;
    paramsHost[i * _teethParamsStride + 1] = params.lowerTeethHeightOffset;
    paramsHost[i * _teethParamsStride + 2] = params.lowerTeethDepthOffset;
  }
  A2F_CHECK_RESULT_WITH_MSG(_teethParams.Init(nva2x::ToConstView(paramsHost)), "Unable to initialize teeth params");

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeethGPU::SetAnimatorData(const HostData& data) {
  A2F_CHECK_RESULT_WITH_MSG(_neutralJaw.Init(data.neutralJaw), "Unable to initialize neutral jaw");
  _nbPoints = data.neutralJaw.Size() / 3;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeethGPU::ComputeJawTransform(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputTransforms, const nva2x::TensorBatchInfo& outputTransformsInfo
  ) {
  A2F_CHECK_ERROR_WITH_MSG(inputDeltasInfo.size == _nbPoints * 3, "Input deltas size must be equal to the number of points times 3", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(outputTransformsInfo.size == 16, "Output transforms size must be equal to 16", nva2x::ErrorCode::eInvalidValue);

  const auto nbTracks = _teethParams.Size() / _teethParamsStride;

  A2F_CHECK_RESULT_WITH_MSG(
    test::ComputeRigidXform(
      outputTransforms.Data(), outputTransformsInfo.offset, outputTransformsInfo.stride,
      inputDeltas.Data(), inputDeltasInfo.offset, inputDeltasInfo.stride,
      _neutralJaw.Data(),
      _teethParams.Data(), _teethParamsStride,
      _nbPoints,
      nbTracks,
      _cudaStream
      ), "Unable to compute jaw transform"
    );

  return nva2x::ErrorCode::eSuccess;
}




//
// GPU implementation, use local memory instead of shared.
//
class MultiTrackAnimatorTeethGPULocal : public MultiTrackAnimatorTeethGPU {
public:
  std::error_code ComputeJawTransform(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputTransforms, const nva2x::TensorBatchInfo& outputTransformsInfo
    ) override; // GPU Async

};

std::error_code MultiTrackAnimatorTeethGPULocal::ComputeJawTransform(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputTransforms, const nva2x::TensorBatchInfo& outputTransformsInfo
  ) {
  A2F_CHECK_ERROR_WITH_MSG(inputDeltasInfo.size == _nbPoints * 3, "Input deltas size must be equal to the number of points times 3", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(outputTransformsInfo.size == 16, "Output transforms size must be equal to 16", nva2x::ErrorCode::eInvalidValue);

  const auto nbTracks = _teethParams.Size() / _teethParamsStride;

  A2F_CHECK_RESULT_WITH_MSG(
    test::ComputeRigidXformLocal(
      outputTransforms.Data(), outputTransformsInfo.offset, outputTransformsInfo.stride,
      inputDeltas.Data(), inputDeltasInfo.offset, inputDeltasInfo.stride,
      _neutralJaw.Data(),
      _teethParams.Data(), _teethParamsStride,
      _nbPoints,
      nbTracks,
      _cudaStream
      ), "Unable to compute jaw transform"
    );

  return nva2x::ErrorCode::eSuccess;
}




//
// GPU implementation, try to use as much parallelism as possible on the GPU.
//
class MultiTrackAnimatorTeethGPUParallel : public MultiTrackAnimatorTeethBase {
public:
  std::error_code Init(const Params& params, std::size_t nbTracks) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

  std::error_code ComputeJawTransform(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputTransforms, const nva2x::TensorBatchInfo& outputTransformsInfo
    ) override; // GPU Async

protected:
  nva2x::DeviceTensorFloat _teethParams;
  std::size_t _teethParamsStride{0};
  nva2x::DeviceTensorFloat _neutralJaw;
  std::size_t _nbPoints{0};
};

std::error_code MultiTrackAnimatorTeethGPUParallel::Init(const Params& params, std::size_t nbTracks) {
  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);

  std::vector<float> paramsHost;
  _teethParamsStride = 4;
  paramsHost.resize(nbTracks * _teethParamsStride);
  for (std::size_t i = 0; i < nbTracks; ++i) {
    paramsHost[i * _teethParamsStride + 0] = params.lowerTeethStrength;
    paramsHost[i * _teethParamsStride + 1] = 0.0f;
    paramsHost[i * _teethParamsStride + 2] = params.lowerTeethHeightOffset;
    paramsHost[i * _teethParamsStride + 3] = params.lowerTeethDepthOffset;
  }
  A2F_CHECK_RESULT_WITH_MSG(_teethParams.Init(nva2x::ToConstView(paramsHost)), "Unable to initialize teeth params");

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeethGPUParallel::SetAnimatorData(const HostData& data) {
  _nbPoints = data.neutralJaw.Size() / 3;

  // Add the neutral jaw mean so it's available.
  std::vector<float> neutralJawHost(data.neutralJaw.Data(), data.neutralJaw.Data() + data.neutralJaw.Size());
  float mean[3] = {0.0f, 0.0f, 0.0f};
  for (std::size_t i = 0; i < _nbPoints; ++i) {
    mean[0] += neutralJawHost[i * 3 + 0];
    mean[1] += neutralJawHost[i * 3 + 1];
    mean[2] += neutralJawHost[i * 3 + 2];
  }
  for (std::size_t i = 0; i < 3; ++i) {
    neutralJawHost.emplace_back(mean[i] / _nbPoints);
  }
  A2F_CHECK_RESULT_WITH_MSG(_neutralJaw.Init(nva2x::ToConstView(neutralJawHost)), "Unable to initialize neutral jaw");

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeethGPUParallel::ComputeJawTransform(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputTransforms, const nva2x::TensorBatchInfo& outputTransformsInfo
  ) {
  A2F_CHECK_ERROR_WITH_MSG(inputDeltasInfo.size == _nbPoints * 3, "Input deltas size must be equal to the number of points times 3", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(outputTransformsInfo.size == 16, "Output transforms size must be equal to 16", nva2x::ErrorCode::eInvalidValue);

  const auto nbTracks = _teethParams.Size() / _teethParamsStride;

  A2F_CHECK_RESULT_WITH_MSG(
    test::ComputeRigidXformParallel(
      outputTransforms.Data(), outputTransformsInfo.offset, outputTransformsInfo.stride,
      inputDeltas.Data(), inputDeltasInfo.offset, inputDeltasInfo.stride,
      _neutralJaw.Data(),
      _teethParams.Data(), _teethParamsStride,
      _nbPoints,
      nbTracks,
      _cudaStream
      ), "Unable to compute jaw transform"
    );

  return nva2x::ErrorCode::eSuccess;
}




//
// GPU implementation, empty one to have a high water mark.
//
class MultiTrackAnimatorTeethGPUEmpty : public MultiTrackAnimatorTeethGPUParallel {
public:
  std::error_code ComputeJawTransform(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputTransforms, const nva2x::TensorBatchInfo& outputTransformsInfo
    ) override; // GPU Async

};

std::error_code MultiTrackAnimatorTeethGPUEmpty::ComputeJawTransform(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputTransforms, const nva2x::TensorBatchInfo& outputTransformsInfo
  ) {
  A2F_CHECK_ERROR_WITH_MSG(inputDeltasInfo.size == _nbPoints * 3, "Input deltas size must be equal to the number of points times 3", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(outputTransformsInfo.size == 16, "Output transforms size must be equal to 16", nva2x::ErrorCode::eInvalidValue);

  const auto nbTracks = _teethParams.Size() / _teethParamsStride;

  A2F_CHECK_RESULT_WITH_MSG(
    test::ComputeRigidXformEmpty(
      outputTransforms.Data(), outputTransformsInfo.offset, outputTransformsInfo.stride,
      inputDeltas.Data(), inputDeltasInfo.offset, inputDeltasInfo.stride,
      _neutralJaw.Data(),
      _teethParams.Data(), _teethParamsStride,
      _nbPoints,
      nbTracks,
      _cudaStream
      ), "Unable to compute jaw transform"
    );

  return nva2x::ErrorCode::eSuccess;
}


} // namespace test




namespace {

using device_creator_func_t = std::function<test::IMultiTrackAnimatorTeeth*()>;
static const std::vector<std::pair<const char*, device_creator_func_t>> kDeviceImplementations {
  {"GPU", []() -> test::IMultiTrackAnimatorTeeth* { return new test::MultiTrackAnimatorTeethGPU; }},
  {"GPU Local", []() -> test::IMultiTrackAnimatorTeeth* { return new test::MultiTrackAnimatorTeethGPULocal; }},
  {"GPU Parallel", []() -> test::IMultiTrackAnimatorTeeth* { return new test::MultiTrackAnimatorTeethGPUParallel; }},
  {"Final", []() -> test::IMultiTrackAnimatorTeeth* { return nva2f::CreateMultiTrackAnimatorTeeth_INTERNAL(); }},
  {"Empty", []() -> test::IMultiTrackAnimatorTeeth* { return new test::MultiTrackAnimatorTeethGPUEmpty; }},
};

using host_creator_func_t = std::function<test::IMultiTrackAnimatorTeethHost*()>;
static const std::vector<std::pair<const char*, host_creator_func_t>> kHostImplementations = []() {
  std::vector<std::pair<const char*, host_creator_func_t>> implementations {
    {"Reference (useDeviceCopy = false)", []() -> test::IMultiTrackAnimatorTeethHost* { return new test::MultiTrackAnimatorTeethHostReference(false); }},
    {"Reference (useDeviceCopy = true)", []() -> test::IMultiTrackAnimatorTeethHost* { return new test::MultiTrackAnimatorTeethHostReference(true); }},
  };

  for (const auto& deviceImplementation : kDeviceImplementations) {
    const auto wrapper = [creator = deviceImplementation.second]() -> test::IMultiTrackAnimatorTeethHost* {
      return new test::MultiTrackAnimatorTeethHostWrapper(nva2x::UniquePtr<test::IMultiTrackAnimatorTeeth>(creator()));
    };
    implementations.emplace_back(deviceImplementation.first, std::move(wrapper));
  }
  return implementations;
}();




struct BatchData {
  std::size_t nbTracks;
  nva2x::CudaStream cudaStream;
  nva2x::UniquePtr<nva2f::IRegressionModel::IGeometryModelInfo> modelInfo;
  nva2f::IMultiTrackAnimatorTeeth::HostData initData;
  nva2f::IMultiTrackAnimatorTeeth::Params params;
  std::size_t nbPoses{100};
  std::size_t nbPoints;

  std::vector<float> targetPoses;
  std::vector<float> targetDeltasHost;
  nva2x::DeviceTensorFloat targetDeltas;

  nva2x::DeviceTensorFloat resultsBuffers;
  nva2x::TensorBatchInfo sourceInfo;
  nva2x::TensorBatchInfo resultInfo;
  nva2x::HostPinnedTensorFloat resultsBuffersHost;
};

BatchData BuildTestData(std::size_t nbTracks, bool useStride = false) {
  BatchData batchData;

  batchData.nbTracks = nbTracks;
  EXPECT_TRUE(!batchData.cudaStream.Init());

  constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
  batchData.modelInfo = nva2x::ToUniquePtr(nva2f::ReadRegressionModelInfo_INTERNAL(modelPath));
  EXPECT_TRUE(batchData.modelInfo);

  batchData.initData.neutralJaw = batchData.modelInfo->GetAnimatorData().GetAnimatorData().teeth.neutralJaw;
  assert(batchData.initData.neutralJaw.Size() % 3 == 0);
  batchData.nbPoints = batchData.initData.neutralJaw.Size() / 3;
  batchData.params = batchData.modelInfo->GetAnimatorParams().teeth;
  batchData.params.lowerTeethStrength = 1.0f;
  batchData.params.lowerTeethHeightOffset = 0.0f;
  batchData.params.lowerTeethDepthOffset = 0.0f;

  // Generate target poses.
  const int seed = static_cast<unsigned int>(time(NULL));
  std::cout << "Current srand seed: " << seed << std::endl;
  std::srand(seed); // make random inputs reproducible

  const auto nbTransforms = batchData.nbPoses * nbTracks;
  batchData.targetPoses.reserve(nbTransforms * batchData.initData.neutralJaw.Size());

  const auto& networkInfo = batchData.modelInfo->GetNetworkInfo().GetNetworkInfo();
  batchData.targetDeltasHost.reserve(nbTransforms * batchData.initData.neutralJaw.Size());
  for (std::size_t i = 0; i < nbTransforms; ++i) {
    auto generateRandomRotation = []() {
      // Similar to Eigen::Quaternion::UnitRandom(), but we control the random number generator.
      const auto u1 = (static_cast<float>(rand()) / RAND_MAX);
      const auto u2 = (static_cast<float>(rand()) / RAND_MAX) * 2 * 3.14159265358979323846264338328f;
      const auto u3 = (static_cast<float>(rand()) / RAND_MAX) * 2 * 3.14159265358979323846264338328f;
      const auto a = sqrtf(1 - u1);
      const auto b = sqrtf(u1);
      return Eigen::Quaternionf(a * sinf(u2), a * cosf(u2), b * sinf(u3), b * cosf(u3));
    };
    auto generateRandomTranslation = []() {
      const auto u1 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
      const auto u2 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
      const auto u3 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
      return Eigen::Vector3f(u1, u2, u3);
    };

    const auto rotation = generateRandomRotation();
    const auto translation = generateRandomTranslation();

    for (std::size_t i = 0; i < batchData.nbPoints; ++i) {
      const auto point = Eigen::Vector3f::Map(batchData.initData.neutralJaw.Data() + i * 3, 3);
      const Eigen::Vector3f transformedPoint = rotation * point + translation;
      batchData.targetPoses.push_back(transformedPoint[0]);
      batchData.targetPoses.push_back(transformedPoint[1]);
      batchData.targetPoses.push_back(transformedPoint[2]);

      const Eigen::Vector3f delta = transformedPoint - point;
      batchData.targetDeltasHost.push_back(delta[0]);
      batchData.targetDeltasHost.push_back(delta[1]);
      batchData.targetDeltasHost.push_back(delta[2]);
    }
  }

  const std::size_t sourceStride = useStride
    ? networkInfo.resultSkinSize + networkInfo.resultTongueSize + networkInfo.resultJawSize + networkInfo.resultEyesSize
    : batchData.initData.neutralJaw.Size();
  if (useStride) {
    EXPECT_TRUE(!batchData.targetDeltas.Allocate(nbTransforms * sourceStride));
    for (std::size_t i = 0; i < nbTransforms; ++i) {
      const auto source = nva2x::ToView(batchData.targetDeltasHost).View(
        i * batchData.initData.neutralJaw.Size(), batchData.initData.neutralJaw.Size()
        );
      const auto target = batchData.targetDeltas.View(
        i * sourceStride, batchData.initData.neutralJaw.Size()
        );
      EXPECT_TRUE(!nva2x::CopyHostToDevice(target, source, batchData.cudaStream.Data()));
    }
  }
  else {
    EXPECT_TRUE(!batchData.targetDeltas.Init(nva2x::ToConstView(batchData.targetDeltasHost)));
  }

  const auto resultsSize = 16 * nbTransforms;
  EXPECT_TRUE(!batchData.resultsBuffers.Allocate(resultsSize));

  batchData.sourceInfo.offset = 0;
  batchData.sourceInfo.size = batchData.initData.neutralJaw.Size();
  batchData.sourceInfo.stride = sourceStride;

  batchData.resultInfo.offset = 0;
  batchData.resultInfo.size = 16;
  batchData.resultInfo.stride = 16;

  EXPECT_TRUE(!batchData.resultsBuffersHost.Allocate(resultsSize));

  EXPECT_TRUE(!cudaDeviceSynchronize());

  return batchData;
}

}




TEST(TestCoreBatchTeethAnimator, Correctness) {
  const auto nbTracks = 10;
  BatchData batchData = BuildTestData(nbTracks);

  // Test implementations.
  auto implementations = kHostImplementations;
  // Ignore the last empty implementation
  implementations.pop_back();
  for (const auto& implementation : implementations) {
    std::cout << "Testing \"" << implementation.first << "\" implementation..." << std::endl;

    ASSERT_TRUE(!nva2x::FillOnDevice(batchData.resultsBuffers, -1.0f, batchData.cudaStream.Data()));
    ASSERT_TRUE(!nva2x::FillOnHost(batchData.resultsBuffersHost, -1.0f));
    ASSERT_TRUE(!batchData.cudaStream.Synchronize());

    const auto animator = nva2x::ToUniquePtr(implementation.second());
    ASSERT_TRUE(!animator->SetCudaStream(batchData.cudaStream.Data()));
    ASSERT_TRUE(!animator->Init(batchData.initData, batchData.params, batchData.nbTracks));

    for (std::size_t pose = 0; pose < batchData.nbPoses; ++pose) {
      const auto source = batchData.targetDeltas.View(
        batchData.sourceInfo.offset + pose * batchData.sourceInfo.stride * nbTracks,
        batchData.sourceInfo.stride * nbTracks
        );
      const auto results = batchData.resultsBuffersHost.View(
        pose * 16 * nbTracks, 16 * nbTracks
        );
      ASSERT_TRUE(!animator->ComputeJawTransform(source, batchData.sourceInfo, results));
    }

    ASSERT_TRUE(!batchData.cudaStream.Synchronize());

    // We reapply the transform to the target poses to get the expected results.
    for (std::size_t pose = 0; pose < batchData.nbPoses * nbTracks; ++pose) {
      const auto transform = Eigen::Matrix4f::Map(batchData.resultsBuffersHost.Data() + pose * 16, 4, 4);
      const auto targetPose = batchData.targetPoses.data() + pose * batchData.initData.neutralJaw.Size();

      for (std::size_t i = 0; i < batchData.nbPoints; ++i) {
        const auto point = Eigen::Vector3f::Map(batchData.initData.neutralJaw.Data() + i * 3, 3);
        const auto expectedPoint = Eigen::Vector3f::Map(targetPose + i * 3, 3);

        const Eigen::Vector3f transformedPoint = (transform * point.homogeneous()).hnormalized();

        ASSERT_NEAR(transformedPoint[0], expectedPoint[0], 1e-4f) << "pose " << pose << " point " << i;
        ASSERT_NEAR(transformedPoint[1], expectedPoint[1], 1e-4f) << "pose " << pose << " point " << i;
        ASSERT_NEAR(transformedPoint[2], expectedPoint[2], 1e-4f) << "pose " << pose << " point " << i;
      }
    }
  }
}

TEST(TestCoreBatchTeethAnimator, PerformanceHost) {
  using clock_t = std::chrono::steady_clock;
  using time_point_t = clock_t::time_point;
  using duration_t = clock_t::duration;

  for (const auto nbTracks : {1, 8, 16, 128}) {
    std::cout << "Benchmarking for " << nbTracks << " tracks..." << std::endl;
    BatchData batchData = BuildTestData(nbTracks);

    // Benchmark implementations.
    for (const auto& implementation : kHostImplementations) {
      std::cout << "  Benchmarking \"" << implementation.first << "\" implementation..." << std::endl;

      const auto animator = nva2x::ToUniquePtr(implementation.second());
      ASSERT_TRUE(!animator->SetCudaStream(batchData.cudaStream.Data()));
      ASSERT_TRUE(!animator->Init(batchData.initData, batchData.params, batchData.nbTracks));

      const std::size_t kWarmupIterations = 10;
      const std::size_t kNbBenchmarkIterations = 100;

      for (std::size_t i = 0; i < kWarmupIterations; ++i) {
        ASSERT_TRUE(!animator->ComputeJawTransform(
          batchData.targetDeltas.View(0, batchData.targetDeltas.Size() / batchData.nbPoses),
          batchData.sourceInfo,
          batchData.resultsBuffersHost.View(0, batchData.resultsBuffers.Size() / batchData.nbPoses)
          ));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      }

      duration_t totalTime = duration_t::zero();
      duration_t minTime = duration_t::max();
      for (std::size_t i = 0; i < kNbBenchmarkIterations; ++i) {
        const auto startTime = clock_t::now();
        ASSERT_TRUE(!animator->ComputeJawTransform(
          batchData.targetDeltas.View(0, batchData.targetDeltas.Size() / batchData.nbPoses),
          batchData.sourceInfo,
          batchData.resultsBuffersHost.View(0, batchData.resultsBuffers.Size() / batchData.nbPoses)
          ));

        // We sync to have a "fair" comparison of getting the results.
        ASSERT_EQ(cudaStreamSynchronize(batchData.cudaStream.Data()), cudaSuccess);

        const auto endTime = clock_t::now();
        const auto duration = endTime - startTime;
        totalTime += duration;
        minTime = std::min(minTime, duration);
      }

      const auto averageTimeMs = std::chrono::duration_cast<std::chrono::nanoseconds>(totalTime).count() / 1e6f / kNbBenchmarkIterations;
      const auto minTimeMs = std::chrono::duration_cast<std::chrono::nanoseconds>(minTime).count() / 1e6f;
      std::cout << "    Avg: " << averageTimeMs << " ms , min: " << minTimeMs << " ms" << std::endl;
    }
  }
}

TEST(TestCoreBatchTeethAnimator, PerformanceDevice) {
  cudaEvent_t start;
  cudaEvent_t end;
  ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);

  for (const auto nbTracks : {1, 8, 16, 128}) {
    std::cout << "Benchmarking for " << nbTracks << " tracks..." << std::endl;
    BatchData batchData = BuildTestData(nbTracks);

    // Benchmark implementations.
    for (const auto& implementation : kDeviceImplementations) {
      std::cout << "  Benchmarking \"" << implementation.first << "\" implementation..." << std::endl;

      const auto animator = nva2x::ToUniquePtr(implementation.second());
      ASSERT_TRUE(!animator->SetCudaStream(batchData.cudaStream.Data()));
      ASSERT_TRUE(!animator->Init(batchData.params, batchData.nbTracks));
      ASSERT_TRUE(!animator->SetAnimatorData(batchData.initData));

      const std::size_t kWarmupIterations = 10;
      const std::size_t kNbBenchmarkIterations = 100;

      for (std::size_t i = 0; i < kWarmupIterations; ++i) {
        ASSERT_TRUE(!animator->ComputeJawTransform(
          batchData.targetDeltas.View(0, batchData.targetDeltas.Size() / batchData.nbPoses),
          batchData.sourceInfo,
          batchData.resultsBuffers.View(0, batchData.resultsBuffers.Size() / batchData.nbPoses),
          batchData.resultInfo
          ));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      }

      float totalTime = 0.0f;
      float minTime = std::numeric_limits<float>::max();
      for (std::size_t i = 0; i < kNbBenchmarkIterations; ++i) {
        ASSERT_EQ(cudaEventRecord(start, batchData.cudaStream.Data()), cudaSuccess);
        ASSERT_TRUE(!animator->ComputeJawTransform(
          batchData.targetDeltas.View(0, batchData.targetDeltas.Size() / batchData.nbPoses),
          batchData.sourceInfo,
          batchData.resultsBuffers.View(0, batchData.resultsBuffers.Size() / batchData.nbPoses),
          batchData.resultInfo
          ));
        ASSERT_EQ(cudaEventRecord(end, batchData.cudaStream.Data()), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(end), cudaSuccess);
        float milliseconds = 0;
        ASSERT_EQ(cudaEventElapsedTime(&milliseconds, start, end), cudaSuccess);
        totalTime += milliseconds;
        minTime = std::min(minTime, milliseconds);
      }

      const auto averageTime = totalTime / kNbBenchmarkIterations;
      std::cout << "    Avg: " << averageTime << " ms , min: " << minTime << " ms" << std::endl;
    }
  }
}
