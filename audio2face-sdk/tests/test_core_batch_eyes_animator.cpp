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

#include "test_core_batch_eyes_animator_cuda.h"

#include <gtest/gtest.h>


namespace test {

using namespace nva2f;


// Interface for benchmarking: data starts on the GPU and ends up on the CPU.
class IMultiTrackAnimatorEyesHost {
public:
  using HostData = IMultiTrackAnimatorEyes::HostData;
  using Params = IMultiTrackAnimatorEyes::Params;

  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;
  virtual std::error_code Init(const HostData& data, float dt,const Params& params, std::size_t nbTracks) = 0;

  virtual std::error_code ComputeEyesRotation(
    nva2x::DeviceTensorFloatConstView inputEyesRotationResult, const nva2x::TensorBatchInfo& inputEyesRotationResultInfo,
    nva2x::HostTensorFloatView outputEyesRotation
    ) = 0;

  virtual void Destroy() = 0;
};



//
// Reference implementation, uses the single track CPU implementation.
//
class MultiTrackAnimatorEyesHostReference : public IMultiTrackAnimatorEyesHost {
public:
  MultiTrackAnimatorEyesHostReference(bool useDeviceCopy);

  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const HostData& data, float dt, const Params& params, std::size_t nbTracks) override; // GPU Async

  std::error_code ComputeEyesRotation(
    nva2x::DeviceTensorFloatConstView inputEyesRotationResult, const nva2x::TensorBatchInfo& inputEyesRotationResultInfo,
    nva2x::HostTensorFloatView outputEyesRotation
    ) override; // GPU Async

  void Destroy() override;

protected:
  cudaStream_t _cudaStream{nullptr};
  std::vector<AnimatorEyes> _animators;
  float _dt{0.0f};
  std::vector<float> _inputEyesRotationResult;

  bool _useDeviceCopy{false};
  nva2x::DeviceTensorFloat _inputEyesRotationResultDevice;
};

MultiTrackAnimatorEyesHostReference::MultiTrackAnimatorEyesHostReference(bool useDeviceCopy)
  : _useDeviceCopy(useDeviceCopy) {
}

std::error_code MultiTrackAnimatorEyesHostReference::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorEyesHostReference::Init(const HostData& data, float dt, const Params& params, std::size_t nbTracks) {
  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  _animators.resize(nbTracks);
  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.Init(params), "Unable to initialize animator");
    A2F_CHECK_RESULT_WITH_MSG(animator.SetAnimatorData(data), "Unable to set animator data");
  }
  _dt = dt;
  _inputEyesRotationResult.resize(4 * nbTracks);

  if (_useDeviceCopy) {
    A2F_CHECK_RESULT_WITH_MSG(
      _inputEyesRotationResultDevice.Allocate(_inputEyesRotationResult.size()),
      "Unable to allocate input eyes rotation result device"
      );
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorEyesHostReference::ComputeEyesRotation(
  nva2x::DeviceTensorFloatConstView inputEyesRotationResult, const nva2x::TensorBatchInfo& inputEyesRotationResultInfo,
  nva2x::HostTensorFloatView outputEyesRotation
  ) {
  if (inputEyesRotationResultInfo.stride != inputEyesRotationResultInfo.size) {
    if (_useDeviceCopy) {
      for (std::size_t trackIndex = 0; trackIndex < _animators.size(); ++trackIndex) {
        const auto outputDevice = _inputEyesRotationResultDevice.View(
          inputEyesRotationResultInfo.size * trackIndex, inputEyesRotationResultInfo.size
          );
        const auto inputDevice = inputEyesRotationResult.View(
          inputEyesRotationResultInfo.offset + trackIndex * inputEyesRotationResultInfo.stride + inputEyesRotationResultInfo.offset, inputEyesRotationResultInfo.size
          );
        A2F_CHECK_RESULT_WITH_MSG(
          nva2x::CopyDeviceToDevice(outputDevice, inputDevice, _cudaStream),
          "Unable to copy input to host"
          );
      }
      A2F_CHECK_RESULT_WITH_MSG(
        nva2x::CopyDeviceToHost(nva2x::ToView(_inputEyesRotationResult), _inputEyesRotationResultDevice, _cudaStream),
        "Unable to copy input to host"
        );
    }
    else {
      for (std::size_t trackIndex = 0; trackIndex < _animators.size(); ++trackIndex) {
        const auto inputHost = nva2x::ToView(_inputEyesRotationResult).View(
          inputEyesRotationResultInfo.size * trackIndex, inputEyesRotationResultInfo.size
          );
        const auto inputDevice = inputEyesRotationResult.View(
          inputEyesRotationResultInfo.offset + trackIndex * inputEyesRotationResultInfo.stride + inputEyesRotationResultInfo.offset, inputEyesRotationResultInfo.size
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
      inputEyesRotationResultInfo.stride == inputEyesRotationResultInfo.size, "Stride must be equal to size", nva2x::ErrorCode::eInvalidValue
      );
    A2F_CHECK_RESULT_WITH_MSG(
      nva2x::CopyDeviceToHost(nva2x::ToView(_inputEyesRotationResult), inputEyesRotationResult, _cudaStream),
      "Unable to copy input to host"
      );
  }

  for (std::size_t trackIndex = 0; trackIndex < _animators.size(); ++trackIndex) {
    const auto inputDevice = inputEyesRotationResult.View(
      trackIndex * inputEyesRotationResultInfo.stride + inputEyesRotationResultInfo.offset, inputEyesRotationResultInfo.size
      );
    const auto inputHost = nva2x::ToView(_inputEyesRotationResult).View(trackIndex * 4, 4);
    const auto outputHostRight = outputEyesRotation.View(trackIndex * 6 + 0, 3);
    const auto outputHostLeft = outputEyesRotation.View(trackIndex * 6 + 3, 3);
    A2F_CHECK_RESULT_WITH_MSG(
      _animators[trackIndex].ComputeEyesRotation(outputHostRight, outputHostLeft, inputHost),
      "Unable to compute eyes rotation"
      );
    A2F_CHECK_RESULT_WITH_MSG(
      _animators[trackIndex].IncrementLiveTime(_dt),
      "Unable to increment live time"
      );
  }
  return nva2x::ErrorCode::eSuccess;
}

void MultiTrackAnimatorEyesHostReference::Destroy() {
  delete this;
}




//
// Wrapper implementation, uses a GPU implementation.
//
class MultiTrackAnimatorEyesHostWrapper : public IMultiTrackAnimatorEyesHost {
public:
  MultiTrackAnimatorEyesHostWrapper(nva2x::UniquePtr<test::IMultiTrackAnimatorEyes> impl);

  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const HostData& data, float dt, const Params& params, std::size_t nbTracks) override; // GPU Async

  std::error_code ComputeEyesRotation(
    nva2x::DeviceTensorFloatConstView inputEyesRotationResult, const nva2x::TensorBatchInfo& inputEyesRotationResultInfo,
    nva2x::HostTensorFloatView outputEyesRotation
    ) override; // GPU Async

  void Destroy() override;

protected:
  nva2x::UniquePtr<test::IMultiTrackAnimatorEyes> _impl;
  cudaStream_t _cudaStream{nullptr};
  nva2x::DeviceTensorFloat _results;
  nva2x::TensorBatchInfo _resultsInfo;
};

MultiTrackAnimatorEyesHostWrapper::MultiTrackAnimatorEyesHostWrapper(
  nva2x::UniquePtr<test::IMultiTrackAnimatorEyes> impl
  ) : _impl(std::move(impl)) {
}

std::error_code MultiTrackAnimatorEyesHostWrapper::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return _impl->SetCudaStream(cudaStream);
}

std::error_code MultiTrackAnimatorEyesHostWrapper::Init(const HostData& data, float dt, const Params& params, std::size_t nbTracks) {
  A2F_CHECK_RESULT_WITH_MSG(
    _results.Allocate(6 * nbTracks),
    "Unable to allocate results"
    );
  _resultsInfo.offset = 0;
  _resultsInfo.size = 6;
  _resultsInfo.stride = 6;
  A2F_CHECK_RESULT_WITH_MSG(_impl->Init(params, nbTracks), "Unable to initialize animator");
  A2F_CHECK_RESULT_WITH_MSG(_impl->SetAnimatorData(data, dt), "Unable to set animator data");
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorEyesHostWrapper::ComputeEyesRotation(
  nva2x::DeviceTensorFloatConstView inputEyesRotationResult, const nva2x::TensorBatchInfo& inputEyesRotationResultInfo,
  nva2x::HostTensorFloatView outputEyesRotation
  ) {
  A2F_CHECK_RESULT_WITH_MSG(
    _impl->ComputeEyesRotation(inputEyesRotationResult, inputEyesRotationResultInfo, _results, _resultsInfo),
    "Unable to compute eyes rotation"
    );
  A2F_CHECK_RESULT_WITH_MSG(
    nva2x::CopyDeviceToHost(outputEyesRotation, _results, _cudaStream),
    "Unable to copy results to host"
    );
  return nva2x::ErrorCode::eSuccess;
}

void MultiTrackAnimatorEyesHostWrapper::Destroy() {
  delete this;
}


//
// Base implementation, used by the other implementations.
//
class MultiTrackAnimatorEyesBase : public IMultiTrackAnimatorEyes {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;

  std::error_code SetParameters(std::size_t trackIndex, const Params& params) override;
  const Params* GetParameters(std::size_t trackIndex) const override;

  std::error_code Reset(std::size_t trackIndex) override;
  std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) override;
  std::error_code SetLiveTime(std::size_t trackIndex, float liveTime) override;

  void Destroy() override;

protected:
  cudaStream_t _cudaStream{nullptr};
};

std::error_code MultiTrackAnimatorEyesBase::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorEyesBase::SetParameters(std::size_t trackIndex, const Params& params) {
  return nva2x::ErrorCode::eUnsupported;
}

const MultiTrackAnimatorEyesBase::Params* MultiTrackAnimatorEyesBase::GetParameters(std::size_t trackIndex) const {
  return nullptr;
}

std::error_code MultiTrackAnimatorEyesBase::Reset(std::size_t trackIndex) {
  return nva2x::ErrorCode::eUnsupported;
}

std::error_code MultiTrackAnimatorEyesBase::SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) {
  return nva2x::ErrorCode::eUnsupported;
}

std::error_code MultiTrackAnimatorEyesBase::SetLiveTime(std::size_t trackIndex, float liveTime) {
  return nva2x::ErrorCode::eUnsupported;
}

void MultiTrackAnimatorEyesBase::Destroy() {
  delete this;
}




//
// GPU implementation, first attempt.
//
class MultiTrackAnimatorEyesGPU : public MultiTrackAnimatorEyesBase {
public:
  std::error_code Init(const Params& params, std::size_t nbTracks) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data, float dt) override; // GPU Async

  std::error_code ComputeEyesRotation(
    nva2x::DeviceTensorFloatConstView inputEyesRotationResult, const nva2x::TensorBatchInfo& inputEyesRotationResultInfo,
    nva2x::DeviceTensorFloatView outputEyesRotation, const nva2x::TensorBatchInfo& outputEyesRotationInfo
    ) override; // GPU Async

protected:
  nva2x::DeviceTensorFloat _eyesParams;
  std::size_t _eyesParamsStride{0};
  nva2x::DeviceTensorFloat _saccadeRot;
  nva2x::DeviceTensorFloat _liveTime;
  float _dt{0.0f};
};

std::error_code MultiTrackAnimatorEyesGPU::Init(const Params& params, std::size_t nbTracks) {
  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);

  std::vector<float> paramsHost;
  _eyesParamsStride = 7;
  paramsHost.resize(nbTracks * _eyesParamsStride);
  for (std::size_t i = 0; i < nbTracks; ++i) {
    paramsHost[i * _eyesParamsStride + 0] = params.eyeballsStrength;
    paramsHost[i * _eyesParamsStride + 1] = params.saccadeStrength;
    paramsHost[i * _eyesParamsStride + 2] = params.rightEyeballRotationOffsetX;
    paramsHost[i * _eyesParamsStride + 3] = params.rightEyeballRotationOffsetY;
    paramsHost[i * _eyesParamsStride + 4] = params.leftEyeballRotationOffsetX;
    paramsHost[i * _eyesParamsStride + 5] = params.leftEyeballRotationOffsetY;
    paramsHost[i * _eyesParamsStride + 6] = params.saccadeSeed;
  }
  A2F_CHECK_RESULT_WITH_MSG(_eyesParams.Init(nva2x::ToConstView(paramsHost)), "Unable to initialize eyes params");
  A2F_CHECK_RESULT_WITH_MSG(_liveTime.Allocate(nbTracks), "Unable to initialize live time");
  A2F_CHECK_RESULT_WITH_MSG(nva2x::FillOnDevice(_liveTime, 0.0f, _cudaStream), "Unable to fill live time");

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorEyesGPU::SetAnimatorData(const HostData& data, float dt) {
  A2F_CHECK_ERROR_WITH_MSG(data.saccadeRot.Data() != nullptr, "Saccade rotation matrix must not be null", nva2x::ErrorCode::eNullPointer);
  A2F_CHECK_ERROR_WITH_MSG(data.saccadeRot.Size() > 0, "Saccade rotation matrix must not be empty", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_RESULT_WITH_MSG(_saccadeRot.Init(data.saccadeRot), "Unable to initialize saccade rot");
  _dt = dt;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorEyesGPU::ComputeEyesRotation(
  nva2x::DeviceTensorFloatConstView inputEyesRotationResult, const nva2x::TensorBatchInfo& inputEyesRotationResultInfo,
  nva2x::DeviceTensorFloatView outputEyesRotation, const nva2x::TensorBatchInfo& outputEyesRotationInfo
  ) {
  A2F_CHECK_ERROR_WITH_MSG(inputEyesRotationResultInfo.size == 4, "Input eyes rotation result size must be equal to 4", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(outputEyesRotationInfo.size == 6, "Output eyes rotation size must be equal to 6", nva2x::ErrorCode::eInvalidValue);

  const auto nbTracks = _eyesParams.Size() / _eyesParamsStride;

  A2F_CHECK_RESULT_WITH_MSG(
    test::ComputeEyesRotation(
      outputEyesRotation.Data(), outputEyesRotationInfo.offset, outputEyesRotationInfo.stride,
      inputEyesRotationResult.Data(), inputEyesRotationResultInfo.offset, inputEyesRotationResultInfo.stride,
      _eyesParams.Data(), _eyesParamsStride,
      _saccadeRot.Data(), _saccadeRot.Size(),
      _dt,
      _liveTime.Data(),
      nbTracks,
      _cudaStream
      ), "Unable to compute eyes rotation"
    );

  return nva2x::ErrorCode::eSuccess;
}




//
// GPU implementation, empty one to have a high water mark.
//
class MultiTrackAnimatorEyesEmpty : public MultiTrackAnimatorEyesGPU {
public:
  std::error_code ComputeEyesRotation(
    nva2x::DeviceTensorFloatConstView inputEyesRotationResult, const nva2x::TensorBatchInfo& inputEyesRotationResultInfo,
    nva2x::DeviceTensorFloatView outputEyesRotation, const nva2x::TensorBatchInfo& outputEyesRotationInfo
    ) override; // GPU Async
};

std::error_code MultiTrackAnimatorEyesEmpty::ComputeEyesRotation(
  nva2x::DeviceTensorFloatConstView inputEyesRotationResult, const nva2x::TensorBatchInfo& inputEyesRotationResultInfo,
  nva2x::DeviceTensorFloatView outputEyesRotation, const nva2x::TensorBatchInfo& outputEyesRotationInfo
  ) {
  A2F_CHECK_ERROR_WITH_MSG(inputEyesRotationResultInfo.size == 4, "Input eyes rotation result size must be equal to 4", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(outputEyesRotationInfo.size == 6, "Output eyes rotation size must be equal to 6", nva2x::ErrorCode::eInvalidValue);

  const auto nbTracks = _eyesParams.Size() / _eyesParamsStride;

  A2F_CHECK_RESULT_WITH_MSG(
    test::ComputeEyesRotationEmpty(
      outputEyesRotation.Data(), outputEyesRotationInfo.offset, outputEyesRotationInfo.stride,
      inputEyesRotationResult.Data(), inputEyesRotationResultInfo.offset, inputEyesRotationResultInfo.stride,
      _eyesParams.Data(), _eyesParamsStride,
      _saccadeRot.Data(), _saccadeRot.Size(),
      _dt,
      _liveTime.Data(),
      nbTracks,
      _cudaStream
      ), "Unable to compute eyes rotation"
    );

  return nva2x::ErrorCode::eSuccess;
}


} // namespace test




namespace {

using device_creator_func_t = std::function<test::IMultiTrackAnimatorEyes*()>;
static const std::vector<std::pair<const char*, device_creator_func_t>> kDeviceImplementations {
  {"GPU", []() -> test::IMultiTrackAnimatorEyes* { return new test::MultiTrackAnimatorEyesGPU; }},
  {"Final", []() -> test::IMultiTrackAnimatorEyes* { return nva2f::CreateMultiTrackAnimatorEyes_INTERNAL(); }},
  {"Empty", []() -> test::IMultiTrackAnimatorEyes* { return new test::MultiTrackAnimatorEyesEmpty; }},
};

using host_creator_func_t = std::function<test::IMultiTrackAnimatorEyesHost*()>;
static const std::vector<std::pair<const char*, host_creator_func_t>> kHostImplementations = []() {
  std::vector<std::pair<const char*, host_creator_func_t>> implementations {
    {"Reference (useDeviceCopy = false)", []() -> test::IMultiTrackAnimatorEyesHost* { return new test::MultiTrackAnimatorEyesHostReference(false); }},
    {"Reference (useDeviceCopy = true)", []() -> test::IMultiTrackAnimatorEyesHost* { return new test::MultiTrackAnimatorEyesHostReference(true); }},
  };

  for (const auto& deviceImplementation : kDeviceImplementations) {
    const auto wrapper = [creator = deviceImplementation.second]() -> test::IMultiTrackAnimatorEyesHost* {
      return new test::MultiTrackAnimatorEyesHostWrapper(nva2x::UniquePtr<test::IMultiTrackAnimatorEyes>(creator()));
    };
    implementations.emplace_back(deviceImplementation.first, std::move(wrapper));
  }
  return implementations;
}();




struct BatchData {
  std::size_t nbTracks;
  nva2x::CudaStream cudaStream;
  nva2x::UniquePtr<nva2f::IRegressionModel::IGeometryModelInfo> modelInfo;
  nva2f::IMultiTrackAnimatorEyes::HostData initData;
  nva2f::IMultiTrackAnimatorEyes::Params params;
  float dt{1.0f / 60.0f};
  std::size_t nbIterations{1000};
  std::size_t nbPoses;

  std::vector<float> sourceDataHost;
  nva2x::TensorBatchInfo sourceInfo;
  nva2x::DeviceTensorFloat sourceData;
  nva2x::TensorBatchInfo resultsInfo;
  nva2x::DeviceTensorFloat resultsBuffers;
  nva2x::HostPinnedTensorFloat resultsBuffersHost;
};

BatchData BuildTestData(std::size_t nbTracks, bool useStride = false) {
  BatchData batchData;

  batchData.nbTracks = nbTracks;
  EXPECT_TRUE(!batchData.cudaStream.Init());

  constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
  batchData.modelInfo = nva2x::ToUniquePtr(nva2f::ReadRegressionModelInfo_INTERNAL(modelPath));
  EXPECT_TRUE(batchData.modelInfo);

  batchData.initData.saccadeRot = batchData.modelInfo->GetAnimatorData().GetAnimatorData().eyes.saccadeRot;
  assert(batchData.initData.saccadeRot.Size() % 2 == 0);
  batchData.params = batchData.modelInfo->GetAnimatorParams().eyes;

  // Generate source data.
  batchData.nbPoses = batchData.nbTracks * batchData.nbIterations;
  batchData.sourceDataHost.resize(batchData.nbPoses * 4);
  FillRandom(batchData.sourceDataHost);

  const std::size_t sourceStride = useStride
#if 0
    // This is too large, use a smaller stride.
    ? networkInfo.resultSkinSize + networkInfo.resultTongueSize + networkInfo.resultJawSize + networkInfo.resultEyesSize
#else
    ? 100
#endif
    : 4;
  if (useStride) {
    EXPECT_TRUE(!batchData.sourceData.Allocate(batchData.nbPoses * sourceStride));
    for (std::size_t i = 0; i < batchData.nbPoses; ++i) {
      const auto source = nva2x::ToView(batchData.sourceDataHost).View(i * 4, 4);
      const auto target = batchData.sourceData.View(
        i * sourceStride, 4
        );
      EXPECT_TRUE(!nva2x::CopyHostToDevice(target, source, batchData.cudaStream.Data()));
    }
  }
  else {
    EXPECT_TRUE(!batchData.sourceData.Init(nva2x::ToConstView(batchData.sourceDataHost)));
  }

  batchData.sourceInfo.offset = 0;
  batchData.sourceInfo.size = 4;
  batchData.sourceInfo.stride = sourceStride;

  batchData.resultsInfo.offset = 0;
  batchData.resultsInfo.size = 6;
  batchData.resultsInfo.stride = 6;

  EXPECT_TRUE(!batchData.resultsBuffers.Allocate(batchData.nbPoses * 6));
  EXPECT_TRUE(!batchData.resultsBuffersHost.Allocate(batchData.nbPoses * 6));

  EXPECT_TRUE(!cudaDeviceSynchronize());

  // Add noise to the params.
  auto addNoise = [](float& value, float amplitude) {
    value += amplitude * (2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 1.0f);
  };
  addNoise(batchData.params.eyeballsStrength, 0.2f);
  addNoise(batchData.params.saccadeStrength, 0.2f);
  addNoise(batchData.params.rightEyeballRotationOffsetX, 0.5f);
  addNoise(batchData.params.rightEyeballRotationOffsetY, 0.5f);
  addNoise(batchData.params.leftEyeballRotationOffsetX, 0.5f);
  addNoise(batchData.params.leftEyeballRotationOffsetY, 0.5f);
  addNoise(batchData.params.saccadeSeed, 4999.0f);
  batchData.params.saccadeSeed = std::abs(batchData.params.saccadeSeed);

  return batchData;
}

}




TEST(TestCoreBatchEyesAnimator, Correctness) {
  const auto nbTracks = 10;
  BatchData batchData = BuildTestData(nbTracks);

  // Generate expected results.
  nva2f::AnimatorEyes singleAnimator;
  ASSERT_TRUE(!singleAnimator.Init(batchData.params));
  ASSERT_TRUE(!singleAnimator.SetAnimatorData(batchData.initData));

  std::vector<float> expectedResultsHost(batchData.resultsBuffers.Size());
  for (std::size_t trackIndex = 0; trackIndex < batchData.nbTracks; ++trackIndex) {
    ASSERT_TRUE(!singleAnimator.Reset());

    for (std::size_t iteration = 0; iteration < batchData.nbIterations; ++iteration) {
      const auto pose = iteration * nbTracks + trackIndex;
      const auto inputEyes = nva2x::ToConstView(batchData.sourceDataHost).View(
        batchData.sourceInfo.offset + pose * batchData.sourceInfo.stride,
        batchData.sourceInfo.size
        );
      const auto outputEyes = nva2x::ToView(expectedResultsHost).View(
        batchData.resultsInfo.offset + pose * batchData.resultsInfo.stride,
        batchData.resultsInfo.size
        );
      ASSERT_TRUE(!singleAnimator.ComputeEyesRotation(outputEyes.View(0,3), outputEyes.View(3,3), inputEyes));
      ASSERT_TRUE(!singleAnimator.IncrementLiveTime(batchData.dt));
    }
    ASSERT_TRUE(!batchData.cudaStream.Synchronize());
  }

  // Test implementations.
  auto implementations = kHostImplementations;
  // Ignore the last empty implementation
  implementations.pop_back();
  for (const auto& implementation : implementations) {
    std::cout << "Testing \"" << implementation.first << "\" implementation..." << std::endl;

    ASSERT_TRUE(!nva2x::FillOnDevice(batchData.resultsBuffers, -1.0f, batchData.cudaStream.Data()));
    ASSERT_TRUE(!nva2x::FillOnHost(batchData.resultsBuffersHost, -1.0f));

    const auto animator = nva2x::ToUniquePtr(implementation.second());
    ASSERT_TRUE(!animator->SetCudaStream(batchData.cudaStream.Data()));
    ASSERT_TRUE(!animator->Init(batchData.initData, batchData.dt, batchData.params, batchData.nbTracks));

    for (std::size_t iteration = 0; iteration < batchData.nbIterations; ++iteration) {
      const auto inputSize = batchData.nbTracks * 4;
      const auto input = batchData.sourceData.View(iteration * inputSize, inputSize);
      const auto outputSize = batchData.nbTracks * 6;
      const auto output = batchData.resultsBuffersHost.View(iteration * outputSize, outputSize);

      ASSERT_TRUE(!animator->ComputeEyesRotation(input, batchData.sourceInfo, output));
    }

    ASSERT_TRUE(!batchData.cudaStream.Synchronize());

    const std::vector<float> resultsHost(
      batchData.resultsBuffersHost.Data(), batchData.resultsBuffersHost.Data() + batchData.resultsBuffersHost.Size()
      );

    // We compare the exact floating point values here.
    ASSERT_EQ(expectedResultsHost, resultsHost);
  }
}

TEST(TestCoreBatchEyesAnimator, PerformanceHost) {
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
      ASSERT_TRUE(!animator->Init(batchData.initData, batchData.dt, batchData.params, batchData.nbTracks));

      const std::size_t kWarmupIterations = 10;
      const std::size_t kNbBenchmarkIterations = 100;

      for (std::size_t i = 0; i < kWarmupIterations; ++i) {
        ASSERT_TRUE(!animator->ComputeEyesRotation(
          batchData.sourceData.View(0, nbTracks * 4),
          batchData.sourceInfo,
          batchData.resultsBuffersHost.View(0, nbTracks * 6)
          ));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      }

      duration_t totalTime = duration_t::zero();
      duration_t minTime = duration_t::max();
      for (std::size_t i = 0; i < kNbBenchmarkIterations; ++i) {
        const auto startTime = clock_t::now();
        ASSERT_TRUE(!animator->ComputeEyesRotation(
          batchData.sourceData.View(0, nbTracks * 4),
          batchData.sourceInfo,
          batchData.resultsBuffersHost.View(0, nbTracks * 6)
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

TEST(TestCoreBatchEyesAnimator, PerformanceDevice) {
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
      ASSERT_TRUE(!animator->SetAnimatorData(batchData.initData, batchData.dt));

      const std::size_t kWarmupIterations = 10;
      const std::size_t kNbBenchmarkIterations = 100;

      for (std::size_t i = 0; i < kWarmupIterations; ++i) {
        ASSERT_TRUE(!animator->ComputeEyesRotation(
          batchData.sourceData.View(0, nbTracks * 4),
          batchData.sourceInfo,
          batchData.resultsBuffers.View(0, nbTracks * 6),
          batchData.resultsInfo
          ));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      }

      float totalTime = 0.0f;
      float minTime = std::numeric_limits<float>::max();
      for (std::size_t i = 0; i < kNbBenchmarkIterations; ++i) {
        ASSERT_EQ(cudaEventRecord(start, batchData.cudaStream.Data()), cudaSuccess);
        ASSERT_TRUE(!animator->ComputeEyesRotation(
          batchData.sourceData.View(0, nbTracks * 4),
          batchData.sourceInfo,
          batchData.resultsBuffers.View(0, nbTracks * 6),
          batchData.resultsInfo
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
