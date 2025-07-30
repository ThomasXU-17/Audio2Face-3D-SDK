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
#include "audio2emotion/internal/multitrack_postprocess.h"
#include "audio2emotion/internal/postprocess.h"
#include "audio2emotion/internal/parse_helper.h"
#include "audio2emotion/internal/logger.h"
#include "audio2emotion/internal/macros.h"
#include "audio2x/error.h"
#include "audio2x/internal/cuda_stream.h"
#include "audio2x/internal/unique_ptr.h"
#include "utils.h"

#include "test_core_batch_post_processor_cuda.h"

#include <gtest/gtest.h>

#include <cmath>

#include <cuda_runtime_api.h>


namespace test {

using namespace nva2e;


// Interface for benchmarking: data starts on the GPU and ends up on the CPU.
class IMultiTrackPostProcessorHost {
public:
  using HostData = IMultiTrackPostProcessor::HostData;
  using Params = IMultiTrackPostProcessor::Params;

  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;
  virtual std::error_code Init(const HostData& data, const Params& params, std::size_t nbTracks) = 0;

  virtual std::error_code PostProcess(
    nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
    nva2x::HostTensorFloatView outputEmotions
    ) = 0;

  virtual void Destroy() = 0;
};



//
// Reference implementation, uses the single track implementation.
//
class MultiTrackPostProcessorHostReference : public IMultiTrackPostProcessorHost {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const HostData& data, const Params& params, std::size_t nbTracks) override; // GPU Async

  std::error_code PostProcess(
    nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
    nva2x::HostTensorFloatView outputEmotions
    ) override; // GPU Async

  void Destroy() override;

protected:
  cudaStream_t _cudaStream{nullptr};
  std::vector<PostProcessor> _postProcessors;
  std::vector<float> _inputEmotions;
  std::size_t _outputEmotionLength{0};
};

std::error_code MultiTrackPostProcessorHostReference::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackPostProcessorHostReference::Init(const HostData& data, const Params& params, std::size_t nbTracks) {
  A2E_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  _postProcessors.resize(nbTracks);
  for (auto& postProcessor : _postProcessors) {
    A2E_CHECK_RESULT_WITH_MSG(postProcessor.Init(data, params), "Unable to initialize post processor");
  }
  _inputEmotions.resize(data.inferenceEmotionLength * nbTracks);
  _outputEmotionLength = data.outputEmotionLength;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackPostProcessorHostReference::PostProcess(
  nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
  nva2x::HostTensorFloatView outputEmotions
  ) {
  // To do a single read, we must not have a stride.
  A2E_CHECK_ERROR_WITH_MSG(
    inputEmotionsInfo.stride == inputEmotionsInfo.size, "Stride must be equal to size", nva2x::ErrorCode::eInvalidValue
    );
  A2E_CHECK_RESULT_WITH_MSG(
    nva2x::CopyDeviceToHost(nva2x::ToView(_inputEmotions), inputEmotions, _cudaStream),
    "Unable to copy input to host"
    );

  for (std::size_t trackIndex = 0; trackIndex < _postProcessors.size(); ++trackIndex) {
    const auto inputDevice = inputEmotions.View(
      trackIndex * inputEmotionsInfo.stride + inputEmotionsInfo.offset, inputEmotionsInfo.size
      );
    const auto inputHost = nva2x::ToView(_inputEmotions).View(trackIndex * inputEmotionsInfo.size, inputEmotionsInfo.size);
    const auto outputHost = outputEmotions.View(trackIndex * _outputEmotionLength, _outputEmotionLength);
    A2E_CHECK_RESULT_WITH_MSG(_postProcessors[trackIndex].PostProcess(outputHost, inputHost), "Unable to post process");
  }
  return nva2x::ErrorCode::eSuccess;
}

void MultiTrackPostProcessorHostReference::Destroy() {
  delete this;
}




//
// Wrapper implementation, uses a GPU implementation.
//
class MultiTrackPostProcessorHostWrapper : public IMultiTrackPostProcessorHost {
public:
  MultiTrackPostProcessorHostWrapper(nva2x::UniquePtr<test::IMultiTrackPostProcessor> impl);

  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const HostData& data, const Params& params, std::size_t nbTracks) override; // GPU Async

  std::error_code PostProcess(
    nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
    nva2x::HostTensorFloatView outputEmotions
    ) override; // GPU Async

  void Destroy() override;

protected:
  nva2x::UniquePtr<test::IMultiTrackPostProcessor> _impl;
  cudaStream_t _cudaStream{nullptr};
  nva2x::DeviceTensorFloat _results;
  nva2x::TensorBatchInfo _resultsInfo;
};

MultiTrackPostProcessorHostWrapper::MultiTrackPostProcessorHostWrapper(
  nva2x::UniquePtr<test::IMultiTrackPostProcessor> impl
  ) : _impl(std::move(impl)) {
}

std::error_code MultiTrackPostProcessorHostWrapper::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return _impl->SetCudaStream(cudaStream);
}

std::error_code MultiTrackPostProcessorHostWrapper::Init(const HostData& data, const Params& params, std::size_t nbTracks) {
  A2E_CHECK_RESULT_WITH_MSG(
    _results.Allocate(data.outputEmotionLength * nbTracks),
    "Unable to allocate results"
    );
  _resultsInfo.offset = 0;
  _resultsInfo.size = data.outputEmotionLength;
  _resultsInfo.stride = data.outputEmotionLength;
  return _impl->Init(data, params, nbTracks);
}

std::error_code MultiTrackPostProcessorHostWrapper::PostProcess(
  nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
  nva2x::HostTensorFloatView outputEmotions
  ) {
  A2E_CHECK_RESULT_WITH_MSG(
    _impl->PostProcess(inputEmotions, inputEmotionsInfo, _results, _resultsInfo),
    "Unable to post process"
    );
  A2E_CHECK_RESULT_WITH_MSG(
    nva2x::CopyDeviceToHost(outputEmotions, _results, _cudaStream),
    "Unable to copy results to host"
    );
  return nva2x::ErrorCode::eSuccess;
}

void MultiTrackPostProcessorHostWrapper::Destroy() {
  delete this;
}




//
// Base implementation, used by the other implementations.
//
class MultiTrackPostProcessorBase : public IMultiTrackPostProcessor {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const HostData& data, const Params& params, std::size_t nbTracks) override; // GPU Async

  std::error_code SetParameters(std::size_t trackIndex, const Params& params) override;
  const Params* GetParameters(std::size_t trackIndex) const override;

  std::error_code Reset(std::size_t trackIndex) override;
  std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) override;

  std::size_t GetInputEmotionsSize() const override;
  std::size_t GetOutputEmotionsSize() const override;

  nva2x::DeviceTensorFloatView GetPreferredEmotion(std::size_t trackIndex) override;
  nva2x::DeviceTensorFloatConstView GetPreferredEmotion(std::size_t trackIndex) const override;

  void Destroy() override;

protected:
  cudaStream_t _cudaStream{nullptr};
  nva2e::PostProcessData _initData;
  std::vector<nva2e::PostProcessParams> _params;
};

std::error_code MultiTrackPostProcessorBase::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackPostProcessorBase::Init(const HostData& data, const Params& params, std::size_t nbTracks) {
  A2E_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  // This is not robust, but for testing don't do deep copy.
  _initData = data;
  _params.clear();
  _params.resize(nbTracks, params);
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackPostProcessorBase::SetParameters(std::size_t trackIndex, const Params& params) {
  return nva2x::ErrorCode::eUnsupported;
}

const MultiTrackPostProcessorBase::Params* MultiTrackPostProcessorBase::GetParameters(std::size_t trackIndex) const {
  return nullptr;
}

std::error_code MultiTrackPostProcessorBase::Reset(std::size_t trackIndex) {
  return nva2x::ErrorCode::eUnsupported;
}

std::error_code MultiTrackPostProcessorBase::SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) {
  return nva2x::ErrorCode::eUnsupported;
}

std::size_t MultiTrackPostProcessorBase::GetInputEmotionsSize() const {
  return 0;
}

std::size_t MultiTrackPostProcessorBase::GetOutputEmotionsSize() const {
  return 0;
}

nva2x::DeviceTensorFloatView MultiTrackPostProcessorBase::GetPreferredEmotion(std::size_t trackIndex) {
  return {};
}

nva2x::DeviceTensorFloatConstView MultiTrackPostProcessorBase::GetPreferredEmotion(std::size_t trackIndex) const {
  return {};
}

void MultiTrackPostProcessorBase::Destroy() {
  delete this;
}




//
// GPU implementation, first attempt.
//
class MultiTrackPostProcessorGPU : public MultiTrackPostProcessorBase {
public:
  std::error_code Init(const HostData& data, const Params& params, std::size_t nbTracks) override;

  std::error_code PostProcess(
    nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
    nva2x::DeviceTensorFloatView outputEmotions, const nva2x::TensorBatchInfo& outputEmotionsInfo
    ) override; // GPU Async

protected:
  nva2x::DeviceTensorInt64 _emotionCorrespondence;
  nva2x::DeviceTensorFloat _postProcessParams;
  std::size_t _postProcessParamsStride{0};
  nva2x::DeviceTensorFloat _preferredEmotion;
  nva2x::DeviceTensorFloat _stateAndWorkBuffers;
  std::size_t _stateAndWorkBuffersStride{0};
};

std::error_code MultiTrackPostProcessorGPU::Init(const HostData& data, const Params& params, std::size_t nbTracks) {
  A2E_CHECK_RESULT(MultiTrackPostProcessorBase::Init(data, params, nbTracks));

  A2E_CHECK_ERROR_WITH_MSG(data.inferenceEmotionLength == data.emotionCorrespondenceSize, "Wrong emotion correspondence size", nva2x::ErrorCode::eMismatch);
  A2E_CHECK_ERROR_WITH_MSG(data.outputEmotionLength == params.beginningEmotion.Size(), "Wrong beginning emotion length", nva2x::ErrorCode::eMismatch);
  A2E_CHECK_ERROR_WITH_MSG(data.outputEmotionLength == params.preferredEmotion.Size(), "Wrong preferred emotion length", nva2x::ErrorCode::eMismatch);

  std::vector<int64_t> emotionCorrespondenceHost(
    data.emotionCorrespondence, data.emotionCorrespondence + data.emotionCorrespondenceSize
    );
  A2E_CHECK_RESULT_WITH_MSG(
    _emotionCorrespondence.Init(nva2x::ToConstView(emotionCorrespondenceHost), _cudaStream),
    "Unable to initialize emotion correspondence"
    );

  _postProcessParamsStride = 8 + data.outputEmotionLength;
  std::vector<float> paramsHost(nbTracks * _postProcessParamsStride);
  std::vector<float> preferredEmotionHost(nbTracks * data.outputEmotionLength);
  for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
    // We store the max emotion as a float, but it's an int.
    assert(params.maxEmotions <= std::numeric_limits<std::uint32_t>::max());
    std::uint32_t maxEmotions = static_cast<std::uint32_t>(params.maxEmotions);
    paramsHost[trackIndex * _postProcessParamsStride + 0] = *reinterpret_cast<const float*>(&maxEmotions);
    paramsHost[trackIndex * _postProcessParamsStride + 1] = params.emotionContrast;
    paramsHost[trackIndex * _postProcessParamsStride + 2] = params.liveBlendCoef;
    paramsHost[trackIndex * _postProcessParamsStride + 3] = params.enablePreferredEmotion ? 1.0f : 0.0f;
    paramsHost[trackIndex * _postProcessParamsStride + 4] = params.preferredEmotionStrength;
    paramsHost[trackIndex * _postProcessParamsStride + 5] = params.liveTransitionTime;
    paramsHost[trackIndex * _postProcessParamsStride + 6] = params.fixedDt;
    paramsHost[trackIndex * _postProcessParamsStride + 7] = params.emotionStrength;
    std::copy(
      nva2x::begin(params.beginningEmotion),
      nva2x::end(params.beginningEmotion),
      paramsHost.begin() + trackIndex * _postProcessParamsStride + 8
      );

    std::copy(
      nva2x::begin(params.preferredEmotion),
      nva2x::end(params.preferredEmotion),
      preferredEmotionHost.begin() + trackIndex * data.outputEmotionLength
      );
  }

  A2E_CHECK_RESULT_WITH_MSG(
    _postProcessParams.Init(nva2x::ToConstView(paramsHost), _cudaStream),
    "Unable to initialize post process params"
    );
  A2E_CHECK_RESULT_WITH_MSG(
    _preferredEmotion.Init(nva2x::ToConstView(preferredEmotionHost), _cudaStream),
    "Unable to initialize preferred emotion"
    );

  _stateAndWorkBuffersStride = 1 + data.inferenceEmotionLength + data.outputEmotionLength * 3;
  A2E_CHECK_RESULT_WITH_MSG(
    _stateAndWorkBuffers.Allocate(_stateAndWorkBuffersStride * nbTracks),
    "Unable to allocate state and work buffers"
    );
  A2E_CHECK_RESULT_WITH_MSG(
    nva2x::FillOnDevice(_stateAndWorkBuffers, 1.0f, _cudaStream),
    "Unable to initialize state and work buffers"
    );

  A2E_CUDA_CHECK_ERROR(cudaStreamSynchronize(_cudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackPostProcessorGPU::PostProcess(
  nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
  nva2x::DeviceTensorFloatView outputEmotions, const nva2x::TensorBatchInfo& outputEmotionsInfo
  ) {
  // FIXME: Add more validation.
  assert(_emotionCorrespondence.Size() == inputEmotionsInfo.size);

  const auto nbTracks = _params.size();

  A2E_CHECK_RESULT_WITH_MSG(
    PostProcessGPU(
      outputEmotions.Data(), outputEmotionsInfo.offset, outputEmotionsInfo.stride, outputEmotionsInfo.size,
      inputEmotions.Data(), inputEmotionsInfo.offset, inputEmotionsInfo.stride, inputEmotionsInfo.size,
      _emotionCorrespondence.Data(),
      _postProcessParams.Data(), _postProcessParamsStride,
      _preferredEmotion.Data(),
      _stateAndWorkBuffers.Data(), _stateAndWorkBuffersStride,
      nbTracks,
      _cudaStream
      ),
    "Unable to post process"
    );

  return nva2x::ErrorCode::eSuccess;
}




//
// GPU implementation, try using shared memory.
//
class MultiTrackPostProcessorGPUShared : public MultiTrackPostProcessorGPU {
public:
  std::error_code Init(const HostData& data, const Params& params, std::size_t nbTracks) override;

  std::error_code PostProcess(
    nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
    nva2x::DeviceTensorFloatView outputEmotions, const nva2x::TensorBatchInfo& outputEmotionsInfo
    ) override; // GPU Async
};

std::error_code MultiTrackPostProcessorGPUShared::Init(const HostData& data, const Params& params, std::size_t nbTracks) {
  A2E_CHECK_RESULT(MultiTrackPostProcessorGPU::Init(data, params, nbTracks));

  _stateAndWorkBuffersStride = 1 + data.outputEmotionLength * 3;
  A2E_CHECK_RESULT_WITH_MSG(
    _stateAndWorkBuffers.Allocate(_stateAndWorkBuffersStride * nbTracks),
    "Unable to allocate state and work buffers"
    );
  A2E_CHECK_RESULT_WITH_MSG(
    nva2x::FillOnDevice(_stateAndWorkBuffers, 1.0f, _cudaStream),
    "Unable to initialize state and work buffers"
    );

  A2E_CUDA_CHECK_ERROR(cudaStreamSynchronize(_cudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackPostProcessorGPUShared::PostProcess(
  nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
  nva2x::DeviceTensorFloatView outputEmotions, const nva2x::TensorBatchInfo& outputEmotionsInfo
  ) {
  // FIXME: Add more validation.
  assert(_emotionCorrespondence.Size() == inputEmotionsInfo.size);

  const auto nbTracks = _params.size();

  A2E_CHECK_RESULT_WITH_MSG(
    PostProcessGPUShared(
      outputEmotions.Data(), outputEmotionsInfo.offset, outputEmotionsInfo.stride, outputEmotionsInfo.size,
      inputEmotions.Data(), inputEmotionsInfo.offset, inputEmotionsInfo.stride, inputEmotionsInfo.size,
      _emotionCorrespondence.Data(),
      _postProcessParams.Data(), _postProcessParamsStride,
      _preferredEmotion.Data(),
      _stateAndWorkBuffers.Data(), _stateAndWorkBuffersStride,
      nbTracks,
      _cudaStream
      ),
    "Unable to post process"
    );

  return nva2x::ErrorCode::eSuccess;
}




//
// GPU implementation, try using local memory.
//
class MultiTrackPostProcessorGPULocal : public MultiTrackPostProcessorGPUShared {
public:
  std::error_code Init(const HostData& data, const Params& params, std::size_t nbTracks) override;

  std::error_code PostProcess(
    nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
    nva2x::DeviceTensorFloatView outputEmotions, const nva2x::TensorBatchInfo& outputEmotionsInfo
    ) override; // GPU Async

protected:
  nva2x::DeviceTensorInt64 _a2fEmotionCorrespondence;
};

std::error_code MultiTrackPostProcessorGPULocal::Init(const HostData& data, const Params& params, std::size_t nbTracks) {
  A2E_CHECK_RESULT(MultiTrackPostProcessorGPUShared::Init(data, params, nbTracks));

  std::vector<int64_t> a2fEmotionCorrespondenceHost(data.outputEmotionLength, -1);
  for (std::size_t i = 0; i < data.emotionCorrespondenceSize; ++i) {
    const auto j = data.emotionCorrespondence[i];
    if (j == -1) {
      continue;
    }
    if (j >= data.outputEmotionLength) {
      return nva2x::ErrorCode::eOutOfBounds;
    }
    a2fEmotionCorrespondenceHost[j] = i;
  }
  A2E_CHECK_RESULT_WITH_MSG(
    _a2fEmotionCorrespondence.Init(nva2x::ToConstView(a2fEmotionCorrespondenceHost), _cudaStream),
    "Unable to initialize A2F emotion correspondence"
    );

  A2E_CUDA_CHECK_ERROR(cudaStreamSynchronize(_cudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackPostProcessorGPULocal::PostProcess(
  nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
  nva2x::DeviceTensorFloatView outputEmotions, const nva2x::TensorBatchInfo& outputEmotionsInfo
  ) {
  // FIXME: Add more validation.
  assert(_emotionCorrespondence.Size() == inputEmotionsInfo.size);
  assert(_a2fEmotionCorrespondence.Size() == outputEmotionsInfo.size);

  const auto nbTracks = _params.size();

  A2E_CHECK_RESULT_WITH_MSG(
    PostProcessGPULocal(
      outputEmotions.Data(), outputEmotionsInfo.offset, outputEmotionsInfo.stride, outputEmotionsInfo.size,
      inputEmotions.Data(), inputEmotionsInfo.offset, inputEmotionsInfo.stride, inputEmotionsInfo.size,
      _emotionCorrespondence.Data(), _a2fEmotionCorrespondence.Data(),
      _postProcessParams.Data(), _postProcessParamsStride,
      _preferredEmotion.Data(),
      _stateAndWorkBuffers.Data(), _stateAndWorkBuffersStride,
      nbTracks,
      _cudaStream
      ),
    "Unable to post process"
    );

  return nva2x::ErrorCode::eSuccess;
}

} // namespace test


namespace {

using device_creator_func_t = std::function<test::IMultiTrackPostProcessor*()>;
static const std::vector<std::pair<const char*, device_creator_func_t>> kDeviceImplementations {
  {"GPU", []() -> test::IMultiTrackPostProcessor* { return new test::MultiTrackPostProcessorGPU; }},
  {"GPU Shared", []() -> test::IMultiTrackPostProcessor* { return new test::MultiTrackPostProcessorGPUShared; }},
  {"GPU Local", []() -> test::IMultiTrackPostProcessor* { return new test::MultiTrackPostProcessorGPULocal; }},
  {"Final (fast)", []() -> test::IMultiTrackPostProcessor* { return new nva2e::MultiTrackPostProcessor(true); }},
  {"Final (slow)", []() -> test::IMultiTrackPostProcessor* { return new nva2e::MultiTrackPostProcessor(false); }},
  {"Final", []() -> test::IMultiTrackPostProcessor* { return nva2e::CreateMultiTrackPostProcessor_INTERNAL(); }},
};

using host_creator_func_t = std::function<test::IMultiTrackPostProcessorHost*()>;
static const std::vector<std::pair<const char*, host_creator_func_t>> kHostImplementations = []() {
  std::vector<std::pair<const char*, host_creator_func_t>> implementations {
    {"Reference", []() -> test::IMultiTrackPostProcessorHost* { return new test::MultiTrackPostProcessorHostReference; }},
  };

  for (const auto& deviceImplementation : kDeviceImplementations) {
    const auto wrapper = [creator = deviceImplementation.second]() -> test::IMultiTrackPostProcessorHost* {
      return new test::MultiTrackPostProcessorHostWrapper(nva2x::UniquePtr<test::IMultiTrackPostProcessor>(creator()));
    };
    implementations.emplace_back(deviceImplementation.first, std::move(wrapper));
  }
  return implementations;
}();




struct BatchData {
  std::size_t nbTracks;
  nva2x::CudaStream cudaStream;
  nva2x::UniquePtr<nva2e::IClassifierModel::IEmotionModelInfo> modelInfo;
  nva2e::PostProcessData initData;
  nva2e::PostProcessParams params;
  std::vector<float> beginningEmotion;
  std::vector<float> preferredEmotion;
  nva2x::DeviceTensorFloat sourceData;
  std::size_t nbIterations{2};

  nva2x::DeviceTensorFloat resultsBuffers;
  nva2x::TensorBatchInfo sourceInfo;
  nva2x::TensorBatchInfo resultInfo;
  nva2x::HostPinnedTensorFloat resultsBuffersHost;
};

BatchData BuildTestData(std::size_t nbTracks) {
  BatchData batchData;

  batchData.nbTracks = nbTracks;
  EXPECT_TRUE(!batchData.cudaStream.Init());

  constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json";
  batchData.modelInfo = nva2x::ToUniquePtr(nva2e::ReadClassifierModelInfo_INTERNAL(modelPath));
  EXPECT_TRUE(batchData.modelInfo);

  batchData.initData = batchData.modelInfo->GetConfigInfo().GetPostProcessData();
  batchData.params = batchData.modelInfo->GetConfigInfo().GetPostProcessParams();

  // Enable settings.
  batchData.params.maxEmotions = 4;

  batchData.beginningEmotion.resize(batchData.initData.outputEmotionLength);
  for (std::size_t i = 0; i < batchData.beginningEmotion.size(); ++i) {
    batchData.beginningEmotion[i] = (1.0f + i) / (1.0f + batchData.beginningEmotion.size());
  }
  batchData.params.beginningEmotion = nva2x::ToConstView(batchData.beginningEmotion);

  batchData.preferredEmotion.resize(batchData.initData.outputEmotionLength);
  for (std::size_t i = 0; i < batchData.preferredEmotion.size(); ++i) {
    batchData.preferredEmotion[i] = 1.0f - (1.0f + i) / (1.0f + batchData.preferredEmotion.size());
  }
  batchData.params.preferredEmotion = nva2x::ToConstView(batchData.preferredEmotion);
  batchData.params.enablePreferredEmotion = true;
  batchData.params.preferredEmotionStrength = 0.5f;

  // Generate source data.
  const auto resultsSize = batchData.initData.outputEmotionLength * batchData.nbTracks * batchData.nbIterations;
  EXPECT_TRUE(!batchData.resultsBuffers.Allocate(resultsSize));

  const auto sourceDataSize = batchData.initData.inferenceEmotionLength * batchData.nbTracks * batchData.nbIterations;
  std::vector<float> sourceDataHost(sourceDataSize);
  FillRandom(sourceDataHost);
  EXPECT_TRUE(!batchData.sourceData.Init(nva2x::ToConstView(sourceDataHost)));

  batchData.sourceInfo.offset = 0;
  batchData.sourceInfo.size = batchData.initData.inferenceEmotionLength;
  batchData.sourceInfo.stride = batchData.initData.inferenceEmotionLength;

  batchData.resultInfo.offset = 0;
  batchData.resultInfo.size = batchData.initData.outputEmotionLength;
  batchData.resultInfo.stride = batchData.initData.outputEmotionLength;

  EXPECT_TRUE(!batchData.resultsBuffersHost.Allocate(resultsSize));

  EXPECT_TRUE(!cudaDeviceSynchronize());

  return batchData;
}

}




TEST(TestCoreBatchPostProcessor, Correctness) {
  const auto nbTracks = 10;
  BatchData batchData = BuildTestData(nbTracks);

  // Generate expected results.
  nva2e::PostProcessor singlePostProcessor;
  ASSERT_TRUE(!singlePostProcessor.Init(batchData.initData, batchData.params));

  std::vector<float> expectedResultsHost(batchData.resultsBuffers.Size());
  nva2x::HostTensorFloat inputHost;
  for (std::size_t trackIndex = 0; trackIndex < batchData.nbTracks; ++trackIndex) {
    ASSERT_TRUE(!singlePostProcessor.Reset());

    for (std::size_t iteration = 0; iteration < batchData.nbIterations; ++iteration) {
      const auto inputEmotions = batchData.sourceData.View(
        (trackIndex + batchData.nbTracks * iteration) * batchData.initData.inferenceEmotionLength,
        batchData.initData.inferenceEmotionLength
        );
      ASSERT_TRUE(!inputHost.Allocate(inputEmotions.Size()));
      ASSERT_TRUE(!nva2x::CopyDeviceToHost(inputHost, inputEmotions, batchData.cudaStream.Data()));
      const auto outputEmotions = nva2x::ToView(expectedResultsHost).View(
        (trackIndex + batchData.nbTracks * iteration) * batchData.initData.outputEmotionLength,
        batchData.initData.outputEmotionLength
        );
      ASSERT_TRUE(!singlePostProcessor.PostProcess(outputEmotions, inputHost));
    }
  }

  // Test implementations.
  for (const auto& implementation : kHostImplementations) {
    std::cout << "Testing \"" << implementation.first << "\" implementation..." << std::endl;

    ASSERT_TRUE(!nva2x::FillOnDevice(batchData.resultsBuffers, -1.0f, batchData.cudaStream.Data()));
    ASSERT_TRUE(!nva2x::FillOnHost(batchData.resultsBuffersHost, -1.0f));
    ASSERT_TRUE(!batchData.cudaStream.Synchronize());

    const auto postProcessor = nva2x::ToUniquePtr(implementation.second());
    ASSERT_TRUE(!postProcessor->SetCudaStream(batchData.cudaStream.Data()));
    ASSERT_TRUE(!postProcessor->Init(batchData.initData, batchData.params, batchData.nbTracks));

    for (std::size_t iteration = 0; iteration < batchData.nbIterations; ++iteration) {
      const auto source = batchData.sourceData.View(
        batchData.initData.inferenceEmotionLength * nbTracks * iteration,
        batchData.initData.inferenceEmotionLength * nbTracks
        );
      const auto results = batchData.resultsBuffersHost.View(
        batchData.initData.outputEmotionLength * nbTracks * iteration,
        batchData.initData.outputEmotionLength * nbTracks
        );
      ASSERT_TRUE(!postProcessor->PostProcess(source, batchData.sourceInfo, results));
    }

    ASSERT_TRUE(!batchData.cudaStream.Synchronize());

    const std::vector<float> resultsHost(
      batchData.resultsBuffersHost.Data(), batchData.resultsBuffersHost.Data() + batchData.resultsBuffersHost.Size()
    );

    // We compare the exact floating point values here.
    // Values computed on the GPU use exp() functions which might be slightly different
    // from the ones computed on the CPU.
    #if 0
    ASSERT_EQ(expectedResultsHost, resultsHost);
    #else
    ASSERT_EQ(expectedResultsHost.size(), resultsHost.size());
    for (std::size_t i = 0; i < expectedResultsHost.size(); ++i) {
      ASSERT_FLOAT_EQ(expectedResultsHost[i], resultsHost[i]) << "at index " << i;
    }
    #endif
  }
}

TEST(TestCoreBatchPostProcessor, PerformanceHost) {
  using clock_t = std::chrono::steady_clock;
  using time_point_t = clock_t::time_point;
  using duration_t = clock_t::duration;

  for (const auto nbTracks : {1, 8, 16, 128}) {
    std::cout << "Benchmarking for " << nbTracks << " tracks..." << std::endl;
    BatchData batchData = BuildTestData(nbTracks);

    // Benchmark implementations.
    for (const auto& implementation : kHostImplementations) {
      std::cout << "  Benchmarking \"" << implementation.first << "\" implementation..." << std::endl;

      const auto postProcessor = nva2x::ToUniquePtr(implementation.second());
      ASSERT_TRUE(!postProcessor->Init(batchData.initData, batchData.params, batchData.nbTracks));
      ASSERT_TRUE(!postProcessor->SetCudaStream(batchData.cudaStream.Data()));

      const std::size_t kWarmupIterations = 10;
      const std::size_t kNbBenchmarkIterations = 100;

      for (std::size_t i = 0; i < kWarmupIterations; ++i) {
        ASSERT_TRUE(!postProcessor->PostProcess(
          batchData.sourceData.View(0, batchData.sourceData.Size() / batchData.nbIterations),
          batchData.sourceInfo,
          batchData.resultsBuffersHost.View(0, batchData.resultsBuffers.Size() / batchData.nbIterations)
          ));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      }

      duration_t totalTime = duration_t::zero();
      duration_t minTime = duration_t::max();
      for (std::size_t i = 0; i < kNbBenchmarkIterations; ++i) {
        const auto startTime = clock_t::now();
        ASSERT_TRUE(!postProcessor->PostProcess(
          batchData.sourceData.View(0, batchData.sourceData.Size() / batchData.nbIterations),
          batchData.sourceInfo,
          batchData.resultsBuffersHost.View(0, batchData.resultsBuffers.Size() / batchData.nbIterations)
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

TEST(TestCoreBatchPostProcessor, PerformanceDevice) {
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

      const auto postProcessor = nva2x::ToUniquePtr(implementation.second());
      ASSERT_TRUE(!postProcessor->Init(batchData.initData, batchData.params, batchData.nbTracks));
      ASSERT_TRUE(!postProcessor->SetCudaStream(batchData.cudaStream.Data()));

      const std::size_t kWarmupIterations = 10;
      const std::size_t kNbBenchmarkIterations = 100;

      for (std::size_t i = 0; i < kWarmupIterations; ++i) {
        ASSERT_TRUE(!postProcessor->PostProcess(
          batchData.sourceData.View(0, batchData.sourceData.Size() / batchData.nbIterations),
          batchData.sourceInfo,
          batchData.resultsBuffers.View(0, batchData.resultsBuffers.Size() / batchData.nbIterations),
          batchData.resultInfo
          ));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      }

      float totalTime = 0.0f;
      float minTime = std::numeric_limits<float>::max();
      for (std::size_t i = 0; i < kNbBenchmarkIterations; ++i) {
        ASSERT_EQ(cudaEventRecord(start, batchData.cudaStream.Data()), cudaSuccess);
        ASSERT_TRUE(!postProcessor->PostProcess(
          batchData.sourceData.View(0, batchData.sourceData.Size() / batchData.nbIterations),
          batchData.sourceInfo,
          batchData.resultsBuffers.View(0, batchData.resultsBuffers.Size() / batchData.nbIterations),
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
