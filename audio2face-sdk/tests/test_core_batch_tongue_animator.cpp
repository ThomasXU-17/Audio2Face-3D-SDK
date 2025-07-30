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

#include "test_core_batch_tongue_animator_cuda.h"

#include <gtest/gtest.h>

#include <cmath>


namespace test {

using namespace nva2f;

//
// Base implementation, used by the other implementations.
//
class MultiTrackAnimatorTongueBase : public IMultiTrackAnimatorTongue {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const Params& params, std::size_t nbTracks) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

  std::error_code SetParameters(std::size_t trackIndex, const Params& params) override;
  const Params* GetParameters(std::size_t trackIndex) const override;

  std::error_code Reset(std::size_t trackIndex) override;
  std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) override;
  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async

  void Destroy() override;

protected:
  cudaStream_t _cudaStream{nullptr};
  std::vector<Params> _params;
  std::size_t _poseSize{0};
};

std::error_code MultiTrackAnimatorTongueBase::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongueBase::Init(const Params& params, std::size_t nbTracks) {
  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  _params.resize(nbTracks, params);
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongueBase::SetAnimatorData(const HostData& data) {
  A2F_CHECK_ERROR_WITH_MSG(data.neutralPose.Size() % 3 == 0, "Neutral pose size must be a multiple of 3", nva2x::ErrorCode::eInvalidValue);

  _poseSize = data.neutralPose.Size();

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongueBase::SetParameters(std::size_t trackIndex, const Params& params) {
  return nva2x::ErrorCode::eUnsupported;
}

const MultiTrackAnimatorTongueBase::Params* MultiTrackAnimatorTongueBase::GetParameters(std::size_t trackIndex) const {
  return nullptr;
}

std::error_code MultiTrackAnimatorTongueBase::Reset(std::size_t trackIndex) {
  return nva2x::ErrorCode::eUnsupported;
}

std::error_code MultiTrackAnimatorTongueBase::SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) {
  return nva2x::ErrorCode::eUnsupported;
}

std::error_code MultiTrackAnimatorTongueBase::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  const auto nbTracks = _params.size();
  A2F_CHECK_ERROR_WITH_MSG(
    inputDeltasInfo.stride * nbTracks == inputDeltas.Size(),
    "Input deltas size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2F_CHECK_ERROR_WITH_MSG(
    outputVerticesInfo.stride * nbTracks == outputVertices.Size(),
    "Output vertices size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );

  const auto poseSize = _poseSize;
  A2F_CHECK_ERROR_WITH_MSG(poseSize == outputVerticesInfo.size, "Output vertices size does not match", nva2x::ErrorCode::eMismatch);
  A2F_CHECK_ERROR_WITH_MSG(poseSize == inputDeltasInfo.size, "Input deltas size does not match", nva2x::ErrorCode::eMismatch);

  return nva2x::ErrorCode::eSuccess;
}

void MultiTrackAnimatorTongueBase::Destroy() {
  delete this;
}




//
// Reference implementation, uses the single track implementation.
//
class MultiTrackAnimatorTongueReference : public MultiTrackAnimatorTongueBase {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const Params& params, std::size_t nbTracks) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async

protected:
  std::vector<AnimatorTongue> _animators;
};

std::error_code MultiTrackAnimatorTongueReference::SetCudaStream(cudaStream_t cudaStream) {
  A2F_CHECK_RESULT(MultiTrackAnimatorTongueBase::SetCudaStream(cudaStream));
  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.SetCudaStream(_cudaStream), "Unable to set CUDA stream on animator");
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongueReference::Init(const Params& params, std::size_t nbTracks) {
  A2F_CHECK_RESULT(MultiTrackAnimatorTongueBase::Init(params, nbTracks));
  _animators.resize(nbTracks);
  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.SetCudaStream(_cudaStream), "Unable to set CUDA stream on animator");
    A2F_CHECK_RESULT_WITH_MSG(animator.Init(params), "Unable to initialize animator");
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongueReference::SetAnimatorData(const HostData& data) {
  A2F_CHECK_RESULT(MultiTrackAnimatorTongueBase::SetAnimatorData(data));

  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.SetAnimatorData(data), "Unable to set animator data");
  }

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongueReference::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  A2F_CHECK_RESULT(MultiTrackAnimatorTongueBase::Animate(inputDeltas, inputDeltasInfo, outputVertices, outputVerticesInfo));
  for (std::size_t i = 0; i < _animators.size(); ++i) {
      const auto input = inputDeltas.View(
        i * inputDeltasInfo.stride + inputDeltasInfo.offset, inputDeltasInfo.size
        );
      const auto output = outputVertices.View(
        i * outputVerticesInfo.stride + outputVerticesInfo.offset, outputVerticesInfo.size
        );
      A2F_CHECK_RESULT_WITH_MSG(_animators[i].Animate(input, -1.0f, output), "Unable to animate animator");
  }
  return nva2x::ErrorCode::eSuccess;
}




//
// Reference (shared) implementation, uses the single track implementation, but shares the animator data.
//
class MultiTrackAnimatorTongueReferenceShared : public MultiTrackAnimatorTongueReference {
public:
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

private:
  AnimatorTongue::Data _data;
};

std::error_code MultiTrackAnimatorTongueReferenceShared::SetAnimatorData(const HostData& data) {
  A2F_CHECK_RESULT(MultiTrackAnimatorTongueBase::SetAnimatorData(data));
  A2F_CHECK_RESULT_WITH_MSG(_data.Init(data, _cudaStream), "Unable to initialize animator data");
  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.SetAnimatorDataView(_data.GetDeviceView()), "Unable to set animator data");
  }

  return nva2x::ErrorCode::eSuccess;
}




//
// Batched kernel implementation.
//
class MultiTrackAnimatorTongueBatched : public MultiTrackAnimatorTongueBase {
public:
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async

private:
  AnimatorTongue::Data _data;
};

std::error_code MultiTrackAnimatorTongueBatched::SetAnimatorData(const HostData& data) {
  A2F_CHECK_RESULT(MultiTrackAnimatorTongueBase::SetAnimatorData(data));
  A2F_CHECK_RESULT_WITH_MSG(_data.Init(data, _cudaStream), "Unable to initialize animator data");

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongueBatched::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  A2F_CHECK_RESULT(MultiTrackAnimatorTongueBase::Animate(inputDeltas, inputDeltasInfo, outputVertices, outputVerticesInfo));

  const auto poseSize = _data.GetDeviceView().neutralPose.Size();
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    AnimateBatched(
      outputVertices.Data(),
      outputVerticesInfo.offset,
      outputVerticesInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _data.GetDeviceView().neutralPose.Data(),
      // FIXME: This is hack, we only pass a single track parameters, but they could be different for each track.
      _params[0].tongueStrength,
      _params[0].tongueHeightOffset,
      _params[0].tongueDepthOffset,
      poseSize,
      nbTracks,
      _cudaStream
      ),
    "Unable to animate animator"
    );

  return nva2x::ErrorCode::eSuccess;
}




//
// Batched kernel implementation, with packed parameters.
//
class MultiTrackAnimatorTongueBatchedParamsPacked : public MultiTrackAnimatorTongueBase {
public:
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async

protected:
  AnimatorTongue::Data _data;
  nva2x::DeviceTensorFloat _tongueParams;
  std::size_t _tongueParamsStride{0};
};

std::error_code MultiTrackAnimatorTongueBatchedParamsPacked::SetAnimatorData(const HostData& data) {
  A2F_CHECK_RESULT(MultiTrackAnimatorTongueBase::SetAnimatorData(data));
  A2F_CHECK_RESULT_WITH_MSG(_data.Init(data, _cudaStream), "Unable to initialize animator data");

  const auto nbTracks = _params.size();

  _tongueParamsStride = 3;
  std::vector<float> paramsHost(nbTracks * _tongueParamsStride);
  for (std::size_t i = 0; i < nbTracks; ++i) {
    paramsHost[i * _tongueParamsStride + 0] = _params[i].tongueStrength;
    paramsHost[i * _tongueParamsStride + 1] = _params[i].tongueHeightOffset;
    paramsHost[i * _tongueParamsStride + 2] = _params[i].tongueDepthOffset;
  }

  A2F_CHECK_RESULT_WITH_MSG(_tongueParams.Init(nva2x::ToConstView(paramsHost), _cudaStream), "Unable to initialize params");

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongueBatchedParamsPacked::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  A2F_CHECK_RESULT(MultiTrackAnimatorTongueBase::Animate(inputDeltas, inputDeltasInfo, outputVertices, outputVerticesInfo));

  const auto poseSize = _data.GetDeviceView().neutralPose.Size();
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    AnimateBatchedParamsPacked(
      outputVertices.Data(),
      outputVerticesInfo.offset,
      outputVerticesInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _data.GetDeviceView().neutralPose.Data(),
      _tongueParams.Data(),
      _tongueParamsStride,
      poseSize,
      nbTracks,
      _cudaStream
      ),
    "Unable to animate animator"
    );

  return nva2x::ErrorCode::eSuccess;
}




//
// Batched kernel implementation, with packed parameters.
//
class MultiTrackAnimatorTongueBatchedParamsPackedVertices : public MultiTrackAnimatorTongueBatchedParamsPacked {
public:
  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async
};

std::error_code MultiTrackAnimatorTongueBatchedParamsPackedVertices::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  A2F_CHECK_RESULT(MultiTrackAnimatorTongueBase::Animate(inputDeltas, inputDeltasInfo, outputVertices, outputVerticesInfo));

  const auto poseSize = _data.GetDeviceView().neutralPose.Size();
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    AnimateBatchedParamsPackedVertices(
      outputVertices.Data(),
      outputVerticesInfo.offset,
      outputVerticesInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _data.GetDeviceView().neutralPose.Data(),
      _tongueParams.Data(),
      _tongueParamsStride,
      poseSize,
      nbTracks,
      _cudaStream
      ),
    "Unable to animate animator"
    );

  return nva2x::ErrorCode::eSuccess;
}




//
// Batched kernel implementation, with packed parameters.
//
class MultiTrackAnimatorTongueBatchedParamsPackedFull : public MultiTrackAnimatorTongueBase {
public:
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async

protected:
  AnimatorTongue::Data _data;
  nva2x::DeviceTensorFloat _tongueParams;
  std::size_t _tongueParamsStride{0};
};

std::error_code MultiTrackAnimatorTongueBatchedParamsPackedFull::SetAnimatorData(const HostData& data) {
  A2F_CHECK_RESULT(MultiTrackAnimatorTongueBase::SetAnimatorData(data));
  A2F_CHECK_RESULT_WITH_MSG(_data.Init(data, _cudaStream), "Unable to initialize animator data");

  const auto nbTracks = _params.size();

  _tongueParamsStride = 4;
  std::vector<float> paramsHost(nbTracks * _tongueParamsStride);
  for (std::size_t i = 0; i < nbTracks; ++i) {
    paramsHost[i * _tongueParamsStride + 0] = 0.0f;
    paramsHost[i * _tongueParamsStride + 1] = _params[i].tongueHeightOffset;
    paramsHost[i * _tongueParamsStride + 2] = _params[i].tongueDepthOffset;
    paramsHost[i * _tongueParamsStride + 3] = _params[i].tongueStrength;
  }

  A2F_CHECK_RESULT_WITH_MSG(_tongueParams.Init(nva2x::ToConstView(paramsHost), _cudaStream), "Unable to initialize params");

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongueBatchedParamsPackedFull::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  A2F_CHECK_RESULT(MultiTrackAnimatorTongueBase::Animate(inputDeltas, inputDeltasInfo, outputVertices, outputVerticesInfo));

  const auto poseSize = _data.GetDeviceView().neutralPose.Size();
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    AnimateBatchedParamsPackedFull(
      outputVertices.Data(),
      outputVerticesInfo.offset,
      outputVerticesInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _data.GetDeviceView().neutralPose.Data(),
      _tongueParams.Data(),
      _tongueParamsStride,
      poseSize,
      nbTracks,
      _cudaStream
      ),
    "Unable to animate animator"
    );

  return nva2x::ErrorCode::eSuccess;
}

} // namespace test


namespace {

using creator_func_t = test::IMultiTrackAnimatorTongue* (*)();
static const std::vector<std::pair<const char*, creator_func_t>> kImplementations {
  {"Reference", []() -> test::IMultiTrackAnimatorTongue* { return new test::MultiTrackAnimatorTongueReference; }},
  {"ReferenceShared", []() -> test::IMultiTrackAnimatorTongue* { return new test::MultiTrackAnimatorTongueReferenceShared; }},
  {"Batched", []() -> test::IMultiTrackAnimatorTongue* { return new test::MultiTrackAnimatorTongueBatched; }},
  {"BatchedParamsPacked", []() -> test::IMultiTrackAnimatorTongue* { return new test::MultiTrackAnimatorTongueBatchedParamsPacked; }},
  {"BatchedParamsPackedVertices", []() -> test::IMultiTrackAnimatorTongue* { return new test::MultiTrackAnimatorTongueBatchedParamsPackedVertices; }},
  {"BatchedParamsPackedFull", []() -> test::IMultiTrackAnimatorTongue* { return new test::MultiTrackAnimatorTongueBatchedParamsPackedFull; }},
  {"Final", []() -> test::IMultiTrackAnimatorTongue* { return nva2f::CreateMultiTrackAnimatorTongue_INTERNAL(); }},
};


struct BatchData {
  std::size_t nbTracks;
  nva2x::CudaStream cudaStream;
  nva2x::UniquePtr<nva2f::IRegressionModel::IGeometryModelInfo> modelInfo;
  nva2f::AnimatorTongue::Params params;
  nva2f::IAnimatorTongue::HostData initData;
  nva2x::DeviceTensorFloat sourceData;
  float dt{-1.0f};
  std::size_t nbIterations{2};

  nva2f::IRegressionModel::ResultBuffers resultsBuffers;
  nva2x::TensorBatchInfo resultsInfo;
};

BatchData BuildTestData(std::size_t nbTracks) {
  BatchData batchData;

  batchData.nbTracks = nbTracks;
  EXPECT_TRUE(!batchData.cudaStream.Init());

  constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
  batchData.modelInfo = nva2x::ToUniquePtr(nva2f::ReadRegressionModelInfo_INTERNAL(modelPath));
  EXPECT_TRUE(batchData.modelInfo);

  batchData.params = batchData.modelInfo->GetAnimatorParams().tongue;
  batchData.initData = batchData.modelInfo->GetAnimatorData().GetAnimatorData().tongue;

  // Generate source data.
  EXPECT_TRUE(!batchData.resultsBuffers.Init(batchData.modelInfo->GetNetworkInfo().GetNetworkInfo(), batchData.nbTracks));

  std::vector<float> sourceDataHost(
    batchData.resultsBuffers.GetResultTensor(batchData.nbTracks).Size() * batchData.nbIterations
    );
  FillRandom(sourceDataHost);
  EXPECT_TRUE(!batchData.sourceData.Init(nva2x::ToConstView(sourceDataHost)));

  batchData.resultsInfo = batchData.resultsBuffers.GetTongueBatchInfo();

  EXPECT_TRUE(!cudaDeviceSynchronize());

  return batchData;
}

}




TEST(TestCoreBatchTongueAnimator, Correctness) {
  const auto nbTracks = 10;
  BatchData batchData = BuildTestData(nbTracks);

  // Generate expected results.
  nva2f::AnimatorTongue singleAnimator;
  ASSERT_TRUE(!singleAnimator.Init(batchData.params));
  ASSERT_TRUE(!singleAnimator.SetCudaStream(batchData.cudaStream.Data()));
  ASSERT_TRUE(!singleAnimator.SetAnimatorData(batchData.initData));

  std::vector<float> expectedResultsHost(batchData.initData.neutralPose.Size() * batchData.nbTracks);
  for (std::size_t trackIndex = 0; trackIndex < batchData.nbTracks; ++trackIndex) {
    const auto inputDeltas = batchData.resultsBuffers.GetResultTongueGeometry(trackIndex);
    const auto outputVertices = inputDeltas;
    ASSERT_TRUE(!singleAnimator.Reset());

    for (std::size_t iteration = 0; iteration < batchData.nbIterations; ++iteration) {
      const auto results = batchData.resultsBuffers.GetResultTensor(batchData.nbTracks);
      const auto source = batchData.sourceData.View(iteration * results.Size(), results.Size());
      ASSERT_TRUE(
        !nva2x::CopyDeviceToDevice(results, source, batchData.cudaStream.Data())
        );
      ASSERT_TRUE(!singleAnimator.Animate(inputDeltas, batchData.dt, outputVertices));
    }

    ASSERT_TRUE(
      !nva2x::CopyDeviceToHost(
        {expectedResultsHost.data() + trackIndex * batchData.initData.neutralPose.Size(),
         batchData.initData.neutralPose.Size()},
        outputVertices,
        batchData.cudaStream.Data()
        )
      );
    ASSERT_TRUE(!batchData.cudaStream.Synchronize());
  }

  // Test implementations.
  for (const auto& implementation : kImplementations) {
    std::cout << "Testing \"" << implementation.first << "\" implementation..." << std::endl;

    ASSERT_TRUE(!nva2x::FillOnDevice(batchData.resultsBuffers.GetResultTensor(), -1.0f, batchData.cudaStream.Data()));

    const auto animator = nva2x::ToUniquePtr(implementation.second());
    ASSERT_TRUE(!animator->Init(batchData.params, batchData.nbTracks));
    ASSERT_TRUE(!animator->SetCudaStream(batchData.cudaStream.Data()));
    ASSERT_TRUE(!animator->SetAnimatorData(batchData.initData));

    for (std::size_t iteration = 0; iteration < batchData.nbIterations; ++iteration) {
      const auto results = batchData.resultsBuffers.GetResultTensor(batchData.nbTracks);
      const auto source = batchData.sourceData.View(iteration * results.Size(), results.Size());
      ASSERT_TRUE(
        !nva2x::CopyDeviceToDevice(results, source, batchData.cudaStream.Data())
        );

      ASSERT_TRUE(!animator->Animate(results, batchData.resultsInfo, results, batchData.resultsInfo));
    }

    std::vector<float> resultsHost(batchData.initData.neutralPose.Size() * batchData.nbTracks);
    for (std::size_t i = 0; i < batchData.nbTracks; ++i) {
      const auto outputShape = batchData.resultsBuffers.GetResultTongueGeometry(i);
      ASSERT_TRUE(
        !nva2x::CopyDeviceToHost(
          {resultsHost.data() + i * batchData.initData.neutralPose.Size(), batchData.initData.neutralPose.Size()},
          outputShape,
          batchData.cudaStream.Data()
          )
        );
    }
    ASSERT_TRUE(!batchData.cudaStream.Synchronize());
    // We compare the exact floating point values here.
    // Note that this seems to work, but during development we've hit issues where tiny
    // modifications to the code would change how code was generated.  Specifically, it
    // affected whether fused multiply-add (fma) instructions were used or not.
    // The current way the code is written seems to be consistent with the way the original
    // implementation was written, but if that comes to change, compiling the code with
    // -fmad=false should be done to confirm that without fma, the results are the same.
    ASSERT_EQ(expectedResultsHost, resultsHost);
  }
}

TEST(TestCoreBatchTongueAnimator, Performance) {
  cudaEvent_t start;
  cudaEvent_t end;
  ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);


  for (const auto nbTracks : {1, 8, 16, 128}) {
    std::cout << "Benchmarking for " << nbTracks << " tracks..." << std::endl;
    BatchData batchData = BuildTestData(nbTracks);

    // Benchmark implementations.
    for (const auto& implementation : kImplementations) {
      std::cout << "  Benchmarking \"" << implementation.first << "\" implementation..." << std::endl;

      const auto results = batchData.resultsBuffers.GetResultTensor(batchData.nbTracks);
      const auto source = batchData.sourceData.View(0, results.Size());
      ASSERT_TRUE(!nva2x::CopyDeviceToDevice(results, source, batchData.cudaStream.Data()));

      const auto animator = nva2x::ToUniquePtr(implementation.second());
      ASSERT_TRUE(!animator->Init(batchData.params, batchData.nbTracks));
      ASSERT_TRUE(!animator->SetCudaStream(batchData.cudaStream.Data()));
      ASSERT_TRUE(!animator->SetAnimatorData(batchData.initData));

      const std::size_t kWarmupIterations = 10;
      const std::size_t kNbBenchmarkIterations = 100;

      for (std::size_t i = 0; i < kWarmupIterations; ++i) {
        ASSERT_TRUE(!animator->Animate(
          batchData.resultsBuffers.GetResultTensor(batchData.nbTracks),
          batchData.resultsInfo,
          batchData.resultsBuffers.GetResultTensor(batchData.nbTracks),
          batchData.resultsInfo
          ));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      }

      float totalTime = 0.0f;
      float minTime = std::numeric_limits<float>::max();
      for (std::size_t i = 0; i < kNbBenchmarkIterations; ++i) {
        ASSERT_EQ(cudaEventRecord(start, batchData.cudaStream.Data()), cudaSuccess);
        ASSERT_TRUE(!animator->Animate(
          batchData.resultsBuffers.GetResultTensor(batchData.nbTracks),
          batchData.resultsInfo,
          batchData.resultsBuffers.GetResultTensor(batchData.nbTracks),
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
