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

#include <gtest/gtest.h>

namespace test {

using namespace nva2f;

//
// Base implementation, used by the other implementations.
//
class MultiTrackAnimatorPcaReconstructionBase : public IMultiTrackAnimatorPcaReconstruction {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(std::size_t nbTracks) override; // GPU Async

  std::error_code Reset(std::size_t trackIndex) override;
  std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) override;

  void Destroy() override;

protected:
  cudaStream_t _cudaStream{nullptr};
  std::size_t _nbTracks{0};
};

std::error_code MultiTrackAnimatorPcaReconstructionBase::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstructionBase::Init(std::size_t nbTracks) {
  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  _nbTracks = nbTracks;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstructionBase::Reset(std::size_t trackIndex) {
  return nva2x::ErrorCode::eUnsupported;
}

std::error_code MultiTrackAnimatorPcaReconstructionBase::SetActiveTracks(
  const std::uint64_t* activeTracks, std::size_t activeTracksSize
  ) {
  return nva2x::ErrorCode::eUnsupported;
}

void MultiTrackAnimatorPcaReconstructionBase::Destroy() {
  delete this;
}





//
// Reference implementation, uses the single track implementation.
//
class MultiTrackAnimatorPcaReconstructionReference : public MultiTrackAnimatorPcaReconstructionBase {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(std::size_t nbTracks) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputPcaCoefs, const nva2x::TensorBatchInfo& inputPcaCoefsInfo,
    nva2x::DeviceTensorFloatView outputShapes, const nva2x::TensorBatchInfo& outputShapesInfo,
    std::size_t batchSize = 0
    ) override; // GPU Async

protected:
  std::vector<AnimatorPcaReconstruction> _animators;
  std::size_t _numShapes{0};
  std::size_t _shapeSize{0};
};

std::error_code MultiTrackAnimatorPcaReconstructionReference::SetCudaStream(cudaStream_t cudaStream) {
  A2F_CHECK_RESULT(MultiTrackAnimatorPcaReconstructionBase::SetCudaStream(cudaStream));
  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.SetCudaStream(_cudaStream), "Unable to set CUDA stream on animator");
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstructionReference::Init(std::size_t nbTracks) {
  A2F_CHECK_RESULT(MultiTrackAnimatorPcaReconstructionBase::Init(nbTracks));
  _animators.resize(nbTracks);
  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.SetCudaStream(_cudaStream), "Unable to set CUDA stream on animator");
    A2F_CHECK_RESULT_WITH_MSG(animator.Init(), "Unable to initialize animator");
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstructionReference::SetAnimatorData(const HostData& data) {
  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.SetAnimatorData(data), "Unable to set animator data");
  }

  _numShapes = data.shapesMatrix.Size() / data.shapeSize;
  _shapeSize = data.shapeSize;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstructionReference::Animate(
  nva2x::DeviceTensorFloatConstView inputPcaCoefs, const nva2x::TensorBatchInfo& inputPcaCoefsInfo,
  nva2x::DeviceTensorFloatView outputShapes, const nva2x::TensorBatchInfo& outputShapesInfo,
  std::size_t batchSize
  ) {
  A2F_CHECK_ERROR_WITH_MSG(batchSize == 0, "Batch size is expected to be 0 in these tests", nva2x::ErrorCode::eInvalidValue);
  const auto nbTracks = _animators.size();
  A2F_CHECK_ERROR_WITH_MSG(
    inputPcaCoefsInfo.stride * nbTracks == inputPcaCoefs.Size(),
    "Input PCA coefs size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2F_CHECK_ERROR_WITH_MSG(
    outputShapesInfo.stride * nbTracks == outputShapes.Size(),
    "Output shapes size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );

  A2F_CHECK_ERROR_WITH_MSG(_shapeSize == outputShapesInfo.size, "Output shapes size does not match", nva2x::ErrorCode::eMismatch);
  A2F_CHECK_ERROR_WITH_MSG(_numShapes == inputPcaCoefsInfo.size, "Input PCA coefs size does not match", nva2x::ErrorCode::eMismatch);

  for (std::size_t i = 0; i < _animators.size(); ++i) {
      const auto input = inputPcaCoefs.View(
        i * inputPcaCoefsInfo.stride + inputPcaCoefsInfo.offset, inputPcaCoefsInfo.size
        );
      const auto output = outputShapes.View(
        i * outputShapesInfo.stride + outputShapesInfo.offset, outputShapesInfo.size
        );
      A2F_CHECK_RESULT_WITH_MSG(_animators[i].Animate(input, output), "Unable to animate animator");
  }
  return nva2x::ErrorCode::eSuccess;
}




//
// Reference (shared) implementation, uses the single track implementation, but shares the animator data.
//
class MultiTrackAnimatorPcaReconstructionReferenceShared : public MultiTrackAnimatorPcaReconstructionReference {
public:
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

private:
  AnimatorPcaReconstruction::Data _data;
};

std::error_code MultiTrackAnimatorPcaReconstructionReferenceShared::SetAnimatorData(const HostData& data) {
  A2F_CHECK_RESULT_WITH_MSG(_data.Init(data, _cudaStream), "Unable to initialize animator data");
  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.SetAnimatorDataView(_data.GetDeviceView()), "Unable to set animator data");
  }

  _numShapes = data.shapesMatrix.Size() / data.shapeSize;
  _shapeSize = data.shapeSize;

  return nva2x::ErrorCode::eSuccess;
}




//
// cublas implementation, used by all implementations using cublas.
//
class MultiTrackAnimatorPcaReconstructionCublas : public MultiTrackAnimatorPcaReconstructionBase {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(std::size_t nbTracks) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

protected:
  CublasHandle _cublasHandle;
  nva2f::AnimatorPcaReconstruction::Data _data;
};

std::error_code MultiTrackAnimatorPcaReconstructionCublas::SetCudaStream(cudaStream_t cudaStream) {
  A2F_CHECK_RESULT(MultiTrackAnimatorPcaReconstructionBase::SetCudaStream(cudaStream));
  if (_cublasHandle.Data() != nullptr) {
    A2F_CHECK_RESULT_WITH_MSG(_cublasHandle.SetCudaStream(_cudaStream), "Unable to set CUDA stream on cublas handle");
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstructionCublas::Init(std::size_t nbTracks) {
  A2F_CHECK_RESULT(MultiTrackAnimatorPcaReconstructionBase::Init(nbTracks));
  A2F_CHECK_RESULT_WITH_MSG(_cublasHandle.Init(), "Unable to initialize cublas handle");
  if (_cublasHandle.Data() != nullptr) {
    A2F_CHECK_RESULT_WITH_MSG(_cublasHandle.SetCudaStream(_cudaStream), "Unable to set CUDA stream on cublas handle");
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstructionCublas::SetAnimatorData(const HostData& data) {
  A2F_CHECK_RESULT_WITH_MSG(_data.Init(data, _cudaStream), "Unable to initialize animator data");
  return nva2x::ErrorCode::eSuccess;
}




//
// Matrix implementation, uses cublasSgemm.
//
class MultiTrackAnimatorPcaReconstructionMatrix : public MultiTrackAnimatorPcaReconstructionCublas {
public:
  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputPcaCoefs, const nva2x::TensorBatchInfo& inputPcaCoefsInfo,
    nva2x::DeviceTensorFloatView outputShapes, const nva2x::TensorBatchInfo& outputShapesInfo,
    std::size_t batchSize = 0
    ) override; // GPU Async
};

std::error_code MultiTrackAnimatorPcaReconstructionMatrix::Animate(
  nva2x::DeviceTensorFloatConstView inputPcaCoefs, const nva2x::TensorBatchInfo& inputPcaCoefsInfo,
  nva2x::DeviceTensorFloatView outputShapes, const nva2x::TensorBatchInfo& outputShapesInfo,
  std::size_t batchSize
  ) {
  A2F_CHECK_ERROR_WITH_MSG(batchSize == 0, "Batch size is expected to be 0 in these tests", nva2x::ErrorCode::eInvalidValue);
  // Could add more validation like in other implementations...
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const auto data = _data.GetDeviceView();
  cublasStatus_t status = cublasSgemm(
    _cublasHandle.Data(), CUBLAS_OP_N, CUBLAS_OP_N,
    static_cast<int>(data.shapeSize), static_cast<int>(_nbTracks), static_cast<int>(inputPcaCoefsInfo.size),
    &alpha, data.shapesMatrix.Data(), static_cast<int>(data.shapeSize),
    inputPcaCoefs.Data() + inputPcaCoefsInfo.offset, static_cast<int>(inputPcaCoefsInfo.stride),
    &beta, outputShapes.Data() + outputShapesInfo.offset, static_cast<int>(outputShapesInfo.stride)
    );
  A2F_CHECK_ERROR_WITH_MSG(status == CUBLAS_STATUS_SUCCESS, "Unable to run matrix multiplication", ErrorCode::eCublasExecutionError);

  return nva2x::ErrorCode::eSuccess;
}




//
// Matrix implementation, uses cublasGemmEx
//
class MultiTrackAnimatorPcaReconstructionMatrixEx : public MultiTrackAnimatorPcaReconstructionCublas {
public:
  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputPcaCoefs, const nva2x::TensorBatchInfo& inputPcaCoefsInfo,
    nva2x::DeviceTensorFloatView outputShapes, const nva2x::TensorBatchInfo& outputShapesInfo,
    std::size_t batchSize = 0
    ) override; // GPU Async
};

std::error_code MultiTrackAnimatorPcaReconstructionMatrixEx::Animate(
  nva2x::DeviceTensorFloatConstView inputPcaCoefs, const nva2x::TensorBatchInfo& inputPcaCoefsInfo,
  nva2x::DeviceTensorFloatView outputShapes, const nva2x::TensorBatchInfo& outputShapesInfo,
  std::size_t batchSize
  ) {
  A2F_CHECK_ERROR_WITH_MSG(batchSize == 0, "Batch size is expected to be 0 in these tests", nva2x::ErrorCode::eInvalidValue);
  // Could add more validation like in other implementations...
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const auto data = _data.GetDeviceView();
  cublasStatus_t status = cublasGemmEx(
    _cublasHandle.Data(), CUBLAS_OP_N, CUBLAS_OP_N,
    static_cast<int>(data.shapeSize), static_cast<int>(_nbTracks), static_cast<int>(inputPcaCoefsInfo.size),
    &alpha, data.shapesMatrix.Data(), cudaDataType_t::CUDA_R_32F, static_cast<int>(data.shapeSize),
    inputPcaCoefs.Data() + inputPcaCoefsInfo.offset, cudaDataType_t::CUDA_R_32F, static_cast<int>(inputPcaCoefsInfo.stride),
    &beta, outputShapes.Data() + outputShapesInfo.offset, cudaDataType_t::CUDA_R_32F, static_cast<int>(outputShapesInfo.stride),
    cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
    cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT
    );
  A2F_CHECK_ERROR_WITH_MSG(status == CUBLAS_STATUS_SUCCESS, "Unable to run matrix multiplication", ErrorCode::eCublasExecutionError);

  return nva2x::ErrorCode::eSuccess;
}




//
// Vector implementation, uses cublasSgemvBatched.
//
class MultiTrackAnimatorPcaReconstructionVector : public MultiTrackAnimatorPcaReconstructionCublas {
public:
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputPcaCoefs, const nva2x::TensorBatchInfo& inputPcaCoefsInfo,
    nva2x::DeviceTensorFloatView outputShapes, const nva2x::TensorBatchInfo& outputShapesInfo,
    std::size_t batchSize = 0
    ) override; // GPU Async

  std::error_code Deallocate();
  ~MultiTrackAnimatorPcaReconstructionVector();

private:
  const float** _paramA{nullptr};

  nva2x::DeviceTensorFloatConstView _inputPcaCoefs;
  nva2x::TensorBatchInfo _inputPcaCoefsInfo;
  nva2x::DeviceTensorFloatView _outputShapes;
  nva2x::TensorBatchInfo _outputShapesInfo;

  const float** _paramx{nullptr};
  float** _paramy{nullptr};
};

std::error_code MultiTrackAnimatorPcaReconstructionVector::SetAnimatorData(const HostData& data) {
  A2F_CHECK_RESULT(MultiTrackAnimatorPcaReconstructionCublas::SetAnimatorData(data));

  std::vector<const float*> paramA(_nbTracks);
  for (std::size_t i = 0; i < _nbTracks; ++i) {
    paramA[i] = _data.GetDeviceView().shapesMatrix.Data();
  }
  A2F_CUDA_CHECK_ERROR(
    cudaMalloc(reinterpret_cast<void**>(&_paramA), sizeof(float*) * _nbTracks),
    nva2x::ErrorCode::eCudaMemoryAllocationError
    );
  A2F_CUDA_CHECK_ERROR(
    cudaMemcpyAsync(_paramA, paramA.data(), sizeof(float*) * _nbTracks, cudaMemcpyHostToDevice, _cudaStream),
    nva2x::ErrorCode::eCudaMemcpyHostToDeviceError
    );
  A2F_CUDA_CHECK_ERROR(
    cudaMalloc(reinterpret_cast<void**>(&_paramx), sizeof(float*) * _nbTracks),
    nva2x::ErrorCode::eCudaMemoryAllocationError
    );
  A2F_CUDA_CHECK_ERROR(
    cudaMalloc(reinterpret_cast<void**>(&_paramy), sizeof(float*) * _nbTracks),
    nva2x::ErrorCode::eCudaMemoryAllocationError
    );

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstructionVector::Animate(
  nva2x::DeviceTensorFloatConstView inputPcaCoefs, const nva2x::TensorBatchInfo& inputPcaCoefsInfo,
  nva2x::DeviceTensorFloatView outputShapes, const nva2x::TensorBatchInfo& outputShapesInfo,
  std::size_t batchSize
  ) {
  A2F_CHECK_ERROR_WITH_MSG(batchSize == 0, "Batch size is expected to be 0 in these tests", nva2x::ErrorCode::eInvalidValue);
  if (_inputPcaCoefs.Data() != inputPcaCoefs.Data() ||
      _inputPcaCoefs.Size() != inputPcaCoefs.Size() ||
      _inputPcaCoefsInfo.offset != inputPcaCoefsInfo.offset ||
      _inputPcaCoefsInfo.size != inputPcaCoefsInfo.size ||
      _inputPcaCoefsInfo.stride != inputPcaCoefsInfo.stride) {
    std::vector<const float*> paramx(_nbTracks);
    for (std::size_t i = 0; i < _nbTracks; ++i) {
      paramx[i] = inputPcaCoefs.Data() + inputPcaCoefsInfo.offset + inputPcaCoefsInfo.stride * i;
    }

    A2F_CUDA_CHECK_ERROR(
      cudaMemcpyAsync(_paramx, paramx.data(), sizeof(float*) * _nbTracks, cudaMemcpyHostToDevice, _cudaStream),
      nva2x::ErrorCode::eCudaMemcpyHostToDeviceError
      );

    _inputPcaCoefs = inputPcaCoefs;
    _inputPcaCoefsInfo = inputPcaCoefsInfo;
  }

  if (_outputShapes.Data() != outputShapes.Data() ||
      _outputShapes.Size() != outputShapes.Size() ||
      _outputShapesInfo.offset != outputShapesInfo.offset ||
      _outputShapesInfo.size != outputShapesInfo.size ||
      _outputShapesInfo.stride != outputShapesInfo.stride) {
    std::vector<float*> paramy(_nbTracks);
    for (std::size_t i = 0; i < _nbTracks; ++i) {
      paramy[i] = outputShapes.Data() + outputShapesInfo.offset + outputShapesInfo.stride * i;
    }

    A2F_CUDA_CHECK_ERROR(
      cudaMemcpyAsync(_paramy, paramy.data(), sizeof(float*) * _nbTracks, cudaMemcpyHostToDevice, _cudaStream),
      nva2x::ErrorCode::eCudaMemcpyHostToDeviceError
      );

    _outputShapes = outputShapes;
    _outputShapesInfo = outputShapesInfo;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  const auto data = _data.GetDeviceView();
  cublasStatus_t status = cublasSgemvBatched(
    _cublasHandle.Data(), CUBLAS_OP_N,
    static_cast<int>(data.shapeSize), static_cast<int>(_inputPcaCoefsInfo.size),
    &alpha, _paramA, static_cast<int>(data.shapeSize),
    _paramx, 1,
    &beta, _paramy, 1,
    _nbTracks
    );
  A2F_CHECK_ERROR_WITH_MSG(status == CUBLAS_STATUS_SUCCESS, "Unable to run matrix multiplication", ErrorCode::eCublasExecutionError);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstructionVector::Deallocate() {
  A2F_CUDA_CHECK_ERROR(cudaFree(_paramA), nva2x::ErrorCode::eCudaMemoryFreeError);
  A2F_CUDA_CHECK_ERROR(cudaFree(_paramx), nva2x::ErrorCode::eCudaMemoryFreeError);
  A2F_CUDA_CHECK_ERROR(cudaFree(_paramy), nva2x::ErrorCode::eCudaMemoryFreeError);
  return nva2x::ErrorCode::eSuccess;
}

MultiTrackAnimatorPcaReconstructionVector::~MultiTrackAnimatorPcaReconstructionVector() {
  Deallocate();
}

} // namespace test


namespace {

using creator_func_t = test::IMultiTrackAnimatorPcaReconstruction* (*)();
static const std::vector<std::pair<const char*, creator_func_t>> kImplementations {
  {"Reference", []() -> test::IMultiTrackAnimatorPcaReconstruction* { return new test::MultiTrackAnimatorPcaReconstructionReference; }},
  {"ReferenceShared", []() -> test::IMultiTrackAnimatorPcaReconstruction* { return new test::MultiTrackAnimatorPcaReconstructionReferenceShared; }},
  {"Matrix", []() -> test::IMultiTrackAnimatorPcaReconstruction* { return new test::MultiTrackAnimatorPcaReconstructionMatrix; }},
  // We disable MatrixEx because it requires a higher threshold for the tolerance.
  // It did not show any performance gain.
  //{"MatrixEx", []() -> test::IMultiTrackAnimatorPcaReconstruction* { return new test::MultiTrackAnimatorPcaReconstructionMatrixEx; }},
  {"Vector", []() -> test::IMultiTrackAnimatorPcaReconstruction* { return new test::MultiTrackAnimatorPcaReconstructionVector; }},
  {"Final", []() -> test::IMultiTrackAnimatorPcaReconstruction* { return nva2f::CreateMultiTrackAnimatorPcaReconstruction_INTERNAL(); }},
};


struct BatchData {
  std::size_t nbTracks;
  nva2x::CudaStream cudaStream;
  nva2x::UniquePtr<nva2f::IRegressionModel::IGeometryModelInfo> modelInfo;
  std::size_t numShapes;
  std::size_t shapeSize;
  nva2f::IAnimatorPcaReconstruction::HostData initData;
  std::vector<float> sourceData;

  nva2f::IRegressionModel::InferenceOutputBuffers inferenceOutputBuffers;
  nva2x::TensorBatchInfo coefsInfo;
  nva2x::DeviceTensorFloat resultsDevice;
  nva2x::TensorBatchInfo shapesInfo;
};

BatchData BuildTestData(std::size_t nbTracks) {
  BatchData batchData;

  batchData.nbTracks = nbTracks;
  EXPECT_TRUE(!batchData.cudaStream.Init());

  constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
  batchData.modelInfo = nva2x::ToUniquePtr(nva2f::ReadRegressionModelInfo_INTERNAL(modelPath));
  EXPECT_TRUE(batchData.modelInfo);

  batchData.numShapes = batchData.modelInfo->GetNetworkInfo().GetNetworkInfo().numShapesSkin;
  batchData.shapeSize = batchData.modelInfo->GetNetworkInfo().GetNetworkInfo().resultSkinSize;

  batchData.initData = batchData.modelInfo->GetAnimatorData().GetSkinPcaReconstructionData();

  // Generate source data.
  batchData.sourceData.resize(batchData.numShapes * batchData.nbTracks);
  FillRandom(batchData.sourceData);

  EXPECT_TRUE(!batchData.inferenceOutputBuffers.Init(batchData.modelInfo->GetNetworkInfo().GetNetworkInfo(), batchData.nbTracks));

  for (std::size_t i = 0; i < batchData.nbTracks; ++i) {
    auto inputPcaCoefs = batchData.inferenceOutputBuffers.GetInferenceResult(i, 1).View(0, batchData.numShapes);
    EXPECT_TRUE(!nva2x::CopyHostToDevice(inputPcaCoefs, {batchData.sourceData.data() + i * batchData.numShapes, batchData.numShapes}));
  }

  batchData.coefsInfo.size = batchData.numShapes;
  batchData.coefsInfo.offset = 0;
  batchData.coefsInfo.stride =
    batchData.modelInfo->GetNetworkInfo().GetNetworkInfo().numShapesSkin +
    batchData.modelInfo->GetNetworkInfo().GetNetworkInfo().numShapesTongue +
    batchData.modelInfo->GetNetworkInfo().GetNetworkInfo().resultJawSize +
    batchData.modelInfo->GetNetworkInfo().GetNetworkInfo().resultEyesSize;

  EXPECT_TRUE(!batchData.resultsDevice.Allocate(batchData.shapeSize * batchData.nbTracks));
  batchData.shapesInfo.size = batchData.shapeSize;
  batchData.shapesInfo.offset = 0;
  batchData.shapesInfo.stride = batchData.shapeSize;

  EXPECT_TRUE(!cudaDeviceSynchronize());

  return batchData;
}

}




TEST(TestCoreBatchPcaAnimator, Correctness) {
  const auto nbTracks = 10;
  BatchData batchData = BuildTestData(nbTracks);

  // Generate expected results.
  nva2f::AnimatorPcaReconstruction singleAnimator;
  ASSERT_TRUE(!singleAnimator.Init());
  ASSERT_TRUE(!singleAnimator.SetCudaStream(batchData.cudaStream.Data()));
  ASSERT_TRUE(!singleAnimator.SetAnimatorData(batchData.initData));

  for (std::size_t i = 0; i < batchData.nbTracks; ++i) {
    const auto inputCoefs = batchData.inferenceOutputBuffers.GetInferenceResultSkin(i);
    const auto outputShapes = batchData.resultsDevice.View(i * batchData.shapeSize, batchData.shapeSize);
    ASSERT_TRUE(!singleAnimator.Animate(inputCoefs, outputShapes));
  }

  std::vector<float> expectedResultsHost(batchData.shapeSize * batchData.nbTracks);
  ASSERT_TRUE(!nva2x::CopyDeviceToHost(nva2x::ToView(expectedResultsHost), batchData.resultsDevice));
  ASSERT_TRUE(!batchData.cudaStream.Synchronize());

  // Test implementations.
  for (const auto& implementation : kImplementations) {
    std::cout << "Testing \"" << implementation.first << "\" implementation..." << std::endl;

    ASSERT_TRUE(!nva2x::FillOnDevice(batchData.resultsDevice, -1.0f, batchData.cudaStream.Data()));

    const auto animator = nva2x::ToUniquePtr(implementation.second());
    ASSERT_TRUE(!animator->Init(batchData.nbTracks));
    ASSERT_TRUE(!animator->SetCudaStream(batchData.cudaStream.Data()));
    ASSERT_TRUE(!animator->SetAnimatorData(batchData.initData));

    ASSERT_TRUE(!animator->Animate(
      batchData.inferenceOutputBuffers.GetInferenceResult(0, batchData.nbTracks),
      batchData.coefsInfo,
      batchData.resultsDevice,
      batchData.shapesInfo
      ));

    std::vector<float> resultsHost(batchData.shapeSize * batchData.nbTracks);
    for (std::size_t i = 0; i < batchData.nbTracks; ++i) {
      const auto outputShape = batchData.resultsDevice.View(i * batchData.shapeSize, batchData.shapeSize);
      ASSERT_TRUE(!nva2x::CopyDeviceToHost({resultsHost.data() + i * batchData.shapeSize, batchData.shapeSize}, outputShape));
    }
    ASSERT_TRUE(!batchData.cudaStream.Synchronize());

    #if 0
    // Even with these library functions which do not involve custom CUDA kernels,
    // the implementations might return slightly different results.
    ASSERT_EQ(expectedResultsHost, resultsHost);
    #elif 0
    // This is the right idea, but Google Test uses 4 ULPs of tolerance, which is not enough.
    for (std::size_t i = 0; i < batchData.shapeSize * batchData.nbTracks; ++i) {
      ASSERT_FLOAT_EQ(expectedResultsHost[i], resultsHost[i]) << " at index " << i;
    }
    #else
    // Even with more ULPs of tolerance, we get some failures in very small numbers.
    // So we go for an absolute tolerance.
    for (std::size_t i = 0; i < batchData.shapeSize * batchData.nbTracks; ++i) {
      ASSERT_NEAR(expectedResultsHost[i], resultsHost[i], 1e-5f) << " at index " << i;
    }
    #endif
  }
}

TEST(TestCoreBatchPcaAnimator, Performance) {
  cudaEvent_t start;
  cudaEvent_t end;
  ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);

  for (const auto nbTracks : {1, 8, 16}) {
    std::cout << "Benchmarking for " << nbTracks << " tracks..." << std::endl;
    BatchData batchData = BuildTestData(nbTracks);

    // Benchmark implementations.
    for (const auto& implementation : kImplementations) {
      std::cout << "  Benchmarking \"" << implementation.first << "\" implementation..." << std::endl;
      const auto animator = nva2x::ToUniquePtr(implementation.second());
      ASSERT_TRUE(!animator->Init(batchData.nbTracks));
      ASSERT_TRUE(!animator->SetCudaStream(batchData.cudaStream.Data()));
      ASSERT_TRUE(!animator->SetAnimatorData(batchData.initData));

      const std::size_t kWarmupIterations = 10;
      const std::size_t kNbBenchmarkIterations = 100;

      for (std::size_t i = 0; i < kWarmupIterations; ++i) {
        ASSERT_TRUE(!animator->Animate(
          batchData.inferenceOutputBuffers.GetInferenceResult(0, batchData.nbTracks),
          batchData.coefsInfo,
          batchData.resultsDevice,
          batchData.shapesInfo
          ));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      }

      float totalTime = 0.0f;
      float minTime = std::numeric_limits<float>::max();
      for (std::size_t i = 0; i < kNbBenchmarkIterations; ++i) {
        ASSERT_EQ(cudaEventRecord(start, batchData.cudaStream.Data()), cudaSuccess);
        ASSERT_TRUE(!animator->Animate(
          batchData.inferenceOutputBuffers.GetInferenceResult(0, batchData.nbTracks),
          batchData.coefsInfo,
          batchData.resultsDevice,
          batchData.shapesInfo
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
