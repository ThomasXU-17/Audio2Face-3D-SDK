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

#include "test_core_batch_skin_animator_cuda.h"

#include <gtest/gtest.h>

#include <cmath>


namespace test {

using namespace nva2f;

//
// Base implementation, used by the other implementations.
//
class MultiTrackAnimatorSkinBase : public IMultiTrackAnimatorSkin {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const Params& params, std::size_t nbTracks) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data, float dt) override; // GPU Async

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
  float _dt{-1.0f};
  std::size_t _poseSize{0};
};

std::error_code MultiTrackAnimatorSkinBase::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinBase::Init(const Params& params, std::size_t nbTracks) {
  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  _params.resize(nbTracks, params);
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinBase::SetAnimatorData(const HostData& data, float dt) {
  A2F_CHECK_ERROR_WITH_MSG(data.neutralPose.Size() % 3 == 0, "Neutral pose size must be a multiple of 3", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(data.neutralPose.Size() == data.lipOpenPoseDelta.Size(), "Neutral pose size does not match lip open pose delta size", nva2x::ErrorCode::eMismatch);
  A2F_CHECK_ERROR_WITH_MSG(data.neutralPose.Size() == data.eyeClosePoseDelta.Size(), "Neutral pose size does not match eye close pose delta size", nva2x::ErrorCode::eMismatch);
  A2F_CHECK_ERROR_WITH_MSG(data.neutralPose.Size() == data.eyeClosePoseDelta.Size(), "Neutral pose size does not match eye close pose delta size", nva2x::ErrorCode::eMismatch);

  _dt = dt;

  _poseSize = data.neutralPose.Size();

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinBase::SetParameters(std::size_t trackIndex, const Params& params) {
  return nva2x::ErrorCode::eUnsupported;
}

const MultiTrackAnimatorSkinBase::Params* MultiTrackAnimatorSkinBase::GetParameters(std::size_t trackIndex) const {
  return nullptr;
}

std::error_code MultiTrackAnimatorSkinBase::Reset(std::size_t trackIndex) {
  return nva2x::ErrorCode::eUnsupported;
}

std::error_code MultiTrackAnimatorSkinBase::SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) {
  return nva2x::ErrorCode::eUnsupported;
}

std::error_code MultiTrackAnimatorSkinBase::Animate(
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

void MultiTrackAnimatorSkinBase::Destroy() {
  delete this;
}




//
// Reference implementation, uses the single track implementation.
//
class MultiTrackAnimatorSkinReference : public MultiTrackAnimatorSkinBase {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const Params& params, std::size_t nbTracks) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data, float dt) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async

private:
  std::vector<AnimatorSkin> _animators;
};

std::error_code MultiTrackAnimatorSkinReference::SetCudaStream(cudaStream_t cudaStream) {
  A2F_CHECK_RESULT(MultiTrackAnimatorSkinBase::SetCudaStream(cudaStream));
  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.SetCudaStream(_cudaStream), "Unable to set CUDA stream on animator");
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinReference::Init(const Params& params, std::size_t nbTracks) {
  A2F_CHECK_RESULT(MultiTrackAnimatorSkinBase::Init(params, nbTracks));
  _animators.resize(nbTracks);
  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.SetCudaStream(_cudaStream), "Unable to set CUDA stream on animator");
    A2F_CHECK_RESULT_WITH_MSG(animator.Init(params), "Unable to initialize animator");
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinReference::SetAnimatorData(const HostData& data, float dt) {
  A2F_CHECK_RESULT(MultiTrackAnimatorSkinBase::SetAnimatorData(data, dt));

  for (auto& animator : _animators) {
    A2F_CHECK_RESULT_WITH_MSG(animator.SetAnimatorData(data), "Unable to set animator data");
  }

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinReference::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  A2F_CHECK_RESULT(MultiTrackAnimatorSkinBase::Animate(inputDeltas, inputDeltasInfo, outputVertices, outputVerticesInfo));
  for (std::size_t i = 0; i < _animators.size(); ++i) {
      const auto input = inputDeltas.View(
        i * inputDeltasInfo.stride + inputDeltasInfo.offset, inputDeltasInfo.size
        );
      const auto output = outputVertices.View(
        i * outputVerticesInfo.stride + outputVerticesInfo.offset, outputVerticesInfo.size
        );
      A2F_CHECK_RESULT_WITH_MSG(_animators[i].Animate(input, _dt, output), "Unable to animate animator");
  }
  return nva2x::ErrorCode::eSuccess;
}




//
// Fused kernel implementation, does all processing in a single kernel.
//
class MultiTrackAnimatorSkinFusedKernel : public MultiTrackAnimatorSkinBase {
public:
  std::error_code SetAnimatorData(const HostData& data, float dt) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async

protected:
  AnimatorSkin::Data _data;
  nva2x::DeviceTensorFloat _faceMaskLower;
  nva2x::DeviceTensorFloat _interpLower;
  nva2x::DeviceTensorFloat _interpUpper;
  std::vector<bool> _interpInitialized;
  static constexpr std::size_t kInterpDegree = 2;
};

std::error_code MultiTrackAnimatorSkinFusedKernel::SetAnimatorData(const HostData& data, float dt) {
  A2F_CHECK_RESULT(MultiTrackAnimatorSkinBase::SetAnimatorData(data, dt));

  A2F_CHECK_RESULT_WITH_MSG(_data.Init(data, _cudaStream), "Unable to initialize animator data");

  const auto poseSize = data.neutralPose.Size();
  const auto numVertices = poseSize / 3;
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    _faceMaskLower.Allocate(numVertices * nbTracks), "Unable to allocate face mask lower"
    );
  for (std::size_t i = 0; i < nbTracks; ++i) {
    A2F_CHECK_RESULT_WITH_MSG(
      CalculateFaceMaskLower(
        _faceMaskLower.Data() + i * numVertices,
        _data.GetDeviceView().neutralPose.Data(),
        poseSize,
        _params[i].faceMaskLevel,
        _params[i].faceMaskSoftness,
        _cudaStream
        ),
      "Unable to calculate face mask lower"
      );
  }

  A2F_CHECK_RESULT_WITH_MSG(_interpLower.Allocate(poseSize * nbTracks * kInterpDegree), "Unable to allocate interp lower");
  A2F_CHECK_RESULT_WITH_MSG(_interpUpper.Allocate(poseSize * nbTracks * kInterpDegree), "Unable to allocate interp upper");
  _interpInitialized.clear();
  _interpInitialized.resize(nbTracks, false);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinFusedKernel::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  const auto poseSize = _data.GetDeviceView().neutralPose.Size();
  const auto numVertices = poseSize / 3;
  const auto nbTracks = _params.size();

  for (std::size_t i = 0; i < nbTracks; ++i) {
    const auto input = inputDeltas.View(
      i * inputDeltasInfo.stride + inputDeltasInfo.offset, inputDeltasInfo.size
      );
    const auto output = outputVertices.View(
      i * outputVerticesInfo.stride + outputVerticesInfo.offset, outputVerticesInfo.size
      );

    A2F_CHECK_RESULT_WITH_MSG(
      AnimateFusedKernel(
        output.Data(),
        input.Data(),
        _data.GetDeviceView().eyeClosePoseDelta.Data(),
        _data.GetDeviceView().lipOpenPoseDelta.Data(),
        _data.GetDeviceView().neutralPose.Data(),
        _faceMaskLower.Data() + i * numVertices,
        _interpLower.Data() + i * poseSize * kInterpDegree,
        _interpUpper.Data() + i * poseSize * kInterpDegree,
        _params[i].skinStrength,
        _params[i].eyelidOpenOffset,
        _params[i].blinkOffset,
        _params[i].blinkStrength,
        _params[i].lipOpenOffset,
        _interpInitialized[i] ? 1.0f - std::pow(0.5f, _dt / _params[i].lowerFaceSmoothing) : -1.0f,
        _interpInitialized[i] ? 1.0f - std::pow(0.5f, _dt / _params[i].upperFaceSmoothing) : -1.0f,
        _params[i].lowerFaceStrength,
        _params[i].upperFaceStrength,
        poseSize,
        _cudaStream
        ),
      "Unable to animate animator"
      );
    _interpInitialized[i] = true;
  }

  return nva2x::ErrorCode::eSuccess;
}




//
// Fused kernel implementation, does all processing in a single kernel, batched.
//
class MultiTrackAnimatorSkinFusedKernelBatched : public MultiTrackAnimatorSkinFusedKernel {
public:
  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) override; // GPU Async
};

std::error_code MultiTrackAnimatorSkinFusedKernelBatched::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  const auto poseSize = _data.GetDeviceView().neutralPose.Size();
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    AnimateFusedKernelBatched(
      outputVertices.Data(),
      outputVerticesInfo.offset,
      outputVerticesInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _data.GetDeviceView().eyeClosePoseDelta.Data(),
      _data.GetDeviceView().lipOpenPoseDelta.Data(),
      _data.GetDeviceView().neutralPose.Data(),
      _faceMaskLower.Data(),
      _interpLower.Data(),
      _interpUpper.Data(),
      // FIXME: This is hack, we only pass a single track parameters, but they could be different for each track.
      _params[0].skinStrength,
      _params[0].eyelidOpenOffset,
      _params[0].blinkOffset,
      _params[0].blinkStrength,
      _params[0].lipOpenOffset,
      _interpInitialized[0] ? 1.0f - std::pow(0.5f, _dt / _params[0].lowerFaceSmoothing) : -1.0f,
      _interpInitialized[0] ? 1.0f - std::pow(0.5f, _dt / _params[0].upperFaceSmoothing) : -1.0f,
      _params[0].lowerFaceStrength,
      _params[0].upperFaceStrength,
      poseSize,
      nbTracks,
      _cudaStream
      ),
    "Unable to animate animator"
    );
  _interpInitialized[0] = true;

  return nva2x::ErrorCode::eSuccess;
}




//
// Fused kernel implementation, does all processing in a single kernel, with support per-track parameters.
//
class MultiTrackAnimatorSkinFusedKernelBatchedParams : public MultiTrackAnimatorSkinBase {
public:
  std::error_code SetAnimatorData(const HostData& data, float dt) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async

private:
  AnimatorSkin::Data _data;
  nva2x::DeviceTensorFloat _faceMaskLower;
  nva2x::DeviceTensorFloat _interpLower;
  nva2x::DeviceTensorFloat _interpUpper;
  nva2x::DeviceTensorFloat _skinStrengths;
  nva2x::DeviceTensorFloat _eyelidOpenOffsets;
  nva2x::DeviceTensorFloat _blinkOffsets;
  nva2x::DeviceTensorFloat _blinkStrengths;
  nva2x::DeviceTensorFloat _lipOpenOffsets;
  nva2x::DeviceTensorFloat _lowerFaceAlphas;
  nva2x::DeviceTensorFloat _upperFaceAlphas;
  nva2x::DeviceTensorFloat _lowerFaceStrengths;
  nva2x::DeviceTensorFloat _upperFaceStrengths;
  nva2x::DeviceTensorBool _initializeds;
  nva2x::DeviceTensorBool _initializedsTrue;
  nva2x::DeviceTensorBoolConstView _initializedsToUse;
  static constexpr std::size_t kInterpDegree = 2;
};

std::error_code MultiTrackAnimatorSkinFusedKernelBatchedParams::SetAnimatorData(const HostData& data, float dt) {
  A2F_CHECK_RESULT(MultiTrackAnimatorSkinBase::SetAnimatorData(data, dt));

  A2F_CHECK_RESULT_WITH_MSG(_data.Init(data, _cudaStream), "Unable to initialize animator data");

  const auto poseSize = data.neutralPose.Size();
  const auto numVertices = poseSize / 3;
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    _faceMaskLower.Allocate(numVertices * nbTracks), "Unable to allocate face mask lower"
    );
  for (std::size_t i = 0; i < nbTracks; ++i) {
    A2F_CHECK_RESULT_WITH_MSG(
      CalculateFaceMaskLower(
        _faceMaskLower.Data() + i * numVertices,
        _data.GetDeviceView().neutralPose.Data(),
        poseSize,
        _params[i].faceMaskLevel,
        _params[i].faceMaskSoftness,
        _cudaStream
        ),
      "Unable to calculate face mask lower"
      );
  }

  A2F_CHECK_RESULT_WITH_MSG(_interpLower.Allocate(poseSize * nbTracks * kInterpDegree), "Unable to allocate interp lower");
  A2F_CHECK_RESULT_WITH_MSG(_interpUpper.Allocate(poseSize * nbTracks * kInterpDegree), "Unable to allocate interp upper");

  std::vector<float> skinStrengthsHost(nbTracks);
  std::vector<float> eyelidOpenOffsetsHost(nbTracks);
  std::vector<float> blinkOffsetsHost(nbTracks);
  std::vector<float> blinkStrengthsHost(nbTracks);
  std::vector<float> lipOpenOffsetsHost(nbTracks);
  std::vector<float> lowerFaceAlphasHost(nbTracks);
  std::vector<float> upperFaceAlphasHost(nbTracks);
  std::vector<float> lowerFaceStrengthsHost(nbTracks);
  std::vector<float> upperFaceStrengthsHost(nbTracks);
  auto initializedsHost = std::make_unique<bool[]>(nbTracks);
  auto initializedsTrueHost = std::make_unique<bool[]>(nbTracks);

  for (std::size_t i = 0; i < nbTracks; ++i) {
    skinStrengthsHost[i] = _params[i].skinStrength;
    eyelidOpenOffsetsHost[i] = _params[i].eyelidOpenOffset;
    blinkOffsetsHost[i] = _params[i].blinkOffset;
    blinkStrengthsHost[i] = _params[i].blinkStrength;
    lipOpenOffsetsHost[i] = _params[i].lipOpenOffset;
    lowerFaceAlphasHost[i] = _params[i].lowerFaceSmoothing > 0.0f ? 1.0f - std::pow(0.5f, dt / _params[i].lowerFaceSmoothing) : -1.0f;
    upperFaceAlphasHost[i] = _params[i].upperFaceSmoothing > 0.0f ? 1.0f - std::pow(0.5f, dt / _params[i].upperFaceSmoothing) : -1.0f;
    lowerFaceStrengthsHost[i] = _params[i].lowerFaceStrength;
    upperFaceStrengthsHost[i] = _params[i].upperFaceStrength;
    initializedsHost[i] = false;
    initializedsTrueHost[i] = true;
  }

  A2F_CHECK_RESULT_WITH_MSG(_skinStrengths.Init(nva2x::ToConstView(skinStrengthsHost), _cudaStream), "Unable to initialize skin strengths");
  A2F_CHECK_RESULT_WITH_MSG(_eyelidOpenOffsets.Init(nva2x::ToConstView(eyelidOpenOffsetsHost), _cudaStream), "Unable to initialize eyelid open offsets");
  A2F_CHECK_RESULT_WITH_MSG(_blinkOffsets.Init(nva2x::ToConstView(blinkOffsetsHost), _cudaStream), "Unable to initialize blink offsets");
  A2F_CHECK_RESULT_WITH_MSG(_blinkStrengths.Init(nva2x::ToConstView(blinkStrengthsHost), _cudaStream), "Unable to initialize blink strengths");
  A2F_CHECK_RESULT_WITH_MSG(_lipOpenOffsets.Init(nva2x::ToConstView(lipOpenOffsetsHost), _cudaStream), "Unable to initialize lip open offsets");
  A2F_CHECK_RESULT_WITH_MSG(_lowerFaceAlphas.Init(nva2x::ToConstView(lowerFaceAlphasHost), _cudaStream), "Unable to initialize lower face alphas");
  A2F_CHECK_RESULT_WITH_MSG(_upperFaceAlphas.Init(nva2x::ToConstView(upperFaceAlphasHost), _cudaStream), "Unable to initialize upper face alphas");
  A2F_CHECK_RESULT_WITH_MSG(_lowerFaceStrengths.Init(nva2x::ToConstView(lowerFaceStrengthsHost), _cudaStream), "Unable to initialize lower face strengths");
  A2F_CHECK_RESULT_WITH_MSG(_upperFaceStrengths.Init(nva2x::ToConstView(upperFaceStrengthsHost), _cudaStream), "Unable to initialize upper face strengths");
  A2F_CHECK_RESULT_WITH_MSG(_initializeds.Init({initializedsHost.get(), nbTracks}, _cudaStream), "Unable to initialize initialized");
  A2F_CHECK_RESULT_WITH_MSG(_initializedsTrue.Init({initializedsTrueHost.get(), nbTracks}, _cudaStream), "Unable to initialize initialized true");

  _initializedsToUse = _initializeds;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinFusedKernelBatchedParams::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  const auto poseSize = _data.GetDeviceView().neutralPose.Size();
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    AnimateFusedKernelBatchedParams(
      outputVertices.Data(),
      outputVerticesInfo.offset,
      outputVerticesInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _data.GetDeviceView().eyeClosePoseDelta.Data(),
      _data.GetDeviceView().lipOpenPoseDelta.Data(),
      _data.GetDeviceView().neutralPose.Data(),
      _faceMaskLower.Data(),
      _interpLower.Data(),
      _interpUpper.Data(),
      _skinStrengths.Data(),
      _eyelidOpenOffsets.Data(),
      _blinkOffsets.Data(),
      _blinkStrengths.Data(),
      _lipOpenOffsets.Data(),
      _lowerFaceAlphas.Data(),
      _upperFaceAlphas.Data(),
      _lowerFaceStrengths.Data(),
      _upperFaceStrengths.Data(),
      _initializedsToUse.Data(),
      poseSize,
      nbTracks,
      _cudaStream
      ),
    "Unable to animate animator"
    );
  _initializedsToUse = _initializedsTrue;

  return nva2x::ErrorCode::eSuccess;
}




//
// Fused kernel implementation, does all processing in a single kernel, with support per-track parameters,
// and packed parameters.
//
class MultiTrackAnimatorSkinFusedKernelBatchedParamsPacked : public MultiTrackAnimatorSkinBase {
public:
  std::error_code SetAnimatorData(const HostData& data, float dt) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async

private:
  AnimatorSkin::Data _data;
  nva2x::DeviceTensorFloat _faceMaskLower;
  nva2x::DeviceTensorFloat _interpLower;
  nva2x::DeviceTensorFloat _interpUpper;
  nva2x::DeviceTensorFloat _skinParams;
  std::size_t _skinParamsStride{0};
  nva2x::DeviceTensorBool _initializeds;
  nva2x::DeviceTensorBool _initializedsTrue;
  nva2x::DeviceTensorBoolConstView _initializedsToUse;
  static constexpr std::size_t kInterpDegree = 2;
};

std::error_code MultiTrackAnimatorSkinFusedKernelBatchedParamsPacked::SetAnimatorData(const HostData& data, float dt) {
  A2F_CHECK_RESULT(MultiTrackAnimatorSkinBase::SetAnimatorData(data, dt));

  A2F_CHECK_RESULT_WITH_MSG(_data.Init(data, _cudaStream), "Unable to initialize animator data");

  const auto poseSize = data.neutralPose.Size();
  const auto numVertices = poseSize / 3;
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    _faceMaskLower.Allocate(numVertices * nbTracks), "Unable to allocate face mask lower"
    );
  for (std::size_t i = 0; i < nbTracks; ++i) {
    A2F_CHECK_RESULT_WITH_MSG(
      CalculateFaceMaskLower(
        _faceMaskLower.Data() + i * numVertices,
        _data.GetDeviceView().neutralPose.Data(),
        poseSize,
        _params[i].faceMaskLevel,
        _params[i].faceMaskSoftness,
        _cudaStream
        ),
      "Unable to calculate face mask lower"
      );
  }

  A2F_CHECK_RESULT_WITH_MSG(_interpLower.Allocate(poseSize * nbTracks * kInterpDegree), "Unable to allocate interp lower");
  A2F_CHECK_RESULT_WITH_MSG(_interpUpper.Allocate(poseSize * nbTracks * kInterpDegree), "Unable to allocate interp upper");

  _skinParamsStride = 9;
  std::vector<float> paramsHost(nbTracks * _skinParamsStride);
  auto initializedsHost = std::make_unique<bool[]>(nbTracks);
  auto initializedsTrueHost = std::make_unique<bool[]>(nbTracks);

  for (std::size_t i = 0; i < nbTracks; ++i) {
    paramsHost[i * _skinParamsStride + 0] = _params[i].skinStrength;
    paramsHost[i * _skinParamsStride + 1] = _params[i].eyelidOpenOffset;
    paramsHost[i * _skinParamsStride + 2] = _params[i].blinkOffset;
    paramsHost[i * _skinParamsStride + 3] = _params[i].blinkStrength;
    paramsHost[i * _skinParamsStride + 4] = _params[i].lipOpenOffset;
    paramsHost[i * _skinParamsStride + 5] = _params[i].lowerFaceSmoothing > 0.0f ? 1.0f - std::pow(0.5f, dt / _params[i].lowerFaceSmoothing) : -1.0f;
    paramsHost[i * _skinParamsStride + 6] = _params[i].upperFaceSmoothing > 0.0f ? 1.0f - std::pow(0.5f, dt / _params[i].upperFaceSmoothing) : -1.0f;
    paramsHost[i * _skinParamsStride + 7] = _params[i].lowerFaceStrength;
    paramsHost[i * _skinParamsStride + 8] = _params[i].upperFaceStrength;
    initializedsHost[i] = false;
    initializedsTrueHost[i] = true;
  }

  A2F_CHECK_RESULT_WITH_MSG(_skinParams.Init(nva2x::ToConstView(paramsHost), _cudaStream), "Unable to initialize skin params");
  A2F_CHECK_RESULT_WITH_MSG(_initializeds.Init({initializedsHost.get(), nbTracks}, _cudaStream), "Unable to initialize initialized");
  A2F_CHECK_RESULT_WITH_MSG(_initializedsTrue.Init({initializedsTrueHost.get(), nbTracks}, _cudaStream), "Unable to initialize initialized true");

  _initializedsToUse = _initializeds;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinFusedKernelBatchedParamsPacked::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  const auto poseSize = _data.GetDeviceView().neutralPose.Size();
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    AnimateFusedKernelBatchedParamsPacked(
      outputVertices.Data(),
      outputVerticesInfo.offset,
      outputVerticesInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _data.GetDeviceView().eyeClosePoseDelta.Data(),
      _data.GetDeviceView().lipOpenPoseDelta.Data(),
      _data.GetDeviceView().neutralPose.Data(),
      _faceMaskLower.Data(),
      _interpLower.Data(),
      _interpUpper.Data(),
      _skinParams.Data(),
      _skinParamsStride,
      _initializedsToUse.Data(),
      poseSize,
      nbTracks,
      _cudaStream
      ),
    "Unable to animate animator"
    );
  _initializedsToUse = _initializedsTrue;

  return nva2x::ErrorCode::eSuccess;
}




//
// Packed implementation, packs the different arrays together.
//
class MultiTrackAnimatorSkinPacked : public MultiTrackAnimatorSkinBase {
public:
  std::error_code SetAnimatorData(const HostData& data, float dt) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async

protected:
  nva2x::DeviceTensorFloat _animatorData;
  std::size_t _animatorDataStride{0};
  nva2x::DeviceTensorFloat _faceMaskLower;
  nva2x::DeviceTensorFloat _interpData;
  std::size_t _interpDataStride{0};
  nva2x::DeviceTensorFloat _skinParams;
  std::size_t _skinParamsStride{0};
  nva2x::DeviceTensorBool _initializeds;
  nva2x::DeviceTensorBool _initializedsTrue;
  nva2x::DeviceTensorBoolConstView _initializedsToUse;
  static constexpr std::size_t kInterpDegree = 2;
};

std::error_code MultiTrackAnimatorSkinPacked::SetAnimatorData(const HostData& data, float dt) {
  A2F_CHECK_RESULT(MultiTrackAnimatorSkinBase::SetAnimatorData(data, dt));

  const auto poseSize = data.neutralPose.Size();
  const auto numVertices = poseSize / 3;
  const auto nbTracks = _params.size();

  _animatorDataStride = 3;
  std::vector<float> animatorDataHost(poseSize * _animatorDataStride);
  for (std::size_t i = 0; i < poseSize; ++i) {
    animatorDataHost[i * _animatorDataStride + 0] = data.eyeClosePoseDelta.Data()[i];
    animatorDataHost[i * _animatorDataStride + 1] = data.lipOpenPoseDelta.Data()[i];
    animatorDataHost[i * _animatorDataStride + 2] = data.neutralPose.Data()[i];
  }
  A2F_CHECK_RESULT_WITH_MSG(_animatorData.Init(nva2x::ToConstView(animatorDataHost), _cudaStream), "Unable to initialize animator data");

  A2F_CHECK_RESULT_WITH_MSG(
    _faceMaskLower.Allocate(numVertices * nbTracks), "Unable to allocate face mask lower"
    );
  for (std::size_t i = 0; i < nbTracks; ++i) {
    A2F_CHECK_RESULT_WITH_MSG(
      CalculateFaceMaskLowerPacked(
        _faceMaskLower.Data() + i * numVertices,
        _animatorData.Data(),
        _animatorDataStride,
        poseSize,
        _params[i].faceMaskLevel,
        _params[i].faceMaskSoftness,
        _cudaStream
        ),
      "Unable to calculate face mask lower"
      );
  }

  _interpDataStride = kInterpDegree * 2;
  A2F_CHECK_RESULT_WITH_MSG(_interpData.Allocate(poseSize * nbTracks * _interpDataStride), "Unable to allocate interp data");

  _skinParamsStride = 9;
  std::vector<float> paramsHost(nbTracks * _skinParamsStride);
  auto initializedsHost = std::make_unique<bool[]>(nbTracks);
  auto initializedsTrueHost = std::make_unique<bool[]>(nbTracks);

  for (std::size_t i = 0; i < nbTracks; ++i) {
    paramsHost[i * _skinParamsStride + 0] = _params[i].skinStrength;
    paramsHost[i * _skinParamsStride + 1] = _params[i].eyelidOpenOffset;
    paramsHost[i * _skinParamsStride + 2] = _params[i].blinkOffset;
    paramsHost[i * _skinParamsStride + 3] = _params[i].blinkStrength;
    paramsHost[i * _skinParamsStride + 4] = _params[i].lipOpenOffset;
    paramsHost[i * _skinParamsStride + 5] = _params[i].lowerFaceSmoothing > 0.0f ? 1.0f - std::pow(0.5f, dt / _params[i].lowerFaceSmoothing) : -1.0f;
    paramsHost[i * _skinParamsStride + 6] = _params[i].upperFaceSmoothing > 0.0f ? 1.0f - std::pow(0.5f, dt / _params[i].upperFaceSmoothing) : -1.0f;
    paramsHost[i * _skinParamsStride + 7] = _params[i].lowerFaceStrength;
    paramsHost[i * _skinParamsStride + 8] = _params[i].upperFaceStrength;
    initializedsHost[i] = false;
    initializedsTrueHost[i] = true;
  }

  A2F_CHECK_RESULT_WITH_MSG(_skinParams.Init(nva2x::ToConstView(paramsHost), _cudaStream), "Unable to initialize skin params");
  A2F_CHECK_RESULT_WITH_MSG(_initializeds.Init({initializedsHost.get(), nbTracks}, _cudaStream), "Unable to initialize initialized");
  A2F_CHECK_RESULT_WITH_MSG(_initializedsTrue.Init({initializedsTrueHost.get(), nbTracks}, _cudaStream), "Unable to initialize initialized true");
  A2F_CUDA_CHECK_ERROR(cudaStreamSynchronize(_cudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError);

  _initializedsToUse = _initializeds;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinPacked::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  const auto poseSize = _animatorData.Size() / _animatorDataStride;
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    AnimatePacked(
      outputVertices.Data(),
      outputVerticesInfo.offset,
      outputVerticesInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _animatorData.Data(),
      _animatorDataStride,
      _faceMaskLower.Data(),
      _interpData.Data(),
      _interpDataStride,
      _skinParams.Data(),
      _skinParamsStride,
      _initializedsToUse.Data(),
      poseSize,
      nbTracks,
      _cudaStream
      ),
    "Unable to animate animator"
    );
  _initializedsToUse = _initializedsTrue;

  return nva2x::ErrorCode::eSuccess;
}




//
// Packed implementation, packs the different arrays together.
//
class MultiTrackAnimatorSkinPackedInOut : public MultiTrackAnimatorSkinPacked {
public:
  std::error_code SetAnimatorData(const HostData& data, float dt) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async

private:
  nva2x::DeviceTensorFloat _outInterpData;
};

std::error_code MultiTrackAnimatorSkinPackedInOut::SetAnimatorData(const HostData& data, float dt) {
  A2F_CHECK_RESULT(MultiTrackAnimatorSkinPacked::SetAnimatorData(data, dt));

  A2F_CHECK_RESULT_WITH_MSG(_outInterpData.Allocate(_interpData.Size()), "Unable to allocate out interp data");

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinPackedInOut::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  const auto poseSize = _animatorData.Size() / _animatorDataStride;
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    AnimatePackedInOut(
      outputVertices.Data(),
      outputVerticesInfo.offset,
      outputVerticesInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _animatorData.Data(),
      _animatorDataStride,
      _faceMaskLower.Data(),
      _interpData.Data(),
      _outInterpData.Data(),
      _interpDataStride,
      _skinParams.Data(),
      _skinParamsStride,
      _initializedsToUse.Data(),
      poseSize,
      nbTracks,
      _cudaStream
      ),
    "Unable to animate animator"
    );
  _initializedsToUse = _initializedsTrue;

  std::swap(_interpData, _outInterpData);

  return nva2x::ErrorCode::eSuccess;
}




//
// Packed implementation, controlling active and initialized state.
//
class MultiTrackAnimatorSkinPackedControl : public MultiTrackAnimatorSkinPacked {
public:
  std::error_code SetAnimatorData(const HostData& data, float dt) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async

protected:
  std::vector<std::uint64_t> _initializedActivesHost;
};

std::error_code MultiTrackAnimatorSkinPackedControl::SetAnimatorData(const HostData& data, float dt) {
  A2F_CHECK_RESULT(MultiTrackAnimatorSkinPacked::SetAnimatorData(data, dt));

  const auto nbTracks = _params.size();

  _initializedActivesHost.clear();
  _initializedActivesHost.resize((nbTracks + 31) / 32, ~std::uint64_t(0));

  A2F_CHECK_RESULT_WITH_MSG(
    _initializedsTrue.Init(
      {reinterpret_cast<const bool*>(_initializedActivesHost.data()),
       _initializedActivesHost.size() * sizeof(std::uint64_t) / sizeof(bool)},
      _cudaStream
      ),
    "Unable to initialize initialized actives true"
    );

  std::fill(_initializedActivesHost.begin(), _initializedActivesHost.end(), 0);
  for (std::size_t i = 0; i < nbTracks; ++i) {
    _initializedActivesHost[i / 32] |= (0b01ULL << (2*(i % 32)));
  }

  A2F_CHECK_RESULT_WITH_MSG(
    _initializeds.Init(
      {reinterpret_cast<const bool*>(_initializedActivesHost.data()),
       _initializedActivesHost.size() * sizeof(std::uint64_t) / sizeof(bool)},
      _cudaStream
      ),
    "Unable to initialize initialized actives"
    );

  _initializedsToUse = _initializeds;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinPackedControl::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  const auto poseSize = _animatorData.Size() / _animatorDataStride;
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    AnimatePackedControl(
      outputVertices.Data(),
      outputVerticesInfo.offset,
      outputVerticesInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _animatorData.Data(),
      _animatorDataStride,
      _faceMaskLower.Data(),
      _interpData.Data(),
      _interpDataStride,
      _skinParams.Data(),
      _skinParamsStride,
      reinterpret_cast<const std::uint64_t*>(_initializedsToUse.Data()),
      poseSize,
      nbTracks,
      _cudaStream
      ),
    "Unable to animate animator"
    );
  _initializedsToUse = _initializedsTrue;

  return nva2x::ErrorCode::eSuccess;
}




//
// Packed implementation, controlling active and initialized state, setting it every frame.
//
class MultiTrackAnimatorSkinPackedControlSet : public MultiTrackAnimatorSkinPackedControl {
public:
  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async
};

std::error_code MultiTrackAnimatorSkinPackedControlSet::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  for (std::size_t i = 0; i < _initializedActivesHost.size(); ++i) {
    A2F_CHECK_RESULT_WITH_MSG(
      AnimatePackedControl_Set(
        reinterpret_cast<std::uint64_t*>(_initializeds.Data()),
        _initializedActivesHost.size(),
        i,
        _initializedActivesHost[i],
        _cudaStream
        ),
      "Unable to set initialized active"
      );
  }

  std::fill(_initializedActivesHost.begin(), _initializedActivesHost.end(), ~std::uint64_t(0));

  _initializedsToUse = _initializeds;

  return MultiTrackAnimatorSkinPackedControl::Animate(inputDeltas, inputDeltasInfo, outputVertices, outputVerticesInfo);
}




//
// Packed implementation, controlling active and initialized state..
//
class MultiTrackAnimatorSkinPackedControl2 : public MultiTrackAnimatorSkinPackedControl {
public:
  std::error_code SetAnimatorData(const HostData& data, float dt) override; // GPU Async

  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) override; // GPU Async
};

std::error_code MultiTrackAnimatorSkinPackedControl2::SetAnimatorData(const HostData& data, float dt) {
  A2F_CHECK_RESULT(MultiTrackAnimatorSkinPackedControl::SetAnimatorData(data, dt));

  const auto nbTracks = _params.size();

  _initializedActivesHost.clear();
  _initializedActivesHost.resize((nbTracks + 63) / 64, ~std::uint64_t(0));

  A2F_CHECK_RESULT_WITH_MSG(
    _initializedsTrue.Init(
      {reinterpret_cast<const bool*>(_initializedActivesHost.data()),
       _initializedActivesHost.size() * sizeof(std::uint64_t) / sizeof(bool)},
      _cudaStream
      ),
    "Unable to initialize initialized actives true"
    );

  std::fill(_initializedActivesHost.begin(), _initializedActivesHost.end(), 0);

  A2F_CHECK_RESULT_WITH_MSG(
    _initializeds.Init(
      {reinterpret_cast<const bool*>(_initializedActivesHost.data()),
       _initializedActivesHost.size() * sizeof(std::uint64_t) / sizeof(bool)},
      _cudaStream
      ),
    "Unable to initialize initialized actives"
    );

  _initializedsToUse = _initializeds;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkinPackedControl2::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  const auto poseSize = _animatorData.Size() / _animatorDataStride;
  const auto nbTracks = _params.size();

  A2F_CHECK_RESULT_WITH_MSG(
    AnimatePackedControl2(
      outputVertices.Data(),
      outputVerticesInfo.offset,
      outputVerticesInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _animatorData.Data(),
      _animatorDataStride,
      _faceMaskLower.Data(),
      _interpData.Data(),
      _interpDataStride,
      _skinParams.Data(),
      _skinParamsStride,
      reinterpret_cast<const std::uint64_t*>(_initializedsToUse.Data()),
      poseSize,
      nbTracks,
      _cudaStream
      ),
    "Unable to animate animator"
    );
  _initializedsToUse = _initializedsTrue;

  return nva2x::ErrorCode::eSuccess;
}

} // namespace test


namespace {

using creator_func_t = test::IMultiTrackAnimatorSkin* (*)();
static const std::vector<std::pair<const char*, creator_func_t>> kImplementations {
  {"Reference", []() -> test::IMultiTrackAnimatorSkin* { return new test::MultiTrackAnimatorSkinReference; }},
  {"FusedKernel", []() -> test::IMultiTrackAnimatorSkin* { return new test::MultiTrackAnimatorSkinFusedKernel; }},
  {"FusedKernelBatched", []() -> test::IMultiTrackAnimatorSkin* { return new test::MultiTrackAnimatorSkinFusedKernelBatched; }},
  {"FusedKernelBatchedParams", []() -> test::IMultiTrackAnimatorSkin* { return new test::MultiTrackAnimatorSkinFusedKernelBatchedParams; }},
  {"FusedKernelBatchedParamsPacked", []() -> test::IMultiTrackAnimatorSkin* { return new test::MultiTrackAnimatorSkinFusedKernelBatchedParamsPacked; }},
  {"Packed", []() -> test::IMultiTrackAnimatorSkin* { return new test::MultiTrackAnimatorSkinPacked; }},
  {"PackedInOut", []() -> test::IMultiTrackAnimatorSkin* { return new test::MultiTrackAnimatorSkinPackedInOut; }},
  {"PackedControl", []() -> test::IMultiTrackAnimatorSkin* { return new test::MultiTrackAnimatorSkinPackedControl; }},
  {"PackedControlSet", []() -> test::IMultiTrackAnimatorSkin* { return new test::MultiTrackAnimatorSkinPackedControlSet; }},
  {"PackedControl2", []() -> test::IMultiTrackAnimatorSkin* { return new test::MultiTrackAnimatorSkinPackedControl2; }},
  {"Final", []() -> test::IMultiTrackAnimatorSkin* { return nva2f::CreateMultiTrackAnimatorSkin_INTERNAL(); }},
};


struct BatchData {
  std::size_t nbTracks;
  nva2x::CudaStream cudaStream;
  nva2x::UniquePtr<nva2f::IRegressionModel::IGeometryModelInfo> modelInfo;
  nva2f::AnimatorSkin::Params params;
  nva2f::IAnimatorSkin::HostData initData;
  nva2x::DeviceTensorFloat sourceData;
  float dt{1.0f / 60.0f};
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

  batchData.params = batchData.modelInfo->GetAnimatorParams().skin;
  batchData.initData = batchData.modelInfo->GetAnimatorData().GetAnimatorData().skin;

  // Generate source data.
  EXPECT_TRUE(!batchData.resultsBuffers.Init(batchData.modelInfo->GetNetworkInfo().GetNetworkInfo(), batchData.nbTracks));

  std::vector<float> sourceDataHost(
    batchData.resultsBuffers.GetResultTensor(batchData.nbTracks).Size() * batchData.nbIterations
    );
  FillRandom(sourceDataHost);
  EXPECT_TRUE(!batchData.sourceData.Init(nva2x::ToConstView(sourceDataHost)));

  batchData.resultsInfo = batchData.resultsBuffers.GetSkinBatchInfo();

  EXPECT_TRUE(!cudaDeviceSynchronize());

  return batchData;
}

}




TEST(TestCoreBatchSkinAnimator, Correctness) {
  const auto nbTracks = 10;
  BatchData batchData = BuildTestData(nbTracks);

  // Generate expected results.
  nva2f::AnimatorSkin singleAnimator;
  ASSERT_TRUE(!singleAnimator.Init(batchData.params));
  ASSERT_TRUE(!singleAnimator.SetCudaStream(batchData.cudaStream.Data()));
  ASSERT_TRUE(!singleAnimator.SetAnimatorData(batchData.initData));

  std::vector<float> expectedResultsHost(batchData.initData.neutralPose.Size() * batchData.nbTracks);
  for (std::size_t trackIndex = 0; trackIndex < batchData.nbTracks; ++trackIndex) {
    const auto inputDeltas = batchData.resultsBuffers.GetResultSkinGeometry(trackIndex);
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
    ASSERT_TRUE(!animator->SetAnimatorData(batchData.initData, batchData.dt));

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
      const auto outputShape = batchData.resultsBuffers.GetResultSkinGeometry(i);
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

TEST(TestCoreBatchSkinAnimator, Performance) {
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
      ASSERT_TRUE(!animator->SetAnimatorData(batchData.initData, batchData.dt));

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
