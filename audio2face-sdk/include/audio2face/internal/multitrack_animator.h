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

#include "audio2face/multitrack_animator.h"
#include "audio2face/internal/animator.h"
#include "audio2face/internal/cublas_handle.h"

#include <vector>

namespace nva2f {

class MultiTrackAnimatorPcaReconstruction : public IMultiTrackAnimatorPcaReconstruction {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(std::size_t nbTracks) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

  std::error_code Reset(std::size_t trackIndex) override;
  std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) override;
  std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputPcaCoefs, const nva2x::TensorBatchInfo& inputPcaCoefsInfo,
    nva2x::DeviceTensorFloatView outputShapes, const nva2x::TensorBatchInfo& outputShapesInfo,
    std::size_t batchSize = 0
    ) override; // GPU Async

  void Destroy() override;

private:
  cudaStream_t _cudaStream{nullptr};
  std::size_t _nbTracks{0};
  CublasHandle _cublasHandle;
  nva2f::AnimatorPcaReconstruction::Data _data;
  bool _initialized{false};
  bool _animatorDataIsSet{false};
};


class MultiTrackAnimatorSkin : public IMultiTrackAnimatorSkin {
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

private:
  std::error_code SetParametersInternal(
    std::size_t trackIndex, const Params& params, std::vector<float>& paramsHost, bool updateMask
    );

  cudaStream_t _cudaStream{nullptr};
  std::vector<Params> _params;

  // This is just a more compact bitset for the active tracks.
  // We would rather use an off-the-shelf bitset, but std ones does not support a dynamic number of bits.
  using bits_type = std::uint64_t;
  static constexpr const std::size_t nb_bits = 8 * sizeof(bits_type);
  std::vector<bits_type> _activeTracks;
  std::vector<bits_type> _initializedTracks;
  nva2x::DeviceTensorUInt64 _activeTracksDevice;
  nva2x::DeviceTensorUInt64 _initializedTracksDevice;

  nva2x::DeviceTensorFloat _animatorData;
  std::size_t _animatorDataStride{0};
  nva2x::DeviceTensorFloat _faceMaskLower;
  nva2x::DeviceTensorFloat _interpData;
  std::size_t _interpDataStride{0};
  nva2x::DeviceTensorFloat _skinParams;
  std::size_t _skinParamsStride{0};
  float _dt{0.f};
  static constexpr std::size_t kInterpDegree = 2;

  bool _initialized{false};
  bool _animatorDataIsSet{false};
};


class MultiTrackAnimatorTongue : public IMultiTrackAnimatorTongue {
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

private:
  std::error_code SetParametersInternal(std::size_t trackIndex, const Params& params, std::vector<float>& paramsHost);

  cudaStream_t _cudaStream{nullptr};
  std::vector<Params> _params;

  AnimatorTongue::Data _data;
  nva2x::DeviceTensorFloat _tongueParams;
  std::size_t _tongueParamsStride{0};

  bool _initialized{false};
  bool _animatorDataIsSet{false};
};


class MultiTrackAnimatorTeeth : public IMultiTrackAnimatorTeeth {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const Params& params, std::size_t nbTracks) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async

  std::error_code SetParameters(std::size_t trackIndex, const Params& params) override;
  const Params* GetParameters(std::size_t trackIndex) const override;

  std::error_code Reset(std::size_t trackIndex) override;
  std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) override;
  std::error_code ComputeJawTransform(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputTransforms, const nva2x::TensorBatchInfo& outputTransformsInfo
    ) override; // GPU Async

  void Destroy() override;

private:
  std::error_code SetParametersInternal(std::size_t trackIndex, const Params& params, std::vector<float>& paramsHost);

  cudaStream_t _cudaStream{nullptr};
  std::vector<Params> _params;

  struct DeviceData {
    nva2x::DeviceTensorFloatConstView neutralJaw;
  };
  class Data {
  public:
    std::error_code Init(const HostData& data, cudaStream_t cudaStream);
    std::error_code Deallocate();
    DeviceData GetDeviceView() const;

  private:
    nva2x::DeviceTensorFloat _neutralJaw;
  };

  Data _data;
  std::size_t _nbPoints{0};
  nva2x::DeviceTensorFloat _teethParams;
  std::size_t _teethParamsStride{0};

  bool _initialized{false};
  bool _animatorDataIsSet{false};
};


class MultiTrackAnimatorEyes : public IMultiTrackAnimatorEyes {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const Params& params, std::size_t nbTracks) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data, float dt) override; // GPU Async

  std::error_code SetParameters(std::size_t trackIndex, const Params& params) override;
  const Params* GetParameters(std::size_t trackIndex) const override;

  std::error_code Reset(std::size_t trackIndex) override;
  std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) override;
  std::error_code SetLiveTime(std::size_t trackIndex, float liveTime) override;
  std::error_code ComputeEyesRotation(
    nva2x::DeviceTensorFloatConstView inputEyesRotationResult, const nva2x::TensorBatchInfo& inputEyesRotationResultInfo,
    nva2x::DeviceTensorFloatView outputEyesRotation, const nva2x::TensorBatchInfo& outputEyesRotationInfo
    ) override; // GPU Async

  void Destroy() override;

private:
  std::error_code SetParametersInternal(std::size_t trackIndex, const Params& params, std::vector<float>& paramsHost);

  cudaStream_t _cudaStream{nullptr};
  std::vector<Params> _params;

  // This is just a more compact bitset for the active tracks.
  // We would rather use an off-the-shelf bitset, but std ones does not support a dynamic number of bits.
  using bits_type = std::uint64_t;
  static constexpr const std::size_t nb_bits = 8 * sizeof(bits_type);
  std::vector<bits_type> _activeTracks;
  nva2x::DeviceTensorUInt64 _activeTracksDevice;

  nva2x::DeviceTensorFloat _eyesParams;
  std::size_t _eyesParamsStride{0};
  nva2x::DeviceTensorFloat _saccadeRot;
  // _liveTime is the only state in this animator.
  // Unlike the skin animator where there is actual data depending on previous frame,
  // here the current time could be passed with each call, which would make this animator stateless.
  nva2x::DeviceTensorFloat _liveTime;
  float _dt{0.0f};

  bool _initialized{false};
  bool _animatorDataIsSet{false};
};


IMultiTrackAnimatorPcaReconstruction *CreateMultiTrackAnimatorPcaReconstruction_INTERNAL();
IMultiTrackAnimatorSkin *CreateMultiTrackAnimatorSkin_INTERNAL();
IMultiTrackAnimatorTongue *CreateMultiTrackAnimatorTongue_INTERNAL();
IMultiTrackAnimatorTeeth *CreateMultiTrackAnimatorTeeth_INTERNAL();
IMultiTrackAnimatorEyes *CreateMultiTrackAnimatorEyes_INTERNAL();

} // namespace nva2f
