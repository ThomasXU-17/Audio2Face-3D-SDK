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

#include "audio2face/animator.h"
#include "audio2face/internal/validator.h"
#include "audio2x/internal/tensor.h"

#include <cublas_v2.h>

REFL_AUTO(
  type(nva2f::AnimatorSkinParams),
  field(lowerFaceSmoothing, nva2f::Validator<float>(0.0023f, 0, 0.1f)),
  field(upperFaceSmoothing, nva2f::Validator<float>(0.001f, 0, 0.1f)),
  field(lowerFaceStrength,  nva2f::Validator<float>(1.3f, 0, 2.f)),
  field(upperFaceStrength,  nva2f::Validator<float>(1.f, 0, 2.f)),
  field(faceMaskLevel,      nva2f::Validator<float>(0.6f, 0, 1.f)),
  field(faceMaskSoftness,   nva2f::Validator<float>(0.0085f, 0.001f, 0.5f)),
  field(skinStrength,       nva2f::Validator<float>(1.f, 0, 2.f)),
  field(blinkStrength,      nva2f::Validator<float>(1.f, 0, 2.f)),
  field(eyelidOpenOffset,   nva2f::Validator<float>(0.06f, -1.f, 1.f)),
  field(lipOpenOffset,      nva2f::Validator<float>(-0.03f, -0.2f, 0.2f)),
  field(blinkOffset,        nva2f::Validator<float>(0, 0, 1.f))
)

REFL_AUTO(
  type(nva2f::AnimatorTongueParams),
  field(tongueStrength,     nva2f::Validator<float>(1.5f, 0, 3.f)),
  field(tongueHeightOffset, nva2f::Validator<float>(0.2f, -3.f, 3.f)),
  field(tongueDepthOffset,  nva2f::Validator<float>(0.13f, -3.f, 3.f))
)

REFL_AUTO(
  type(nva2f::AnimatorTeethParams),
  field(lowerTeethStrength,     nva2f::Validator<float>(1.0f, 0, 2.f)),
  field(lowerTeethHeightOffset, nva2f::Validator<float>(0.0f, -3.f, 3.f)),
  field(lowerTeethDepthOffset,  nva2f::Validator<float>(0.0f, -3.f, 3.f))
)

REFL_AUTO(
  type(nva2f::AnimatorEyesParams),
  field(eyeballsStrength,             nva2f::Validator<float>(1.0f, 0, 2.f)),
  field(saccadeStrength,              nva2f::Validator<float>(1.0f, 0, 2.f)),
  field(rightEyeballRotationOffsetX,  nva2f::Validator<float>(0.0f, -10.f, 10.f)),
  field(rightEyeballRotationOffsetY,  nva2f::Validator<float>(0.0f, -10.f, 10.f)),
  field(leftEyeballRotationOffsetX,   nva2f::Validator<float>(0.0f, -10.f, 10.f)),
  field(leftEyeballRotationOffsetY,   nva2f::Validator<float>(0.0f, -10.f, 10.f)),
  field(saccadeSeed,                  nva2f::Validator<float>(0.0f, 0.0f, 4999.0f))
)

namespace nva2f {

class Interpolator {
public:
  Interpolator();
  ~Interpolator();

  Interpolator(Interpolator&&) = default;
  Interpolator& operator=(Interpolator&&) = default;

  std::error_code SetCudaStream(cudaStream_t cudaStream);
  std::error_code Init(float smoothing, size_t size);
  void SetSmoothing(float value);
  std::error_code Reset();
  std::error_code Update(nva2x::DeviceTensorFloatConstView raw, float dt, nva2x::DeviceTensorFloatView smoothed); // GPU Async

private:
  cudaStream_t _cudaStream;
  float _smoothing;
  unsigned int _degree;
  size_t _size;
  nva2x::DeviceTensorFloat _dataArr;
  bool _dataArrInitialized;
  bool _initialized;
};

class AnimatorPcaReconstruction : public IAnimatorPcaReconstruction {
public:
  AnimatorPcaReconstruction();
  ~AnimatorPcaReconstruction();

  AnimatorPcaReconstruction(AnimatorPcaReconstruction&&) = default;
  AnimatorPcaReconstruction& operator=(AnimatorPcaReconstruction&&) = default;

  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init() override;
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async
  std::error_code SetAnimatorDataView(const DeviceData& data) override;

  std::error_code Reset() override;
  std::error_code Animate(nva2x::DeviceTensorFloatConstView inputPcaCoefs,
                          nva2x::DeviceTensorFloatView outputDeltas) override; // GPU Async

  void Destroy() override;

  std::error_code Deallocate();

  class Data {
  public:
    std::error_code Init(const HostData& data, cudaStream_t cudaStream);
    std::error_code Deallocate();
    DeviceData GetDeviceView() const;

  private:
    nva2x::DeviceTensorFloat _shapesMatrix;
    std::size_t _shapeSize{0};
  };

private:
  cudaStream_t _cudaStream;
  cublasHandle_t _cublasHandle;
  Data _data;
  DeviceData _dataView;
  size_t _numShapes;
  bool _initialized;
  bool _animatorDataIsSet;
};

class AnimatorSkin : public IAnimatorSkin {
public:
  AnimatorSkin();
  ~AnimatorSkin();

  AnimatorSkin(AnimatorSkin&&) = default;
  AnimatorSkin& operator=(AnimatorSkin&&) = default;

  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const Params& params) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async
  std::error_code SetAnimatorDataView(const DeviceData& data) override;

  std::error_code SetLowerFaceSmoothing(float value) override;
  std::error_code SetUpperFaceSmoothing(float value) override;
  std::error_code SetLowerFaceStrength(float value) override;
  std::error_code SetUpperFaceStrength(float value) override;
  std::error_code SetFaceMaskLevel(float value) override;    // GPU Async
  std::error_code SetFaceMaskSoftness(float value) override; // GPU Async
  std::error_code SetSkinStrength(float value) override;
  std::error_code SetBlinkStrength(float value) override;
  std::error_code SetEyelidOpenOffset(float value) override;
  std::error_code SetLipOpenOffset(float value) override;

  std::error_code SetParameters(const Params& params) override;
  const Params& GetParameters() const override;

  std::error_code Reset() override;
  std::error_code SetBlinkOffset(float value) override;
  std::error_code Animate(nva2x::DeviceTensorFloatConstView inputDeltas, float dt,
                          nva2x::DeviceTensorFloatView outputVertices) override; // GPU Async

  void Destroy() override;

  std::error_code CalculateFaceMaskLower(); // GPU Async
  std::error_code Deallocate();

  class Data {
  public:
    std::error_code Init(const HostData& data, cudaStream_t cudaStream);
    std::error_code Deallocate();
    DeviceData GetDeviceView() const;

  private:
    nva2x::DeviceTensorFloat _neutralPose;
    nva2x::DeviceTensorFloat _lipOpenPoseDelta;
    nva2x::DeviceTensorFloat _eyeClosePoseDelta;
  };

private:
  cudaStream_t _cudaStream;
  Params _params;
  Data _data;
  DeviceData _dataView;
  nva2x::DeviceTensorFloat _faceMaskLower;
  nva2x::DeviceTensorFloat _smoothedLower;
  nva2x::DeviceTensorFloat _smoothedUpper;
  Interpolator _interpLower;
  Interpolator _interpUpper;
  bool _initialized;
  bool _animatorDataIsSet;
  ValidatorProxy<Params> _proxy;
};

class AnimatorTongue : public IAnimatorTongue {
public:
  AnimatorTongue();
  ~AnimatorTongue();

  AnimatorTongue(AnimatorTongue&&) = default;
  AnimatorTongue& operator=(AnimatorTongue&&) = default;

  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const Params& params) override; // GPU Async
  std::error_code SetAnimatorData(const HostData& data) override; // GPU Async
  std::error_code SetAnimatorDataView(const DeviceData& data) override;

  std::error_code SetTongueStrength(float value) override;
  std::error_code SetTongueHeightOffset(float value) override;
  std::error_code SetTongueDepthOffset(float value) override;

  std::error_code SetParameters(const Params& params) override;
  const Params& GetParameters() const override;

  std::error_code Reset() override;
  std::error_code Animate(nva2x::DeviceTensorFloatConstView inputDeltas, float dt,
                          nva2x::DeviceTensorFloatView outputVertices) override; // GPU Async

  void Destroy() override;

  std::error_code Deallocate();

  class Data {
  public:
    std::error_code Init(const HostData& data, cudaStream_t cudaStream);
    std::error_code Deallocate();
    DeviceData GetDeviceView() const;

  private:
    nva2x::DeviceTensorFloat _neutralPose;
  };

private:
  cudaStream_t _cudaStream;
  Params _params;
  Data _data;
  DeviceData _dataView;
  bool _initialized;
  bool _animatorDataIsSet;
  ValidatorProxy<Params> _proxy;
};

class AnimatorTeeth : public IAnimatorTeeth {
public:
  AnimatorTeeth();
  ~AnimatorTeeth();

  AnimatorTeeth(AnimatorTeeth&&) = default;
  AnimatorTeeth& operator=(AnimatorTeeth&&) = default;

  std::error_code Init(const Params& params) override;
  std::error_code SetAnimatorData(const HostData& data) override;
  std::error_code SetAnimatorDataView(const HostData& data) override;

  std::error_code SetLowerTeethStrength(float value) override;
  std::error_code SetLowerTeethHeightOffset(float value) override;
  std::error_code SetLowerTeethDepthOffset(float value) override;

  std::error_code SetParameters(const Params& params) override;
  const Params& GetParameters() const override;

  std::error_code Reset() override;
  std::error_code ComputeJawTransform(nva2x::HostTensorFloatView jawTransform,
                                      nva2x::HostTensorFloatConstView jawResultPose) override;

  void Destroy() override;

  std::error_code Deallocate();

  class Data {
  public:
    std::error_code Init(const HostData& data);
    std::error_code Deallocate();
    HostData GetHostView() const;

  private:
    nva2x::HostTensorFloat _neutralJaw;
  };

private:
  Params _params;
  Data _data;
  HostData _dataView;
  nva2x::HostTensorFloat _jawPose;
  bool _initialized;
  bool _animatorDataIsSet;
  ValidatorProxy<Params> _proxy;
};

class AnimatorEyes : public IAnimatorEyes {
public:
  AnimatorEyes();
  ~AnimatorEyes();

  AnimatorEyes(AnimatorEyes&&) = default;
  AnimatorEyes& operator=(AnimatorEyes&&) = default;

  std::error_code Init(const Params& params) override;
  std::error_code SetAnimatorData(const HostData& data) override;
  std::error_code SetAnimatorDataView(const HostData& data) override;

  std::error_code SetEyeballsStrength(float value) override;
  std::error_code SetSaccadeStrength(float value) override;
  std::error_code SetRightEyeballRotationOffsetX(float value) override;
  std::error_code SetRightEyeballRotationOffsetY(float value) override;
  std::error_code SetLeftEyeballRotationOffsetX(float value) override;
  std::error_code SetLeftEyeballRotationOffsetY(float value) override;
  std::error_code SetSaccadeSeed(float value) override;

  std::error_code SetParameters(const Params& params) override;
  const Params& GetParameters() const override;

  std::error_code Reset() override;
  std::error_code SetFrameIndex(int frameIdx) override;
  std::error_code IncrementLiveTime(float deltatime) override;
  std::error_code ComputeEyesRotation(nva2x::HostTensorFloatView rightEyeRotation,
                                      nva2x::HostTensorFloatView leftEyeRotation,
                                      nva2x::HostTensorFloatConstView eyesRotationResult) override;

  void Destroy() override;

  std::error_code Deallocate();

  class Data {
  public:
    std::error_code Init(const HostData& data);
    std::error_code Deallocate();
    HostData GetHostView() const;

  private:
    nva2x::HostTensorFloat _saccadeRot;
  };

private:
  Params _params;
  Data _data;
  HostData _dataView;
  int _frameIdx;
  float _liveTime;
  bool _initialized;
  bool _animatorDataIsSet;
  ValidatorProxy<Params> _proxy;
};

struct AnimatorData {
    nva2x::HostTensorFloat neutralSkin;
    nva2x::HostTensorFloat lipOpenPoseDelta;
    nva2x::HostTensorFloat eyeClosePoseDelta;
    nva2x::HostTensorFloat neutralTongue;
    nva2x::HostTensorFloat neutralJaw;
    nva2x::HostTensorFloat saccadeRot;

    std::error_code Init(AnimatorDataView data);
};


IAnimatorPcaReconstruction *CreateAnimatorPcaReconstruction_INTERNAL();
IAnimatorSkin *CreateAnimatorSkin_INTERNAL();
IAnimatorTongue *CreateAnimatorTongue_INTERNAL();
IAnimatorTeeth *CreateAnimatorTeeth_INTERNAL();
IAnimatorEyes *CreateAnimatorEyes_INTERNAL();

bool AreEqual_INTERNAL(const AnimatorSkinParams& a, const AnimatorSkinParams& b);
bool AreEqual_INTERNAL(const AnimatorTongueParams& a, const AnimatorTongueParams& b);
bool AreEqual_INTERNAL(const AnimatorTeethParams& a, const AnimatorTeethParams& b);
bool AreEqual_INTERNAL(const AnimatorEyesParams& a, const AnimatorEyesParams& b);

} // namespace nva2f
