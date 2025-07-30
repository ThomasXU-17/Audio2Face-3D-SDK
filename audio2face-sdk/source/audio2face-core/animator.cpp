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
#include "audio2face/internal/animator.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2face/internal/math_utils.h"
#include "audio2x/error.h"

#include <algorithm>
#include <cassert>

namespace nva2f {

//////////////////////////////////////////////

constexpr auto kSkinType = refl::reflect<AnimatorSkinParams>();
constexpr auto kSkinMembers = filter(kSkinType.members, [](auto member) {
  return refl::descriptor::has_attribute<Validator>(member);
});
static_assert(kSkinMembers.size == 11);
MAKE_ERROR_CATEGORY_NAME_TRAITS(AnimatorSkinParams, "AnimatorSkinParams", kSkinMembers);

constexpr auto kTongueType = refl::reflect<AnimatorTongueParams>();
constexpr auto kTongueMembers = filter(kTongueType.members, [](auto member) {
  return refl::descriptor::has_attribute<Validator>(member);
});
static_assert(kTongueMembers.size == 3);
MAKE_ERROR_CATEGORY_NAME_TRAITS(AnimatorTongueParams, "AnimatorTongueParams", kTongueMembers);

constexpr auto kTeethType = refl::reflect<AnimatorTeethParams>();
constexpr auto kTeethMembers = filter(kTeethType.members, [](auto member) {
  return refl::descriptor::has_attribute<Validator>(member);
});
static_assert(kTeethMembers.size == 3);
MAKE_ERROR_CATEGORY_NAME_TRAITS(AnimatorTeethParams, "AnimatorTeethParams", kTeethMembers);

constexpr auto kEyesType = refl::reflect<AnimatorEyesParams>();
constexpr auto kEyesMembers = filter(kEyesType.members, [](auto member) {
  return refl::descriptor::has_attribute<Validator>(member);
});
static_assert(kEyesMembers.size == 7);
MAKE_ERROR_CATEGORY_NAME_TRAITS(AnimatorEyesParams, "AnimatorEyesParams", kEyesMembers);

//////////////////////////////////////////////

Interpolator::Interpolator()
    : _cudaStream(nullptr), _smoothing(0.f), _degree(2u),
      _size(0), _dataArrInitialized(false), _initialized(false) {
  LOG_DEBUG("Interpolator::Interpolator()");
}

Interpolator::~Interpolator() { LOG_DEBUG("Interpolator::~Interpolator()"); }

std::error_code Interpolator::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code Interpolator::Init(float smoothing, size_t size) {
  _initialized = false;
  _smoothing = smoothing;
  _degree = 2u;
  _size = size;
  _dataArrInitialized = false;
  CHECK_RESULT_WITH_MSG(_dataArr.Allocate(_size * (_degree + 1)),
      "Unable to allocate data for Interpolator");
  _initialized = true;
  return nva2x::ErrorCode::eSuccess;
}

void Interpolator::SetSmoothing(float value) { _smoothing = value; }

std::error_code Interpolator::Reset() {
  _dataArrInitialized = false;
  return nva2x::ErrorCode::eSuccess;
}

//////////////////////////////////////////////

IAnimatorPcaReconstruction::~IAnimatorPcaReconstruction() = default;

AnimatorPcaReconstruction::AnimatorPcaReconstruction()
    : _cudaStream(nullptr), _cublasHandle(nullptr), _numShapes(0), _initialized(false),
      _animatorDataIsSet(false) {
  LOG_DEBUG("AnimatorPcaReconstruction::AnimatorPcaReconstruction()");
}

AnimatorPcaReconstruction::~AnimatorPcaReconstruction() {
  LOG_DEBUG("AnimatorPcaReconstruction::~AnimatorPcaReconstruction()");
  Deallocate();
}

std::error_code AnimatorPcaReconstruction::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  if (_cublasHandle) {
    cublasSetStream(_cublasHandle, _cudaStream);
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorPcaReconstruction::Init() {
  _initialized = false;
  CHECK_RESULT_WITH_MSG(Deallocate(), "Unable to destroy AnimatorPcaReconstruction");
  cublasCreate(&_cublasHandle);
  cublasSetStream(_cublasHandle, _cudaStream);
  _initialized = true;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorPcaReconstruction::SetAnimatorData(const HostData& data) {
  Data ownedData;
  CHECK_RESULT_WITH_MSG(ownedData.Init(data, _cudaStream), "Unable to initialize owned data");

  CHECK_RESULT(SetAnimatorDataView(ownedData.GetDeviceView()));
  _data = std::move(ownedData);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorPcaReconstruction::SetAnimatorDataView(const DeviceData& data) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorPcaReconstruction is not initialized", nva2x::ErrorCode::eNotInitialized);
  CHECK_ERROR_WITH_MSG(data.shapeSize > 0, "AnimatorPcaReconstruction: Shape size must not be zero", nva2x::ErrorCode::eInvalidValue);
  CHECK_ERROR_WITH_MSG(data.shapesMatrix.Size() % data.shapeSize == 0, "AnimatorPcaReconstruction: Mismatch in shape size", nva2x::ErrorCode::eMismatch);
  _animatorDataIsSet = false;

  // Point to external data.
  CHECK_RESULT_WITH_MSG(_data.Deallocate(), "Unable to deallocate owned data");
  _dataView = data;
  _numShapes =  _dataView.shapesMatrix.Size() / _dataView.shapeSize;

  _animatorDataIsSet = true;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorPcaReconstruction::Reset() {
  CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "AnimatorPcaReconstruction: Animator Data is not set", ErrorCode::eDataNotSet);
  return nva2x::ErrorCode::eSuccess;
}

void AnimatorPcaReconstruction::Destroy() {
  LOG_DEBUG("AnimatorPcaReconstruction::Destroy()");
  delete this;
}

std::error_code AnimatorPcaReconstruction::Deallocate() {
  if (_cublasHandle != nullptr) {
    LOG_DEBUG("Destroying cublas handle");
    cublasDestroy(_cublasHandle);
    _cublasHandle = nullptr;
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorPcaReconstruction::Data::Init(const HostData& data, cudaStream_t cudaStream) {
  CHECK_ERROR_WITH_MSG(data.shapeSize > 0, "AnimatorPcaReconstruction::Data: Shape size must not be zero", nva2x::ErrorCode::eInvalidValue);
  CHECK_ERROR_WITH_MSG(data.shapesMatrix.Size() % data.shapeSize == 0, "AnimatorPcaReconstruction::Data: Mismatch in shape size", nva2x::ErrorCode::eMismatch);

  _shapeSize = data.shapeSize;
  CHECK_RESULT_WITH_MSG(_shapesMatrix.Init(data.shapesMatrix, cudaStream),
      "Unable to initialize shapes matrix");

  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorPcaReconstruction::Data::Deallocate() {
  CHECK_RESULT_WITH_MSG(_shapesMatrix.Deallocate(), "Unable to deallocate shapes matrix");
  return nva2x::ErrorCode::eSuccess;
}

AnimatorPcaReconstruction::DeviceData AnimatorPcaReconstruction::Data::GetDeviceView() const {
 return {_shapesMatrix, _shapeSize};
}

//////////////////////////////////////////////

IAnimatorSkin::~IAnimatorSkin() = default;

AnimatorSkin::AnimatorSkin()
    : _cudaStream(nullptr), _initialized(false),
      _animatorDataIsSet(false), _proxy(_params) {
  LOG_DEBUG("AnimatorSkin::AnimatorSkin()");
}

AnimatorSkin::~AnimatorSkin() {
  LOG_DEBUG("AnimatorSkin::~AnimatorSkin()");
  Deallocate();
}

std::error_code AnimatorSkin::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  CHECK_RESULT_WITH_MSG(_interpLower.SetCudaStream(cudaStream),
             "Unable to set CUDA Stream for interp lower");
  CHECK_RESULT_WITH_MSG(_interpUpper.SetCudaStream(cudaStream),
             "Unable to set CUDA Stream for interp upper");
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorSkin::Init(const Params& params) {
  _initialized = false;
  CHECK_RESULT_WITH_MSG(Deallocate(), "Unable to destroy AnimatorSkin");
  _params = params;
  _initialized = true;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorSkin::SetAnimatorData(const HostData& data) {
  Data ownedData;
  CHECK_RESULT_WITH_MSG(ownedData.Init(data, _cudaStream), "Unable to initialize owned data");

  CHECK_RESULT(SetAnimatorDataView(ownedData.GetDeviceView()));
  _data = std::move(ownedData);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorSkin::SetAnimatorDataView(const DeviceData& data) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorSkin is not initialized", nva2x::ErrorCode::eNotInitialized);
  _animatorDataIsSet = false;

  // Point to external data.
  CHECK_RESULT_WITH_MSG(_data.Deallocate(), "Unable to deallocate owned data");
  _dataView = data;

  CHECK_RESULT_WITH_MSG(_faceMaskLower.Allocate(data.neutralPose.Size() / 3),
      "Unable to allocate face mask lower");
  CHECK_RESULT_WITH_MSG(_smoothedLower.Allocate(data.neutralPose.Size()),
      "Unable to allocate smoothed lower pose");
  CHECK_RESULT_WITH_MSG(_smoothedUpper.Allocate(data.neutralPose.Size()),
      "Unable to allocate smoothed upper pose");
  CHECK_RESULT_WITH_MSG(
      _interpLower.Init(_params.lowerFaceSmoothing, data.neutralPose.Size()),
      "Unable to init lower interpolator");
  CHECK_RESULT_WITH_MSG(
      _interpUpper.Init(_params.upperFaceSmoothing, data.neutralPose.Size()),
      "Unable to init upper interpolator");
  CHECK_RESULT_WITH_MSG(CalculateFaceMaskLower(), "Unable to calculate face mask lower");

  _animatorDataIsSet = true;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorSkin::SetLowerFaceSmoothing(float value) {
  // _animatorDataIsSet guarantees readiness of _interpLower
  CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "AnimatorSkin: Animator Data is not set", ErrorCode::eDataNotSet);
  CHECK_RESULT(_proxy.lowerFaceSmoothing(value));
  _interpLower.SetSmoothing(value);
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorSkin::SetUpperFaceSmoothing(float value) {
  // _animatorDataIsSet guarantees readiness of _interpUpper
  CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "AnimatorSkin: Animator Data is not set", ErrorCode::eDataNotSet);
  CHECK_RESULT(_proxy.upperFaceSmoothing(value));
  _interpUpper.SetSmoothing(value);
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorSkin::SetLowerFaceStrength(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorSkin is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.lowerFaceStrength(value);
}

std::error_code AnimatorSkin::SetUpperFaceStrength(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorSkin is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.upperFaceStrength(value);
}

std::error_code AnimatorSkin::SetFaceMaskLevel(float value) {
  // _animatorDataIsSet guarantees readiness of _neutralPose, _faceMaskLower
  CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "AnimatorSkin: Animator Data is not set", ErrorCode::eDataNotSet);
  const auto valueBefore = _params.faceMaskLevel;
  CHECK_RESULT(_proxy.faceMaskLevel(value));
  if (valueBefore != value) {
    CHECK_RESULT_WITH_MSG(CalculateFaceMaskLower(), "Unable to calculate face mask lower");
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorSkin::SetFaceMaskSoftness(float value) {
  // _animatorDataIsSet guarantees readiness of _neutralPose, _faceMaskLower
  CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "AnimatorSkin: Animator Data is not set", ErrorCode::eDataNotSet);
  const auto valueBefore = _params.faceMaskSoftness;
  CHECK_RESULT(_proxy.faceMaskSoftness(value));
  if (valueBefore != value) {
    CHECK_RESULT_WITH_MSG(CalculateFaceMaskLower(), "Unable to calculate face mask lower");
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorSkin::SetSkinStrength(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorSkin is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.skinStrength(value);
}

std::error_code AnimatorSkin::SetBlinkStrength(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorSkin is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.blinkStrength(value);
}

std::error_code AnimatorSkin::SetEyelidOpenOffset(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorSkin is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.eyelidOpenOffset(value);
}

std::error_code AnimatorSkin::SetLipOpenOffset(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorSkin is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.lipOpenOffset(value);
}

std::error_code AnimatorSkin::Reset() {
  // _animatorDataIsSet guarantees readiness of _interpLower, _interpUpper
  CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "AnimatorSkin: Animator Data is not set", ErrorCode::eDataNotSet);
  CHECK_RESULT_WITH_MSG(_interpLower.Reset(), "Unable to reset lower interpolator");
  CHECK_RESULT_WITH_MSG(_interpUpper.Reset(), "Unable to reset upper interpolator");
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorSkin::SetBlinkOffset(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorSkin is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.blinkOffset(value);
}

void AnimatorSkin::Destroy() {
  LOG_DEBUG("AnimatorSkin::Destroy()");
  delete this;
}

std::error_code AnimatorSkin::Deallocate() {
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorSkin::SetParameters(const Params& params) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorSkin is not initialized", nva2x::ErrorCode::eNotInitialized);
  CHECK_RESULT(SetLowerFaceSmoothing(params.lowerFaceSmoothing));
  CHECK_RESULT(SetUpperFaceSmoothing(params.upperFaceSmoothing));
  CHECK_RESULT(SetLowerFaceStrength(params.lowerFaceStrength));
  CHECK_RESULT(SetUpperFaceStrength(params.upperFaceStrength));
  CHECK_RESULT(SetFaceMaskLevel(params.faceMaskLevel));
  CHECK_RESULT(SetFaceMaskSoftness(params.faceMaskSoftness));
  CHECK_RESULT(SetSkinStrength(params.skinStrength));
  CHECK_RESULT(SetBlinkStrength(params.blinkStrength));
  CHECK_RESULT(SetEyelidOpenOffset(params.eyelidOpenOffset));
  CHECK_RESULT(SetLipOpenOffset(params.lipOpenOffset));
  CHECK_RESULT(SetBlinkOffset(params.blinkOffset));
  return nva2x::ErrorCode::eSuccess;
}

const AnimatorSkin::Params& AnimatorSkin::GetParameters() const {
  return _params;
}

std::error_code AnimatorSkin::Data::Init(const HostData& data, cudaStream_t cudaStream) {
  CHECK_RESULT_WITH_MSG(_neutralPose.Init(data.neutralPose, cudaStream),
      "Unable to initialize neutral pose");
  CHECK_RESULT_WITH_MSG(_lipOpenPoseDelta.Init(data.lipOpenPoseDelta, cudaStream),
      "Unable to initialize lip open pose delta");
  CHECK_RESULT_WITH_MSG(_eyeClosePoseDelta.Init(data.eyeClosePoseDelta, cudaStream),
      "Unable to initialize eye close pose delta");
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorSkin::Data::Deallocate() {
  CHECK_RESULT_WITH_MSG(_neutralPose.Deallocate(), "Unable to deallocate neutral pose");
  CHECK_RESULT_WITH_MSG(_lipOpenPoseDelta.Deallocate(), "Unable to deallocate lip open pose delta");
  CHECK_RESULT_WITH_MSG(_eyeClosePoseDelta.Deallocate(), "Unable to deallocate eye close pose delta");
  return nva2x::ErrorCode::eSuccess;
}

AnimatorSkin::DeviceData AnimatorSkin::Data::GetDeviceView() const {
 return {_neutralPose, _lipOpenPoseDelta, _eyeClosePoseDelta};
}

const RangeConfig<float> IAnimatorSkin::GetRangeConfig(float Params::* P) {
  return GetRangeConfigImpl<float, Params>(P);
}

//////////////////////////////////////////////

IAnimatorTongue::~IAnimatorTongue() = default;

AnimatorTongue::AnimatorTongue()
    : _cudaStream(nullptr),
      _initialized(false), _animatorDataIsSet(false), _proxy(_params) {
  LOG_DEBUG("AnimatorTongue::AnimatorTongue()");
}

AnimatorTongue::~AnimatorTongue() {
  LOG_DEBUG("AnimatorTongue::~AnimatorTongue()");
  Deallocate();
}

std::error_code AnimatorTongue::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorTongue::Init(const Params& params) {
  _initialized = false;
  CHECK_RESULT_WITH_MSG(Deallocate(), "Unable to destroy AnimatorTongue");
  _params = params;
  _initialized = true;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorTongue::SetAnimatorData(const HostData& data) {
  Data ownedData;
  CHECK_RESULT_WITH_MSG(ownedData.Init(data, _cudaStream), "Unable to initialize owned data");

  CHECK_RESULT(SetAnimatorDataView(ownedData.GetDeviceView()));
  _data = std::move(ownedData);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorTongue::SetAnimatorDataView(const DeviceData& data) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorTongue is not initialized", nva2x::ErrorCode::eNotInitialized);
  _animatorDataIsSet = false;

  // Point to external data.
  CHECK_RESULT_WITH_MSG(_data.Deallocate(), "Unable to deallocate owned data");
  _dataView = data;

  _animatorDataIsSet = true;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorTongue::SetTongueStrength(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorTongue is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.tongueStrength(value);
}

std::error_code AnimatorTongue::SetTongueHeightOffset(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorTongue is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.tongueHeightOffset(value);
}

std::error_code AnimatorTongue::SetTongueDepthOffset(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorTongue is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.tongueDepthOffset(value);
}

std::error_code AnimatorTongue::SetParameters(const Params& params) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorTongue is not initialized", nva2x::ErrorCode::eNotInitialized);
  CHECK_RESULT_WITH_MSG(SetTongueStrength(params.tongueStrength),"");
  CHECK_RESULT_WITH_MSG(SetTongueHeightOffset(params.tongueHeightOffset),"");
  CHECK_RESULT_WITH_MSG(SetTongueDepthOffset(params.tongueDepthOffset),"");
  return nva2x::ErrorCode::eSuccess;
}

const AnimatorTongue::Params& AnimatorTongue::GetParameters() const {
  return _params;
}

std::error_code AnimatorTongue::Reset() { return nva2x::ErrorCode::eSuccess; }

void AnimatorTongue::Destroy() {
  LOG_DEBUG("AnimatorTongue::Destroy()");
  delete this;
}

std::error_code AnimatorTongue::Deallocate() {
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorTongue::Data::Init(const HostData& data, cudaStream_t cudaStream) {
  CHECK_RESULT_WITH_MSG(_neutralPose.Init(data.neutralPose, cudaStream),
      "Unable to initialize neutral pose");
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorTongue::Data::Deallocate() {
  CHECK_RESULT_WITH_MSG(_neutralPose.Deallocate(), "Unable to deallocate neutral pose");
  return nva2x::ErrorCode::eSuccess;
}

AnimatorTongue::DeviceData AnimatorTongue::Data::GetDeviceView() const {
 return {_neutralPose};
}

const RangeConfig<float> IAnimatorTongue::GetRangeConfig(float Params::* P) {
  return GetRangeConfigImpl<float, Params>(P);
}

//////////////////////////////////////////////

IAnimatorTeeth::~IAnimatorTeeth() = default;

AnimatorTeeth::AnimatorTeeth()
    : _initialized(false), _animatorDataIsSet(false), _proxy(_params) {
  LOG_DEBUG("AnimatorTeeth::AnimatorTeeth()");
}

AnimatorTeeth::~AnimatorTeeth() {
  LOG_DEBUG("AnimatorTeeth::~AnimatorTeeth()");
}

std::error_code AnimatorTeeth::Init(const Params& params) {
  _initialized = false;
  CHECK_RESULT_WITH_MSG(Deallocate(), "Unable to destroy AnimatorTeeth");
  _params = params;
  _initialized = true;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorTeeth::SetAnimatorData(const HostData& data) {
  Data ownedData;
  CHECK_RESULT_WITH_MSG(ownedData.Init(data), "Unable to initialize owned data");

  CHECK_RESULT(SetAnimatorDataView(ownedData.GetHostView()));
  _data = std::move(ownedData);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorTeeth::SetAnimatorDataView(const HostData& data) {
  CHECK_ERROR_WITH_MSG(data.neutralJaw.Data(), "Neutral jaw is null", nva2x::ErrorCode::eNullPointer);
  CHECK_ERROR_WITH_MSG(data.neutralJaw.Size() % 3 == 0, "Neutral jaw is not a multiple of 3 floats", nva2x::ErrorCode::eInvalidValue);
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorTeeth is not initialized", nva2x::ErrorCode::eNotInitialized);
  _animatorDataIsSet = false;

  // Point to external data.
  CHECK_RESULT_WITH_MSG(_data.Deallocate(), "Unable to deallocate owned data");
  _dataView = data;

  CHECK_RESULT_WITH_MSG(_jawPose.Allocate(data.neutralJaw.Size()),
      "Unable to allocate jaw pose");

  _animatorDataIsSet = true;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorTeeth::SetLowerTeethStrength(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorTeeth is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.lowerTeethStrength(value);
}

std::error_code AnimatorTeeth::SetLowerTeethHeightOffset(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorTeeth is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.lowerTeethHeightOffset(value);
}

std::error_code AnimatorTeeth::SetLowerTeethDepthOffset(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorTeeth is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.lowerTeethDepthOffset(value);
}

std::error_code AnimatorTeeth::SetParameters(const Params& params) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorTeeth is not initialized", nva2x::ErrorCode::eNotInitialized);
  CHECK_RESULT_WITH_MSG(SetLowerTeethStrength(params.lowerTeethStrength),"");
  CHECK_RESULT_WITH_MSG(SetLowerTeethHeightOffset(params.lowerTeethHeightOffset),"");
  CHECK_RESULT_WITH_MSG(SetLowerTeethDepthOffset(params.lowerTeethDepthOffset),"");
  return nva2x::ErrorCode::eSuccess;
}

const AnimatorTeeth::Params& AnimatorTeeth::GetParameters() const {
  return _params;
}

std::error_code AnimatorTeeth::Reset() { return nva2x::ErrorCode::eSuccess; }

std::error_code AnimatorTeeth::ComputeJawTransform(
  nva2x::HostTensorFloatView jawTransform, nva2x::HostTensorFloatConstView jawResultPose)
{
  // Add the strength and offset parameters.
  // There might be a more optimal way to do this, but the number of points should
  // be small.
  CHECK_ERROR_WITH_MSG(jawResultPose.Size() == _dataView.neutralJaw.Size(), "Mismatched size for jaw results in AnimatorTeeth", nva2x::ErrorCode::eMismatch);
  CHECK_ERROR_WITH_MSG(jawTransform.Size() == 16, "Mismatched size for jaw transform in AnimatorTeeth", nva2x::ErrorCode::eMismatch);

  assert(_dataView.neutralJaw.Size() == _jawPose.Size());
  assert(_dataView.neutralJaw.Size() % 3 == 0);
  const std::size_t nbPoints = _dataView.neutralJaw.Size() / 3;
  float* jawPose = _jawPose.Data();
  const float* jawResultPosePtr = jawResultPose.Data();
  for (std::size_t i = 0; i < nbPoints; ++i)
  {
    jawPose[3*i + 0] = jawResultPosePtr[3*i + 0] * _params.lowerTeethStrength;
    jawPose[3*i + 1] = jawResultPosePtr[3*i + 1] * _params.lowerTeethStrength + _params.lowerTeethHeightOffset;
    jawPose[3*i + 2] = jawResultPosePtr[3*i + 2] * _params.lowerTeethStrength + _params.lowerTeethDepthOffset;
  }

  // Add neutral pose.
  std::transform(
      _dataView.neutralJaw.Data(), _dataView.neutralJaw.Data() + _dataView.neutralJaw.Size(),
      jawPose,
      jawPose,
      std::plus<>()
  );
  const auto status = rigidXform(jawTransform.Data(), jawPose, _dataView.neutralJaw.Data(), nbPoints);

  return status;
}

void AnimatorTeeth::Destroy() {
  LOG_DEBUG("AnimatorTeeth::Destroy()");
  delete this;
}

std::error_code AnimatorTeeth::Deallocate() {
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorTeeth::Data::Init(const HostData& data) {
  CHECK_RESULT_WITH_MSG(_neutralJaw.Init(data.neutralJaw),
      "Unable to initialize neutral jaw");
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorTeeth::Data::Deallocate() {
  CHECK_RESULT_WITH_MSG(_neutralJaw.Deallocate(), "Unable to deallocate neutral jaw");
  return nva2x::ErrorCode::eSuccess;
}

AnimatorTeeth::HostData AnimatorTeeth::Data::GetHostView() const {
 return {_neutralJaw};
}

const RangeConfig<float> IAnimatorTeeth::GetRangeConfig(float Params::* P) {
  return GetRangeConfigImpl<float, Params>(P);
}

//////////////////////////////////////////////

IAnimatorEyes::~IAnimatorEyes() = default;

AnimatorEyes::AnimatorEyes()
    : _frameIdx(0), _liveTime(0.0f)
    , _initialized(false), _animatorDataIsSet(false), _proxy(_params) {
  LOG_DEBUG("AnimatorEyes::AnimatorEyes()");
}

AnimatorEyes::~AnimatorEyes() {
  LOG_DEBUG("AnimatorEyes::~AnimatorEyes()");
}

std::error_code AnimatorEyes::Init(const Params& params) {
  _initialized = false;
  CHECK_RESULT_WITH_MSG(Deallocate(), "Unable to destroy AnimatorEyes");
  _params = params;
  _initialized = true;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorEyes::SetAnimatorData(const HostData& data) {
  Data ownedData;
  CHECK_RESULT_WITH_MSG(ownedData.Init(data), "Unable to initialize owned data");

  CHECK_RESULT(SetAnimatorDataView(ownedData.GetHostView()));
  _data = std::move(ownedData);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorEyes::SetAnimatorDataView(const HostData& data) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorEyes is not initialized", nva2x::ErrorCode::eNotInitialized);
  _animatorDataIsSet = false;

  // Point to external data.
  CHECK_RESULT_WITH_MSG(_data.Deallocate(), "Unable to deallocate owned data");
  _dataView = data;

  _animatorDataIsSet = true;

  // Make sure frame index is updated to the new seed.
  CHECK_RESULT_WITH_MSG(IncrementLiveTime(0.0f), "Unable to update frame index");

  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorEyes::SetEyeballsStrength(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorEyes is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.eyeballsStrength(value);
}

std::error_code AnimatorEyes::SetSaccadeStrength(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorEyes is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.saccadeStrength(value);
}

std::error_code AnimatorEyes::SetRightEyeballRotationOffsetX(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorEyes is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.rightEyeballRotationOffsetX(value);
}

std::error_code AnimatorEyes::SetRightEyeballRotationOffsetY(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorEyes is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.rightEyeballRotationOffsetY(value);
}

std::error_code AnimatorEyes::SetLeftEyeballRotationOffsetX(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorEyes is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.leftEyeballRotationOffsetX(value);
}

std::error_code AnimatorEyes::SetLeftEyeballRotationOffsetY(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorEyes is not initialized", nva2x::ErrorCode::eNotInitialized);
  return _proxy.leftEyeballRotationOffsetY(value);
}

std::error_code AnimatorEyes::SetSaccadeSeed(float value) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorEyes is not initialized", nva2x::ErrorCode::eNotInitialized);
  const auto status = _proxy.saccadeSeed(value);
  if (_animatorDataIsSet) {
    // Make sure frame index is updated to the new seed.
    CHECK_RESULT_WITH_MSG(IncrementLiveTime(0.0f), "Unable to update frame index");
  }
  return status;
}

std::error_code AnimatorEyes::SetParameters(const Params& params) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorEyes is not initialized", nva2x::ErrorCode::eNotInitialized);
  CHECK_RESULT_WITH_MSG(SetEyeballsStrength(params.eyeballsStrength),"");
  CHECK_RESULT_WITH_MSG(SetSaccadeStrength(params.saccadeStrength),"");
  CHECK_RESULT_WITH_MSG(SetRightEyeballRotationOffsetX(params.rightEyeballRotationOffsetX),"");
  CHECK_RESULT_WITH_MSG(SetRightEyeballRotationOffsetY(params.rightEyeballRotationOffsetY),"");
  CHECK_RESULT_WITH_MSG(SetLeftEyeballRotationOffsetX(params.leftEyeballRotationOffsetX),"");
  CHECK_RESULT_WITH_MSG(SetLeftEyeballRotationOffsetY(params.leftEyeballRotationOffsetY),"");
  CHECK_RESULT_WITH_MSG(SetSaccadeSeed(params.saccadeSeed),"");
  return nva2x::ErrorCode::eSuccess;
}

const AnimatorEyes::Params& AnimatorEyes::GetParameters() const {
  return _params;
}

std::error_code AnimatorEyes::Reset() {
  _frameIdx = 0;
  _liveTime = 0.0f;
  // Make sure frame index is updated to the new seed.
  CHECK_RESULT_WITH_MSG(IncrementLiveTime(0.0f), "Unable to update frame index");
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorEyes::SetFrameIndex(int frameIdx) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorEyes is not initialized", nva2x::ErrorCode::eNotInitialized);
  const auto maxNbFrames = _dataView.saccadeRot.Size() / 2;
  CHECK_ERROR_WITH_MSG(maxNbFrames > 0, "Saccade rotation matrix not initialized", nva2x::ErrorCode::eNotInitialized);
  _frameIdx = (static_cast<int>(_params.saccadeSeed) + frameIdx) % maxNbFrames;
  if (_frameIdx < 0)
  {
    _frameIdx += maxNbFrames;
  }
  assert(_frameIdx >= 0);
  assert(_frameIdx < maxNbFrames);
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorEyes::IncrementLiveTime(float deltatime) {
  CHECK_ERROR_WITH_MSG(_initialized, "AnimatorEyes is not initialized", nva2x::ErrorCode::eNotInitialized);
  const auto maxNbFrames = _dataView.saccadeRot.Size() / 2;
  CHECK_ERROR_WITH_MSG(maxNbFrames > 0, "Saccade rotation matrix not initialized", nva2x::ErrorCode::eNotInitialized);

  auto wrap = [max=static_cast<float>(maxNbFrames)](float value) {
    float result = std::fmod(value, max);
    if (result < 0.0f) {
      result += max;
    }
    return result;
  };

  // Assume the saccade information was meant for 30 FPS, otherwise there would be more
  // movement with a higher FPS.
  constexpr float fps = 30.0f;
  const float increment = deltatime * fps;

  // Increment live time and wrap so it doesn't grow indefinitely
  _liveTime += increment;
  _liveTime = wrap(_liveTime);

  // Compute frame index from live time and seed
  float totalTime = _params.saccadeSeed + _liveTime;
  totalTime = wrap(totalTime);
  _frameIdx = static_cast<int>(totalTime);
  assert(_frameIdx >= 0);
  assert(_frameIdx < maxNbFrames);
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorEyes::ComputeEyesRotation(
  nva2x::HostTensorFloatView rightEyeRotationView,
  nva2x::HostTensorFloatView leftEyeRotationView,
  nva2x::HostTensorFloatConstView eyeRotationResultView)
{
  CHECK_ERROR_WITH_MSG(rightEyeRotationView.Size() == 3, "Mismatched size for right eye results in AnimatorEyes", nva2x::ErrorCode::eMismatch);
  CHECK_ERROR_WITH_MSG(leftEyeRotationView.Size() == 3, "Mismatched size for left eye results in AnimatorEyes", nva2x::ErrorCode::eMismatch);
  CHECK_ERROR_WITH_MSG(eyeRotationResultView.Size() == 4, "Mismatched size for eyes results in AnimatorEyes", nva2x::ErrorCode::eMismatch);

  float* rightEyeRotation = rightEyeRotationView.Data();
  float* leftEyeRotation = leftEyeRotationView.Data();
  const float* eyeRotationResult = eyeRotationResultView.Data();

  rightEyeRotation[0] = _params.rightEyeballRotationOffsetX;
  rightEyeRotation[1] = _params.rightEyeballRotationOffsetY;
  leftEyeRotation[0] = _params.leftEyeballRotationOffsetX;
  leftEyeRotation[1] = _params.leftEyeballRotationOffsetY;

  rightEyeRotation[0] += _params.eyeballsStrength * eyeRotationResult[0];
  rightEyeRotation[1] += _params.eyeballsStrength * eyeRotationResult[1];
  leftEyeRotation[0] += _params.eyeballsStrength * eyeRotationResult[2];
  leftEyeRotation[1] += _params.eyeballsStrength * eyeRotationResult[3];

  const float* saccadeRot = _dataView.saccadeRot.Data() + 2 * _frameIdx;
  rightEyeRotation[0] += _params.saccadeStrength * saccadeRot[0];
  rightEyeRotation[1] += _params.saccadeStrength * saccadeRot[1];
  leftEyeRotation[0] += _params.saccadeStrength * saccadeRot[0];
  leftEyeRotation[1] += _params.saccadeStrength * saccadeRot[1];

  rightEyeRotation[2] = 0.0f;
  leftEyeRotation[2] = 0.0f;

  return nva2x::ErrorCode::eSuccess;
}

void AnimatorEyes::Destroy() {
  LOG_DEBUG("AnimatorEyes::Destroy()");
  delete this;
}

std::error_code AnimatorEyes::Deallocate() {
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorEyes::Data::Init(const HostData& data) {
  CHECK_RESULT_WITH_MSG(_saccadeRot.Init(data.saccadeRot),
      "Unable to initialize saccade rotations");
  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimatorEyes::Data::Deallocate() {
  CHECK_RESULT_WITH_MSG(_saccadeRot.Deallocate(), "Unable to deallocate saccade rotations");
  return nva2x::ErrorCode::eSuccess;
}

AnimatorEyes::HostData AnimatorEyes::Data::GetHostView() const {
 return {_saccadeRot};
}

const RangeConfig<float> IAnimatorEyes::GetRangeConfig(float Params::* P) {
  return GetRangeConfigImpl<float, Params>(P);
}

//////////////////////////////////////////////

std::error_code AnimatorData::Init(AnimatorDataView data) {
  A2F_CHECK_RESULT(neutralSkin.Init(data.skin.neutralPose));
  A2F_CHECK_RESULT(lipOpenPoseDelta.Init(data.skin.lipOpenPoseDelta));
  A2F_CHECK_RESULT(eyeClosePoseDelta.Init(data.skin.eyeClosePoseDelta));
  A2F_CHECK_RESULT(neutralTongue.Init(data.tongue.neutralPose));
  A2F_CHECK_RESULT(neutralJaw.Init(data.teeth.neutralJaw));
  A2F_CHECK_RESULT(saccadeRot.Init(data.eyes.saccadeRot));
  return nva2x::ErrorCode::eSuccess;
}

//////////////////////////////////////////////

IAnimatorPcaReconstruction *CreateAnimatorPcaReconstruction_INTERNAL() {
  LOG_DEBUG("CreateAnimatorPcaReconstruction_INTERNAL()");
  return new AnimatorPcaReconstruction();
}

IAnimatorSkin *CreateAnimatorSkin_INTERNAL() {
  LOG_DEBUG("CreateAnimatorSkin_INTERNAL()");
  return new AnimatorSkin();
}

IAnimatorTongue *CreateAnimatorTongue_INTERNAL() {
  LOG_DEBUG("CreateAnimatorTongue_INTERNAL()");
  return new AnimatorTongue();
}

IAnimatorTeeth *CreateAnimatorTeeth_INTERNAL() {
  LOG_DEBUG("CreateAnimatorTeeth_INTERNAL()");
  return new AnimatorTeeth();
}

IAnimatorEyes *CreateAnimatorEyes_INTERNAL() {
  LOG_DEBUG("CreateAnimatorEyes_INTERNAL()");
  return new AnimatorEyes();
}

bool AreEqual_INTERNAL(const AnimatorSkinParams& a, const AnimatorSkinParams& b) {
  return a.lowerFaceSmoothing == b.lowerFaceSmoothing &&
         a.upperFaceSmoothing == b.upperFaceSmoothing &&
         a.lowerFaceStrength == b.lowerFaceStrength &&
         a.upperFaceStrength == b.upperFaceStrength &&
         a.faceMaskLevel == b.faceMaskLevel &&
         a.faceMaskSoftness == b.faceMaskSoftness &&
         a.skinStrength == b.skinStrength &&
         a.blinkStrength == b.blinkStrength &&
         a.eyelidOpenOffset == b.eyelidOpenOffset &&
         a.lipOpenOffset == b.lipOpenOffset &&
         a.blinkOffset == b.blinkOffset;
}

bool AreEqual_INTERNAL(const AnimatorTongueParams& a, const AnimatorTongueParams& b) {
  return a.tongueStrength == b.tongueStrength &&
         a.tongueHeightOffset == b.tongueHeightOffset &&
         a.tongueDepthOffset == b.tongueDepthOffset;
}

bool AreEqual_INTERNAL(const AnimatorTeethParams& a, const AnimatorTeethParams& b) {
  return a.lowerTeethStrength == b.lowerTeethStrength &&
         a.lowerTeethHeightOffset == b.lowerTeethHeightOffset &&
         a.lowerTeethDepthOffset == b.lowerTeethDepthOffset;
}

bool AreEqual_INTERNAL(const AnimatorEyesParams& a, const AnimatorEyesParams& b) {
  return a.eyeballsStrength == b.eyeballsStrength &&
         a.saccadeStrength == b.saccadeStrength &&
         a.rightEyeballRotationOffsetX == b.rightEyeballRotationOffsetX &&
         a.rightEyeballRotationOffsetY == b.rightEyeballRotationOffsetY &&
         a.leftEyeballRotationOffsetX == b.leftEyeballRotationOffsetX &&
         a.leftEyeballRotationOffsetY == b.leftEyeballRotationOffsetY &&
         a.saccadeSeed == b.saccadeSeed;
}

} // namespace nva2f
