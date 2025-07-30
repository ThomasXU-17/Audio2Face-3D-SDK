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

#include "audio2x/tensor.h"

#include <limits>

// This file contains the single-track animator classes.
// These are obsolete and their use has been replaced by the multi-track animator classes.
// The multi-track animator classes (multitrack_animator.h) provide more efficient batch
// processing and replace certain CPU implementations with GPU ones.

namespace nva2f {

// Parameters for skin animation / post-processing.
struct AnimatorSkinParams {
  float lowerFaceSmoothing;
  float upperFaceSmoothing;
  float lowerFaceStrength;
  float upperFaceStrength;
  float faceMaskLevel;
  float faceMaskSoftness;
  float skinStrength;
  float blinkStrength;
  float eyelidOpenOffset;
  float lipOpenOffset;
  float blinkOffset;
};

// Parameters for tongue animation / post-processing.
struct AnimatorTongueParams {
  float tongueStrength;
  float tongueHeightOffset;
  float tongueDepthOffset;
};

// Parameters for teeth animation / post-processing.
struct AnimatorTeethParams {
  float lowerTeethStrength;
  float lowerTeethHeightOffset;
  float lowerTeethDepthOffset;
};

// Parameters for eyes animation / post-processing.
struct AnimatorEyesParams {
  float eyeballsStrength;
  float saccadeStrength;
  float rightEyeballRotationOffsetX;
  float rightEyeballRotationOffsetY;
  float leftEyeballRotationOffsetX;
  float leftEyeballRotationOffsetY;
  float saccadeSeed;
};

// Range configuration structure for animator parameters with default, minimum, and maximum values.
template<typename T>
struct RangeConfig {
    T default_value;
    T minimum;
    T maximum;
    const char* description{""};

    static const RangeConfig<T> kDefault;
};

// Equality comparison operator for RangeConfig.
template<typename T>
bool operator==(const nva2f::RangeConfig<T>& a, const nva2f::RangeConfig<T>& b) {
    return (a.default_value == b.default_value)&&(a.minimum == b.minimum)&&(a.maximum == b.maximum);
}

// Default RangeConfig instance with full range limits.
template<typename T>
const RangeConfig<T> RangeConfig<T>::kDefault{0,std::numeric_limits<T>::min(),std::numeric_limits<T>::max(),"default"};

// Interface for PCA reconstruction animator that converts PCA coefficients to vertex deltas.
class IAnimatorPcaReconstruction {
public:
  // Host memory data structure containing shape matrix and size information.
  struct HostData {
    nva2x::HostTensorFloatConstView shapesMatrix;
    std::size_t shapeSize{0};
  };
  // Device memory data structure containing shape matrix and size information.
  struct DeviceData {
    nva2x::DeviceTensorFloatConstView shapesMatrix;
    std::size_t shapeSize{0};
  };

  // Set the CUDA stream for GPU operations.
  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;
  // Initialize the animator.
  virtual std::error_code Init() = 0;
  // Set animator data from host memory.
  // Using this function will allocate device memory for the animator data.
  virtual std::error_code SetAnimatorData(const HostData& data) = 0; // GPU Async
  // Set animator data from device memory.
  // Using this function will refer the view given as a parameter without copying the data.
  // This is useful to share data between multiple animators (i.e. multi-track).
  virtual std::error_code SetAnimatorDataView(const DeviceData& data) = 0;

  // Reset the animator state.
  virtual std::error_code Reset() = 0;
  // Animate PCA coefficients to produce vertex deltas.
  virtual std::error_code Animate(nva2x::DeviceTensorFloatConstView inputPcaCoefs,
                                  nva2x::DeviceTensorFloatView outputDeltas) = 0; // GPU Async

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IAnimatorPcaReconstruction();
};

// Interface for skin animation that handles face deformation and blending.
class IAnimatorSkin {
public:
  // Host memory data structure containing neutral pose and delta poses.
  struct HostData {
    nva2x::HostTensorFloatConstView neutralPose;
    nva2x::HostTensorFloatConstView lipOpenPoseDelta;
    nva2x::HostTensorFloatConstView eyeClosePoseDelta;
  };
  // Device memory data structure containing neutral pose and delta poses.
  struct DeviceData {
    nva2x::DeviceTensorFloatConstView neutralPose;
    nva2x::DeviceTensorFloatConstView lipOpenPoseDelta;
    nva2x::DeviceTensorFloatConstView eyeClosePoseDelta;
  };
  using Params = AnimatorSkinParams;
  // Combined initialization data structure.
  struct InitData {
    HostData data;
    Params params;
  };

  // Set the CUDA stream for GPU operations.
  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;
  // Initialize the animator with parameters.
  virtual std::error_code Init(const Params& params) = 0; // GPU Async
  // Set animator data from host memory.
  // Using this function will allocate device memory for the animator data.
  virtual std::error_code SetAnimatorData(const HostData& data) = 0; // GPU Async
  // Set animator data from device memory.
  // Using this function will refer the view given as a parameter without copying the data.
  // This is useful to share data between multiple animators (i.e. multi-track).
  virtual std::error_code SetAnimatorDataView(const DeviceData& data) = 0;

  // Set the smoothing factor for lower face animation.
  virtual std::error_code SetLowerFaceSmoothing(float value) = 0;
  // Set the smoothing factor for upper face animation.
  virtual std::error_code SetUpperFaceSmoothing(float value) = 0;
  // Set the strength factor for lower face animation.
  virtual std::error_code SetLowerFaceStrength(float value) = 0;
  // Set the strength factor for upper face animation.
  virtual std::error_code SetUpperFaceStrength(float value) = 0;
  // Set the face mask level for distinguishing between upper and lower face.
  // This invalidates the face mask which will be regenerated on the GPU.
  virtual std::error_code SetFaceMaskLevel(float value) = 0;
  // Set the face mask softness for distinguishing between upper and lower face.
  // This invalidates the face mask which will be regenerated on the GPU.
  virtual std::error_code SetFaceMaskSoftness(float value) = 0;
  // Set the overall skin animation strength.
  virtual std::error_code SetSkinStrength(float value) = 0;
  // Set the blink animation strength.
  virtual std::error_code SetBlinkStrength(float value) = 0;
  // Set the eyelid open offset for blink animation.
  virtual std::error_code SetEyelidOpenOffset(float value) = 0;
  // Set the lip open offset for lip animation.
  virtual std::error_code SetLipOpenOffset(float value) = 0;
  // Set the blink offset for blink animation.
  virtual std::error_code SetBlinkOffset(float value) = 0;

  // Set all parameters at once.
  virtual std::error_code SetParameters(const Params& params) = 0;
  // Get the current parameters.
  virtual const Params& GetParameters() const = 0;

  // Get the range configuration for a parameter.
  static const RangeConfig<float> GetRangeConfig(float Params::*);

  // Reset the animator state.
  virtual std::error_code Reset() = 0;
  // Animate input deltas to produce output vertices.
  virtual std::error_code Animate(nva2x::DeviceTensorFloatConstView inputDeltas, float dt,
                                  nva2x::DeviceTensorFloatView outputVertices) = 0; // GPU Async

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IAnimatorSkin();
};

// Interface for tongue animation that handles tongue deformation and positioning.
class IAnimatorTongue {
public:
  // Host memory data structure containing neutral pose.
  struct HostData {
    nva2x::HostTensorFloatConstView neutralPose;
  };
  // Device memory data structure containing neutral pose.
  struct DeviceData {
    nva2x::DeviceTensorFloatConstView neutralPose;
  };
  using Params = AnimatorTongueParams;
  // Combined initialization data structure.
  struct InitData {
    HostData data;
    Params params;
  };

  // Set the CUDA stream for GPU operations.
  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;
  // Initialize the animator with parameters.
  virtual std::error_code Init(const Params& params) = 0; // GPU Async
  // Set animator data from host memory.
  // Using this function will allocate device memory for the animator data.
  virtual std::error_code SetAnimatorData(const HostData& data) = 0; // GPU Async
  // Set animator data from device memory.
  // Using this function will refer the view given as a parameter without copying the data.
  // This is useful to share data between multiple animators (i.e. multi-track).
  virtual std::error_code SetAnimatorDataView(const DeviceData& data) = 0;

  // Set the tongue animation strength.
  virtual std::error_code SetTongueStrength(float value) = 0;
  // Set the tongue height offset.
  virtual std::error_code SetTongueHeightOffset(float value) = 0;
  // Set the tongue depth offset.
  virtual std::error_code SetTongueDepthOffset(float value) = 0;

  // Set all parameters at once.
  virtual std::error_code SetParameters(const Params& params) = 0;
  // Get the current parameters.
  virtual const Params& GetParameters() const = 0;

  // Get the range configuration for a parameter.
  static const RangeConfig<float> GetRangeConfig(float Params::*);

  // Reset the animator state.
  virtual std::error_code Reset() = 0;
  // Animate input deltas to produce output vertices.
  virtual std::error_code Animate(nva2x::DeviceTensorFloatConstView inputDeltas, float dt,
                                  nva2x::DeviceTensorFloatView outputVertices) = 0; // GPU Async

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IAnimatorTongue();
};

// Interface for teeth animation that handles jaw transformation.
class IAnimatorTeeth {
public:
  // Host memory data structure containing neutral jaw pose.
  struct HostData {
    nva2x::HostTensorFloatConstView neutralJaw;
  };
  using Params = AnimatorTeethParams;
  // Combined initialization data structure.
  struct InitData {
    HostData data;
    Params params;
  };

  // Initialize the animator with parameters.
  virtual std::error_code Init(const Params& params) = 0;
  // Set animator data from host memory.
  // Using this function will allocate device memory for the animator data.
  virtual std::error_code SetAnimatorData(const HostData& data) = 0;
  // Set animator data view from host memory.
  // Using this function will refer the view given as a parameter without copying the data.
  // This is useful to share data between multiple animators (i.e. multi-track).
  virtual std::error_code SetAnimatorDataView(const HostData& data) = 0;

  // Set the lower teeth animation strength.
  virtual std::error_code SetLowerTeethStrength(float value) = 0;
  // Set the lower teeth height offset.
  virtual std::error_code SetLowerTeethHeightOffset(float value) = 0;
  // Set the lower teeth depth offset.
  virtual std::error_code SetLowerTeethDepthOffset(float value) = 0;

  // Set all parameters at once.
  virtual std::error_code SetParameters(const Params& params) = 0;
  // Get the current parameters.
  virtual const Params& GetParameters() const = 0;

  // Get the range configuration for a parameter.
  static const RangeConfig<float> GetRangeConfig(float Params::*);

  // Reset the animator state.
  virtual std::error_code Reset() = 0;
  // Compute jaw transform from jaw result pose.
  virtual std::error_code ComputeJawTransform(nva2x::HostTensorFloatView jawTransform,
                                              nva2x::HostTensorFloatConstView jawResultPose) = 0;

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IAnimatorTeeth();
};

// Interface for eye animation that handles eyeball movement and saccade.
class IAnimatorEyes {
public:
  // Host memory data structure containing saccade rotation data.
  struct HostData {
    nva2x::HostTensorFloatConstView saccadeRot;
  };
  using Params = AnimatorEyesParams;
  // Combined initialization data structure.
  struct InitData {
    HostData data;
    Params params;
  };

  // Initialize the animator with parameters.
  virtual std::error_code Init(const Params& params) = 0;
  // Set animator data from host memory.
  // Using this function will allocate device memory for the animator data.
  virtual std::error_code SetAnimatorData(const HostData& data) = 0;
  // Set animator data view from host memory.
  // Using this function will refer the view given as a parameter without copying the data.
  // This is useful to share data between multiple animators (i.e. multi-track).
  virtual std::error_code SetAnimatorDataView(const HostData& data) = 0;

  // Set the eyeball animation strength.
  virtual std::error_code SetEyeballsStrength(float value) = 0;
  // Set the saccade animation strength.
  virtual std::error_code SetSaccadeStrength(float value) = 0;
  // Set the right eyeball rotation offset in X axis.
  virtual std::error_code SetRightEyeballRotationOffsetX(float value) = 0;
  // Set the right eyeball rotation offset in Y axis.
  virtual std::error_code SetRightEyeballRotationOffsetY(float value) = 0;
  // Set the left eyeball rotation offset in X axis.
  virtual std::error_code SetLeftEyeballRotationOffsetX(float value) = 0;
  // Set the left eyeball rotation offset in Y axis.
  virtual std::error_code SetLeftEyeballRotationOffsetY(float value) = 0;
  // Set the saccade seed for random eye movement.
  virtual std::error_code SetSaccadeSeed(float value) = 0;

  // Set all parameters at once.
  virtual std::error_code SetParameters(const Params& params) = 0;
  // Get the current parameters.
  virtual const Params& GetParameters() const = 0;

  // Get the range configuration for a parameter.
  static const RangeConfig<float> GetRangeConfig(float Params::*);

  // Reset the animator state.
  virtual std::error_code Reset() = 0;
  // Set the current frame index for saccade animation.
  virtual std::error_code SetFrameIndex(int frameIndex) = 0;
  // Increment the live time for saccade animation.
  virtual std::error_code IncrementLiveTime(float deltatime) = 0;
  // Compute eye rotations for both eyes.
  virtual std::error_code ComputeEyesRotation(nva2x::HostTensorFloatView rightEyeRotation,
                                              nva2x::HostTensorFloatView leftEyeRotation,
                                              nva2x::HostTensorFloatConstView eyesRotationResult) = 0;

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IAnimatorEyes();
};

// Combined parameters structure for all animator types.
struct AnimatorParams {
    AnimatorSkinParams skin;
    AnimatorTongueParams tongue;
    AnimatorTeethParams teeth;
    AnimatorEyesParams eyes;
};

// Combined data view structure for all animator types.
struct AnimatorDataView {
    IAnimatorSkin::HostData skin;
    IAnimatorTongue::HostData tongue;
    IAnimatorTeeth::HostData teeth;
    IAnimatorEyes::HostData eyes;
};

} // namespace nva2f
