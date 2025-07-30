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

#include "audio2face/executor_regression.h"
#include "audio2face/executor_diffusion.h"
#include "audio2face/executor_blendshapesolve.h"
#include "audio2face/executor.h"
#include "audio2x/cuda_stream.h"

namespace nva2f {

namespace IRegressionModel {

// Interface for accessing network information in regression models.
class INetworkInfo {
public:
    // Get the network configuration information.
    virtual const NetworkInfo& GetNetworkInfo() const = 0;
    // Get the number of emotions using as inference input.
    virtual std::size_t GetEmotionsCount() const = 0;
    // Get the name of the emotion at the specified index.
    virtual const char* GetEmotionName(std::size_t index) const = 0;
    // Get the default emotion tensor.
    virtual nva2x::HostTensorFloatConstView GetDefaultEmotion() const = 0;
    // Get the identity name for this network.
    virtual const char* GetIdentityName() const = 0;

    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~INetworkInfo();
};

// Interface for accessing animator data in regression models.
class IAnimatorData {
public:
    // Get a view on the animator data.
    virtual AnimatorDataView GetAnimatorData() const = 0;
    // Get the skin PCA reconstruction data.
    virtual IAnimatorPcaReconstruction::HostData GetSkinPcaReconstructionData() const = 0;
    // Get the tongue PCA reconstruction data.
    virtual IAnimatorPcaReconstruction::HostData GetTonguePcaReconstructionData() const = 0;
    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~IAnimatorData();
};

// Interface for accessing model information in regression models.
class IGeometryModelInfo {
public:
  // Get the network information.
  virtual const INetworkInfo& GetNetworkInfo() const = 0;
  // Get the animator data.
  virtual const IAnimatorData& GetAnimatorData() const = 0;
  // Get the animator parameters.
  virtual const AnimatorParams& GetAnimatorParams() const = 0;

  // Get geometry executor creation parameters with the specified runtime settings.
  virtual GeometryExecutorCreationParameters GetExecutorCreationParameters(
      IGeometryExecutor::ExecutionOption executionOption,
      std::size_t frameRateNumerator,
      std::size_t frameRateDenominator
  ) const = 0;
  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IGeometryModelInfo();
};

// Interface for accessing blendshape solve model information in regression models.
class IBlendshapeSolveModelInfo {
public:
  // Get blendshape solve executor creation parameters with the specified runtime settings.
  virtual BlendshapeSolveExecutorCreationParameters GetExecutorCreationParameters(
      IGeometryExecutor::ExecutionOption executionOption
      ) const = 0;
  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IBlendshapeSolveModelInfo();
};

} // namespace IRegressionModel

namespace IDiffusionModel {

// Interface for accessing network information in diffusion models.
class INetworkInfo {
public:
    // Get the network configuration information.
    virtual const NetworkInfo& GetNetworkInfo() const = 0;
    // Get the number of emotions using as inference input.
    virtual std::size_t GetEmotionsCount() const = 0;
    // Get the name of the emotion at the specified index.
    virtual const char* GetEmotionName(std::size_t index) const = 0;
    // Get the default emotion tensor.
    virtual nva2x::HostTensorFloatConstView GetDefaultEmotion() const = 0;
    // Get the number of identities supported by this network.
    virtual std::size_t GetIdentityLength() const = 0;
    // Get the name of the identity at the specified index.
    virtual const char* GetIdentityName(std::size_t index) const = 0;

    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~INetworkInfo();
};

// Interface for accessing animator data in diffusion models.
class IAnimatorData {
public:
    // Get a view on the animator data.
    virtual AnimatorDataView GetAnimatorData() const = 0;
    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~IAnimatorData();
};

// Interface for accessing model information in diffusion models.
class IGeometryModelInfo {
public:
  // Get the network information.
  virtual const INetworkInfo& GetNetworkInfo() const = 0;
  // Get the animator data for the specified identity index.
  virtual const IAnimatorData* GetAnimatorData(std::size_t identityIndex) const = 0;
  // Get the animator parameters for the specified identity index.
  virtual const AnimatorParams* GetAnimatorParams(std::size_t identityIndex) const = 0;

  // Get geometry executor creation parameters with the specified runtime settings.
  virtual GeometryExecutorCreationParameters GetExecutorCreationParameters(
      IGeometryExecutor::ExecutionOption executionOption,
      std::size_t identityIndex,
      bool constantNoise
      ) const = 0;
  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IGeometryModelInfo();
};

// Interface for accessing blendshape solve model information in diffusion models.
class IBlendshapeSolveModelInfo {
public:
  // Get blendshape solve executor creation parameters with the specified runtime settings.
  virtual BlendshapeSolveExecutorCreationParameters GetExecutorCreationParameters(
      IGeometryExecutor::ExecutionOption executionOption,
      std::size_t identityIndex
      ) const = 0;
  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IBlendshapeSolveModelInfo();
};

}  // namespace IDiffusionModel

// Interface for accessing blendshape solver configuration.
class IBlendshapeSolverConfig {
public:
    // Get the blendshape solver parameters.
    virtual const BlendshapeSolverParams& GetBlendshapeSolverParams() const = 0;
    // Get the blendshape solver configuration.
    virtual BlendshapeSolverConfig GetBlendshapeSolverConfig() const = 0;
    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~IBlendshapeSolverConfig();
};

// Interface for accessing blendshape solver data.
class IBlendshapeSolverData {
public:
    // Get a view on the blendshape solver data.
    virtual BlendshapeSolverDataView GetBlendshapeSolverDataView() const = 0;
    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~IBlendshapeSolverData();
};


// Interface for accessing geometry executor bundle components.
// It contains everything needed to run the geometry executor.
class IGeometryExecutorBundle {
public:
    // Get the CUDA stream for GPU operations.
    virtual nva2x::ICudaStream& GetCudaStream() = 0;
    // Get the CUDA stream for GPU operations (const version).
    virtual const nva2x::ICudaStream& GetCudaStream() const = 0;

    // Get the audio accumulator for the specified track index.
    virtual nva2x::IAudioAccumulator& GetAudioAccumulator(std::size_t trackIndex) = 0;
    // Get the audio accumulator for the specified track index (const version).
    virtual const nva2x::IAudioAccumulator& GetAudioAccumulator(std::size_t trackIndex) const = 0;

    // Get the emotion accumulator for the specified track index.
    virtual nva2x::IEmotionAccumulator& GetEmotionAccumulator(std::size_t trackIndex) = 0;
    // Get the emotion accumulator for the specified track index (const version).
    virtual const nva2x::IEmotionAccumulator& GetEmotionAccumulator(std::size_t trackIndex) const = 0;

    // Returns the geometry executor.
    virtual IGeometryExecutor& GetExecutor() = 0;
    // Get the geometry executor (const version).
    virtual const IGeometryExecutor& GetExecutor() const = 0;

    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~IGeometryExecutorBundle();
};

// Interface for accessing blendshape executor bundle components.
// It contains everything needed to run the blendshape executor.
class IBlendshapeExecutorBundle {
public:
    // Get the CUDA stream for GPU operations.
    virtual nva2x::ICudaStream& GetCudaStream() = 0;
    // Get the CUDA stream for GPU operations (const version).
    virtual const nva2x::ICudaStream& GetCudaStream() const = 0;

    // Get the audio accumulator for the specified track index.
    virtual nva2x::IAudioAccumulator& GetAudioAccumulator(std::size_t trackIndex) = 0;
    // Get the audio accumulator for the specified track index (const version).
    virtual const nva2x::IAudioAccumulator& GetAudioAccumulator(std::size_t trackIndex) const = 0;

    // Get the emotion accumulator for the specified track index.
    virtual nva2x::IEmotionAccumulator& GetEmotionAccumulator(std::size_t trackIndex) = 0;
    // Get the emotion accumulator for the specified track index (const version).
    virtual const nva2x::IEmotionAccumulator& GetEmotionAccumulator(std::size_t trackIndex) const = 0;

    // Get the blendshape executor.
    virtual IBlendshapeExecutor& GetExecutor() = 0;
    // Get the blendshape executor (const version).
    virtual const IBlendshapeExecutor& GetExecutor() const = 0;

    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~IBlendshapeExecutorBundle();
};

} // namespace nva2f
