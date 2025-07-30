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

#include "audio2emotion/executor_classifier.h"
#include "audio2emotion/executor_postprocess.h"
#include "audio2emotion/executor.h"
#include "audio2x/cuda_stream.h"

namespace nva2e {

namespace IClassifierModel {

// Interface for network information used by emotion classification models.
class INetworkInfo {
public:
    // Get network information with the specified buffer length, since buffer length is a runtime parameter.
    virtual NetworkInfo GetNetworkInfo(std::size_t bufferLength) const = 0;
    // Get the number of emotions returned by the model.
    virtual std::size_t GetEmotionsCount() const = 0;
    // Get the name of the emotion at the specified index.
    virtual const char* GetEmotionName(std::size_t index) const = 0;

    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~INetworkInfo();
};

// Interface for configuration information used by emotion classification models.
class IConfigInfo {
public:
    // Get the post-process data configuration.
    virtual const PostProcessData& GetPostProcessData() const = 0;
    // Get the post-process parameters configuration.
    virtual const PostProcessParams& GetPostProcessParams() const = 0;
    // Get the input strength parameter for the model.
    virtual float GetInputStrength() const = 0;

    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~IConfigInfo();
};

// Interface for whole emotion model information that combines network and configuration data.
class IEmotionModelInfo {
public:
  // Get the network information for this emotion model.
  virtual const INetworkInfo& GetNetworkInfo() const = 0;
  // Get the configuration information for this emotion model.
  virtual const IConfigInfo& GetConfigInfo() const = 0;
  // Get executor creation parameters with the specified runtime settings.
  virtual EmotionExecutorCreationParameters GetExecutorCreationParameters(
    std::size_t bufferLength,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    std::size_t inferencesToSkip
  ) const = 0;
  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IEmotionModelInfo();
};

} // namespace IClassifierModel

namespace IPostProcessModel {

// Type aliases for network information, same as in classifier model.
using INetworkInfo = IClassifierModel::INetworkInfo;

// Type aliases for configuration information, same as in classifier model.
using IConfigInfo = IClassifierModel::IConfigInfo;

// Interface for whole emotion model information for post-processing model which does not run inference.
class IEmotionModelInfo {
public:
  // Get the network information for this emotion model.
  virtual const INetworkInfo& GetNetworkInfo() const = 0;
  // Get the configuration information for this emotion model.
  virtual const IConfigInfo& GetConfigInfo() const = 0;
  // Get executor creation parameters with the specified runtime settings.
  virtual EmotionExecutorCreationParameters GetExecutorCreationParameters(
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator
  ) const = 0;
  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IEmotionModelInfo();
};

} // namespace IPostProcessModel

// Interface for a bundle containing emotion executor and related components.
// It contains everything needed to run the emotion executor.
class IEmotionExecutorBundle {
public:
    // Get the CUDA stream used for GPU operations.
    virtual nva2x::ICudaStream& GetCudaStream() = 0;
    // Get the CUDA stream used for GPU operations (const version).
    virtual const nva2x::ICudaStream& GetCudaStream() const = 0;

    // Get the audio accumulator for the specified track index.
    virtual nva2x::IAudioAccumulator& GetAudioAccumulator(std::size_t trackIndex) = 0;
    // Get the audio accumulator for the specified track index (const version).
    virtual const nva2x::IAudioAccumulator& GetAudioAccumulator(std::size_t trackIndex) const = 0;

    // Get the preferred emotion accumulator for the specified track index.
    virtual nva2x::IEmotionAccumulator& GetPreferredEmotionAccumulator(std::size_t trackIndex) = 0;
    // Get the preferred emotion accumulator for the specified track index (const version).
    virtual const nva2x::IEmotionAccumulator& GetPreferredEmotionAccumulator(std::size_t trackIndex) const = 0;

    // Get the emotion executor instance.
    virtual IEmotionExecutor& GetExecutor() = 0;
    // Get the emotion executor instance (const version).
    virtual const IEmotionExecutor& GetExecutor() const = 0;

    // Delete this object.
    virtual void Destroy() = 0;

protected:
    virtual ~IEmotionExecutorBundle();
};

} // namespace nva2e
