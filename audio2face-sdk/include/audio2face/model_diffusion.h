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
#include "audio2face/model_shared.h"

namespace nva2f {

namespace IDiffusionModel {

  // Network information details for diffusion model.
  struct NetworkInfo {
    // Number of emotions per track as inference input.
    std::size_t emotionLength;
    // Number of identities in the model.
    std::size_t identityLength;
    // Number of floats in skin output.
    std::size_t skinDim;
    // Number of floats in tongue output.
    std::size_t tongueDim;
    // Number of floats in jaw output.
    std::size_t jawDim;
    // Number of floats in eyes output.
    std::size_t eyesDim;
    // Number of diffusion steps.
    std::size_t numDiffusionSteps;
    // Number of GRU layers.
    std::size_t numGruLayers;
    // Dimension of the GRU layer.
    std::size_t gruLatentDim;
    // Number of discarded frames to truncate from the left in inference output.
    std::size_t numFramesLeftTruncate;
    // Number of discarded frames to truncate from the right in inference output.
    std::size_t numFramesRightTruncate;
    // Number of good frames in the center of inference output.
    std::size_t numFramesCenter;
    // Number of samples in inference audio buffer input.
    std::size_t bufferLength;
    // Number of samples to pad from the left before using audio input.
    std::size_t paddingLeft;
    // Number of samples to pad from the right after using audio input.
    std::size_t paddingRight;
    // Sampling rate of the audio buffer input.
    std::size_t bufferSamplerate;
  };

  // Indices of the buffers in bindings for this model.
  constexpr std::size_t kEmotionTensorIndex = 0;
  constexpr std::size_t kIdentityTensorIndex = 1;
  constexpr std::size_t kInputLatentsTensorIndex = 2;
  constexpr std::size_t kNoiseTensorIndex = 3;
  constexpr std::size_t kWindowTensorIndex = 4;
  constexpr std::size_t kOutputLatentsTensorIndex = 5;
  constexpr std::size_t kPredictionTensorIndex = 6;

  // Buffers required for this model's inference as inputs.
  class IInferenceInputBuffers {
  public:
    // Return the number of copies of the tensors.
    virtual std::size_t GetCount() const = 0;
    // Return the number of frames in each copy.
    virtual std::size_t GetFrameCount() const = 0;

    // Return the full tensors, used to create bindings.
    virtual nva2x::DeviceTensorFloatView GetEmotionTensor(std::size_t count = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetEmotionTensor(std::size_t count = 0) const = 0;
    virtual nva2x::DeviceTensorFloatView GetInputTensor(std::size_t count = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetInputTensor(std::size_t count = 0) const = 0;
    virtual nva2x::DeviceTensorFloatView GetIdentityTensor(std::size_t count = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetIdentityTensor(std::size_t count = 0) const = 0;
    virtual nva2x::DeviceTensorFloatView GetNoiseTensor(std::size_t count = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetNoiseTensor(std::size_t count = 0) const = 0;

    // Return views to specific parts of the tensors, used to read/write data.
    virtual nva2x::DeviceTensorFloatView GetEmotions(std::size_t frameIndex, std::size_t index = 0) = 0;
    virtual nva2x::DeviceTensorFloatView GetInput(std::size_t index = 0) = 0;
    virtual nva2x::DeviceTensorFloatView GetIdentity(std::size_t index = 0) = 0;
    virtual nva2x::DeviceTensorFloatView GetNoise(std::size_t index = 0) = 0;

    // Delete this object.
    virtual void Destroy() = 0;

  protected:
    virtual ~IInferenceInputBuffers();
  };

  // Buffers required for this model's inference as state kept between inferences.
  class IInferenceStateBuffers {
  public:
    // Return the number of copies of the tensors.
    virtual std::size_t GetCount() const = 0;

    // Return the full tensors, used to create bindings.
    virtual nva2x::DeviceTensorFloatView GetInputGRUStateTensor(std::size_t count = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetInputGRUStateTensor(std::size_t count = 0) const = 0;
    virtual nva2x::DeviceTensorFloatView GetOutputGRUStateTensor(std::size_t count = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetOutputGRUStateTensor(std::size_t count = 0) const = 0;

    // GRU state is interleaved, so a full view for a given index can't be provided.
    // Instead, a function copying the data from output to input is provided.
    virtual std::error_code CopyOutputToInputGRUState(cudaStream_t cudaStream, std::size_t index = 0) = 0;
    // GRU state is interleaved, so a full view for a given index can't be provided.
    // Instead, a function zeroing the state is provided.
    virtual std::error_code Reset(cudaStream_t cudaStream, std::size_t index = 0) = 0;

    // Swap input and output GRU state tensors.
    virtual std::error_code Swap() = 0;

    // Delete this object.
    virtual void Destroy() = 0;

  protected:
    virtual ~IInferenceStateBuffers();
  };

    // Output buffers required for this model's inference.
  class IInferenceOutputBuffers {
  public:
    // Return the number of copies of the tensors.
    virtual std::size_t GetCount() const = 0;
    // Return the number of frames in each copy.
    virtual std::size_t GetFrameCount() const = 0;

    // Return the full tensors, used to create bindings.
    virtual nva2x::DeviceTensorFloatView GetResultTensor(std::size_t count = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetResultTensor(std::size_t count = 0) const = 0;

    // Return views to specific parts of the tensors, used to read/write data.
    virtual nva2x::DeviceTensorFloatView GetInferenceResult(std::size_t index = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetInferenceResult(std::size_t index = 0) const = 0;
    virtual nva2x::DeviceTensorFloatConstView GetInferenceResultSkin(std::size_t frameIndex, std::size_t index = 0) const = 0;
    virtual nva2x::DeviceTensorFloatConstView GetInferenceResultTongue(std::size_t frameIndex, std::size_t index = 0) const = 0;
    virtual nva2x::DeviceTensorFloatConstView GetInferenceResultJaw(std::size_t frameIndex, std::size_t index = 0) const = 0;
    virtual nva2x::DeviceTensorFloatConstView GetInferenceResultEyes(std::size_t frameIndex, std::size_t index = 0) const = 0;

    // Return the batch information for the tensors, used to pass to batched processing.
    virtual nva2x::TensorBatchInfo GetSkinBatchInfo(std::size_t frameIndex) const = 0;
    virtual nva2x::TensorBatchInfo GetTongueBatchInfo(std::size_t frameIndex) const = 0;
    virtual nva2x::TensorBatchInfo GetJawBatchInfo(std::size_t frameIndex) const = 0;
    virtual nva2x::TensorBatchInfo GetEyesBatchInfo(std::size_t frameIndex) const = 0;

    // Delete this object.
    virtual void Destroy() = 0;

  protected:
    virtual ~IInferenceOutputBuffers();
  };

  // Buffers required for this model's results.
  using IResultBuffers = ISharedModel::IGeometryResultBuffers;

}  // namespace IDiffusionModel

} // namespace nva2f
