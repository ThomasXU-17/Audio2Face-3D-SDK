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

namespace IRegressionModel {

  // Network information details for regression model.
  struct NetworkInfo {
    // Number of implicit emotions per track as inference input.
    std::size_t implicitEmotionLength;
    // Number of explicit emotions per track as inference input.
    std::size_t explicitEmotionLength;
    // Number of shapes, i.e. PCA coefficients, in skin output.
    std::size_t numShapesSkin;
    // Number of shapes, i.e. PCA coefficients, in tongue output.
    std::size_t numShapesTongue;
    // Number of floats in skin geometry output.
    std::size_t resultSkinSize;
    // Number of floats in tongue geometry output.
    std::size_t resultTongueSize;
    // Number of floats in jaw pose output.
    std::size_t resultJawSize;
    // Number of floats in eyes rotation output.
    std::size_t resultEyesSize;
    // Number of samples in inference audio buffer input.
    std::size_t bufferLength;
    // Offset from the start of the inference audio buffer to which the output results will be timed.
    std::size_t bufferOffset;
    // Sampling rate of the audio buffer input.
    std::size_t bufferSamplerate;
  };

  // Indices of the buffers in bindings for this model.
  constexpr std::size_t kEmotionTensorIndex = 0;
  constexpr std::size_t kInputTensorIndex = 1;
  constexpr std::size_t kResultTensorIndex = 2;

  // Buffers required for this model's inference as inputs.
  class IInferenceInputBuffers {
  public:
    // Return the number of copies of the tensors.
    virtual std::size_t GetCount() const = 0;

    // Return the full tensors, used to create bindings.
    virtual nva2x::DeviceTensorFloatView GetEmotionTensor(std::size_t count = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetEmotionTensor(std::size_t count = 0) const = 0;
    virtual nva2x::DeviceTensorFloatView GetInputTensor(std::size_t count = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetInputTensor(std::size_t count = 0) const = 0;

    // Return views to specific parts of the tensors, used to read/write data.
    virtual nva2x::DeviceTensorFloatView GetImplicitEmotions(std::size_t index = 0) = 0;
    virtual nva2x::DeviceTensorFloatView GetExplicitEmotions(std::size_t index = 0) = 0;
    virtual nva2x::DeviceTensorFloatView GetInput(std::size_t index = 0) = 0;

    // Delete this object.
    virtual void Destroy() = 0;

  protected:
    virtual ~IInferenceInputBuffers();
  };

  // Buffers required for this model's inference as outputs.
  using IInferenceOutputBuffers = ISharedModel::IGeometryInferenceOutputBuffers;

  // Buffers required for this model's results.
  using IResultBuffers = ISharedModel::IGeometryResultBuffers;

};

} // namespace nva2f
