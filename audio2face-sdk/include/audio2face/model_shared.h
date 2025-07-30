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

namespace nva2f {

namespace ISharedModel {

  // Output buffers required for this model's inference.
  class IGeometryInferenceOutputBuffers {
  public:
    // Return the number of copies of the tensors.
    virtual std::size_t GetCount() const = 0;
    // Return the size of each frame result.
    virtual std::size_t GetResultSize() const = 0;

    // Return the full tensors, used to create bindings.
    virtual nva2x::DeviceTensorFloatView GetResultTensor(std::size_t count = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetResultTensor(std::size_t count = 0) const = 0;

    // Return views to specific parts of the tensors, used to read/write data.
    virtual nva2x::DeviceTensorFloatConstView GetInferenceResultSkin(std::size_t index = 0) const = 0;
    virtual nva2x::DeviceTensorFloatConstView GetInferenceResultTongue(std::size_t index = 0) const = 0;
    virtual nva2x::DeviceTensorFloatConstView GetInferenceResultJaw(std::size_t index = 0) const = 0;
    virtual nva2x::DeviceTensorFloatConstView GetInferenceResultEyes(std::size_t index = 0) const = 0;

    // Return the batch information for the tensors, used to pass to batched processing.
    virtual nva2x::TensorBatchInfo GetSkinBatchInfo() const = 0;
    virtual nva2x::TensorBatchInfo GetTongueBatchInfo() const = 0;
    virtual nva2x::TensorBatchInfo GetJawBatchInfo() const = 0;
    virtual nva2x::TensorBatchInfo GetEyesBatchInfo() const = 0;

    // Delete this object.
    virtual void Destroy() = 0;

  protected:
    virtual ~IGeometryInferenceOutputBuffers();
  };

  // Buffers required for this model's results.
  class IGeometryResultBuffers {
  public:
    // Return the number of copies of the tensors.
    virtual std::size_t GetCount() const = 0;
    // Return the size of each frame result.
    virtual std::size_t GetResultSize() const = 0;

    // Return views to specific parts of the tensors, used to read/write data.
    virtual nva2x::DeviceTensorFloatView GetResultTensor(std::size_t index = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetResultTensor(std::size_t index = 0) const = 0;
    virtual nva2x::DeviceTensorFloatView GetResultSkinGeometry(std::size_t index = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetResultSkinGeometry(std::size_t index = 0) const = 0;
    virtual nva2x::DeviceTensorFloatView GetResultTongueGeometry(std::size_t index = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetResultTongueGeometry(std::size_t index = 0) const = 0;
    virtual nva2x::DeviceTensorFloatView GetResultJawTransform(std::size_t index = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetResultJawTransform(std::size_t index = 0) const = 0;
    virtual nva2x::DeviceTensorFloatView GetResultEyesRotation(std::size_t index = 0) = 0;
    virtual nva2x::DeviceTensorFloatConstView GetResultEyesRotation(std::size_t index = 0) const = 0;

    // Return the batch information for the tensors, used to pass to batched processing.
    virtual nva2x::TensorBatchInfo GetSkinBatchInfo() const = 0;
    virtual nva2x::TensorBatchInfo GetTongueBatchInfo() const = 0;
    virtual nva2x::TensorBatchInfo GetJawBatchInfo() const = 0;
    virtual nva2x::TensorBatchInfo GetEyesBatchInfo() const = 0;

    // Delete this object.
    virtual void Destroy() = 0;

  protected:
    virtual ~IGeometryResultBuffers();
  };

};

} // namespace nva2f
