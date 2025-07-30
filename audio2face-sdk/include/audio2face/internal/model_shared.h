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

#include "audio2x/internal/tensor.h"
#include "audio2face/model_shared.h"

namespace nva2f {

namespace ISharedModel {

  class GeometryInferenceOutputBuffers : public IGeometryInferenceOutputBuffers {
  public:
    std::error_code Init(
      std::size_t skinSize, std::size_t tongueSize, std::size_t jawSize, std::size_t eyesSize,
      std::size_t count = 1
      );
    nva2x::DeviceTensorFloatView GetInferenceResult(std::size_t index, std::size_t count);
    nva2x::DeviceTensorFloatConstView GetInferenceResult(std::size_t index, std::size_t count) const;

    std::size_t GetCount() const override;
    std::size_t GetResultSize() const override;

    nva2x::DeviceTensorFloatView GetResultTensor(std::size_t count = 0) override;
    nva2x::DeviceTensorFloatConstView GetResultTensor(std::size_t count = 0) const override;

    nva2x::DeviceTensorFloatConstView GetInferenceResultSkin(std::size_t index = 0) const override;
    nva2x::DeviceTensorFloatConstView GetInferenceResultTongue(std::size_t index = 0) const override;
    nva2x::DeviceTensorFloatConstView GetInferenceResultJaw(std::size_t index = 0) const override;
    nva2x::DeviceTensorFloatConstView GetInferenceResultEyes(std::size_t index = 0) const override;

    nva2x::TensorBatchInfo GetSkinBatchInfo() const override;
    nva2x::TensorBatchInfo GetTongueBatchInfo() const override;
    nva2x::TensorBatchInfo GetJawBatchInfo() const override;
    nva2x::TensorBatchInfo GetEyesBatchInfo() const override;

    void Destroy() override;

  private:
    nva2x::DeviceTensorFloat _inferenceResult;

    std::size_t _count{0};
    std::size_t _inferenceResultSize{0};
    std::size_t _skinOffset{0};
    std::size_t _skinSize{0};
    std::size_t _tongueOffset{0};
    std::size_t _tongueSize{0};
    std::size_t _jawOffset{0};
    std::size_t _jawSize{0};
    std::size_t _eyesOffset{0};
    std::size_t _eyesSize{0};
  };

  class GeometryResultBuffers : public IGeometryResultBuffers {
  public:
    std::error_code Init(
      std::size_t skinSize, std::size_t tongueSize,
      std::size_t count = 1
      );

    std::size_t GetCount() const override;
    std::size_t GetResultSize() const override;

    nva2x::DeviceTensorFloatView GetResultTensor(std::size_t count = 0) override;
    nva2x::DeviceTensorFloatConstView GetResultTensor(std::size_t count = 0) const override;
    nva2x::DeviceTensorFloatView GetResultSkinGeometry(std::size_t index = 0) override;
    nva2x::DeviceTensorFloatConstView GetResultSkinGeometry(std::size_t index = 0) const override;
    nva2x::DeviceTensorFloatView GetResultTongueGeometry(std::size_t index = 0) override;
    nva2x::DeviceTensorFloatConstView GetResultTongueGeometry(std::size_t index = 0) const override;
    nva2x::DeviceTensorFloatView GetResultJawTransform(std::size_t index = 0) override;
    nva2x::DeviceTensorFloatConstView GetResultJawTransform(std::size_t index = 0) const override;
    nva2x::DeviceTensorFloatView GetResultEyesRotation(std::size_t index = 0) override;
    nva2x::DeviceTensorFloatConstView GetResultEyesRotation(std::size_t index = 0) const override;

    nva2x::TensorBatchInfo GetSkinBatchInfo() const override;
    nva2x::TensorBatchInfo GetTongueBatchInfo() const override;
    nva2x::TensorBatchInfo GetJawBatchInfo() const override;
    nva2x::TensorBatchInfo GetEyesBatchInfo() const override;

    void Destroy() override;

  private:
    nva2x::DeviceTensorFloat _result;

    std::size_t _count{0};
    std::size_t _resultSize{0};
    std::size_t _skinGeometryOffset{0};
    std::size_t _skinGeometrySize{0};
    std::size_t _tongueGeometryOffset{0};
    std::size_t _tongueGeometrySize{0};
    std::size_t _jawTransformOffset{0};
    std::size_t _jawTransformSize{0};
    std::size_t _eyesRotationOffset{0};
    std::size_t _eyesRotationSize{0};
  };

} // namespace ISharedModel

} // namespace nva2f
