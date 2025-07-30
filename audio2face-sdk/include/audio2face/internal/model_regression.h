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

#include "audio2x/internal/inference_engine.h"
#include "audio2x/internal/tensor.h"
#include "audio2face/model_regression.h"
#include "audio2face/internal/model_shared.h"

namespace nva2f {

namespace IRegressionModel {

  class InferenceInputBuffers : public IInferenceInputBuffers {
  public:
    std::error_code Init(const NetworkInfo& networkInfo, std::size_t count = 1);

    std::size_t GetCount() const override;

    // Return the full tensors, used to create bindings.
    nva2x::DeviceTensorFloatView GetEmotionTensor(std::size_t count = 0) override;
    nva2x::DeviceTensorFloatConstView GetEmotionTensor(std::size_t count = 0) const override;
    nva2x::DeviceTensorFloatView GetInputTensor(std::size_t count = 0) override;
    nva2x::DeviceTensorFloatConstView GetInputTensor(std::size_t count = 0) const override;

    // Return views to specific parts of the tensors, used to read/write data.
    nva2x::DeviceTensorFloatView GetImplicitEmotions(std::size_t index = 0) override;
    nva2x::DeviceTensorFloatView GetExplicitEmotions(std::size_t index = 0) override;
    nva2x::DeviceTensorFloatView GetInput(std::size_t index = 0) override;

    void Destroy() override;

  private:
    nva2x::DeviceTensorFloat _emotion;
    nva2x::DeviceTensorFloat _input;

    std::size_t _count{0};
    std::size_t _implicitEmotionSize{0};
    std::size_t _explicitEmotionSize{0};
    std::size_t _emotionSize{0};
    std::size_t _inputSize{0};
  };

  class InferenceOutputBuffers : public ISharedModel::GeometryInferenceOutputBuffers {
  public:
    std::error_code Init(const NetworkInfo& networkInfo, std::size_t count = 1);
  };

  class ResultBuffers : public ISharedModel::GeometryResultBuffers {
  public:
    std::error_code Init(const NetworkInfo& networkInfo, std::size_t count = 1);
  };

  InferenceInputBuffers* CreateInferenceInputBuffers(const NetworkInfo& networkInfo, std::size_t count);
  InferenceOutputBuffers* CreateInferenceOutputBuffers(const NetworkInfo& networkInfo, std::size_t count);
  ResultBuffers* CreateResultBuffers(const NetworkInfo& networkInfo, std::size_t count);

  const nva2x::BufferBindingsDescription& GetBindingsDescription();
  nva2x::BufferBindings* CreateBindings();

} // namespace IRegressionModel

IRegressionModel::IInferenceInputBuffers* CreateInferenceInputBuffersForRegressionModel_INTERNAL(
  const IRegressionModel::NetworkInfo& networkInfo, std::size_t count
  );
IRegressionModel::IInferenceOutputBuffers* CreateInferenceOutputBuffersForRegressionModel_INTERNAL(
  const IRegressionModel::NetworkInfo& networkInfo, std::size_t count
  );
IRegressionModel::IResultBuffers* CreateResultBuffersForRegressionModel_INTERNAL(
  const IRegressionModel::NetworkInfo& networkInfo, std::size_t count
  );

const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForRegressionModel_INTERNAL();

nva2x::IBufferBindings* CreateBindingsForRegressionModel_INTERNAL();

} // namespace nva2f
