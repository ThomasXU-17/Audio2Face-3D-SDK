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
#include "audio2face/internal/model_regression.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"

#include <array>
#include <memory>

namespace nva2f {

namespace IRegressionModel {

IInferenceInputBuffers::~IInferenceInputBuffers() = default;

std::error_code InferenceInputBuffers::Init(const NetworkInfo& networkInfo, std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count > 0,
    "Count must be greater than 1",
    nva2x::ErrorCode::eInvalidValue
  );

  const std::size_t emotionSize = networkInfo.implicitEmotionLength + networkInfo.explicitEmotionLength;

  A2F_CHECK_RESULT_WITH_MSG(_emotion.Allocate(emotionSize * count), "Unable to allocate emotion buffer");
  A2F_CHECK_RESULT_WITH_MSG(_input.Allocate(networkInfo.bufferLength * count), "Unable to allocate input buffer");

  _count = count;
  _implicitEmotionSize = networkInfo.implicitEmotionLength;
  _explicitEmotionSize = networkInfo.explicitEmotionLength;
  _emotionSize = emotionSize;
  _inputSize = networkInfo.bufferLength;

  return nva2x::ErrorCode::eSuccess;
}

std::size_t InferenceInputBuffers::GetCount() const {
  return _count;
}

nva2x::DeviceTensorFloatView InferenceInputBuffers::GetEmotionTensor(std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _emotion.View(0, count * _emotionSize);
}

nva2x::DeviceTensorFloatConstView InferenceInputBuffers::GetEmotionTensor(std::size_t count) const {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _emotion.View(0, count * _emotionSize);
}

nva2x::DeviceTensorFloatView InferenceInputBuffers::GetInputTensor(std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _input.View(0, count * _inputSize);
}

nva2x::DeviceTensorFloatConstView InferenceInputBuffers::GetInputTensor(std::size_t count) const {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _input.View(0, count * _inputSize);
}

nva2x::DeviceTensorFloatView InferenceInputBuffers::GetImplicitEmotions(std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Count " << index << " must be smaller than count " << _count,
    {});
  return _emotion.View(index * _emotionSize, _implicitEmotionSize);
}

nva2x::DeviceTensorFloatView InferenceInputBuffers::GetExplicitEmotions(std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Count " << index << " must be smaller than count " << _count,
    {});
  return _emotion.View(index * _emotionSize + _implicitEmotionSize, _explicitEmotionSize);
}

nva2x::DeviceTensorFloatView InferenceInputBuffers::GetInput(std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Count " << index << " must be smaller than count " << _count,
    {});
  return _input.View(index * _inputSize, _inputSize);
}

void InferenceInputBuffers::Destroy() {
  delete this;
}

std::error_code InferenceOutputBuffers::Init(const NetworkInfo& networkInfo, std::size_t count) {
  return GeometryInferenceOutputBuffers::Init(
    networkInfo.numShapesSkin, networkInfo.numShapesTongue, networkInfo.resultJawSize, networkInfo.resultEyesSize,
    count
  );
}

std::error_code ResultBuffers::Init(const NetworkInfo& networkInfo, std::size_t count) {
  return GeometryResultBuffers::Init(
    networkInfo.resultSkinSize, networkInfo.resultTongueSize,
    count
  );
}

} // namespace IRegressionModel

IRegressionModel::InferenceInputBuffers* IRegressionModel::CreateInferenceInputBuffers(
  const NetworkInfo& networkInfo, std::size_t count
  ) {
  LOG_DEBUG("IRegressionModel::CreateInferenceInputBuffers()");
  auto buffers = std::make_unique<InferenceInputBuffers>();
  if (buffers->Init(networkInfo, count)) {
    LOG_ERROR("Unable to allocate regression model inference input buffers");
    return nullptr;
  }
  return buffers.release();
}

IRegressionModel::InferenceOutputBuffers* IRegressionModel::CreateInferenceOutputBuffers(
  const NetworkInfo& networkInfo, std::size_t count
  ) {
  LOG_DEBUG("IRegressionModel::CreateInferenceOutputBuffers()");
  auto buffers = std::make_unique<InferenceOutputBuffers>();
  if (buffers->Init(networkInfo, count)) {
    LOG_ERROR("Unable to allocate regression model inference output buffers");
    return nullptr;
  }
  return buffers.release();
}

IRegressionModel::ResultBuffers* IRegressionModel::CreateResultBuffers(
  const NetworkInfo& networkInfo, std::size_t count
  ) {
  LOG_DEBUG("IRegressionModel::CreateResultBuffers()");
  auto buffers = std::make_unique<ResultBuffers>();
  if (buffers->Init(networkInfo, count)) {
    LOG_ERROR("Unable to allocate regression model result buffers");
    return nullptr;
  }
  return buffers.release();
}


const nva2x::BufferBindingsDescription& IRegressionModel::GetBindingsDescription() {
  using IOType = nva2x::IBufferBindingsDescription::IOType;
  using DimensionType = nva2x::IBufferBindingsDescription::DimensionType;
  static constexpr std::array<nva2x::BindingDescription, 3> kDescriptions = {{
    {"emotion", IOType::INPUT, {{DimensionType::BATCH, DimensionType::FIXED, DimensionType::FIXED}}},
    {"input", IOType::INPUT, {{DimensionType::BATCH, DimensionType::FIXED, DimensionType::FIXED}}},
    {"result", IOType::OUTPUT, {{DimensionType::BATCH, DimensionType::FIXED, DimensionType::FIXED}}},
  }};
  // Validate everything is as expected at compile-time.
  static_assert(nva2x::IsSorted(kDescriptions.data(), kDescriptions.size()));
  static_assert(0 == nva2x::CompareCStr("emotion", kDescriptions[kEmotionTensorIndex].name));
  static_assert(0 == nva2x::CompareCStr("input", kDescriptions[kInputTensorIndex].name));
  static_assert(0 == nva2x::CompareCStr("result", kDescriptions[kResultTensorIndex].name));
  static_assert(2 == nva2x::GetInputCount(kDescriptions.data(), kDescriptions.size()));
  static_assert(1 == nva2x::GetBatchIndexCount(kDescriptions[kEmotionTensorIndex]));
  static_assert(1 == nva2x::GetBatchIndexCount(kDescriptions[kInputTensorIndex]));
  static_assert(1 == nva2x::GetBatchIndexCount(kDescriptions[kResultTensorIndex]));

  static const nva2x::BufferBindingsDescription descriptions({kDescriptions.begin(), kDescriptions.end()});
  return descriptions;
}

nva2x::BufferBindings* IRegressionModel::CreateBindings() {
  return new nva2x::BufferBindings(GetBindingsDescription());
}

IRegressionModel::IInferenceInputBuffers* CreateInferenceInputBuffersForRegressionModel_INTERNAL(
  const IRegressionModel::NetworkInfo& networkInfo, std::size_t count
  ) {
  return IRegressionModel::CreateInferenceInputBuffers(networkInfo, count);
}

IRegressionModel::IInferenceOutputBuffers* CreateInferenceOutputBuffersForRegressionModel_INTERNAL(
  const IRegressionModel::NetworkInfo& networkInfo, std::size_t count
  ) {
  return IRegressionModel::CreateInferenceOutputBuffers(networkInfo, count);
}

IRegressionModel::IResultBuffers* CreateResultBuffersForRegressionModel_INTERNAL(
  const IRegressionModel::NetworkInfo& networkInfo, std::size_t count
  ) {
  return IRegressionModel::CreateResultBuffers(networkInfo, count);
}

const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForRegressionModel_INTERNAL() {
  return IRegressionModel::GetBindingsDescription();
}

nva2x::IBufferBindings* CreateBindingsForRegressionModel_INTERNAL() {
  return IRegressionModel::CreateBindings();
}

} // namespace nva2f
