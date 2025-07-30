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
#include "audio2face/internal/model_diffusion.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"

#include <algorithm>
#include <array>
#include <memory>
#include <random>


namespace nva2f {

namespace IDiffusionModel {

IInferenceInputBuffers::~IInferenceInputBuffers() = default;
IInferenceStateBuffers::~IInferenceStateBuffers() = default;
IInferenceOutputBuffers::~IInferenceOutputBuffers() = default;

std::error_code InferenceInputBuffers::Init(const NetworkInfo& networkInfo, std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count > 0,
    "Count must be greater than 1",
    nva2x::ErrorCode::eInvalidValue
  );

  A2F_CHECK_RESULT_WITH_MSG(_emotion.Allocate(networkInfo.emotionLength * networkInfo.numFramesCenter * count), "Unable to allocate emotion buffer");
  A2F_CHECK_RESULT_WITH_MSG(_input.Allocate(networkInfo.bufferLength * count), "Unable to allocate input buffer");
  A2F_CHECK_RESULT_WITH_MSG(_identity.Allocate(networkInfo.identityLength * count), "Unable to allocate identity buffer");

  const std::size_t noiseSize =
    (networkInfo.numDiffusionSteps + 1) *
    (networkInfo.numFramesLeftTruncate + networkInfo.numFramesRightTruncate + networkInfo.numFramesCenter) *
    (networkInfo.skinDim + networkInfo.tongueDim + networkInfo.jawDim + networkInfo.eyesDim);
  A2F_CHECK_RESULT_WITH_MSG(_noise.Allocate(noiseSize * count), "Unable to allocate noise buffer");

  _count = count;
  _frameCount = networkInfo.numFramesCenter;
  _emotionSize = networkInfo.emotionLength;
  _inputSize = networkInfo.bufferLength;
  _identitySize = networkInfo.identityLength;
  _noiseSize = noiseSize;

  return nva2x::ErrorCode::eSuccess;
}

std::size_t InferenceInputBuffers::GetCount() const {
  return _count;
}

std::size_t InferenceInputBuffers::GetFrameCount() const {
  return _frameCount;
}

nva2x::DeviceTensorFloatView InferenceInputBuffers::GetEmotionTensor(std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _emotion.View(0, count * _frameCount * _emotionSize);
}

nva2x::DeviceTensorFloatConstView InferenceInputBuffers::GetEmotionTensor(std::size_t count) const {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _emotion.View(0, count * _frameCount * _emotionSize);
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

nva2x::DeviceTensorFloatView InferenceInputBuffers::GetIdentityTensor(std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _identity.View(0, count * _identitySize);
}

nva2x::DeviceTensorFloatConstView InferenceInputBuffers::GetIdentityTensor(std::size_t count) const {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _identity.View(0, count * _identitySize);
}

nva2x::DeviceTensorFloatView InferenceInputBuffers::GetNoiseTensor(std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _noise.View(0, count * _noiseSize);
}

nva2x::DeviceTensorFloatConstView InferenceInputBuffers::GetNoiseTensor(std::size_t count) const {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _noise.View(0, count * _noiseSize);
}

nva2x::DeviceTensorFloatView InferenceInputBuffers::GetEmotions(std::size_t frameIndex, std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    frameIndex < _frameCount,
    "Frame index " << frameIndex << " must be smaller than frame count " << _frameCount,
    {});
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Count " << index << " must be smaller than count " << _count,
    {});
  return _emotion.View((index * _frameCount + frameIndex) * _emotionSize, _emotionSize);
}

nva2x::DeviceTensorFloatView InferenceInputBuffers::GetInput(std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Count " << index << " must be smaller than count " << _count,
    {});
  return _input.View(index * _inputSize, _inputSize);
}

nva2x::DeviceTensorFloatView InferenceInputBuffers::GetIdentity(std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Count " << index << " must be smaller than count " << _count,
    {});
  return _identity.View(index * _identitySize, _identitySize);
}

nva2x::DeviceTensorFloatView InferenceInputBuffers::GetNoise(std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Count " << index << " must be smaller than count " << _count,
    {});
  return _noise.View(index * _noiseSize, _noiseSize);
}

void InferenceInputBuffers::Destroy() {
  delete this;
}

std::error_code InferenceStateBuffers::Init(const NetworkInfo& networkInfo, std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count > 0,
    "Count must be greater than 1",
    nva2x::ErrorCode::eInvalidValue
  );

  const auto nbSlices = networkInfo.numDiffusionSteps * networkInfo.numGruLayers;
  const auto sliceSize = networkInfo.gruLatentDim;
  const auto gruSize = nbSlices * sliceSize;

  A2F_CHECK_RESULT_WITH_MSG(_inputGRU.Allocate(gruSize * count), "Unable to allocate GRU input buffer");
  A2F_CHECK_RESULT_WITH_MSG(_outputGRU.Allocate(gruSize * count), "Unable to allocate GRU output buffer");

  _count = count;
  _gruSize = gruSize;
  _nbSlices = nbSlices;
  _sliceSize = sliceSize;

  return nva2x::ErrorCode::eSuccess;
}

std::size_t InferenceStateBuffers::GetCount() const {
  return _count;
}

nva2x::DeviceTensorFloatView InferenceStateBuffers::GetInputGRUStateTensor(std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _inputGRU.View(0, count * _gruSize);
}

nva2x::DeviceTensorFloatConstView InferenceStateBuffers::GetInputGRUStateTensor(std::size_t count) const {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _inputGRU.View(0, count * _gruSize);
}

nva2x::DeviceTensorFloatView InferenceStateBuffers::GetOutputGRUStateTensor(std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _outputGRU.View(0, count * _gruSize);
}

nva2x::DeviceTensorFloatConstView InferenceStateBuffers::GetOutputGRUStateTensor(std::size_t count) const {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _outputGRU.View(0, count * _gruSize);
}

std::error_code InferenceStateBuffers::CopyOutputToInputGRUState(cudaStream_t cudaStream, std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});

  // The per-batch data is interleaved in the GRU state buffer, so multiple slices
  // have to be copied.
  const auto nbSlices = _nbSlices;
  for (std::size_t sliceIndex = 0; sliceIndex < nbSlices; ++sliceIndex) {
    const auto sizeToCopy = _sliceSize;
    const auto offsetToCopy = (sliceIndex * _count + index) * sizeToCopy;
    const auto source = _outputGRU.View(offsetToCopy, sizeToCopy);
    const auto target = _inputGRU.View(offsetToCopy, sizeToCopy);
    A2F_CHECK_RESULT_WITH_MSG(
      nva2x::CopyDeviceToDevice(target, source, cudaStream),
      "Unable to copy GRU state"
      );
  }

  return nva2x::ErrorCode::eSuccess;
}

std::error_code InferenceStateBuffers::Reset(cudaStream_t cudaStream, std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});

  // The per-batch data is interleaved in the GRU state buffer, so multiple slices
  // have to be zeroed.
  const auto nbSlices = _nbSlices;
  for (std::size_t sliceIndex = 0; sliceIndex < nbSlices; ++sliceIndex) {
    const auto sizeToZero = _sliceSize;
    const auto offsetToZero = (sliceIndex * _count + index) * sizeToZero;
    A2F_CHECK_RESULT_WITH_MSG(
      nva2x::FillOnDevice(_inputGRU.View(offsetToZero, sizeToZero), 0.0f, cudaStream),
      "Unable to reset GRU input buffer"
      );
  }

  return nva2x::ErrorCode::eSuccess;
}

std::error_code InferenceStateBuffers::Swap() {
  std::swap(_inputGRU, _outputGRU);
  return nva2x::ErrorCode::eSuccess;
}

void InferenceStateBuffers::Destroy() {
  delete this;
}

std::error_code InferenceOutputBuffers::Init(const NetworkInfo& networkInfo, std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count > 0,
    "Count must be greater than 1",
    nva2x::ErrorCode::eInvalidValue
  );

  const std::size_t frameCount = networkInfo.numFramesLeftTruncate + networkInfo.numFramesRightTruncate +
    networkInfo.numFramesCenter;

  A2F_CHECK_RESULT_WITH_MSG(
    _buffers.Init(
      networkInfo.skinDim, networkInfo.tongueDim, networkInfo.jawDim, networkInfo.eyesDim,
      count * frameCount
      ),
    "Unable to allocate inference result buffer"
    );

  _count = count;
  _frameCount = frameCount;
  _goodFramesOffset = networkInfo.numFramesLeftTruncate;
  _goodFramesCount = networkInfo.numFramesCenter;

  return nva2x::ErrorCode::eSuccess;
}

std::size_t InferenceOutputBuffers::GetCount() const {
  return _count;
}

std::size_t InferenceOutputBuffers::GetFrameCount() const {
  return _frameCount;
}

nva2x::DeviceTensorFloatView InferenceOutputBuffers::GetResultTensor(std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _buffers.GetResultTensor(count * _frameCount);
}

nva2x::DeviceTensorFloatConstView InferenceOutputBuffers::GetResultTensor(std::size_t count) const {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _buffers.GetResultTensor(count * _frameCount);
}

nva2x::DeviceTensorFloatView InferenceOutputBuffers::GetInferenceResult(std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _buffers.GetInferenceResult(index * _frameCount + _goodFramesOffset, _goodFramesCount);
}

nva2x::DeviceTensorFloatConstView InferenceOutputBuffers::GetInferenceResult(std::size_t index) const {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _buffers.GetInferenceResult(index * _frameCount + _goodFramesOffset, _goodFramesCount);
}

nva2x::DeviceTensorFloatConstView InferenceOutputBuffers::GetInferenceResultSkin(std::size_t frameIndex, std::size_t index) const {
  A2F_CHECK_ERROR_WITH_MSG(
    frameIndex < _goodFramesCount,
    "Frame index " << frameIndex << " must be smaller than good frame count " << _goodFramesCount,
    {});
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _buffers.GetInferenceResultSkin(index * _frameCount + _goodFramesOffset + frameIndex);
}

nva2x::DeviceTensorFloatConstView InferenceOutputBuffers::GetInferenceResultTongue(std::size_t frameIndex, std::size_t index) const {
  A2F_CHECK_ERROR_WITH_MSG(
    frameIndex < _goodFramesCount,
    "Frame index " << frameIndex << " must be smaller than good frame count " << _goodFramesCount,
    {});
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _buffers.GetInferenceResultTongue(index * _frameCount + _goodFramesOffset + frameIndex);
}

nva2x::DeviceTensorFloatConstView InferenceOutputBuffers::GetInferenceResultJaw(std::size_t frameIndex, std::size_t index) const {
  A2F_CHECK_ERROR_WITH_MSG(
    frameIndex < _goodFramesCount,
    "Frame index " << frameIndex << " must be smaller than good frame count " << _goodFramesCount,
    {});
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _buffers.GetInferenceResultJaw(index * _frameCount + _goodFramesOffset + frameIndex);
}

nva2x::DeviceTensorFloatConstView InferenceOutputBuffers::GetInferenceResultEyes(std::size_t frameIndex, std::size_t index) const {
  A2F_CHECK_ERROR_WITH_MSG(
    frameIndex < _goodFramesCount,
    "Frame index " << frameIndex << " must be smaller than good frame count " << _goodFramesCount,
    {});
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _buffers.GetInferenceResultEyes(index * _frameCount + _goodFramesOffset + frameIndex);
}

nva2x::TensorBatchInfo InferenceOutputBuffers::GetSkinBatchInfo(std::size_t frameIndex) const {
  A2F_CHECK_ERROR_WITH_MSG(
    frameIndex < _goodFramesCount,
    "Frame index " << frameIndex << " must be smaller than good frame count " << _goodFramesCount,
    {});
  const auto skin = GetInferenceResultSkin(frameIndex, 0);
  const auto offset = skin.Data() - GetResultTensor().Data();
  assert(offset >= 0);
  const auto size = skin.Size();
  const auto stride = _buffers.GetResultSize() * _frameCount;
  return {static_cast<std::size_t>(offset), size, stride};
}

nva2x::TensorBatchInfo InferenceOutputBuffers::GetTongueBatchInfo(std::size_t frameIndex) const {
  A2F_CHECK_ERROR_WITH_MSG(
    frameIndex < _goodFramesCount,
    "Frame index " << frameIndex << " must be smaller than good frame count " << _goodFramesCount,
    {});
  const auto tongue = GetInferenceResultTongue(frameIndex, 0);
  const auto offset = tongue.Data() - GetResultTensor().Data();
  assert(offset >= 0);
  const auto size = tongue.Size();
  const auto stride = _buffers.GetResultSize() * _frameCount;
  return {static_cast<std::size_t>(offset), size, stride};
}

nva2x::TensorBatchInfo InferenceOutputBuffers::GetJawBatchInfo(std::size_t frameIndex) const {
  A2F_CHECK_ERROR_WITH_MSG(
    frameIndex < _goodFramesCount,
    "Frame index " << frameIndex << " must be smaller than good frame count " << _goodFramesCount,
    {});
  const auto jaw = GetInferenceResultJaw(frameIndex, 0);
  const auto offset = jaw.Data() - GetResultTensor().Data();
  assert(offset >= 0);
  const auto size = jaw.Size();
  const auto stride = _buffers.GetResultSize() * _frameCount;
  return {static_cast<std::size_t>(offset), size, stride};
}

nva2x::TensorBatchInfo InferenceOutputBuffers::GetEyesBatchInfo(std::size_t frameIndex) const {
  A2F_CHECK_ERROR_WITH_MSG(
    frameIndex < _goodFramesCount,
    "Frame index " << frameIndex << " must be smaller than good frame count " << _goodFramesCount,
    {});
  const auto eyes = GetInferenceResultEyes(frameIndex, 0);
  const auto offset = eyes.Data() - GetResultTensor().Data();
  assert(offset >= 0);
  const auto size = eyes.Size();
  const auto stride = _buffers.GetResultSize() * _frameCount;
  return {static_cast<std::size_t>(offset), size, stride};
}

void InferenceOutputBuffers::Destroy() {
  delete this;
}

std::error_code ResultBuffers::Init(const NetworkInfo& networkInfo, std::size_t count) {
  return GeometryResultBuffers::Init(
    networkInfo.skinDim, networkInfo.tongueDim,
    count
  );
}

} // namespace IDiffusionModel

IDiffusionModel::InferenceInputBuffers* IDiffusionModel::CreateInferenceInputBuffers(
  const NetworkInfo& networkInfo, std::size_t count
  ) {
  LOG_DEBUG("IDiffusionModel::CreateInferenceInputBuffers()");
  auto buffers = std::make_unique<InferenceInputBuffers>();
  if (buffers->Init(networkInfo, count)) {
    LOG_ERROR("Unable to allocate regression model inference input buffers");
    return nullptr;
  }
  return buffers.release();
}

IDiffusionModel::InferenceStateBuffers* IDiffusionModel::CreateInferenceStateBuffers(
  const NetworkInfo& networkInfo, std::size_t count
  ) {
  LOG_DEBUG("IDiffusionModel::CreateInferenceStateBuffers()");
  auto buffers = std::make_unique<InferenceStateBuffers>();
  if (buffers->Init(networkInfo, count)) {
    LOG_ERROR("Unable to allocate regression model inference state buffers");
    return nullptr;
  }
  return buffers.release();
}

IDiffusionModel::InferenceOutputBuffers* IDiffusionModel::CreateInferenceOutputBuffers(
  const NetworkInfo& networkInfo, std::size_t count
  ) {
  LOG_DEBUG("IDiffusionModel::CreateInferenceOutputBuffers()");
  auto buffers = std::make_unique<InferenceOutputBuffers>();
  if (buffers->Init(networkInfo, count)) {
    LOG_ERROR("Unable to allocate regression model inference output buffers");
    return nullptr;
  }
  return buffers.release();
}

IDiffusionModel::ResultBuffers* IDiffusionModel::CreateResultBuffers(
  const NetworkInfo& networkInfo, std::size_t count
  ) {
  LOG_DEBUG("IDiffusionModel::CreateResultBuffers()");
  auto buffers = std::make_unique<ResultBuffers>();
  if (buffers->Init(networkInfo, count)) {
    LOG_ERROR("Unable to allocate regression model result buffers");
    return nullptr;
  }
  return buffers.release();
}


const nva2x::BufferBindingsDescription& IDiffusionModel::GetBindingsDescription() {
  using IOType = nva2x::IBufferBindingsDescription::IOType;
  using DimensionType = nva2x::IBufferBindingsDescription::DimensionType;
  static constexpr std::array<nva2x::BindingDescription, 7> kDescriptions = {{
    {"emotion", IOType::INPUT, {{DimensionType::BATCH, DimensionType::FIXED, DimensionType::FIXED}}},
    {"identity", IOType::INPUT, {{DimensionType::BATCH, DimensionType::FIXED}}},
    {"input_latents", IOType::INPUT, {{DimensionType::FIXED, DimensionType::FIXED, DimensionType::BATCH, DimensionType::FIXED}}},
    {"noise", IOType::INPUT, {{DimensionType::BATCH, DimensionType::FIXED, DimensionType::FIXED, DimensionType::FIXED}}},
    {"window", IOType::INPUT, {{DimensionType::BATCH, DimensionType::FIXED}}},
    {"output_latents", IOType::OUTPUT, {{DimensionType::FIXED, DimensionType::FIXED, DimensionType::BATCH, DimensionType::FIXED}}},
    {"prediction", IOType::OUTPUT, {{DimensionType::BATCH, DimensionType::FIXED, DimensionType::FIXED}}},
  }};
  // Validate everything is as expected at compile-time.
  static_assert(nva2x::IsSorted(kDescriptions.data(), kDescriptions.size()));
  static_assert(0 == nva2x::CompareCStr("emotion", kDescriptions[kEmotionTensorIndex].name));
  static_assert(0 == nva2x::CompareCStr("identity", kDescriptions[kIdentityTensorIndex].name));
  static_assert(0 == nva2x::CompareCStr("input_latents", kDescriptions[kInputLatentsTensorIndex].name));
  static_assert(0 == nva2x::CompareCStr("noise", kDescriptions[kNoiseTensorIndex].name));
  static_assert(0 == nva2x::CompareCStr("window", kDescriptions[kWindowTensorIndex].name));
  static_assert(0 == nva2x::CompareCStr("output_latents", kDescriptions[kOutputLatentsTensorIndex].name));
  static_assert(0 == nva2x::CompareCStr("prediction", kDescriptions[kPredictionTensorIndex].name));
  static_assert(5 == nva2x::GetInputCount(kDescriptions.data(), kDescriptions.size()));
  static_assert(1 == nva2x::GetBatchIndexCount(kDescriptions[kEmotionTensorIndex]));
  static_assert(1 == nva2x::GetBatchIndexCount(kDescriptions[kIdentityTensorIndex]));
  static_assert(1 == nva2x::GetBatchIndexCount(kDescriptions[kInputLatentsTensorIndex]));
  static_assert(1 == nva2x::GetBatchIndexCount(kDescriptions[kNoiseTensorIndex]));
  static_assert(1 == nva2x::GetBatchIndexCount(kDescriptions[kWindowTensorIndex]));
  static_assert(1 == nva2x::GetBatchIndexCount(kDescriptions[kOutputLatentsTensorIndex]));
  static_assert(1 == nva2x::GetBatchIndexCount(kDescriptions[kPredictionTensorIndex]));

  static const nva2x::BufferBindingsDescription descriptions({kDescriptions.begin(), kDescriptions.end()});
  return descriptions;
}

nva2x::BufferBindings* IDiffusionModel::CreateBindings() {
  return new nva2x::BufferBindings(GetBindingsDescription());
}

IDiffusionModel::IInferenceInputBuffers* CreateInferenceInputBuffersForDiffusionModel_INTERNAL(
  const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count
  ) {
  return IDiffusionModel::CreateInferenceInputBuffers(networkInfo, count);
}

IDiffusionModel::IInferenceStateBuffers* CreateInferenceStateBuffersForDiffusionModel_INTERNAL(
  const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count
  ) {
  return IDiffusionModel::CreateInferenceStateBuffers(networkInfo, count);
}

IDiffusionModel::IInferenceOutputBuffers* CreateInferenceOutputBuffersForDiffusionModel_INTERNAL(
  const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count
  ) {
  return IDiffusionModel::CreateInferenceOutputBuffers(networkInfo, count);
}

IDiffusionModel::IResultBuffers* CreateResultBuffersForDiffusionModel_INTERNAL(
  const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count
  ) {
  return IDiffusionModel::CreateResultBuffers(networkInfo, count);
}

const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForDiffusionModel_INTERNAL() {
  return IDiffusionModel::GetBindingsDescription();
}

nva2x::IBufferBindings* CreateBindingsForDiffusionModel_INTERNAL() {
  return IDiffusionModel::CreateBindings();
}

} // namespace nva2f
