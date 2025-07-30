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
#include "audio2face/internal/model_shared.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"

#include <array>
#include <cassert>
#include <memory>

namespace nva2f {

namespace ISharedModel {

IGeometryInferenceOutputBuffers::~IGeometryInferenceOutputBuffers() = default;
IGeometryResultBuffers::~IGeometryResultBuffers() = default;

std::error_code GeometryInferenceOutputBuffers::Init(
  std::size_t skinSize, std::size_t tongueSize, std::size_t jawSize, std::size_t eyesSize, std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count > 0,
    "Count must be greater than 1",
    nva2x::ErrorCode::eInvalidValue
  );

  const std::size_t resultSize = skinSize + tongueSize + jawSize + eyesSize;

  A2F_CHECK_RESULT_WITH_MSG(_inferenceResult.Allocate(resultSize * count), "Unable to allocate inference result buffer");

  _count = count;
  _inferenceResultSize = resultSize;
  _skinOffset = 0;
  _skinSize = skinSize;
  _tongueOffset = _skinOffset + _skinSize;
  _tongueSize = tongueSize;
  _jawOffset = _tongueOffset + _tongueSize;
  _jawSize = jawSize;
  _eyesOffset = _jawOffset + _jawSize;
  _eyesSize = eyesSize;

  return nva2x::ErrorCode::eSuccess;
}

nva2x::DeviceTensorFloatView GeometryInferenceOutputBuffers::GetInferenceResult(std::size_t index, std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count - index,
    "Count " << count << " is too large",
    {});
  return _inferenceResult.View(index * _inferenceResultSize, count * _inferenceResultSize);
}

nva2x::DeviceTensorFloatConstView GeometryInferenceOutputBuffers::GetInferenceResult(std::size_t index, std::size_t count) const {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count - index,
    "Count " << count << " is too large",
    {});
  return _inferenceResult.View(index * _inferenceResultSize, count * _inferenceResultSize);
}


std::size_t GeometryInferenceOutputBuffers::GetCount() const {
  return _count;
}

std::size_t GeometryInferenceOutputBuffers::GetResultSize() const {
  return _inferenceResultSize;
}

nva2x::DeviceTensorFloatView GeometryInferenceOutputBuffers::GetResultTensor(std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _inferenceResult.View(0, count * _inferenceResultSize);
}

nva2x::DeviceTensorFloatConstView GeometryInferenceOutputBuffers::GetResultTensor(std::size_t count) const {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _inferenceResult.View(0, count * _inferenceResultSize);
}

nva2x::DeviceTensorFloatConstView GeometryInferenceOutputBuffers::GetInferenceResultSkin(std::size_t index) const {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _inferenceResult.View(index * _inferenceResultSize + _skinOffset, _skinSize);
}

nva2x::DeviceTensorFloatConstView GeometryInferenceOutputBuffers::GetInferenceResultTongue(std::size_t index) const {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _inferenceResult.View(index * _inferenceResultSize + _tongueOffset, _tongueSize);
}

nva2x::DeviceTensorFloatConstView GeometryInferenceOutputBuffers::GetInferenceResultJaw(std::size_t index) const {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _inferenceResult.View(index * _inferenceResultSize + _jawOffset, _jawSize);
}

nva2x::DeviceTensorFloatConstView GeometryInferenceOutputBuffers::GetInferenceResultEyes(std::size_t index) const  {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _inferenceResult.View(index * _inferenceResultSize + _eyesOffset, _eyesSize);
}

nva2x::TensorBatchInfo GeometryInferenceOutputBuffers::GetSkinBatchInfo() const {
  return {_skinOffset, _skinSize, _inferenceResultSize};
}

nva2x::TensorBatchInfo GeometryInferenceOutputBuffers::GetTongueBatchInfo() const {
  return {_tongueOffset, _tongueSize, _inferenceResultSize};
}

nva2x::TensorBatchInfo GeometryInferenceOutputBuffers::GetJawBatchInfo() const {
  return {_jawOffset, _jawSize, _inferenceResultSize};
}

nva2x::TensorBatchInfo GeometryInferenceOutputBuffers::GetEyesBatchInfo() const {
  return {_eyesOffset, _eyesSize, _inferenceResultSize};
}

void GeometryInferenceOutputBuffers::Destroy() {
  delete this;
}


std::error_code GeometryResultBuffers::Init(
  std::size_t skinSize, std::size_t tongueSize, std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count > 0,
    "Count must be greater than 1",
    nva2x::ErrorCode::eInvalidValue
  );

  const std::size_t jawSize = 16;
  const std::size_t eyesSize = 6;
  const std::size_t resultSize = skinSize + tongueSize + jawSize + eyesSize;

  A2F_CHECK_RESULT_WITH_MSG(_result.Allocate(resultSize * count), "Unable to allocate result buffer");

  _count = count;
  _resultSize = resultSize;
  _skinGeometryOffset = 0;
  _skinGeometrySize = skinSize;
  _tongueGeometryOffset = _skinGeometryOffset + _skinGeometrySize;
  _tongueGeometrySize = tongueSize;
  _jawTransformOffset = _tongueGeometryOffset + _tongueGeometrySize;
  _jawTransformSize = jawSize;
  _eyesRotationOffset = _jawTransformOffset + _jawTransformSize;
  _eyesRotationSize = eyesSize;

  return nva2x::ErrorCode::eSuccess;
}

std::size_t GeometryResultBuffers::GetCount() const {
  return _count;
}

std::size_t GeometryResultBuffers::GetResultSize() const {
  return _resultSize;
}

nva2x::DeviceTensorFloatView GeometryResultBuffers::GetResultTensor(std::size_t count) {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _result.View(0, count * _resultSize);
}

nva2x::DeviceTensorFloatConstView GeometryResultBuffers::GetResultTensor(std::size_t count) const {
  A2F_CHECK_ERROR_WITH_MSG(
    count <= _count,
    "Count " << count << " must be smaller or equal than count " << _count,
    {});
  if (count == 0) {
    count = _count;
  }
  return _result.View(0, count * _resultSize);
}

nva2x::DeviceTensorFloatView GeometryResultBuffers::GetResultSkinGeometry(std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _result.View(index * _resultSize + _skinGeometryOffset, _skinGeometrySize);
}

nva2x::DeviceTensorFloatConstView GeometryResultBuffers::GetResultSkinGeometry(std::size_t index) const {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _result.View(index * _resultSize + _skinGeometryOffset, _skinGeometrySize);
}

nva2x::DeviceTensorFloatView GeometryResultBuffers::GetResultTongueGeometry(std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _result.View(index * _resultSize + _tongueGeometryOffset, _tongueGeometrySize);
}

nva2x::DeviceTensorFloatConstView GeometryResultBuffers::GetResultTongueGeometry(std::size_t index) const {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _result.View(index * _resultSize + _tongueGeometryOffset, _tongueGeometrySize);
}

nva2x::DeviceTensorFloatView GeometryResultBuffers::GetResultJawTransform(std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _result.View(index * _resultSize + _jawTransformOffset, _jawTransformSize);
}

nva2x::DeviceTensorFloatConstView GeometryResultBuffers::GetResultJawTransform(std::size_t index) const {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _result.View(index * _resultSize + _jawTransformOffset, _jawTransformSize);
}

nva2x::DeviceTensorFloatView GeometryResultBuffers::GetResultEyesRotation(std::size_t index) {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _result.View(index * _resultSize + _eyesRotationOffset, _eyesRotationSize);
}

nva2x::DeviceTensorFloatConstView GeometryResultBuffers::GetResultEyesRotation(std::size_t index) const {
  A2F_CHECK_ERROR_WITH_MSG(
    index < _count,
    "Index " << index << " must be smaller than count " << _count,
    {});
  return _result.View(index * _resultSize + _eyesRotationOffset, _eyesRotationSize);
}

nva2x::TensorBatchInfo GeometryResultBuffers::GetSkinBatchInfo() const {
  return {_skinGeometryOffset, _skinGeometrySize, _resultSize};
}

nva2x::TensorBatchInfo GeometryResultBuffers::GetTongueBatchInfo() const {
  return {_tongueGeometryOffset, _tongueGeometrySize, _resultSize};
}

nva2x::TensorBatchInfo GeometryResultBuffers::GetJawBatchInfo() const {
  return {_jawTransformOffset, _jawTransformSize, _resultSize};
}

nva2x::TensorBatchInfo GeometryResultBuffers::GetEyesBatchInfo() const {
  return {_eyesRotationOffset, _eyesRotationSize, _resultSize};
}

void GeometryResultBuffers::Destroy() {
  delete this;
}

} // namespace ISharedModel

} // namespace nva2f
