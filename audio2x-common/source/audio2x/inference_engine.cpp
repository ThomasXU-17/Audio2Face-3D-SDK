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
#include "audio2x/internal/inference_engine.h"
#include "audio2x/internal/audio2x.h"
#include "audio2x/internal/logger_trt.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/error.h"
#include "audio2x/cuda_utils.h"

#include <NvInferPlugin.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>

namespace nva2x {

IBufferBindingsDescription::~IBufferBindingsDescription() = default;
IBufferBindings::~IBufferBindings() = default;
IInferenceEngine::~IInferenceEngine() = default;

BufferBindingsDescription::BufferBindingsDescription(
  std::vector<BindingDescription> descriptions
  ) : _descriptions{std::move(descriptions)} {
  A2X_LOG_DEBUG("BufferBindingsDescription::BufferBindingsDescription()");
  assert(IsSorted(_descriptions.data(), _descriptions.size()));
  const auto inputCount = GetInputCount(_descriptions.data(), _descriptions.size());
  assert(inputCount >= 0);
  _inputCount = static_cast<std::size_t>(inputCount);
}

BufferBindingsDescription::~BufferBindingsDescription() {
  A2X_LOG_DEBUG("BufferBindingsDescription::~BufferBindingsDescription()");
}

std::size_t BufferBindingsDescription::Count() const {
  return _descriptions.size();
}

const char* BufferBindingsDescription::GetName(std::size_t index) const {
  if (index >= _descriptions.size()) {
    return nullptr;
  }
  return _descriptions[index].name;
}

IBufferBindingsDescription::IOType BufferBindingsDescription::GetIOType(std::size_t index) const {
  if (index >= _descriptions.size()) {
    return IBufferBindingsDescription::IOType::UNKNOWN;
  }
  return _descriptions[index].type;
}

std::size_t BufferBindingsDescription::GetNbDimensions(std::size_t index) const {
  if (index >= _descriptions.size()) {
    return 0;
  }
  return _descriptions[index].dimensions.Size();
}

IBufferBindingsDescription::DimensionType BufferBindingsDescription::GetDimensionType(std::size_t index, std::size_t dimensionIndex) const {
  if (index >= _descriptions.size()) {
    return IBufferBindingsDescription::DimensionType::UNKNOWN;
  }
  if (dimensionIndex >= _descriptions[index].dimensions.Size()) {
    return IBufferBindingsDescription::DimensionType::UNKNOWN;
  }
  return _descriptions[index].dimensions.Get(dimensionIndex);
}

BufferBindings::BufferBindings(const BufferBindingsDescription& description)
  : _description{description}
  , _inputBindings{_description.InputCount()}
  , _outputBindings{_description.OutputCount()}
  , _dynamicDimensions{_description.Count()} {
  A2X_LOG_DEBUG("BufferBindings::BufferBindings()");
}

BufferBindings::~BufferBindings() {
  A2X_LOG_DEBUG("BufferBindings::~BufferBindings()");
}

const BufferBindingsDescription& BufferBindings::GetDescription() const {
  return _description;
}

DeviceTensorVoidConstView BufferBindings::GetInputBinding(std::size_t index) const {
  if (index >= _description.Count()) {
    return {};
  }
  if (_description.GetIOType(index) != IBufferBindingsDescription::IOType::INPUT) {
    return {};
  }
  if (index >= _inputBindings.size()) {
    return {};
  }
  return _inputBindings[index];
}

DeviceTensorVoidView BufferBindings::GetOutputBinding(std::size_t index) const {
  if (index >= _description.Count()) {
    return {};
  }
  if (_description.GetIOType(index) != IBufferBindingsDescription::IOType::OUTPUT) {
    return {};
  }
  if (index < _inputBindings.size()) {
    return {};
  }
  index -= _inputBindings.size();
  return _outputBindings[index];
}

std::error_code BufferBindings::SetInputBinding(std::size_t index, DeviceTensorVoidConstView buffer) {
  if (index >= _description.Count()) {
    return ErrorCode::eOutOfBounds;
  }
  if (_description.GetIOType(index) != IBufferBindingsDescription::IOType::INPUT) {
    return ErrorCode::eMismatch;
  }
  if (index >= _inputBindings.size()) {
    return ErrorCode::eOutOfBounds;
  }
  _inputBindings[index] = buffer;

  return ErrorCode::eSuccess;
}

std::error_code BufferBindings::SetOutputBinding(std::size_t index, DeviceTensorVoidView buffer) {
  if (index >= _description.Count()) {
    return ErrorCode::eOutOfBounds;
  }
  if (_description.GetIOType(index) != IBufferBindingsDescription::IOType::OUTPUT) {
    return ErrorCode::eMismatch;
  }
  if (index < _inputBindings.size()) {
    return ErrorCode::eOutOfBounds;
  }
  index -= _inputBindings.size();
  _outputBindings[index] = buffer;

  return ErrorCode::eSuccess;
}

void BufferBindings::Destroy() {
  A2X_LOG_DEBUG("BufferBindings::Destroy()");
  delete this;
}

std::error_code BufferBindings::SetDynamicDimension(std::size_t index, std::size_t dimensionIndex, std::size_t dimensionSize) {
  A2X_CHECK_ERROR_WITH_MSG(index < _description.Count(), "Index out of bounds", ErrorCode::eOutOfBounds);
  A2X_CHECK_ERROR_WITH_MSG(
    _description.GetIOType(index) == IBufferBindingsDescription::IOType::INPUT,
    "Dynamic dimension can only be set on input bindings",
    ErrorCode::eMismatch
  );
  A2X_CHECK_ERROR_WITH_MSG(
    dimensionIndex < _description.GetNbDimensions(index),
    "Dimension index out of bounds",
    ErrorCode::eOutOfBounds
  );
  A2X_CHECK_ERROR_WITH_MSG(
    _description.GetDimensionType(index, dimensionIndex) == IBufferBindingsDescription::DimensionType::DYNAMIC,
    "Dimension is not dynamic",
    ErrorCode::eMismatch
  );
  _dynamicDimensions[index][dimensionIndex] = dimensionSize;
  return ErrorCode::eSuccess;
}

const std::size_t* BufferBindings::GetDynamicDimension(std::size_t index, std::size_t dimensionIndex) const {
  A2X_CHECK_ERROR_WITH_MSG(index < _description.Count(), "Index out of bound", nullptr);
  A2X_CHECK_ERROR_WITH_MSG(
    _description.GetIOType(index) == IBufferBindingsDescription::IOType::INPUT,
    "Dynamic dimension can only be set on input bindings",
    nullptr
  );
  A2X_CHECK_ERROR_WITH_MSG(
    dimensionIndex < _description.GetNbDimensions(index),
    "Dimension index out of bound",
    nullptr
  );
  A2X_CHECK_ERROR_WITH_MSG(
    _description.GetDimensionType(index, dimensionIndex) == IBufferBindingsDescription::DimensionType::DYNAMIC,
    "Dimension is not dynamic",
    nullptr
  );
  const auto it = _dynamicDimensions[index].find(dimensionIndex);
  if (it == _dynamicDimensions[index].end()) {
    return nullptr;
  }
  return &it->second;
}


InferenceEngine::InferenceEngine()
    : _runtime(nullptr), _engine(nullptr), _context(nullptr), _initialized(false) {
  A2X_LOG_DEBUG("InferenceEngine::InferenceEngine()");
}

InferenceEngine::~InferenceEngine() {
  A2X_LOG_DEBUG("InferenceEngine::~InferenceEngine()");
  Deallocate();
}

std::error_code InferenceEngine::Init(const void *networkData, size_t networkDataSize) {
  _initialized = false;

  A2X_CHECK_RESULT_WITH_MSG(Deallocate(), "Unable to destroy TensorRT Inference Engine");

  A2X_CHECK_ERROR_WITH_MSG(initLibNvInferPlugins(&trtLogger, ""),
             "Unable to initialize Plugins", ErrorCode::eInitNvInferPluginsFailed);

  _runtime = nvinfer1::createInferRuntime(trtLogger);
  A2X_CHECK_ERROR_WITH_MSG(nullptr != _runtime, "Unable to create TensorRT Inference Runtime", ErrorCode::eCreateInferRuntimeFailed)
#ifdef WANT_NVINFER_DISPATCH
  _runtime->setEngineHostCodeAllowed(true);
#endif
  _engine = _runtime->deserializeCudaEngine(networkData, networkDataSize);
  A2X_CHECK_ERROR_WITH_MSG(nullptr != _engine, "Unable to deserialize TensorRT Engine", ErrorCode::eDeserializeCudaEngineFailed);

  _context = _engine->createExecutionContext();
  A2X_CHECK_ERROR_WITH_MSG(nullptr != _context, "Unable to create TensorRT Execution Context", ErrorCode::eCreateExecutionContextFailed);

  _initialized = true;

  return ErrorCode::eSuccess;
}

std::int64_t InferenceEngine::GetMaxBatchSize(const IBufferBindingsDescription& bindingsDescription) const {
  // If this error is hit, multiple optimization profiles could be supported,
  // but this code would need to be adapted.
  A2X_CHECK_ERROR_WITH_MSG(
    _engine->getNbOptimizationProfiles() == 1,
    "Expected a single optimization profile",
    -1
  );

  std::int64_t maxBatchSize = -1;

  const std::size_t nbBindings = bindingsDescription.Count();
  for (std::size_t i = 0; i < nbBindings; ++i) {
    const char* name = bindingsDescription.GetName(i);
    const nvinfer1::Dims dims = _engine->getTensorShape(name);
    const nvinfer1::Dims maxDims = _engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
    if (maxDims.nbDims < 0) {
      // Not an input tensor, skipping.
      assert(bindingsDescription.GetIOType(i) == IBufferBindingsDescription::IOType::OUTPUT);
      continue;
    }
    A2X_CHECK_ERROR_WITH_MSG(
      dims.nbDims == maxDims.nbDims,
      "Mismatch in engine dimensions, " << name << " is " << dims.nbDims << " vs " << maxDims.nbDims,
      -1
    );
    const auto nbDimensions = bindingsDescription.GetNbDimensions(i);
    A2X_CHECK_ERROR_WITH_MSG(
      static_cast<std::size_t>(maxDims.nbDims) == nbDimensions,
      "Mismatch in engine dimensions, " << name << " is " << dims.nbDims << " vs " << nbDimensions,
      -1
    );

    for (std::size_t j = 0; j < nbDimensions; ++j) {
      const auto dimensionType = bindingsDescription.GetDimensionType(i, j);
      if (dimensionType == IBufferBindingsDescription::DimensionType::BATCH) {
        const auto maxDim = maxDims.d[j];
        if (maxBatchSize == -1 || maxBatchSize > maxDim) {
          maxBatchSize = maxDim;
        }
      }
    }
  }

  return maxBatchSize == -1 ? 1 : maxBatchSize;
}

std::error_code InferenceEngine::CheckBindings(const IBufferBindingsDescription& bindingsDescription) const {
  // Build a description
  std::vector<BindingDescription> sourceDescription;
  const std::int32_t nbBindings = _engine->getNbIOTensors();
  A2X_CHECK_ERROR_WITH_MSG(
    nbBindings >= 0,
    "Wrong number of engine I/O bindings",
    ErrorCode::eMismatchEngineIOBindings
    );
  sourceDescription.reserve(nbBindings);
  for (std::int32_t i = 0; i < nbBindings; ++i) {
    const char* name = _engine->getIOTensorName(i);

    const auto mode = _engine->getTensorIOMode(name);
    A2X_CHECK_ERROR_WITH_MSG(
      mode == nvinfer1::TensorIOMode::kINPUT || mode == nvinfer1::TensorIOMode::kOUTPUT,
      "Wrong mode for engine I/O binding, " << name << " has mode " << static_cast<int>(mode),
      ErrorCode::eMismatchEngineIOBindings
    );
    const auto bindingMode = (
      mode == nvinfer1::TensorIOMode::kINPUT
      ?
      IBufferBindingsDescription::IOType::INPUT
      :
      IBufferBindingsDescription::IOType::OUTPUT
    );

    // We store the size in the batch index for comparison.
    const nvinfer1::Dims dims = _engine->getTensorShape(name);
    A2X_CHECK_ERROR_WITH_MSG(
      dims.nbDims >= 0,
      "Wrong dims for engine I/O binding, " << name << " is " << dims.nbDims,
      ErrorCode::eMismatchEngineIOBindings
    );
    BindingDescription::Dimensions dimensions;
    for (std::int32_t j = 0; j < dims.nbDims; ++j) {
      const auto dimensionType = dims.d[j] < 0
        ? IBufferBindingsDescription::DimensionType::DYNAMIC
        : IBufferBindingsDescription::DimensionType::FIXED;
      dimensions.Add(dimensionType);
    }

    sourceDescription.emplace_back(BindingDescription{name, bindingMode, dimensions});
  }
  std::sort(sourceDescription.begin(), sourceDescription.end(), CompareBindingDescription);

  // Check the description.
  A2X_CHECK_ERROR_WITH_MSG(
    sourceDescription.size() == bindingsDescription.Count(),
    "Mismatch in engine I/O bindings",
    ErrorCode::eMismatchEngineIOBindings
    );
  for (std::size_t i = 0; i < sourceDescription.size(); ++i) {
    const char* name = sourceDescription[i].name;
    const char* bindingName = bindingsDescription.GetName(i);
    A2X_CHECK_ERROR_WITH_MSG(
      0 == std::strcmp(name, bindingName),
      "Mismatch in engine I/O bindings name, index " << i << " is " << name << " vs " << bindingName,
      ErrorCode::eMismatchEngineIOBindings
    );

    const auto mode = sourceDescription[i].type;
    const auto bindingMode = bindingsDescription.GetIOType(i);
    A2X_CHECK_ERROR_WITH_MSG(
      mode == bindingMode,
      "Mismatch in engine I/O bindings mode, " << name << " is " << static_cast<int>(mode) << " vs " << static_cast<int>(bindingMode),
      ErrorCode::eMismatchEngineIOBindings
    );

    const auto dimsSize = sourceDescription[i].dimensions.Size();
    const auto bindingDimsSize = bindingsDescription.GetNbDimensions(i);
    A2X_CHECK_ERROR_WITH_MSG(
      dimsSize == bindingDimsSize,
      "Mismatch in engine I/O bindings dimensions size, " << name << " has " << dimsSize << " vs " << bindingDimsSize,
      ErrorCode::eMismatchEngineIOBindings
    );

    for (std::size_t j = 0; j < dimsSize; ++j) {
      const auto dimsType = sourceDescription[i].dimensions.Get(j);
      const auto bindingDimsType = bindingsDescription.GetDimensionType(i, j);
      A2X_CHECK_ERROR_WITH_MSG(
        // If it matches, we are fine.
        (dimsType == bindingDimsType) ||
        // If the binding is dynamic, we accept batch:
        // - Batch is a form of dynamic dimension.
        // - Dynamic would have been the match above.
        (dimsType == IBufferBindingsDescription::DimensionType::DYNAMIC &&
         bindingDimsType == IBufferBindingsDescription::DimensionType::BATCH) ||
        // If the binding is fixed, we accept dynamic and batch:
        // - As long as the dynamic size is the same, we are fine.
        // - As long as the batch size is 1, we are fine.
        (dimsType == IBufferBindingsDescription::DimensionType::FIXED &&
         (bindingDimsType == IBufferBindingsDescription::DimensionType::DYNAMIC ||
          bindingDimsType == IBufferBindingsDescription::DimensionType::BATCH)),
        "Mismatch in engine I/O bindings dimensions type, " << name << " for dimension " << j
            << " has " << static_cast<int>(dimsType) << " vs " << static_cast<int>(bindingDimsType),
        ErrorCode::eMismatchEngineIOBindings
      );
    }
  }

  return ErrorCode::eSuccess;
}

std::error_code InferenceEngine::BindBuffers(const IBufferBindings& bindings, int batchSize) const {
  const auto& description = bindings.GetDescription();
  for (std::size_t i = 0, count = description.Count(); i < count; ++i) {
    const auto descriptionType = description.GetIOType(i);
    const auto descriptionName = description.GetName(i);
    if (descriptionType == IBufferBindingsDescription::IOType::INPUT) {
      const void* buffer = bindings.GetInputBinding(i).Data();
      A2X_CHECK_ERROR_WITH_MSG(
        buffer,
        "Mismatch in engine I/O bindings for " << descriptionName,
        ErrorCode::eMismatchEngineIOBindings
      );
      A2X_CHECK_ERROR_WITH_MSG(
        _context->setInputTensorAddress(descriptionName, buffer),
        "Unable to set input tensor address for " << descriptionName,
        ErrorCode::eSetTensorAddress
      );

      nvinfer1::Dims dims = _context->getTensorShape(descriptionName);
      const auto descriptionDimsSize = description.GetNbDimensions(i);
      A2X_CHECK_ERROR_WITH_MSG(
        dims.nbDims == static_cast<std::int64_t>(descriptionDimsSize),
        "Mismatch in engine I/O bindings dimensions size, " << descriptionName << " has " << dims.nbDims << " vs " << descriptionDimsSize,
        ErrorCode::eMismatchEngineIOBindings
      );
      for (std::size_t j = 0; j < descriptionDimsSize; ++j) {
        const auto dimsType = description.GetDimensionType(i, j);
        if (dimsType == IBufferBindingsDescription::DimensionType::BATCH) {
            dims.d[j] = batchSize;
        }
        else if (dimsType == IBufferBindingsDescription::DimensionType::DYNAMIC) {
          const auto dynamicDimension = bindings.GetDynamicDimension(i, j);
          A2X_CHECK_ERROR_WITH_MSG(
            dynamicDimension != nullptr,
            "Dynamic dimension not set for " << descriptionName << ", dimension " << j,
            ErrorCode::eInvalidValue
          );
          dims.d[j] = *dynamicDimension;
        }
      }
      A2X_CHECK_ERROR_WITH_MSG(
        _context->setInputShape(descriptionName, dims),
        "Unable to set input shape for " << descriptionName,
        ErrorCode::eSetInputShape
      );
    }
    else if (descriptionType == IBufferBindingsDescription::IOType::OUTPUT) {
      void* buffer = bindings.GetOutputBinding(i).Data();
      A2X_CHECK_ERROR_WITH_MSG(
        buffer,
        "Mismatch in engine I/O bindings for " << descriptionName,
        ErrorCode::eMismatchEngineIOBindings
      );
      A2X_CHECK_ERROR_WITH_MSG(
        _context->setTensorAddress(descriptionName, buffer),
        "Unable to set tensor address for " << descriptionName,
        ErrorCode::eSetTensorAddress
      );
    }
    else {
      return ErrorCode::eInvalidValue;
    }
  }

  A2X_CHECK_ERROR_WITH_MSG(_context->inferShapes(0, nullptr) == 0,
                           "Not all inputs dimensions are specified", ErrorCode::eNotAllInputDimensionsSpecified);

  // Validate the shapes size
  for (std::size_t i = 0, count = description.Count(); i < count; ++i) {
    const auto descriptionType = description.GetIOType(i);
    const auto descriptionName = description.GetName(i);
    std::size_t bufferSize = 0;
    if (descriptionType == IBufferBindingsDescription::IOType::INPUT) {
      bufferSize = bindings.GetInputBinding(i).Size();
    }
    else if (descriptionType == IBufferBindingsDescription::IOType::OUTPUT) {
      bufferSize = bindings.GetOutputBinding(i).Size();
    }
    else {
      return ErrorCode::eInvalidValue;
    }

    nvinfer1::Dims dims = _context->getTensorShape(descriptionName);
    std::size_t bindingSize = 1;
    for (std::int32_t j = 0; j < dims.nbDims; ++j) {
      if (dims.d[j] < 0) {
        return ErrorCode::eInvalidValue;
      }
      bindingSize *= dims.d[j];
    }
    A2X_CHECK_ERROR_WITH_MSG(
      bindingSize == bufferSize,
      "Mismatch in engine I/O bindings for " << descriptionName << ", expected size " << bindingSize << " but got " << bufferSize,
      ErrorCode::eMismatchEngineIOBindings
    );
  }

  return ErrorCode::eSuccess;
}

std::error_code InferenceEngine::Run(cudaStream_t cudaStream) const {
  A2X_CHECK_ERROR_WITH_MSG(_context->enqueueV3(cudaStream),
                           "Unable to enqueue TensorRT inference", ErrorCode::eEnqueueFailed);

  return ErrorCode::eSuccess;
}

void InferenceEngine::Destroy() {
  A2X_LOG_DEBUG("InferenceEngine::Destroy()");
  delete this;
}

std::error_code InferenceEngine::Deallocate() {
  if (_context != nullptr) {
    A2X_LOG_DEBUG("Destroying TensorRT context");
    delete _context;
    _context = nullptr;
  }
  if (_engine != nullptr) {
    A2X_LOG_DEBUG("Destroying TensorRT engine");
    delete _engine;
    _engine = nullptr;
  }
  if (_runtime != nullptr) {
    A2X_LOG_DEBUG("Destroying TensorRT runtime");
    delete _runtime;
    _runtime = nullptr;
  }
  return ErrorCode::eSuccess;
}


} // namespace nva2x


nva2x::IInferenceEngine* nva2x::internal::CreateInferenceEngine() {
  A2X_LOG_DEBUG("CreateInferenceEngine()");
  return new InferenceEngine();
}
