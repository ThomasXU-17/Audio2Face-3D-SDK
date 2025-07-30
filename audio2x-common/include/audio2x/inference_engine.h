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

#include <cstddef>
#include <cstdint>
#include <system_error>

namespace nva2x {

// Interface for describing the structure and properties of tensor bindings for inference.
// This class provides metadata about input/output tensors including their names, types,
// dimensions, and whether dimensions are fixed, batch-based, or dynamic.
class IBufferBindingsDescription {
public:
  // Enumeration defining whether a tensor is an input or output.
  enum class IOType : std::uint32_t { UNKNOWN, INPUT, OUTPUT };

  // Enumeration defining the type of a tensor dimension.
  // FIXED: Dimension has a constant size
  // BATCH: Dimension size matches batch size
  // DYNAMIC: Dimension size is runtime-configurable
  enum class DimensionType : std::uint32_t { UNKNOWN, FIXED, BATCH, DYNAMIC };

  // Get the total number of tensor bindings (inputs + outputs).
  virtual std::size_t Count() const = 0;

  // Get the name of the tensor at the specified index.
  virtual const char* GetName(std::size_t index) const = 0;

  // Get the I/O type (input/output) of the binding at the specified index.
  virtual IOType GetIOType(std::size_t index) const = 0;

  // Get the number of dimensions in the tensor shape at the specified index.
  virtual std::size_t GetNbDimensions(std::size_t index) const = 0;

  // Get the dimension type (fixed/batch/dynamic) at the given tensor and dimension indices.
  // The index is the index of the tensor in the bindings description.
  // The dimension index is the index of the dimension in the tensor shape.
  virtual DimensionType GetDimensionType(std::size_t index, std::size_t dimensionIndex) const = 0;

protected:
  virtual ~IBufferBindingsDescription();
};

// Interface for managing tensor buffer bindings for inference execution.
// This class handles the actual GPU memory buffers for input/output tensors
// and provides methods to set/retrieve tensor data and dynamic dimensions.
class IBufferBindings {
public:
  // Get the description object that defines the structure of these bindings.
  virtual const IBufferBindingsDescription& GetDescription() const = 0;

  // Get a const view to the input tensor buffer at the specified index.
  virtual DeviceTensorVoidConstView GetInputBinding(std::size_t index) const = 0;

  // Get a view to the output tensor buffer at the specified index.
  virtual DeviceTensorVoidView GetOutputBinding(std::size_t index) const = 0;

  // Set the input tensor buffer at the specified index.
  virtual std::error_code SetInputBinding(std::size_t index, DeviceTensorVoidConstView buffer) = 0;

  // Set the output tensor buffer at the specified index.
  virtual std::error_code SetOutputBinding(std::size_t index, DeviceTensorVoidView buffer) = 0;

  // Set the size of a dynamic dimension for an input tensor.
  // The index is the index of the tensor in the bindings description.
  // The dimension index is the index of the dimension in the tensor shape.
  virtual std::error_code SetDynamicDimension(std::size_t index, std::size_t dimensionIndex, std::size_t dimensionSize) = 0;

  // Get the current size of a dynamic dimension for an input tensor.
  // The index is the index of the tensor in the bindings description.
  // The dimension index is the index of the dimension in the tensor shape.
  // Return pointer to the dimension size, or nullptr if not dynamic or not set.
  virtual const std::size_t* GetDynamicDimension(std::size_t index, std::size_t dimensionIndex) const = 0;

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IBufferBindings();
};

// Interface for TensorRT-based inference engine.
// This class manages the TensorRT runtime, engine, and execution context
// and provides methods to initialize, configure, and run inference.
class IInferenceEngine {
public:
  // Initialize the inference engine with TensorRT network data.
  virtual std::error_code Init(const void *networkData, size_t networkDataSize) = 0;

  // Get the maximum supported batch size for the given bindings description
  // This is determined by the TensorRT engine's optimization profile.
  // Returns -1 if unable to determine or on error.
  virtual std::int64_t GetMaxBatchSize(const IBufferBindingsDescription& bindingsDescription) const = 0;

  // Validate that the provided bindings description matches the engine's requirements.
  // It checks tensor names, types, dimensions, and dimension types.
  virtual std::error_code CheckBindings(const IBufferBindingsDescription& bindingsDescription) const = 0;

  // Binds buffers to the engine for execution.
  virtual std::error_code BindBuffers(const IBufferBindings& bindings, int batchSize) const = 0;

  // Execute inference asynchronously on the GPU.
  virtual std::error_code Run(cudaStream_t cudaStream) const = 0; // GPU Async

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IInferenceEngine();
};

// Create a new inference engine instance.
AUDIO2X_SDK_EXPORT IInferenceEngine* CreateInferenceEngine();

} // namespace nva2x
