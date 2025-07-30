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

#include "audio2x/inference_engine.h"
#include "audio2x/internal/integer_array.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100 5204)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <NvInfer.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <cstring>
#include <unordered_map>
#include <vector>

namespace nva2x {

struct BindingDescription {
  const char* name{nullptr};
  IBufferBindingsDescription::IOType type{IBufferBindingsDescription::IOType::UNKNOWN};

  using Dimensions = IntegerArray<IBufferBindingsDescription::DimensionType, std::uint32_t, 2>;
  Dimensions dimensions;
};

class BufferBindingsDescription : public IBufferBindingsDescription {
public:
  BufferBindingsDescription(std::vector<BindingDescription> descriptions);
  ~BufferBindingsDescription() override;

  std::size_t Count() const override;
  const char* GetName(std::size_t index) const override;
  IOType GetIOType(std::size_t index) const override;
  std::size_t GetNbDimensions(std::size_t index) const override;
  DimensionType GetDimensionType(std::size_t index, std::size_t dimensionIndex) const override;

  inline std::size_t InputCount() const { return _inputCount; }
  inline std::size_t OutputCount() const { return _descriptions.size() - _inputCount; }

private:
  std::vector<BindingDescription> _descriptions;
  std::size_t _inputCount;
};

// C++20 offers better ways to handle this, with operator<=> and constexpr,
// but for now redo a couple of things ourselves.
#if __cplusplus >= 202002L
  #if defined(_MSC_VER)
    #pragma message("Now that C++20 is available, the code below could be made much simpler")
  #elif defined(__GNUC__)
    #warning "Now that C++20 is available, the code below could be made much simpler"
  #endif
#endif

// In C++20, std::strcmp is constexpr
constexpr int CompareCStr(const char* left, const char* right) {
  while (*left && (*left == *right)) {
    ++left;
    ++right;
  }
  return static_cast<int>(*left) - static_cast<int>(*right);
}

constexpr int CompareBindingDescription3Way(
  const BindingDescription& left, const BindingDescription& right
) {
  // The descriptions are sorted with the inputs first then output, and sorted
  // by name with each category.
  if (left.type != right.type) {
    return static_cast<int>(left.type) - static_cast<int>(right.type);
  }

  return CompareCStr(left.name, right.name);
}

constexpr bool CompareBindingDescription(
  const BindingDescription& left, const BindingDescription& right
) {
  return CompareBindingDescription3Way(left, right) < 0;
}

constexpr bool IsSorted(const BindingDescription* descriptions, std::size_t count) {
  for (std::size_t i = 1; i < count; ++i) {
    if (CompareBindingDescription3Way(descriptions[i - 1], descriptions[i]) > 0) {
      return false;
    }
  }
  return true;
}

constexpr std::int64_t GetInputCount(const BindingDescription* descriptions, std::size_t count) {
  constexpr auto get_first_non_input_index = [](
    const BindingDescription* descriptions, std::size_t count
    ) -> std::size_t {
      for (std::size_t i = 0; i < count; ++i) {
        if (descriptions[i].type != IBufferBindingsDescription::IOType::INPUT) {
          return i;
        }
      }
      return count;
    };
  const std::size_t split_index = get_first_non_input_index(descriptions, count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto expected = i < split_index
       ? IBufferBindingsDescription::IOType::INPUT
       : IBufferBindingsDescription::IOType::OUTPUT;
    if (descriptions[i].type != expected) {
      return -1;
    }
  }
  return static_cast<int>(split_index);
}

constexpr std::size_t GetBatchIndexCount(const BindingDescription& description) {
  std::size_t batch_index_count = 0;
  for (std::size_t i = 0; i < description.dimensions.Size(); ++i) {
    if (description.dimensions.Get(i) == IBufferBindingsDescription::DimensionType::BATCH) {
      ++batch_index_count;
    }
  }
  return batch_index_count;
}


class BufferBindings : public IBufferBindings {
public:
  BufferBindings(const BufferBindingsDescription& description);
  ~BufferBindings() override;

  const BufferBindingsDescription& GetDescription() const override;

  DeviceTensorVoidConstView GetInputBinding(std::size_t index) const override;
  DeviceTensorVoidView GetOutputBinding(std::size_t index) const override;

  std::error_code SetInputBinding(std::size_t index, DeviceTensorVoidConstView buffer) override;
  std::error_code SetOutputBinding(std::size_t index, DeviceTensorVoidView buffer) override;

  std::error_code SetDynamicDimension(std::size_t index, std::size_t dimensionIndex, std::size_t dimensionSize) override;
  const std::size_t* GetDynamicDimension(std::size_t index, std::size_t dimensionIndex) const override;

  void Destroy() override;


private:
  const BufferBindingsDescription& _description;
  // Eventually, these will be tensor views.
  std::vector<DeviceTensorVoidConstView> _inputBindings;
  std::vector<DeviceTensorVoidView> _outputBindings;

  std::vector<std::unordered_map<std::size_t, std::size_t>> _dynamicDimensions;
};


class InferenceEngine : public IInferenceEngine {
public:
  InferenceEngine();
  ~InferenceEngine() override;
  std::error_code Init(const void *networkData, size_t networkDataSize) override;
  std::int64_t GetMaxBatchSize(const IBufferBindingsDescription& bindingsDescription) const override;
  std::error_code CheckBindings(const IBufferBindingsDescription& bindingsDescription) const override;
  std::error_code BindBuffers(const IBufferBindings& bindings, int batchSize) const override;
  std::error_code Run(cudaStream_t cudaStream) const override;  // GPU Async
  void Destroy() override;
  std::error_code Deallocate();

private:
  nvinfer1::IRuntime *_runtime;
  nvinfer1::ICudaEngine *_engine;
  nvinfer1::IExecutionContext *_context;
  bool _initialized;
};

} // namespace nva2x
