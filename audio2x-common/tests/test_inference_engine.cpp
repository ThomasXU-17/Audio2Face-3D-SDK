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
#include "audio2x/internal/cuda_stream.h"
#include "audio2x/internal/io.h"
#include "audio2x/internal/tensor_dict.h"
#include "utils.h"

#include <gtest/gtest.h>

#include <thread>
#include <chrono>

namespace {

template <bool Batched>
void TestInferenceEngine() {
  nva2x::HostTensorDict testData;
  ASSERT_TRUE(!testData.ReadFromFile(TEST_DATA_DIR "_data/generated/audio2x-common/tests/data/test_data_inference.bin"));

  constexpr const char* trt_path = Batched ?
      TEST_DATA_DIR "_data/generated/audio2x-common/tests/data/test_data_inference_network_batched.trt"
      :
      TEST_DATA_DIR "_data/generated/audio2x-common/tests/data/test_data_inference_network.trt";

  nva2x::DataBytes networkDataBytes;
  ASSERT_TRUE(!networkDataBytes.ReadFromFile(trt_path));

  auto audioBufferHostPtr = testData.At("input");
  auto emotionHostPtr = testData.At("emotion");
  auto inferenceResultHostTruePtr = testData.At("result");
  ASSERT_NE(audioBufferHostPtr, nullptr);
  ASSERT_NE(emotionHostPtr, nullptr);
  ASSERT_NE(inferenceResultHostTruePtr, nullptr);
  const nva2x::HostTensorFloat &audioBufferHost = *audioBufferHostPtr;
  const nva2x::HostTensorFloat &emotionHost = *emotionHostPtr;
  const nva2x::HostTensorFloat &inferenceResultHostTrue = *inferenceResultHostTruePtr;

  nva2x::HostTensorFloat inferenceResultHost;
  ASSERT_TRUE(!inferenceResultHost.Allocate(inferenceResultHostTrue.Size()));

  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::DeviceTensorFloat audioBufferDevice, emotionDevice, inferenceResultDevice;
  ASSERT_TRUE(!audioBufferDevice.Init(audioBufferHost, cudaStream.Data()));
  ASSERT_TRUE(!emotionDevice.Init(emotionHost, cudaStream.Data()));
  ASSERT_TRUE(!inferenceResultDevice.Allocate(inferenceResultHost.Size()));

  nva2x::InferenceEngine inference;
  ASSERT_TRUE(!inference.Init(networkDataBytes.Data(), networkDataBytes.Size()));

  using DimensionType = nva2x::IBufferBindingsDescription::DimensionType;
  constexpr auto batchDimension = Batched ? DimensionType::BATCH : DimensionType::FIXED;
  constexpr std::array<nva2x::BindingDescription, 3> kDescriptions = {{
    {"emotion", nva2x::IBufferBindingsDescription::IOType::INPUT,
     {{batchDimension, DimensionType::FIXED, DimensionType::FIXED}}},
    {"input", nva2x::IBufferBindingsDescription::IOType::INPUT,
     {{batchDimension, DimensionType::FIXED, DimensionType::FIXED}}},
    {"result", nva2x::IBufferBindingsDescription::IOType::OUTPUT,
     {{batchDimension, DimensionType::FIXED, DimensionType::FIXED}}},
  }};
  const nva2x::BufferBindingsDescription descriptions({kDescriptions.begin(), kDescriptions.end()});
  ASSERT_TRUE(!inference.CheckBindings(descriptions));

  nva2x::BufferBindings bindings(descriptions);
  ASSERT_TRUE(!bindings.SetInputBinding(0, emotionDevice));
  ASSERT_TRUE(!bindings.SetInputBinding(1, audioBufferDevice));
  ASSERT_TRUE(!bindings.SetOutputBinding(2, inferenceResultDevice));
  ASSERT_TRUE(!inference.BindBuffers(bindings, 1));

  ASSERT_TRUE(!inference.Run(cudaStream.Data()));
  ASSERT_TRUE(!nva2x::CopyDeviceToHost(inferenceResultHost, inferenceResultDevice, cudaStream.Data()));
  ASSERT_TRUE(!cudaStream.Synchronize());

  for (unsigned int i = 0; i < inferenceResultHost.Size(); ++i) {
    ASSERT_NEAR(inferenceResultHostTrue.Data()[i],
                inferenceResultHost.Data()[i], 1e-3) // relaxed from 1e-6 to 1e-3 due to cuda 12 upgrade.
        << "i = " << i;
  }

  // Test assigning wrong bindings.
  ASSERT_TRUE(bindings.SetInputBinding(2, emotionDevice));
  ASSERT_TRUE(bindings.SetInputBinding(3, emotionDevice));
  ASSERT_TRUE(bindings.SetInputBinding(100, emotionDevice));
  ASSERT_TRUE(bindings.SetOutputBinding(0, inferenceResultDevice));
  ASSERT_TRUE(bindings.SetOutputBinding(1, inferenceResultDevice));
  ASSERT_TRUE(bindings.SetOutputBinding(3, inferenceResultDevice));
  ASSERT_TRUE(bindings.SetOutputBinding(100, inferenceResultDevice));

  // Test querying with wrong indices.
  ASSERT_NE(nullptr, bindings.GetInputBinding(0).Data());
  ASSERT_NE(nullptr, bindings.GetInputBinding(1).Data());
  ASSERT_EQ(nullptr, bindings.GetInputBinding(2).Data());
  ASSERT_EQ(nullptr, bindings.GetInputBinding(3).Data());
  ASSERT_EQ(nullptr, bindings.GetInputBinding(100).Data());
  ASSERT_EQ(nullptr, bindings.GetOutputBinding(0).Data());
  ASSERT_EQ(nullptr, bindings.GetOutputBinding(1).Data());
  ASSERT_NE(nullptr, bindings.GetOutputBinding(2).Data());
  ASSERT_EQ(nullptr, bindings.GetOutputBinding(3).Data());
  ASSERT_EQ(nullptr, bindings.GetOutputBinding(100).Data());

  // Test assigning wrong sizes.
  {
    nva2x::BufferBindings bindingsWrong(descriptions);
    ASSERT_TRUE(!bindingsWrong.SetInputBinding(0, emotionDevice.View(0, emotionDevice.Size() - 1)));
    ASSERT_TRUE(!bindingsWrong.SetInputBinding(1, audioBufferDevice));
    ASSERT_TRUE(!bindingsWrong.SetOutputBinding(2, inferenceResultDevice));
    ASSERT_TRUE(inference.BindBuffers(bindingsWrong, 1));
  }
  {
    nva2x::BufferBindings bindingsWrong(descriptions);
    ASSERT_TRUE(!bindingsWrong.SetInputBinding(0, emotionDevice));
    ASSERT_TRUE(!bindingsWrong.SetInputBinding(1, audioBufferDevice.View(0, audioBufferDevice.Size() - 1)));
    ASSERT_TRUE(!bindingsWrong.SetOutputBinding(2, inferenceResultDevice));
    ASSERT_TRUE(inference.BindBuffers(bindingsWrong, 1));
  }
  {
    nva2x::BufferBindings bindingsWrong(descriptions);
    ASSERT_TRUE(!bindingsWrong.SetInputBinding(0, emotionDevice));
    ASSERT_TRUE(!bindingsWrong.SetInputBinding(1, audioBufferDevice));
    ASSERT_TRUE(!bindingsWrong.SetOutputBinding(2, inferenceResultDevice.View(0, inferenceResultDevice.Size() - 1)));
    ASSERT_TRUE(inference.BindBuffers(bindingsWrong, 1));
  }
  {
    nva2x::BufferBindings bindingsWrong(descriptions);
    const std::size_t nbBatches = 8;
    nva2x::DeviceTensorFloat audioBufferDeviceBatch, emotionDeviceBatch, inferenceResultDeviceBatch;
    ASSERT_TRUE(!audioBufferDeviceBatch.Allocate(audioBufferDevice.Size() * nbBatches));
    ASSERT_TRUE(!emotionDeviceBatch.Allocate(emotionDevice.Size() * nbBatches));
    ASSERT_TRUE(!inferenceResultDeviceBatch.Allocate(inferenceResultDevice.Size() * nbBatches));

    ASSERT_TRUE(!bindingsWrong.SetInputBinding(0, emotionDeviceBatch));
    ASSERT_TRUE(!bindingsWrong.SetInputBinding(1, audioBufferDeviceBatch));
    ASSERT_TRUE(!bindingsWrong.SetOutputBinding(2, inferenceResultDeviceBatch));
    ASSERT_TRUE(inference.BindBuffers(bindingsWrong, 1));

    if (Batched) {
      ASSERT_TRUE(!inference.BindBuffers(bindingsWrong, nbBatches));
    }
  }
}

}

TEST(InferenceEngine, Inference) {
  TestInferenceEngine<false>();
  TestInferenceEngine<true>();
}

TEST(InferenceEngine, Bindings) {
  // Test wrong bindings.
  using DimensionType = nva2x::IBufferBindingsDescription::DimensionType;
  {
    constexpr std::array<nva2x::BindingDescription, 3> kDescriptionsWrong = {{
      {"emotion", nva2x::IBufferBindingsDescription::IOType::INPUT,
       {{DimensionType::FIXED, DimensionType::FIXED, DimensionType::BATCH}}},
      {"result", nva2x::IBufferBindingsDescription::IOType::OUTPUT,
       {{DimensionType::FIXED, DimensionType::FIXED, DimensionType::BATCH}}},
      {"input", nva2x::IBufferBindingsDescription::IOType::INPUT,
       {{DimensionType::FIXED, DimensionType::FIXED, DimensionType::BATCH}}},
    }};
    ASSERT_FALSE(nva2x::IsSorted(kDescriptionsWrong.data(), kDescriptionsWrong.size()));
  }
  {
    constexpr std::array<nva2x::BindingDescription, 3> kDescriptionsWrong = {{
      {"input", nva2x::IBufferBindingsDescription::IOType::INPUT,
       {{DimensionType::FIXED, DimensionType::FIXED, DimensionType::BATCH}}},
      {"emotion", nva2x::IBufferBindingsDescription::IOType::INPUT,
       {{DimensionType::FIXED, DimensionType::FIXED, DimensionType::BATCH}}},
      {"result", nva2x::IBufferBindingsDescription::IOType::OUTPUT,
       {{DimensionType::FIXED, DimensionType::FIXED, DimensionType::BATCH}}},
    }};
    ASSERT_FALSE(nva2x::IsSorted(kDescriptionsWrong.data(), kDescriptionsWrong.size()));
  }
}
