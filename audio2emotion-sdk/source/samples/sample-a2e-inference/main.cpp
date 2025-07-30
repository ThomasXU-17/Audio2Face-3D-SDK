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

#include "audio2emotion/audio2emotion.h"

#include "audio2x/cuda_utils.h"
#include "audio2x/cuda_stream.h"
#include "audio2x/inference_engine.h"
#include "audio2x/io.h"
#include "audio2x/tensor.h"

#include <iostream>
#include <memory>
#include <string>
#include <cmath>

#define CHECK_SUCCESS(func)                                                    \
  {                                                                            \
    std::error_code error = func;                                              \
    if (error) {                                                               \
      std::cout << "Error: Failed to execute: " << #func;                      \
      std::cout << ", Reason: "<< error.message() << std::endl;                \
      return false;                                                            \
    }                                                                          \
  }

#define CHECK_NOT_NULL(expression)                                             \
  {                                                                            \
    if ((expression) == nullptr) {                                             \
      std::cout << "Error: " << #expression << " is NULL" << std::endl;        \
      return false;                                                            \
    }                                                                          \
  }

struct Destroyer {
    template <typename T>
    void operator()(T* obj) const {
        obj->Destroy();
    }
};
template <typename T>
using UniquePtr = std::unique_ptr<T, Destroyer>;
template <typename T>
UniquePtr<T> ToUniquePtr(T* ptr) { return UniquePtr<T>(ptr); }

bool sample(void) {
    std::cout << "==============================" << std::endl;
    std::cout << "Audio2Emotion Inference Engine Example" << std::endl;
    std::cout << "Inference API" << std::endl;

    std::string netPath = TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/network.trt";
    constexpr int batchSize = 10;
    constexpr int windowSize = 30000;
    unsigned int xLength = windowSize * batchSize;
    unsigned int zLength = 6 * batchSize;

    constexpr int deviceID = 0;
    CHECK_SUCCESS(nva2x::SetCudaDeviceIfNeeded(deviceID));
    auto cudaStream = ToUniquePtr(nva2x::CreateCudaStream());
    CHECK_NOT_NULL(cudaStream);

    auto xDevice = ToUniquePtr(nva2x::CreateDeviceTensorFloat(xLength));
    CHECK_NOT_NULL(xDevice);
    auto zDevice = ToUniquePtr(nva2x::CreateDeviceTensorFloat(zLength));
    CHECK_NOT_NULL(zDevice);
    auto zHost = ToUniquePtr(nva2x::CreateHostTensorFloat(zLength));
    CHECK_NOT_NULL(zHost);

    const auto& bindingsDescription = nva2e::GetBindingsDescriptionForClassifierModel();

    auto modelDataBytes = ToUniquePtr(nva2x::CreateDataBytes());
    CHECK_SUCCESS(modelDataBytes->ReadFromFile(netPath.c_str()));
    auto engine = ToUniquePtr(nva2x::CreateInferenceEngine());
    CHECK_SUCCESS(engine->Init(modelDataBytes->Data(), modelDataBytes->Size()));
    CHECK_SUCCESS(engine->CheckBindings(bindingsDescription));

    auto bufferBindings = ToUniquePtr(nva2e::CreateBindingsForClassifierModel());
    CHECK_SUCCESS(bufferBindings->SetInputBinding(nva2e::IClassifierModel::kInputTensorIndex, *xDevice));
    CHECK_SUCCESS(bufferBindings->SetOutputBinding(nva2e::IClassifierModel::kResultTensorIndex, *zDevice));
    CHECK_SUCCESS(bufferBindings->SetDynamicDimension(nva2e::IClassifierModel::kInputTensorIndex, 1, windowSize));
    CHECK_SUCCESS(engine->BindBuffers(*bufferBindings, batchSize));

    // Fill with actual audio.
    CHECK_SUCCESS(nva2x::FillOnDevice(*xDevice, 0.0f, cudaStream->Data()));
    CHECK_SUCCESS(engine->Run(cudaStream->Data()));
    CHECK_SUCCESS(nva2x::CopyDeviceToHost(*zHost, *zDevice, cudaStream->Data()));
    CHECK_SUCCESS(cudaStream->Synchronize());

    std::cout << "Finished" << std::endl;
    std::cout << "==============================" << std::endl;

    return true;
}

int main(void) {
    if (!sample()) {
        return 1;
    }
    return 0;
}
