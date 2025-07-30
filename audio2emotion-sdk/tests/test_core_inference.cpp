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
#include "audio2emotion/model.h"
#include "audio2emotion/internal/model.h"
#include "utils.h"

#include "audio2x/internal/cuda_stream.h"
#include "audio2x/internal/inference_engine.h"
#include "audio2x/internal/io.h"
#include "audio2x/internal/tensor.h"
#include "audio2x/internal/tensor_dict.h"

#include <gtest/gtest.h>

class TestCoreInferenceEngine : public ::testing::Test
{
};

TEST_F(TestCoreInferenceEngine, TestInference)
{
    nva2x::HostTensorDict testData;
    ASSERT_TRUE(!testData.ReadFromFile(TEST_DATA_DIR "_data/generated/audio2emotion-sdk/tests/data/test_data_inference.bin"));

    nva2x::DataBytes modelDataBytes;
    ASSERT_TRUE(!modelDataBytes.ReadFromFile(TEST_DATA_DIR "_data/generated/audio2emotion-sdk/tests/data/test_data_inference_model.trt"));

    auto xHostPtr = testData.At("x");
    auto zHostTruePtr = testData.At("z");
    ASSERT_NE(xHostPtr, nullptr);
    ASSERT_NE(zHostTruePtr, nullptr);
    const nva2x::HostTensorFloat& xHost = *xHostPtr;
    const nva2x::HostTensorFloat& zHostTrue = *zHostTruePtr;

    nva2x::HostTensorFloat zHost;
    ASSERT_TRUE(!zHost.Allocate(zHostTrue.Size()));

    nva2x::CudaStream cudaStream;
    ASSERT_TRUE(!cudaStream.Init());

    nva2x::DeviceTensorFloat xDevice, zDevice;
    ASSERT_TRUE(!xDevice.Allocate(xHost.Size()));
    ASSERT_TRUE(!zDevice.Allocate(zHost.Size()));

    const auto& bindingsDescription = nva2e::IClassifierModel::GetBindingsDescription();

    nva2x::InferenceEngine engine;
    ASSERT_TRUE(!engine.Init(modelDataBytes.Data(), modelDataBytes.Size()));
    ASSERT_TRUE(!engine.CheckBindings(bindingsDescription));

    nva2x::BufferBindings bufferBindings(bindingsDescription);
    ASSERT_TRUE(!bufferBindings.SetInputBinding(nva2e::IClassifierModel::kInputTensorIndex, xDevice));
    ASSERT_TRUE(!bufferBindings.SetOutputBinding(nva2e::IClassifierModel::kResultTensorIndex, zDevice));
    ASSERT_TRUE(!bufferBindings.SetDynamicDimension(nva2e::IClassifierModel::kInputTensorIndex, 1, 30000));
    ASSERT_TRUE(!engine.BindBuffers(bufferBindings, 10));

    ASSERT_TRUE(!engine.Run(cudaStream.Data()));

    ASSERT_TRUE(!nva2x::CopyHostToDevice(xDevice, xHost, cudaStream.Data()));
    ASSERT_TRUE(!engine.Run(cudaStream.Data()));
    ASSERT_TRUE(!nva2x::CopyDeviceToHost(zHost, zDevice, cudaStream.Data()));
    ASSERT_TRUE(!cudaStream.Synchronize());

    for (unsigned int i = 0; i < zHost.Size(); ++i) {
        ASSERT_NEAR(zHostTrue.Data()[i], zHost.Data()[i], EPS) << "i = " << i;
    }
}
