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
#include "audio2face/internal/emotion.h"
#include "audio2x/internal/cuda_stream.h"
#include "audio2x/internal/io.h"
#include "audio2x/internal/tensor_dict.h"
#include "utils.h"

#include <gtest/gtest.h>

class TestCoreEmotion : public ::testing::Test {};

TEST_F(TestCoreEmotion, TestEmotionIO) {
  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2f::EmotionDatabase emotionDatabase;
  ASSERT_TRUE(!emotionDatabase.SetCudaStream(cudaStream.Data()));
  ASSERT_TRUE(!emotionDatabase.InitFromFile(
      TEST_DATA_DIR "_data/audio2x-sdk-test-data/audio2face-sdk-test-data/test_data_implicit_emo_db.npz"));
  ASSERT_TRUE(!cudaStream.Synchronize());
}

TEST_F(TestCoreEmotion, TestGetEmotion) {
  nva2x::HostTensorDict testData;
  ASSERT_TRUE(!testData.ReadFromFile(TEST_DATA_DIR "_data/audio2x-sdk-test-data/audio2face-sdk-test-data/test_data_emotion.npz"));

  auto emoVec1HostTruePtr = testData.At("emo_vec1");
  auto emoVec2HostTruePtr = testData.At("emo_vec2");
  auto emoVec3HostTruePtr = testData.At("emo_vec3");
  ASSERT_NE(emoVec1HostTruePtr, nullptr);
  ASSERT_NE(emoVec2HostTruePtr, nullptr);
  ASSERT_NE(emoVec3HostTruePtr, nullptr);
  const nva2x::HostTensorFloat &emoVec1HostTrue = *emoVec1HostTruePtr;
  const nva2x::HostTensorFloat &emoVec2HostTrue = *emoVec2HostTruePtr;
  const nva2x::HostTensorFloat &emoVec3HostTrue = *emoVec3HostTruePtr;

  nva2x::HostTensorFloat emoVec1Host, emoVec2Host, emoVec3Host;
  ASSERT_TRUE(!emoVec1Host.Allocate(emoVec1HostTrue.Size()));
  ASSERT_TRUE(!emoVec2Host.Allocate(emoVec2HostTrue.Size()));
  ASSERT_TRUE(!emoVec3Host.Allocate(emoVec3HostTrue.Size()));

  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::DeviceTensorFloat emoVec1Device, emoVec2Device, emoVec3Device;
  ASSERT_TRUE(!emoVec1Device.Allocate(emoVec1Host.Size()));
  ASSERT_TRUE(!emoVec2Device.Allocate(emoVec2Host.Size()));
  ASSERT_TRUE(!emoVec3Device.Allocate(emoVec3Host.Size()));

  nva2f::EmotionDatabase emotionDatabase;
  ASSERT_TRUE(!emotionDatabase.SetCudaStream(cudaStream.Data()));
  ASSERT_TRUE(!emotionDatabase.InitFromFile(
      TEST_DATA_DIR "_data/audio2x-sdk-test-data/audio2face-sdk-test-data/test_data_implicit_emo_db.npz"));
  ASSERT_TRUE(!cudaStream.Synchronize());

  //////////////////////////////////////////////

  ASSERT_TRUE(!emotionDatabase.GetEmotion("g1c_neutral", 100, emoVec1Device));
  ASSERT_TRUE(!nva2x::CopyDeviceToHost(emoVec1Host, emoVec1Device, cudaStream.Data()));
  ASSERT_TRUE(!cudaStream.Synchronize());

  for (unsigned int i = 0; i < emoVec1Host.Size(); ++i) {
    ASSERT_NEAR(emoVec1HostTrue.Data()[i], emoVec1Host.Data()[i], 1e-9)
        << "i = " << i;
  }

  //////////////////////////////////////////////

  ASSERT_TRUE(!emotionDatabase.GetEmotion("g2a_neutral", 10, emoVec2Device));
  ASSERT_TRUE(!nva2x::CopyDeviceToHost(emoVec2Host, emoVec2Device, cudaStream.Data()));
  ASSERT_TRUE(!cudaStream.Synchronize());

  for (unsigned int i = 0; i < emoVec2Host.Size(); ++i) {
    ASSERT_NEAR(emoVec2HostTrue.Data()[i], emoVec2Host.Data()[i], 1e-9)
        << "i = " << i;
  }

  //////////////////////////////////////////////

  ASSERT_TRUE(
      !emotionDatabase.GetEmotion("p3_neutral", 280, emoVec3Device));
  ASSERT_TRUE(!nva2x::CopyDeviceToHost(emoVec3Host, emoVec3Device, cudaStream.Data()));
  ASSERT_TRUE(!cudaStream.Synchronize());

  for (unsigned int i = 0; i < emoVec3Host.Size(); ++i) {
    ASSERT_NEAR(emoVec3HostTrue.Data()[i], emoVec3Host.Data()[i], 1e-9)
        << "i = " << i;
  }
}
