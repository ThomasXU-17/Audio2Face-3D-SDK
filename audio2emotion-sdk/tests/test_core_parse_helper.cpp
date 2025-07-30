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
#include "audio2emotion/internal/parse_helper.h"
#include "audio2emotion/internal/executor_classifier.h"
#include "audio2emotion/internal/executor_postprocess.h"
#include "audio2x/internal/unique_ptr.h"
#include <gtest/gtest.h>

#include <filesystem>

namespace {
    std::string GetModelPath(const char* filename) {
        return std::string(TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/") + filename;
    }

  constexpr const char* kEmotionNames[] = {"angry", "disgust", "fear", "happy", "neutral", "sad"};
}


TEST(TestCoreParseHelper, PostProcessParams) {
  const auto configPath = GetModelPath("model_config.json");
  for (auto readFunc : {nva2e::ReadClassifierConfigInfo_INTERNAL, nva2e::ReadPostProcessConfigInfo_INTERNAL}) {
    auto config = nva2x::ToUniquePtr(
      readFunc(configPath.c_str(), std::size(kEmotionNames), kEmotionNames)
      );
    ASSERT_NE(config, nullptr);

    const auto postProcessData = config->GetPostProcessData();
    EXPECT_EQ(postProcessData.inferenceEmotionLength, std::size(kEmotionNames));
    EXPECT_EQ(postProcessData.outputEmotionLength, 10);
    EXPECT_EQ(postProcessData.emotionCorrespondenceSize, 6);
    EXPECT_EQ(postProcessData.emotionCorrespondence[0], 1);
    EXPECT_EQ(postProcessData.emotionCorrespondence[1], 3);
    EXPECT_EQ(postProcessData.emotionCorrespondence[2], 4);
    EXPECT_EQ(postProcessData.emotionCorrespondence[3], 6);
    EXPECT_EQ(postProcessData.emotionCorrespondence[4], -1);
    EXPECT_EQ(postProcessData.emotionCorrespondence[5], 9);

    const auto postProcessParams = config->GetPostProcessParams();
    EXPECT_EQ(postProcessParams.emotionContrast, 1.0f);
    EXPECT_EQ(postProcessParams.maxEmotions, 6);
    EXPECT_EQ(postProcessParams.beginningEmotion.Size(), 10);
    EXPECT_EQ(postProcessParams.beginningEmotion.Data()[0], 0.0f);
    EXPECT_EQ(postProcessParams.beginningEmotion.Data()[1], 0.0f);
    EXPECT_EQ(postProcessParams.beginningEmotion.Data()[2], 0.0f);
    EXPECT_EQ(postProcessParams.beginningEmotion.Data()[3], 0.0f);
    EXPECT_EQ(postProcessParams.beginningEmotion.Data()[4], 0.0f);
    EXPECT_EQ(postProcessParams.beginningEmotion.Data()[5], 0.0f);
    EXPECT_EQ(postProcessParams.beginningEmotion.Data()[6], 0.0f);
    EXPECT_EQ(postProcessParams.beginningEmotion.Data()[7], 0.0f);
    EXPECT_EQ(postProcessParams.beginningEmotion.Data()[8], 0.0f);
    EXPECT_EQ(postProcessParams.beginningEmotion.Data()[9], 0.0f);

    EXPECT_EQ(postProcessParams.preferredEmotion.Size(), 10);
    EXPECT_EQ(postProcessParams.preferredEmotion.Data()[0], 0.0f);
    EXPECT_EQ(postProcessParams.preferredEmotion.Data()[1], 0.0f);
    EXPECT_EQ(postProcessParams.preferredEmotion.Data()[2], 0.0f);
    EXPECT_EQ(postProcessParams.preferredEmotion.Data()[3], 0.0f);
    EXPECT_EQ(postProcessParams.preferredEmotion.Data()[4], 0.0f);
    EXPECT_EQ(postProcessParams.preferredEmotion.Data()[5], 0.0f);
    EXPECT_EQ(postProcessParams.preferredEmotion.Data()[6], 0.0f);
    EXPECT_EQ(postProcessParams.preferredEmotion.Data()[7], 0.0f);
    EXPECT_EQ(postProcessParams.preferredEmotion.Data()[8], 0.0f);
    EXPECT_EQ(postProcessParams.preferredEmotion.Data()[9], 0.0f);

    EXPECT_EQ(postProcessParams.liveBlendCoef, 0.7f);
    EXPECT_EQ(postProcessParams.enablePreferredEmotion, false);
    EXPECT_EQ(postProcessParams.preferredEmotionStrength, 0.5f);
    EXPECT_EQ(postProcessParams.liveTransitionTime, 0.5f);
    EXPECT_EQ(postProcessParams.fixedDt, 0.033f);
    EXPECT_EQ(postProcessParams.emotionStrength, 0.6f);

    const auto inputStrength = config->GetInputStrength();
    EXPECT_EQ(inputStrength, 1.0f);
  }
}

TEST(TestCoreParseHelper, PostProcessParamsError) {
  for (auto readFunc : {nva2e::ReadClassifierConfigInfo_INTERNAL, nva2e::ReadPostProcessConfigInfo_INTERNAL}) {
    {
      const auto configPath = GetModelPath("config_error.json");
      ASSERT_FALSE(std::filesystem::exists(configPath));
      auto config = nva2x::ToUniquePtr(readFunc(
        configPath.c_str(), std::size(kEmotionNames), kEmotionNames
        ));
      ASSERT_EQ(config, nullptr);
    }
    {
      const auto configPath = GetModelPath("trt_info.json");
      ASSERT_TRUE(std::filesystem::exists(configPath));
      auto config = nva2x::ToUniquePtr(readFunc(
        configPath.c_str(), std::size(kEmotionNames), kEmotionNames
        ));
      ASSERT_EQ(config, nullptr);
    }
  }
}

TEST(TestCoreParseHelper, NetworkInfo) {
  const auto networkInfoPath = GetModelPath("network_info.json");
  for (auto readFunc : {nva2e::ReadClassifierNetworkInfo_INTERNAL, nva2e::ReadPostProcessNetworkInfo_INTERNAL}) {
    auto networkInfo = nva2x::ToUniquePtr(readFunc(networkInfoPath.c_str()));
    ASSERT_NE(networkInfo, nullptr);

    EXPECT_EQ(networkInfo->GetNetworkInfo(60000).bufferLength, 60000U);
    EXPECT_EQ(networkInfo->GetNetworkInfo(60000).bufferSamplerate, 16000U);

    EXPECT_EQ(networkInfo->GetEmotionsCount(), 6U);
    EXPECT_STREQ(networkInfo->GetEmotionName(0), kEmotionNames[0]);
    EXPECT_STREQ(networkInfo->GetEmotionName(1), kEmotionNames[1]);
    EXPECT_STREQ(networkInfo->GetEmotionName(2), kEmotionNames[2]);
    EXPECT_STREQ(networkInfo->GetEmotionName(3), kEmotionNames[3]);
    EXPECT_STREQ(networkInfo->GetEmotionName(4), kEmotionNames[4]);
    EXPECT_STREQ(networkInfo->GetEmotionName(5), kEmotionNames[5]);
  }
}

TEST(TestCoreParseHelper, NetworkInfoError) {
  for (auto readFunc : {nva2e::ReadClassifierConfigInfo_INTERNAL, nva2e::ReadPostProcessConfigInfo_INTERNAL}) {
    {
      const auto networkInfoPath = GetModelPath("network_info_error.json");
      ASSERT_FALSE(std::filesystem::exists(networkInfoPath));
      auto networkInfo = nva2x::ToUniquePtr(nva2e::ReadClassifierNetworkInfo_INTERNAL(networkInfoPath.c_str()));
      ASSERT_EQ(networkInfo, nullptr);
    }
    {
      const auto networkInfoPath = GetModelPath("trt_info.json");
      ASSERT_TRUE(std::filesystem::exists(networkInfoPath));
      auto networkInfo = nva2x::ToUniquePtr(nva2e::ReadClassifierNetworkInfo_INTERNAL(networkInfoPath.c_str()));
      ASSERT_EQ(networkInfo, nullptr);
    }
  }
}

TEST(TestCoreParseHelper, ClassifierEmotionExecutor) {
  const auto modelPath = GetModelPath("model.json");
  auto modelInfo = nva2x::ToUniquePtr(nva2e::ReadClassifierModelInfo_INTERNAL(modelPath.c_str()));
  ASSERT_NE(modelInfo, nullptr);
  const auto classifierParams = modelInfo->GetExecutorCreationParameters(
    60000, 30, 1, 0
    );

  auto audioAccumulator = nva2x::ToUniquePtr(nva2x::CreateAudioAccumulator(16000U, 0));
  ASSERT_NE(audioAccumulator, nullptr);

  nva2e::EmotionExecutorCreationParameters params;
  params.cudaStream = nullptr;
  params.nbTracks = 1;
  auto sharedAudioAccumulator = audioAccumulator.get();
  params.sharedAudioAccumulators = &sharedAudioAccumulator;

  auto executor = nva2x::ToUniquePtr(nva2e::CreateClassifierEmotionExecutor_INTERNAL(params, classifierParams));
  ASSERT_NE(executor, nullptr);
}

TEST(TestCoreParseHelper, ClassifierEmotionExecutorBundle) {
  const auto modelPath = GetModelPath("model.json");
  nva2e::IClassifierModel::IEmotionModelInfo* modelInfo = nullptr;
  auto bundle = nva2x::ToUniquePtr(
    nva2e::ReadClassifierEmotionExecutorBundle_INTERNAL(
      1, modelPath.c_str(), 60000, 30, 1, 0, &modelInfo
      )
    );
  ASSERT_NE(bundle, nullptr);
  ASSERT_NE(bundle->GetCudaStream().Data(), nullptr);
  ASSERT_EQ(bundle->GetAudioAccumulator(0).NbAccumulatedSamples(), 0U);
  ASSERT_EQ(bundle->GetPreferredEmotionAccumulator(0).GetEmotionSize(), 10U);
  ASSERT_EQ(bundle->GetExecutor().HasExecutionStarted(0), false);

  ASSERT_NE(modelInfo, nullptr);
  EXPECT_EQ(modelInfo->GetNetworkInfo().GetNetworkInfo(60000).bufferLength, 60000U);
  EXPECT_EQ(modelInfo->GetNetworkInfo().GetNetworkInfo(60000).bufferSamplerate, 16000U);

  EXPECT_EQ(modelInfo->GetNetworkInfo().GetEmotionsCount(), 6U);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(0), kEmotionNames[0]);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(1), kEmotionNames[1]);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(2), kEmotionNames[2]);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(3), kEmotionNames[3]);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(4), kEmotionNames[4]);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(5), kEmotionNames[5]);
}

TEST(TestCoreParseHelper, PostProcessEmotionExecutor) {
  const auto modelPath = GetModelPath("model.json");
  auto modelInfo = nva2x::ToUniquePtr(nva2e::ReadPostProcessModelInfo_INTERNAL(modelPath.c_str()));
  ASSERT_NE(modelInfo, nullptr);
  const auto postProcessParams = modelInfo->GetExecutorCreationParameters(
    30, 1
    );

  auto audioAccumulator = nva2x::ToUniquePtr(nva2x::CreateAudioAccumulator(16000U, 0));
  ASSERT_NE(audioAccumulator, nullptr);

  nva2e::EmotionExecutorCreationParameters params;
  params.cudaStream = nullptr;
  params.nbTracks = 1;
  auto sharedAudioAccumulator = audioAccumulator.get();
  params.sharedAudioAccumulators = &sharedAudioAccumulator;

  auto executor = nva2x::ToUniquePtr(nva2e::CreatePostProcessEmotionExecutor_INTERNAL(params, postProcessParams));
  ASSERT_NE(executor, nullptr);
}

TEST(TestCoreParseHelper, PostProcessEmotionExecutorBundle) {
  const auto modelPath = GetModelPath("model.json");
  nva2e::IPostProcessModel::IEmotionModelInfo* modelInfo = nullptr;
  auto bundle = nva2x::ToUniquePtr(
    nva2e::ReadPostProcessEmotionExecutorBundle_INTERNAL(
      1, modelPath.c_str(), 30, 1, &modelInfo
      )
    );
  ASSERT_NE(bundle, nullptr);
  ASSERT_NE(bundle->GetCudaStream().Data(), nullptr);
  ASSERT_EQ(bundle->GetAudioAccumulator(0).NbAccumulatedSamples(), 0U);
  ASSERT_EQ(bundle->GetPreferredEmotionAccumulator(0).GetEmotionSize(), 10U);
  ASSERT_EQ(bundle->GetExecutor().HasExecutionStarted(0), false);

  ASSERT_NE(modelInfo, nullptr);
  EXPECT_EQ(modelInfo->GetNetworkInfo().GetNetworkInfo(60000).bufferLength, 60000U);
  EXPECT_EQ(modelInfo->GetNetworkInfo().GetNetworkInfo(60000).bufferSamplerate, 16000U);

  EXPECT_EQ(modelInfo->GetNetworkInfo().GetEmotionsCount(), 6U);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(0), kEmotionNames[0]);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(1), kEmotionNames[1]);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(2), kEmotionNames[2]);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(3), kEmotionNames[3]);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(4), kEmotionNames[4]);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(5), kEmotionNames[5]);
}
