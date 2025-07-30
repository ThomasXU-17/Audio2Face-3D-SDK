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
#include "audio2face/internal/parse_helper.h"
#include "audio2face/internal/executor_regression.h"
#include "audio2face/internal/executor_diffusion.h"
#include "audio2face/internal/executor_blendshapesolve.h"
#include "audio2x/internal/tensor_dict.h"
#include "audio2x/internal/unique_ptr.h"
#include <gtest/gtest.h>

#include <filesystem>

namespace {
    std::string GetRegressionPath(const char* filename, const char* actor="mark") {
        return std::string(TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/") + actor + "/" + filename;
    }

    std::string GetDiffusionPath(const char* filename) {
        return std::string(TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/") + filename;
    }
}


TEST(TestCoreParseHelper, RegressionAnimatorParams) {
  const auto configPath = GetRegressionPath("model_config.json");
  float inputStrength;
  nva2f::AnimatorParams animatorParams;
  const auto error = nva2f::ReadAnimatorParams_INTERNAL(configPath.c_str(), inputStrength, animatorParams);
  ASSERT_TRUE(!error);

  EXPECT_EQ(inputStrength, 1.3f);
  EXPECT_EQ(animatorParams.skin.lowerFaceSmoothing, 0.0023f);
  EXPECT_EQ(animatorParams.skin.upperFaceSmoothing, 0.001f);
  EXPECT_EQ(animatorParams.skin.lowerFaceStrength, 1.4f);
  EXPECT_EQ(animatorParams.skin.upperFaceStrength, 1.0f);
  EXPECT_EQ(animatorParams.skin.faceMaskLevel, 0.6f);
  EXPECT_EQ(animatorParams.skin.faceMaskSoftness, 0.0085f);
  EXPECT_EQ(animatorParams.skin.skinStrength, 1.1f);
  EXPECT_EQ(animatorParams.skin.blinkStrength, 1.0f);
  EXPECT_EQ(animatorParams.skin.eyelidOpenOffset, 0.06f);
  EXPECT_EQ(animatorParams.skin.lipOpenOffset, -0.03f);
  EXPECT_EQ(animatorParams.skin.blinkOffset, 0.0f);

  EXPECT_EQ(animatorParams.tongue.tongueStrength, 1.5f);
  EXPECT_EQ(animatorParams.tongue.tongueHeightOffset, 0.2f);
  EXPECT_EQ(animatorParams.tongue.tongueDepthOffset, 0.13f);

  EXPECT_EQ(animatorParams.teeth.lowerTeethStrength, 1.3f);
  EXPECT_EQ(animatorParams.teeth.lowerTeethHeightOffset, -0.1f);
  EXPECT_EQ(animatorParams.teeth.lowerTeethDepthOffset, 0.0f);

  EXPECT_EQ(animatorParams.eyes.eyeballsStrength, 1.0f);
  EXPECT_EQ(animatorParams.eyes.saccadeStrength, 0.9f);
  EXPECT_EQ(animatorParams.eyes.rightEyeballRotationOffsetX, 0.0f);
  EXPECT_EQ(animatorParams.eyes.rightEyeballRotationOffsetY, -2.0f);
  EXPECT_EQ(animatorParams.eyes.leftEyeballRotationOffsetX, 0.0f);
  EXPECT_EQ(animatorParams.eyes.leftEyeballRotationOffsetY, 2.0f);
  EXPECT_EQ(animatorParams.eyes.saccadeSeed, 0.0f);
}


TEST(TestCoreParseHelper, RegressionNetworkInfo) {
  const auto networkInfoPath = GetRegressionPath("network_info.json");
  auto networkInfo = nva2x::ToUniquePtr(nva2f::ReadRegressionNetworkInfo_INTERNAL(networkInfoPath.c_str()));
  ASSERT_NE(networkInfo, nullptr);

  EXPECT_EQ(networkInfo->GetNetworkInfo().implicitEmotionLength, 16U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().explicitEmotionLength, 10U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().numShapesSkin, 272U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().numShapesTongue, 10U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().resultSkinSize, 61520U * 3);
  EXPECT_EQ(networkInfo->GetNetworkInfo().resultTongueSize, 5602U * 3);
  EXPECT_EQ(networkInfo->GetNetworkInfo().resultJawSize, 15U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().resultEyesSize, 4U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().bufferLength, 8320U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().bufferOffset, 4160U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().bufferSamplerate, 16000U);

  EXPECT_EQ(networkInfo->GetEmotionsCount(), 10U);
  EXPECT_STREQ(networkInfo->GetEmotionName(0), "amazement");
  EXPECT_STREQ(networkInfo->GetEmotionName(1), "anger");
  EXPECT_STREQ(networkInfo->GetEmotionName(2), "cheekiness");
  EXPECT_STREQ(networkInfo->GetEmotionName(3), "disgust");
  EXPECT_STREQ(networkInfo->GetEmotionName(4), "fear");
  EXPECT_STREQ(networkInfo->GetEmotionName(5), "grief");
  EXPECT_STREQ(networkInfo->GetEmotionName(6), "joy");
  EXPECT_STREQ(networkInfo->GetEmotionName(7), "outofbreath");
  EXPECT_STREQ(networkInfo->GetEmotionName(8), "pain");
  EXPECT_STREQ(networkInfo->GetEmotionName(9), "sadness");

  EXPECT_EQ(networkInfo->GetDefaultEmotion().Size(), 10U);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[0], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[1], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[2], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[3], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[4], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[5], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[6], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[7], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[8], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[9], 0.0f);

  EXPECT_STREQ(networkInfo->GetIdentityName(), "mark");
}

TEST(TestCoreParseHelper, RegressionNetworkInfoError) {
  {
    const auto networkInfoPath = GetRegressionPath("network_info_error.json");
    ASSERT_FALSE(std::filesystem::exists(networkInfoPath));
    auto networkInfo = nva2x::ToUniquePtr(nva2f::ReadRegressionNetworkInfo_INTERNAL(networkInfoPath.c_str()));
    ASSERT_EQ(networkInfo, nullptr);
  }
  {
    const auto networkInfoPath = GetRegressionPath("model.json");
    ASSERT_TRUE(std::filesystem::exists(networkInfoPath));
    auto networkInfo = nva2x::ToUniquePtr(nva2f::ReadRegressionNetworkInfo_INTERNAL(networkInfoPath.c_str()));
    ASSERT_EQ(networkInfo, nullptr);
  }
}

TEST(TestCoreParseHelper, RegressionAnimatorData) {
  const auto animatorDataPath = GetRegressionPath("model_data.npz");
  auto animatorData = nva2x::ToUniquePtr(nva2f::ReadRegressionAnimatorData_INTERNAL(animatorDataPath.c_str()));
  ASSERT_NE(animatorData, nullptr);

  const auto animatorDataView = animatorData->GetAnimatorData();
  EXPECT_EQ(animatorDataView.skin.neutralPose.Size(), 184560U);
  EXPECT_EQ(animatorDataView.skin.lipOpenPoseDelta.Size(), 184560U);
  EXPECT_EQ(animatorDataView.skin.eyeClosePoseDelta.Size(), 184560U);
  EXPECT_EQ(animatorDataView.tongue.neutralPose.Size(), 16806);
  EXPECT_EQ(animatorDataView.teeth.neutralJaw.Size(), 15U);
  EXPECT_EQ(animatorDataView.eyes.saccadeRot.Size(), 10000U);

  const auto skinPcaReconstructionData = animatorData->GetSkinPcaReconstructionData();
  EXPECT_EQ(skinPcaReconstructionData.shapesMatrix.Size(), 184560U * 272);
  EXPECT_EQ(skinPcaReconstructionData.shapeSize, 184560U);

  const auto tonguePcaReconstructionData = animatorData->GetTonguePcaReconstructionData();
  EXPECT_EQ(tonguePcaReconstructionData.shapesMatrix.Size(), 16806U * 10);
  EXPECT_EQ(tonguePcaReconstructionData.shapeSize, 16806U);
}

TEST(TestCoreParseHelper, RegressionAnimatorDataNoUnexpectedTensors) {
  for(const auto& actor : {
    "claire",
    "james",
    "mark"
  }) {
    const auto animatorDataPath = GetRegressionPath("model_data.npz", actor);
    nva2x::HostTensorDict data;
    ASSERT_TRUE(!data.ReadFromFile(animatorDataPath.c_str()));

    const std::vector<std::string> expectedTensorNames = {
      "shapes_matrix_skin",
      "shapes_mean_skin",
      "lip_open_pose_delta",
      "eye_close_pose_delta",
      "shapes_matrix_tongue",
      "shapes_mean_tongue",
      "neutral_jaw",
      "saccade_rot_matrix",
    };

    for(const auto& tensorName : expectedTensorNames) {
      ASSERT_NE(data.At(tensorName.c_str()), nullptr)
        << "Failed to read tensor: '" << tensorName
        << "' in file: " << animatorDataPath;
    }
    ASSERT_EQ(data.Size(), expectedTensorNames.size())
      << "Unexpected tensor found in: " << animatorDataPath;
  }
}

TEST(TestCoreParseHelper, RegressionAnimatorDataError) {
  {
    const auto animatorDataPath = GetRegressionPath("data_error.npz");
    ASSERT_FALSE(std::filesystem::exists(animatorDataPath));
    auto animatorData = nva2f::ReadRegressionAnimatorData_INTERNAL(animatorDataPath.c_str());
    ASSERT_EQ(animatorData, nullptr);
  }
  {
    const auto animatorDataPath = GetRegressionPath("bs_skin.npz");
    ASSERT_TRUE(std::filesystem::exists(animatorDataPath));
    auto animatorData = nva2f::ReadRegressionAnimatorData_INTERNAL(animatorDataPath.c_str());
    ASSERT_EQ(animatorData, nullptr);
  }
}

TEST(TestCoreParseHelper, RegressionGeometryExecutor) {
  const auto modelPath = GetRegressionPath("model.json");
  auto modelInfo = nva2x::ToUniquePtr(nva2f::ReadRegressionModelInfo_INTERNAL(modelPath.c_str()));
  ASSERT_NE(modelInfo, nullptr);
  const auto regressionParams = modelInfo->GetExecutorCreationParameters(
    nva2f::IGeometryExecutor::ExecutionOption::All, 30, 1
    );

  auto audioAccumulator = nva2x::ToUniquePtr(nva2x::CreateAudioAccumulator(16000U, 0));
  ASSERT_NE(audioAccumulator, nullptr);

  auto emotionAccumulator = nva2x::ToUniquePtr(
    nva2x::CreateEmotionAccumulator(regressionParams.networkInfo.explicitEmotionLength, 100, 0)
    );
  ASSERT_NE(emotionAccumulator, nullptr);

  nva2f::GeometryExecutorCreationParameters params;
  params.cudaStream = nullptr;
  params.nbTracks = 1;
  const auto sharedAudioAccumulator = audioAccumulator.get();
  params.sharedAudioAccumulators = &sharedAudioAccumulator;
  const auto sharedEmotionAccumulator = emotionAccumulator.get();
  params.sharedEmotionAccumulators = &sharedEmotionAccumulator;

  auto executor = nva2x::ToUniquePtr(nva2f::CreateRegressionGeometryExecutor_INTERNAL(params, regressionParams));
  ASSERT_NE(executor, nullptr);
}

TEST(TestCoreParseHelper, RegressionGeometryExecutorBundle) {
  const auto modelPath = GetRegressionPath("model.json");
  nva2f::IRegressionModel::IGeometryModelInfo* modelInfo = nullptr;
  auto bundle = nva2x::ToUniquePtr(
    nva2f::ReadRegressionGeometryExecutorBundle_INTERNAL(
      1, modelPath.c_str(), nva2f::IGeometryExecutor::ExecutionOption::All, 30, 1, &modelInfo
      )
    );
  ASSERT_NE(bundle, nullptr);
  ASSERT_NE(bundle->GetCudaStream().Data(), nullptr);
  ASSERT_EQ(bundle->GetAudioAccumulator(0).NbAccumulatedSamples(), 0U);
  ASSERT_EQ(bundle->GetEmotionAccumulator(0).GetEmotionSize(), 10U);
  ASSERT_EQ(bundle->GetExecutor().HasExecutionStarted(0), false);

  ASSERT_NE(modelInfo, nullptr);
  EXPECT_EQ(modelInfo->GetNetworkInfo().GetEmotionsCount(), 10U);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(0), "amazement");
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(9), "sadness");
}

TEST(TestCoreParseHelper, RegressionBlendshapeExecutorBundle) {
  const auto modelPath = GetRegressionPath("model.json");
  {
    nva2f::IRegressionModel::IGeometryModelInfo* modelInfo = nullptr;
    nva2f::IRegressionModel::IBlendshapeSolveModelInfo* blendshapeSolveModelInfo = nullptr;
    auto bundle = nva2x::ToUniquePtr(
      nva2f::ReadRegressionBlendshapeSolveExecutorBundle_INTERNAL(
        1, modelPath.c_str(), nva2f::IGeometryExecutor::ExecutionOption::SkinTongue, false, 30, 1, &modelInfo, &blendshapeSolveModelInfo
        )
      );
    ASSERT_NE(bundle, nullptr);
    ASSERT_NE(bundle->GetCudaStream().Data(), nullptr);
    ASSERT_EQ(bundle->GetAudioAccumulator(0).NbAccumulatedSamples(), 0U);
    ASSERT_EQ(bundle->GetEmotionAccumulator(0).GetEmotionSize(), 10U);
    ASSERT_EQ(bundle->GetExecutor().HasExecutionStarted(0), false);

    ASSERT_NE(modelInfo, nullptr);
    EXPECT_EQ(modelInfo->GetNetworkInfo().GetEmotionsCount(), 10U);
    EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(0), "amazement");
    EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(9), "sadness");

    ASSERT_NE(blendshapeSolveModelInfo, nullptr);
  }
  {
    auto bundle = nva2x::ToUniquePtr(
      nva2f::ReadRegressionBlendshapeSolveExecutorBundle_INTERNAL(
        1, modelPath.c_str(), nva2f::IGeometryExecutor::ExecutionOption::SkinTongue, true, 30, 1, nullptr, nullptr
        )
      );
    ASSERT_NE(bundle, nullptr);
    ASSERT_NE(bundle->GetCudaStream().Data(), nullptr);
    ASSERT_EQ(bundle->GetAudioAccumulator(0).NbAccumulatedSamples(), 0U);
    ASSERT_EQ(bundle->GetEmotionAccumulator(0).GetEmotionSize(), 10U);
    ASSERT_EQ(bundle->GetExecutor().HasExecutionStarted(0), false);
  }
}


TEST(TestCoreParseHelper, DiffusionAnimatorParams) {
  const auto configPath = GetDiffusionPath("model_config_Claire.json");
  float inputStrength;
  nva2f::AnimatorParams animatorParams;
  const auto error = nva2f::ReadAnimatorParams_INTERNAL(configPath.c_str(), inputStrength, animatorParams);
  ASSERT_TRUE(!error);

  EXPECT_EQ(inputStrength, 1.0f);
  EXPECT_EQ(animatorParams.skin.lowerFaceSmoothing, 0.006f);
  EXPECT_EQ(animatorParams.skin.upperFaceSmoothing, 0.001f);
  EXPECT_EQ(animatorParams.skin.lowerFaceStrength, 1.0f);
  EXPECT_EQ(animatorParams.skin.upperFaceStrength, 1.0f);
  EXPECT_EQ(animatorParams.skin.faceMaskLevel, 0.6f);
  EXPECT_EQ(animatorParams.skin.faceMaskSoftness, 0.0085f);
  EXPECT_EQ(animatorParams.skin.skinStrength, 1.0f);
  EXPECT_EQ(animatorParams.skin.blinkStrength, 1.0f);
  EXPECT_EQ(animatorParams.skin.eyelidOpenOffset, 0.0f);
  EXPECT_EQ(animatorParams.skin.lipOpenOffset, 0.0f);
  EXPECT_EQ(animatorParams.skin.blinkOffset, 0.0f);

  EXPECT_EQ(animatorParams.tongue.tongueStrength, 1.3f);
  EXPECT_EQ(animatorParams.tongue.tongueHeightOffset, 0.0f);
  EXPECT_EQ(animatorParams.tongue.tongueDepthOffset, 0.0f);

  EXPECT_EQ(animatorParams.teeth.lowerTeethStrength, 1.0f);
  EXPECT_EQ(animatorParams.teeth.lowerTeethHeightOffset, 0.0f);
  EXPECT_EQ(animatorParams.teeth.lowerTeethDepthOffset, 0.0f);

  EXPECT_EQ(animatorParams.eyes.eyeballsStrength, 1.0f);
  EXPECT_EQ(animatorParams.eyes.saccadeStrength, 0.6f);
  EXPECT_EQ(animatorParams.eyes.rightEyeballRotationOffsetX, 0.0f);
  EXPECT_EQ(animatorParams.eyes.rightEyeballRotationOffsetY, 0.0f);
  EXPECT_EQ(animatorParams.eyes.leftEyeballRotationOffsetX, 0.0f);
  EXPECT_EQ(animatorParams.eyes.leftEyeballRotationOffsetY, 0.0f);
  EXPECT_EQ(animatorParams.eyes.saccadeSeed, 0.0f);
}


TEST(TestCoreParseHelper, DiffusionNetworkInfo) {
  const auto networkInfoPath = GetDiffusionPath("network_info.json");
  auto networkInfo = nva2x::ToUniquePtr(nva2f::ReadDiffusionNetworkInfo_INTERNAL(networkInfoPath.c_str()));
  ASSERT_NE(networkInfo, nullptr);

  EXPECT_EQ(networkInfo->GetNetworkInfo().skinDim, 72006U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().tongueDim, 16806U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().jawDim, 15U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().eyesDim, 4U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().numDiffusionSteps, 2U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().numGruLayers, 2U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().gruLatentDim, 256U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().numFramesLeftTruncate, 15U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().numFramesRightTruncate, 15U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().numFramesCenter, 30U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().bufferLength, 16000U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().paddingLeft, 16000U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().paddingRight, 16000U);
  EXPECT_EQ(networkInfo->GetNetworkInfo().bufferSamplerate, 16000U);

  EXPECT_EQ(networkInfo->GetEmotionsCount(), 10U);
  EXPECT_STREQ(networkInfo->GetEmotionName(0), "amazement");
  EXPECT_STREQ(networkInfo->GetEmotionName(1), "anger");
  EXPECT_STREQ(networkInfo->GetEmotionName(2), "cheekiness");
  EXPECT_STREQ(networkInfo->GetEmotionName(3), "disgust");
  EXPECT_STREQ(networkInfo->GetEmotionName(4), "fear");
  EXPECT_STREQ(networkInfo->GetEmotionName(5), "grief");
  EXPECT_STREQ(networkInfo->GetEmotionName(6), "joy");
  EXPECT_STREQ(networkInfo->GetEmotionName(7), "outofbreath");
  EXPECT_STREQ(networkInfo->GetEmotionName(8), "pain");
  EXPECT_STREQ(networkInfo->GetEmotionName(9), "sadness");

  EXPECT_EQ(networkInfo->GetDefaultEmotion().Size(), 10U);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[0], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[1], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[2], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[3], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[4], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[5], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[6], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[7], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[8], 0.0f);
  EXPECT_EQ(networkInfo->GetDefaultEmotion().Data()[9], 0.0f);

  EXPECT_EQ(networkInfo->GetIdentityLength(), 3U);
  EXPECT_STREQ(networkInfo->GetIdentityName(0), "Claire");
  EXPECT_STREQ(networkInfo->GetIdentityName(1), "James");
  EXPECT_STREQ(networkInfo->GetIdentityName(2), "Mark");
}

TEST(TestCoreParseHelper, DiffusionNetworkInfoError) {
  {
    const auto networkInfoPath = GetDiffusionPath("network_info_error.json");
    ASSERT_FALSE(std::filesystem::exists(networkInfoPath));
    auto networkInfo = nva2f::ReadDiffusionNetworkInfo_INTERNAL(networkInfoPath.c_str());
    ASSERT_EQ(networkInfo, nullptr);
  }
  {
    const auto networkInfoPath = GetDiffusionPath("model.json");
    ASSERT_TRUE(std::filesystem::exists(networkInfoPath));
    auto networkInfo = nva2f::ReadDiffusionNetworkInfo_INTERNAL(networkInfoPath.c_str());
    ASSERT_EQ(networkInfo, nullptr);
  }
}

TEST(TestCoreParseHelper, DiffusionAnimatorData) {
  const auto animatorDataPath = GetDiffusionPath("model_data_Claire.npz");
  auto animatorData = nva2x::ToUniquePtr(nva2f::ReadDiffusionAnimatorData_INTERNAL(animatorDataPath.c_str()));
  ASSERT_NE(animatorData, nullptr);

  const auto animatorDataView = animatorData->GetAnimatorData();
  EXPECT_EQ(animatorDataView.skin.neutralPose.Size(), 72006U);
  EXPECT_EQ(animatorDataView.skin.lipOpenPoseDelta.Size(), 72006U);
  EXPECT_EQ(animatorDataView.skin.eyeClosePoseDelta.Size(), 72006U);
  EXPECT_EQ(animatorDataView.tongue.neutralPose.Size(), 16806);
  EXPECT_EQ(animatorDataView.teeth.neutralJaw.Size(), 15U);
  EXPECT_EQ(animatorDataView.eyes.saccadeRot.Size(), 10000U);
}

TEST(TestCoreParseHelper, DiffusionAnimatorDataNoUnexpectedTensors) {
  for(const auto& filename : {
    "model_data_Claire.npz",
    "model_data_James.npz",
    "model_data_Mark.npz"
  }) {
    const auto animatorDataPath = GetDiffusionPath(filename);
    nva2x::HostTensorDict data;
    ASSERT_TRUE(!data.ReadFromFile(animatorDataPath.c_str()));

    const std::vector<std::string> expectedTensorNames = {
      "neutral_skin",
      "neutral_tongue",
      "neutral_jaw",
      "lip_open_pose_delta",
      "eye_close_pose_delta",
      "saccade_rot_matrix",
    };

    for(const auto& tensorName : expectedTensorNames) {
      ASSERT_NE(data.At(tensorName.c_str()), nullptr)
        << "Failed to read tensor: '" << tensorName
        << "' in file: " << animatorDataPath;
    }
    ASSERT_EQ(data.Size(), expectedTensorNames.size())
      << "Unexpected tensor found in: " << animatorDataPath;
  }
}

TEST(TestCoreParseHelper, DiffusionAnimatorDataError) {
  {
    const auto animatorDataPath = GetDiffusionPath("model_data_Claire_error.npz");
    ASSERT_FALSE(std::filesystem::exists(animatorDataPath));
    auto animatorData = nva2f::ReadDiffusionAnimatorData_INTERNAL(animatorDataPath.c_str());
    ASSERT_EQ(animatorData, nullptr);
  }
  {
    const auto animatorDataPath = GetDiffusionPath("bs_skin_Claire.npz");
    ASSERT_TRUE(std::filesystem::exists(animatorDataPath));
    auto animatorData = nva2f::ReadDiffusionAnimatorData_INTERNAL(animatorDataPath.c_str());
    ASSERT_EQ(animatorData, nullptr);
  }
}

TEST(TestCoreParseHelper, DiffusionGeometryExecutor) {
  const auto modelPath = GetDiffusionPath("model.json");
  auto modelInfo = nva2x::ToUniquePtr(nva2f::ReadDiffusionModelInfo_INTERNAL(modelPath.c_str()));
  ASSERT_NE(modelInfo, nullptr);
  const auto diffusionParams = modelInfo->GetExecutorCreationParameters(
      nva2f::IGeometryExecutor::ExecutionOption::All, 0, false
      );

  auto audioAccumulator = nva2x::ToUniquePtr(nva2x::CreateAudioAccumulator(16000U, 0));
  ASSERT_NE(audioAccumulator, nullptr);
  auto emotionAccumulator = nva2x::ToUniquePtr(
    nva2x::CreateEmotionAccumulator(diffusionParams.networkInfo.emotionLength, 100, 0)
    );
  ASSERT_NE(emotionAccumulator, nullptr);

  nva2f::GeometryExecutorCreationParameters params;
  params.cudaStream = nullptr;
  params.nbTracks = 1;
  const auto sharedAudioAccumulator = audioAccumulator.get();
  params.sharedAudioAccumulators = &sharedAudioAccumulator;
  const auto sharedEmotionAccumulator = emotionAccumulator.get();
  params.sharedEmotionAccumulators = &sharedEmotionAccumulator;

  auto executor = nva2x::ToUniquePtr(nva2f::CreateDiffusionGeometryExecutor_INTERNAL(params, diffusionParams));
  ASSERT_NE(executor, nullptr);
}

TEST(TestCoreParseHelper, DiffusionGeometryExecutorBundle) {
  const auto modelPath = GetDiffusionPath("model.json");
  nva2f::IDiffusionModel::IGeometryModelInfo* modelInfo = nullptr;
  auto bundle = nva2x::ToUniquePtr(
    nva2f::ReadDiffusionGeometryExecutorBundle_INTERNAL(
      1, modelPath.c_str(), nva2f::IGeometryExecutor::ExecutionOption::All, 0, false, &modelInfo
      )
    );
  ASSERT_NE(bundle, nullptr);
  ASSERT_NE(bundle->GetCudaStream().Data(), nullptr);
  ASSERT_EQ(bundle->GetAudioAccumulator(0).NbAccumulatedSamples(), 0U);
  ASSERT_EQ(bundle->GetEmotionAccumulator(0).GetEmotionSize(), 10U);
  ASSERT_EQ(bundle->GetExecutor().HasExecutionStarted(0), false);

  ASSERT_NE(modelInfo, nullptr);
  EXPECT_EQ(modelInfo->GetNetworkInfo().GetEmotionsCount(), 10U);
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(0), "amazement");
  EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(9), "sadness");
}

TEST(TestCoreParseHelper, DiffusionBlendshapeExecutorBundle) {
  const auto modelPath = GetDiffusionPath("model.json");
  {
    nva2f::IDiffusionModel::IGeometryModelInfo* modelInfo = nullptr;
    nva2f::IDiffusionModel::IBlendshapeSolveModelInfo* blendshapeSolveModelInfo = nullptr;
    auto bundle = nva2x::ToUniquePtr(
      nva2f::ReadDiffusionBlendshapeSolveExecutorBundle_INTERNAL(
        1, modelPath.c_str(), nva2f::IGeometryExecutor::ExecutionOption::SkinTongue, false, 0, false, &modelInfo, &blendshapeSolveModelInfo
        )
      );
    ASSERT_NE(bundle, nullptr);
    ASSERT_NE(bundle->GetCudaStream().Data(), nullptr);
    ASSERT_EQ(bundle->GetAudioAccumulator(0).NbAccumulatedSamples(), 0U);
    ASSERT_EQ(bundle->GetEmotionAccumulator(0).GetEmotionSize(), 10U);
    ASSERT_EQ(bundle->GetExecutor().HasExecutionStarted(0), false);

    ASSERT_NE(modelInfo, nullptr);
    EXPECT_EQ(modelInfo->GetNetworkInfo().GetEmotionsCount(), 10U);
    EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(0), "amazement");
    EXPECT_STREQ(modelInfo->GetNetworkInfo().GetEmotionName(9), "sadness");

    ASSERT_NE(blendshapeSolveModelInfo, nullptr);
  }
  {
    auto bundle = nva2x::ToUniquePtr(
      nva2f::ReadDiffusionBlendshapeSolveExecutorBundle_INTERNAL(
        1, modelPath.c_str(), nva2f::IGeometryExecutor::ExecutionOption::SkinTongue, true, 0, false, nullptr, nullptr
        )
      );
    ASSERT_NE(bundle, nullptr);
    ASSERT_NE(bundle->GetCudaStream().Data(), nullptr);
    ASSERT_EQ(bundle->GetAudioAccumulator(0).NbAccumulatedSamples(), 0U);
    ASSERT_EQ(bundle->GetEmotionAccumulator(0).GetEmotionSize(), 10U);
    ASSERT_EQ(bundle->GetExecutor().HasExecutionStarted(0), false);
  }
}


TEST(TestCoreParseHelper, BlendshapeSolverConfig) {
  // Read skin.
  {
    const auto configPath = GetRegressionPath("bs_skin_config.json");
    auto blendshapeSolverConfigLoaded = nva2x::ToUniquePtr(
      nva2f::ReadBlendshapeSolverConfig_INTERNAL(configPath.c_str())
      );
    ASSERT_NE(blendshapeSolverConfigLoaded, nullptr);

    const auto blendshapeSolverParams = blendshapeSolverConfigLoaded->GetBlendshapeSolverParams();
    EXPECT_EQ(blendshapeSolverParams.L1Reg, 0.5f);
    EXPECT_EQ(blendshapeSolverParams.L2Reg, 0.5f);
    EXPECT_EQ(blendshapeSolverParams.SymmetryReg, 100.0f);
    EXPECT_EQ(blendshapeSolverParams.TemporalReg, 0.3f);
    EXPECT_EQ(blendshapeSolverParams.templateBBSize, 53.4505028f);
    EXPECT_EQ(blendshapeSolverParams.tolerance, 1e-10f);

    const auto blendshapeSolverConfig = blendshapeSolverConfigLoaded->GetBlendshapeSolverConfig();
    EXPECT_EQ(blendshapeSolverConfig.numBlendshapes, 52U);
    EXPECT_NE(blendshapeSolverConfig.activePoses, nullptr);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[0], 1);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[1], 0);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[2], 0);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[3], 0);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[4], 0);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[5], 1);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[51], 0);

    EXPECT_NE(blendshapeSolverConfig.cancelPoses, nullptr);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[0], -1);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[1], -1);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[2], -1);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[3], -1);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[4], -1);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[5], -1);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[51], -1);

    EXPECT_NE(blendshapeSolverConfig.symmetryPoses, nullptr);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[0], 0);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[1], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[2], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[3], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[4], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[5], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[6], 1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[7], 0);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[51], -1);

    EXPECT_EQ(blendshapeSolverConfig.multipliers.Size(), 52U);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[0], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[1], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[2], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[3], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[4], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[5], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[14], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[15], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[51], 1.0f);

    EXPECT_EQ(blendshapeSolverConfig.offsets.Size(), 52U);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[0], 0.0f);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[1], 0.0f);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[2], 0.0f);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[3], 0.0f);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[4], 0.0f);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[5], 0.0f);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[51], 0.0f);
  }

  // Read tongue.
  {
    const auto configPath = GetRegressionPath("bs_tongue_config.json");
    auto blendshapeSolverConfigLoaded = nva2x::ToUniquePtr(
      nva2f::ReadBlendshapeSolverConfig_INTERNAL(configPath.c_str())
      );
    ASSERT_NE(blendshapeSolverConfigLoaded, nullptr);

    const auto blendshapeSolverParams = blendshapeSolverConfigLoaded->GetBlendshapeSolverParams();
    EXPECT_EQ(blendshapeSolverParams.L1Reg, 1.0f);
    EXPECT_EQ(blendshapeSolverParams.L2Reg, 3.5f);
    EXPECT_EQ(blendshapeSolverParams.SymmetryReg, 100.0f);
    EXPECT_EQ(blendshapeSolverParams.TemporalReg, 5.0f);
    EXPECT_EQ(blendshapeSolverParams.templateBBSize, 7.9358663f);
    EXPECT_EQ(blendshapeSolverParams.tolerance, 1e-10f);

    const auto blendshapeSolverConfig = blendshapeSolverConfigLoaded->GetBlendshapeSolverConfig();
    EXPECT_EQ(blendshapeSolverConfig.numBlendshapes, 16U);
    EXPECT_NE(blendshapeSolverConfig.activePoses, nullptr);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[0], 1);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[1], 1);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[2], 1);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[3], 1);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[4], 1);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[5], 1);
    EXPECT_EQ(blendshapeSolverConfig.activePoses[15], 1);

    EXPECT_NE(blendshapeSolverConfig.cancelPoses, nullptr);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[0], -1);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[1], -1);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[2], -1);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[3], -1);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[4], -1);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[5], -1);
    EXPECT_EQ(blendshapeSolverConfig.cancelPoses[15], -1);

    EXPECT_NE(blendshapeSolverConfig.symmetryPoses, nullptr);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[0], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[1], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[2], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[3], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[4], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[5], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[6], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[7], -1);
    EXPECT_EQ(blendshapeSolverConfig.symmetryPoses[15], -1);

    EXPECT_EQ(blendshapeSolverConfig.multipliers.Size(), 16U);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[0], 2.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[1], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[2], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[3], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[4], 3.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[5], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[13], 2.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[14], 1.0f);
    EXPECT_EQ(blendshapeSolverConfig.multipliers.Data()[15], 1.0f);

    EXPECT_EQ(blendshapeSolverConfig.offsets.Size(), 16U);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[0], 0.0f);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[1], 0.0f);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[2], 0.0f);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[3], 0.0f);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[4], 0.0f);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[5], 0.0f);
    EXPECT_EQ(blendshapeSolverConfig.offsets.Data()[15], 0.0f);
  }
}

TEST(TestCoreParseHelper, BlendshapeSolverConfigError) {
  {
    const auto configPath = GetRegressionPath("bs_tongue_config_error.json");
    ASSERT_FALSE(std::filesystem::exists(configPath));
    auto blendshapeSolverConfigLoaded = nva2x::ToUniquePtr(
      nva2f::ReadBlendshapeSolverConfig_INTERNAL(configPath.c_str())
      );
    ASSERT_EQ(blendshapeSolverConfigLoaded, nullptr);
  }
  {
    const auto configPath = GetRegressionPath("model.json");
    ASSERT_TRUE(std::filesystem::exists(configPath));
    auto blendshapeSolverConfigLoaded = nva2x::ToUniquePtr(
      nva2f::ReadBlendshapeSolverConfig_INTERNAL(configPath.c_str())
      );
    ASSERT_EQ(blendshapeSolverConfigLoaded, nullptr);
  }
}

TEST(TestCoreParseHelper, BlendshapeSolverData) {
  // Read skin.
  {
    const auto dataPath = GetRegressionPath("bs_skin.npz");
    auto blendshapeSolverData = nva2x::ToUniquePtr(
      nva2f::ReadBlendshapeSolverData_INTERNAL(dataPath.c_str())
      );
    ASSERT_NE(blendshapeSolverData, nullptr);

    const auto blendshapeDataView = blendshapeSolverData->GetBlendshapeSolverDataView();
    EXPECT_EQ(blendshapeDataView.neutralPose.Size(), 61520U * 3);
    EXPECT_EQ(blendshapeDataView.deltaPoses.Size(), 61520U * 3 * 52U);
    EXPECT_NE(blendshapeDataView.poseMask, nullptr);
    EXPECT_EQ(blendshapeDataView.poseMaskSize, 10085U);
    EXPECT_NE(blendshapeDataView.poseNames, nullptr);
    EXPECT_EQ(blendshapeDataView.poseNamesSize, 52U);
    EXPECT_STREQ(blendshapeDataView.poseNames[0], "eyeBlinkLeft");
    EXPECT_STREQ(blendshapeDataView.poseNames[51], "tongueOut");
  }

  // Read tongue.
  {
    const auto dataPath = GetRegressionPath("bs_tongue.npz");
    auto blendshapeSolverData = nva2x::ToUniquePtr(
      nva2f::ReadBlendshapeSolverData_INTERNAL(dataPath.c_str())
      );
    ASSERT_NE(blendshapeSolverData, nullptr);

    const auto blendshapeDataView = blendshapeSolverData->GetBlendshapeSolverDataView();
    EXPECT_EQ(blendshapeDataView.neutralPose.Size(), 5602U * 3);
    EXPECT_EQ(blendshapeDataView.deltaPoses.Size(), 5602U * 3 * 16U);
    EXPECT_EQ(blendshapeDataView.poseMask, nullptr);
    EXPECT_EQ(blendshapeDataView.poseMaskSize, 0U);
    EXPECT_NE(blendshapeDataView.poseNames, nullptr);
    EXPECT_EQ(blendshapeDataView.poseNamesSize, 16U);
    EXPECT_STREQ(blendshapeDataView.poseNames[0], "tongueTipUp");
    EXPECT_STREQ(blendshapeDataView.poseNames[15], "tongueNarrow");
  }
}

TEST(TestCoreParseHelper, BlendshapeSolverDataError) {
  {
    const auto dataPath = GetRegressionPath("bs_arkit_error.npz");
    ASSERT_FALSE(std::filesystem::exists(dataPath));
    auto blendshapeSolverData= nva2x::ToUniquePtr(
      nva2f::ReadBlendshapeSolverData_INTERNAL(dataPath.c_str())
      );
    ASSERT_EQ(blendshapeSolverData, nullptr);
  }
  {
    const auto dataPath = GetRegressionPath("model_data.npz");
    ASSERT_TRUE(std::filesystem::exists(dataPath));
    auto blendshapeSolverData= nva2x::ToUniquePtr(
      nva2f::ReadBlendshapeSolverData_INTERNAL(dataPath.c_str())
      );
    ASSERT_EQ(blendshapeSolverData, nullptr);
  }
}

TEST(TestCoreParseHelper, RegressionBlendshapeSolveExecutor) {
  const auto modelPath = GetRegressionPath("model.json");
  auto modelInfo = nva2x::ToUniquePtr(
    nva2f::ReadRegressionModelInfo_INTERNAL(modelPath.c_str())
    );
  ASSERT_NE(modelInfo, nullptr);
  const auto regressionParams = modelInfo->GetExecutorCreationParameters(
    nva2f::IGeometryExecutor::ExecutionOption::All, 30, 1
    );

  auto audioAccumulator = nva2x::ToUniquePtr(nva2x::CreateAudioAccumulator(16000U, 0));
  ASSERT_NE(audioAccumulator, nullptr);
  auto emotionAccumulator = nva2x::ToUniquePtr(
    nva2x::CreateEmotionAccumulator(regressionParams.networkInfo.explicitEmotionLength, 100, 0)
    );
  ASSERT_NE(emotionAccumulator, nullptr);

  nva2f::GeometryExecutorCreationParameters params;
  params.cudaStream = nullptr;
  params.nbTracks = 1;
  const auto sharedAudioAccumulator = audioAccumulator.get();
  params.sharedAudioAccumulators = &sharedAudioAccumulator;
  const auto sharedEmotionAccumulator = emotionAccumulator.get();
  params.sharedEmotionAccumulators = &sharedEmotionAccumulator;

  auto geometryExecutor = nva2x::ToUniquePtr(
    nva2f::CreateRegressionGeometryExecutor_INTERNAL(params, regressionParams)
    );
  ASSERT_NE(geometryExecutor, nullptr);

  auto blendshapeSolveInfo = nva2x::ToUniquePtr(
    nva2f::ReadRegressionBlendshapeSolveModelInfo_INTERNAL(modelPath.c_str())
    );
  ASSERT_NE(blendshapeSolveInfo, nullptr);
  const auto blendshapeSolveParams = blendshapeSolveInfo->GetExecutorCreationParameters(
      nva2f::IGeometryExecutor::ExecutionOption::SkinTongue
      );
  nva2f::HostBlendshapeSolveExecutorCreationParameters creationParams;
  creationParams.initializationSkinParams = blendshapeSolveParams.initializationSkinParams;
  creationParams.initializationTongueParams = blendshapeSolveParams.initializationTongueParams;
  auto executor = nva2x::ToUniquePtr(
    nva2f::CreateHostBlendshapeSolveExecutor_INTERNAL(geometryExecutor.release(), creationParams)
    );
  ASSERT_NE(executor, nullptr);
}

TEST(TestCoreParseHelper, DiffusionBlendshapeSolveExecutor) {
  const auto modelPath = GetDiffusionPath("model.json");
  auto modelInfo = nva2x::ToUniquePtr(
    nva2f::ReadDiffusionModelInfo_INTERNAL(modelPath.c_str())
    );
  ASSERT_NE(modelInfo, nullptr);
  const auto diffusionParams = modelInfo->GetExecutorCreationParameters(
      nva2f::IGeometryExecutor::ExecutionOption::All, 0, false
      );

  auto audioAccumulator = nva2x::ToUniquePtr(nva2x::CreateAudioAccumulator(16000U, 0));
  ASSERT_NE(audioAccumulator, nullptr);
  auto emotionAccumulator = nva2x::ToUniquePtr(
    nva2x::CreateEmotionAccumulator(diffusionParams.networkInfo.emotionLength, 100, 0)
    );
  ASSERT_NE(emotionAccumulator, nullptr);

  nva2f::GeometryExecutorCreationParameters params;
  params.cudaStream = nullptr;
  params.nbTracks = 1;
  const auto sharedAudioAccumulator = audioAccumulator.get();
  params.sharedAudioAccumulators = &sharedAudioAccumulator;
  const auto sharedEmotionAccumulator = emotionAccumulator.get();
  params.sharedEmotionAccumulators = &sharedEmotionAccumulator;

  auto geometryExecutor = nva2x::ToUniquePtr(
    nva2f::CreateDiffusionGeometryExecutor_INTERNAL(params, diffusionParams)
    );
  ASSERT_NE(geometryExecutor, nullptr);

  auto blendshapeSolveInfo = nva2x::ToUniquePtr(
    nva2f::ReadDiffusionBlendshapeSolveModelInfo_INTERNAL(modelPath.c_str())
    );
  ASSERT_NE(blendshapeSolveInfo, nullptr);
  const auto blendshapeSolveParams = blendshapeSolveInfo->GetExecutorCreationParameters(
      nva2f::IGeometryExecutor::ExecutionOption::SkinTongue, 0
      );
  nva2f::HostBlendshapeSolveExecutorCreationParameters creationParams;
  creationParams.initializationSkinParams = blendshapeSolveParams.initializationSkinParams;
  creationParams.initializationTongueParams = blendshapeSolveParams.initializationTongueParams;
  auto executor = nva2x::ToUniquePtr(
    nva2f::CreateHostBlendshapeSolveExecutor_INTERNAL(geometryExecutor.release(), creationParams)
    );
  ASSERT_NE(executor, nullptr);
}
