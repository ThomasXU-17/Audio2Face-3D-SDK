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
#include "audio2x/internal/audio_utils.h"
#include "audio2x/internal/audio2x.h"
#include "audio2face/internal/executor_regression.h"
#include "audio2face/internal/executor_diffusion.h"

#include <gtest/gtest.h>

#include <any>

// For IGeometryExecutor::ExecutionOption operators.
using namespace ::nva2f::internal;

namespace {

  std::vector<float> GetAudio(const char* filename) {
    auto audio = nva2x::get_file_wav_content(filename);
    EXPECT_TRUE(audio.has_value());
    return audio.value();
  }

  std::vector<float> GetAudio() {
    constexpr char filename[] = TEST_DATA_DIR "sample-data/audio_4sec_16k_s16le.wav";
    return GetAudio(filename);
  }

  template <typename T>
  struct TestData {
    T data;
    std::any ownedData;
  };

  constexpr std::size_t kNbTracks = 8;
  constexpr std::size_t kEmotionSize = 10;

  TestData<nva2f::GeometryExecutorCreationParameters> GetParameters(std::size_t emotionSize) {

    auto cudaStream = nva2x::ToSharedPtr(nva2x::internal::CreateCudaStream());
    EXPECT_TRUE(cudaStream);

    auto audioAccumulators = std::make_shared<std::vector<nva2x::UniquePtr<nva2x::IAudioAccumulator>>>(kNbTracks);
    auto emotionAccumulators = std::make_shared<std::vector<nva2x::UniquePtr<nva2x::IEmotionAccumulator>>>(kNbTracks);
    auto sharedAudioAccumulators = std::make_shared<std::vector<const nva2x::IAudioAccumulator*>>(kNbTracks);
    auto sharedEmotionAccumulators = std::make_shared<std::vector<const nva2x::IEmotionAccumulator*>>(kNbTracks);
    for (std::size_t i = 0; i < kNbTracks; ++i) {
      (*audioAccumulators)[i] = nva2x::ToUniquePtr(nva2x::internal::CreateAudioAccumulator(16000, 0));
      EXPECT_TRUE((*audioAccumulators)[i]);
      (*emotionAccumulators)[i] = nva2x::ToUniquePtr(nva2x::internal::CreateEmotionAccumulator(emotionSize, 300, 0));
      EXPECT_TRUE((*emotionAccumulators)[i]);
      (*sharedAudioAccumulators)[i] = (*audioAccumulators)[i].get();
      (*sharedEmotionAccumulators)[i] = (*emotionAccumulators)[i].get();
    }

    TestData<nva2f::GeometryExecutorCreationParameters> params;
    params.data.cudaStream = cudaStream->Data();
    params.data.nbTracks = kNbTracks;
    params.data.sharedAudioAccumulators = sharedAudioAccumulators->data();
    params.data.sharedEmotionAccumulators = sharedEmotionAccumulators->data();

    params.ownedData = std::vector<std::any>{
      std::move(cudaStream),
      std::move(audioAccumulators),
      std::move(emotionAccumulators),
      std::move(sharedAudioAccumulators),
      std::move(sharedEmotionAccumulators),
    };

    return params;
  }

  TestData<nva2f::IRegressionModel::GeometryExecutorCreationParameters> GetRegressionParameters() {
    constexpr char filename[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
    auto modelInfo = nva2x::ToSharedPtr(nva2f::ReadRegressionModelInfo_INTERNAL(filename));
    EXPECT_TRUE(modelInfo);
    EXPECT_EQ(modelInfo->GetNetworkInfo().GetEmotionsCount(), kEmotionSize);

    const auto regressionParams = modelInfo->GetExecutorCreationParameters(
      nva2f::IGeometryExecutor::ExecutionOption::All, 60, 1
      );

    return {regressionParams, std::move(modelInfo)};
  }

  TestData<nva2f::IDiffusionModel::GeometryExecutorCreationParameters> GetDiffusionParameters() {
    constexpr char filename[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json";
    auto modelInfo = nva2x::ToSharedPtr(nva2f::ReadDiffusionModelInfo_INTERNAL(filename));
    EXPECT_TRUE(modelInfo);
    EXPECT_EQ(modelInfo->GetNetworkInfo().GetEmotionsCount(), kEmotionSize);

    const auto diffusionParams = modelInfo->GetExecutorCreationParameters(
      nva2f::IGeometryExecutor::ExecutionOption::All, 0, false
      );

    return {diffusionParams, std::move(modelInfo)};
  }

  template <typename BundleType>
  void InitBundle(BundleType& bundle) {
    const auto nbTracks = bundle.GetExecutor().GetNbTracks();
    const auto audio = GetAudio();
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
      std::vector<float> emotion(bundle.GetEmotionAccumulator(trackIndex).GetEmotionSize(), 0.0f);
      EXPECT_EQ(bundle.GetEmotionAccumulator(trackIndex).GetEmotionSize(), kEmotionSize);
      EXPECT_TRUE(!bundle.GetEmotionAccumulator(trackIndex).Reset());
      EXPECT_TRUE(!bundle.GetEmotionAccumulator(trackIndex).Accumulate(
        0, nva2x::ToConstView(emotion), bundle.GetCudaStream().Data()
        ));
      EXPECT_TRUE(!bundle.GetEmotionAccumulator(trackIndex).Close());

      auto& audioAccumulator = bundle.GetAudioAccumulator(trackIndex);
      EXPECT_TRUE(!audioAccumulator.Reset());
      EXPECT_TRUE(!audioAccumulator.Accumulate(nva2x::ToConstView(audio), bundle.GetCudaStream().Data()));
      EXPECT_TRUE(!audioAccumulator.Close());
    }
  }

  nva2x::UniquePtr<nva2f::IGeometryExecutorBundle> CreateGeometryExecutorRegressionBundle(
    nva2f::IGeometryExecutor::ExecutionOption executionOption, bool init
  ) {
    constexpr char filename[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
    auto bundle = nva2x::ToUniquePtr(
      nva2f::ReadRegressionGeometryExecutorBundle_INTERNAL(
        kNbTracks,
        filename,
        executionOption,
        60, 1,
        nullptr
        )
      );
    EXPECT_TRUE(bundle);

    if (init) {
      InitBundle(*bundle);
    }

    return bundle;
  }

  nva2x::UniquePtr<nva2f::IGeometryExecutorBundle> CreateGeometryExecutorDiffusionBundle(
    nva2f::IGeometryExecutor::ExecutionOption executionOption, bool init
  ) {
    constexpr char filename[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json";
    auto bundle = nva2x::ToUniquePtr(
      nva2f::ReadDiffusionGeometryExecutorBundle_INTERNAL(
        kNbTracks,
        filename,
        executionOption,
        0,
        false,
        nullptr
        )
      );
    EXPECT_TRUE(bundle);

    if (init) {
      InitBundle(*bundle);
    }

    return bundle;
  }

  nva2x::UniquePtr<nva2f::IBlendshapeExecutorBundle> CreateBlendshapeSolveExecutorRegressionBundle(
    bool useGpuSolver, nva2f::IGeometryExecutor::ExecutionOption executionOption, bool init
  ) {
    constexpr char filename[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
    auto bundle = nva2x::ToUniquePtr(
      nva2f::ReadRegressionBlendshapeSolveExecutorBundle_INTERNAL(
        kNbTracks,
        filename,
        executionOption,
        useGpuSolver,
        60, 1,
        nullptr,
        nullptr
        )
      );
    EXPECT_TRUE(bundle);

    if (init) {
      InitBundle(*bundle);
    }

    return bundle;
  }

  nva2x::UniquePtr<nva2f::IBlendshapeExecutorBundle> CreateBlendshapeSolveExecutorDiffusionBundle(
    bool useGpuSolver, nva2f::IGeometryExecutor::ExecutionOption executionOption, bool init
  ) {
    constexpr char filename[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json";
    auto bundle = nva2x::ToUniquePtr(
      nva2f::ReadDiffusionBlendshapeSolveExecutorBundle_INTERNAL(
        kNbTracks,
        filename,
        executionOption,
        useGpuSolver,
        0,
        false,
        nullptr,
        nullptr
        )
      );
    EXPECT_TRUE(bundle);

    if (init) {
      InitBundle(*bundle);
    }

    return bundle;
  }

}


TEST(TestCoreExecutor, SanityGeometryExecutor) {
  // Compute expected number of frames.
  const auto audio = GetAudio();
  const auto nbFrames = (audio.size() * 60 + 16000 - 1) / 16000;

  // Record timestamps to compare executors.
  std::vector<std::vector<nva2f::IGeometryExecutor::timestamp_t>> expectedTimestamps;

  for (const auto createFunc : {&CreateGeometryExecutorRegressionBundle, &CreateGeometryExecutorDiffusionBundle}) {
    auto bundle = createFunc(nva2f::IGeometryExecutor::ExecutionOption::All, true);
    ASSERT_TRUE(bundle);
    const auto nbTracks = bundle->GetExecutor().GetNbTracks();

    expectedTimestamps.resize(nbTracks);

    // Validate executors are at 60 FPS.
    std::size_t frameRateNumerator = 0;
    std::size_t frameRateDenominator = 0;
    bundle->GetExecutor().GetFrameRate(frameRateNumerator, frameRateDenominator);
    ASSERT_EQ(frameRateNumerator, 60);
    ASSERT_EQ(frameRateDenominator, 1);

    // Check the number of expected frames is the right one.
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
      ASSERT_EQ(bundle->GetExecutor().GetTotalNbFrames(trackIndex), nbFrames);
    }

    // Not having a callback should trigger an error, even if there is available data.
    ASSERT_LT(0, nva2x::GetNbReadyTracks(bundle->GetExecutor()));
    ASSERT_TRUE(bundle->GetExecutor().Execute(nullptr));

    // Count the actual number of executed frames.
    struct CallbackData {
      std::vector<std::size_t> nbExecutedFrames;
      nva2f::IGeometryExecutor* executor;
      const std::vector<std::vector<nva2f::IGeometryExecutor::timestamp_t>>* expectedTimestamps;
      std::vector<std::vector<nva2f::IGeometryExecutor::timestamp_t>> timestamps;
    };
    CallbackData callbackData;
    callbackData.nbExecutedFrames.resize(nbTracks);
    callbackData.executor = &bundle->GetExecutor();
    callbackData.timestamps.resize(nbTracks);
    callbackData.expectedTimestamps = &expectedTimestamps;
    auto callback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
      auto* callbackData = static_cast<CallbackData*>(userdata);
      const auto trackIndex = results.trackIndex;

      // Test that the received timestamp matches the one computed by the executor on query.
      EXPECT_EQ(callbackData->executor->GetFrameTimestamp(callbackData->nbExecutedFrames[trackIndex]), results.timeStampCurrentFrame);

      // Test that all executors have the same timestamps.
      if ((*callbackData->expectedTimestamps)[trackIndex].empty()) {
        // Just collecting time stamps, it's the first executor we run.
        callbackData->timestamps[trackIndex].emplace_back(results.timeStampCurrentFrame);
      } else {
        // Compare with expected timestamps.
        EXPECT_EQ(callbackData->timestamps[trackIndex].size(), 0);
        EXPECT_EQ((*callbackData->expectedTimestamps)[trackIndex][callbackData->nbExecutedFrames[trackIndex]], results.timeStampCurrentFrame);
      }

      ++callbackData->nbExecutedFrames[trackIndex];
      return true;
    };
    // We should be able to set a null callback.
    ASSERT_TRUE(!bundle->GetExecutor().SetResultsCallback(nullptr, nullptr));
    // We shouldn't be able to set a null callback with non-null userdata.
    ASSERT_TRUE(bundle->GetExecutor().SetResultsCallback(nullptr, &callbackData));
    ASSERT_TRUE(!bundle->GetExecutor().SetResultsCallback(callback, &callbackData));

    while (nva2x::GetNbReadyTracks(bundle->GetExecutor()) > 0) {
      std::size_t nbExecutedTracks = 0;
      ASSERT_TRUE(!bundle->GetExecutor().Execute(&nbExecutedTracks));
      ASSERT_EQ(nbExecutedTracks, nbTracks);

      for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        const auto timestamp = bundle->GetExecutor().GetNextEmotionTimestampToRead(trackIndex);
        ASSERT_TRUE(!bundle->GetEmotionAccumulator(trackIndex).DropEmotionsBefore(timestamp));

        const auto sample = bundle->GetExecutor().GetNextAudioSampleToRead(trackIndex);
        ASSERT_TRUE(!bundle->GetAudioAccumulator(trackIndex).DropSamplesBefore(sample));
      }
    }

    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
      ASSERT_EQ(bundle->GetExecutor().GetTotalNbFrames(trackIndex), nbFrames);
      ASSERT_EQ(bundle->GetExecutor().GetNbAvailableExecutions(trackIndex), 0);
      ASSERT_EQ(callbackData.nbExecutedFrames[trackIndex], nbFrames);
    }

    expectedTimestamps = std::move(callbackData.timestamps);
  }
}

TEST(TestCoreExecutor, SanityBlendshapeExecutor) {
  // Compute expected number of frames.
  const auto audio = GetAudio();
  const auto nbFrames = (audio.size() * 60 + 16000 - 1) / 16000;

  // Record timestamps to compare executors.
  std::vector<std::vector<nva2f::IBlendshapeExecutor::timestamp_t>> expectedTimestamps;

  for (const auto createFunc : {&CreateBlendshapeSolveExecutorRegressionBundle, &CreateBlendshapeSolveExecutorDiffusionBundle}) {
    for (const auto useGpuSolver : {false, true}) {
      auto bundle = createFunc(useGpuSolver, nva2f::IGeometryExecutor::ExecutionOption::Skin, true);
      ASSERT_TRUE(bundle);
      const auto nbTracks = bundle->GetExecutor().GetNbTracks();

      expectedTimestamps.resize(nbTracks);

      // Validate executors are at 60 FPS.
      std::size_t frameRateNumerator = 0;
      std::size_t frameRateDenominator = 0;
      bundle->GetExecutor().GetFrameRate(frameRateNumerator, frameRateDenominator);
      ASSERT_EQ(frameRateNumerator, 60);
      ASSERT_EQ(frameRateDenominator, 1);

      // Check the number of expected frames is the right one.
      for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        ASSERT_EQ(bundle->GetExecutor().GetTotalNbFrames(trackIndex), nbFrames);
      }

      // Not having a callback should trigger an error, even if there is available data.
      ASSERT_LT(0, nva2x::GetNbReadyTracks(bundle->GetExecutor()));
      ASSERT_TRUE(bundle->GetExecutor().Execute(nullptr));

      // Count the actual number of executed frames.
      struct CallbackData {
        std::vector<std::size_t> nbExecutedFrames;
        nva2f::IBlendshapeExecutor* executor;
        const std::vector<std::vector<nva2f::IBlendshapeExecutor::timestamp_t>>* expectedTimestamps;
        std::vector<std::vector<nva2f::IBlendshapeExecutor::timestamp_t>> timestamps;
      };
      CallbackData callbackData;
      callbackData.nbExecutedFrames.resize(nbTracks);
      callbackData.executor = &bundle->GetExecutor();
      callbackData.timestamps.resize(nbTracks);
      callbackData.expectedTimestamps = &expectedTimestamps;
      static constexpr auto callback = [](void* userdata, const auto& results) {
        auto* callbackData = static_cast<CallbackData*>(userdata);
        const auto trackIndex = results.trackIndex;

        // Test that the received timestamp matches the one computed by the executor on query.
        EXPECT_EQ(callbackData->executor->GetFrameTimestamp(callbackData->nbExecutedFrames[trackIndex]), results.timeStampCurrentFrame);

        // Test that all executors have the same timestamps.
        if ((*callbackData->expectedTimestamps)[trackIndex].empty()) {
          // Just collecting time stamps, it's the first executor we run.
          callbackData->timestamps[trackIndex].emplace_back(results.timeStampCurrentFrame);
        } else {
          // Compare with expected timestamps.
          EXPECT_EQ(callbackData->timestamps[trackIndex].size(), 0);
          EXPECT_EQ((*callbackData->expectedTimestamps)[trackIndex][callbackData->nbExecutedFrames[trackIndex]], results.timeStampCurrentFrame);
        }

        ++callbackData->nbExecutedFrames[trackIndex];
        return true;
      };
      if (useGpuSolver) {
        // We should be able to set a null callback.
        using null_callback_t = nva2f::IBlendshapeExecutor::device_results_callback_t;
        ASSERT_TRUE(!bundle->GetExecutor().SetResultsCallback(null_callback_t{}, nullptr));
        // We shouldn't be able to set a null callback with non-null userdata.
        ASSERT_TRUE(bundle->GetExecutor().SetResultsCallback(null_callback_t{}, &callbackData));

        ASSERT_TRUE(!bundle->GetExecutor().SetResultsCallback(callback, &callbackData));
      } else {
        // We should be able to set a null callback.
        using null_callback_t = nva2f::IBlendshapeExecutor::host_results_callback_t;
        ASSERT_TRUE(!bundle->GetExecutor().SetResultsCallback(null_callback_t{}, nullptr));
        // We shouldn't be able to set a null callback with non-null userdata.
        ASSERT_TRUE(bundle->GetExecutor().SetResultsCallback(null_callback_t{}, &callbackData));

        auto hostCallback = [](
          void* userdata, const nva2f::IBlendshapeExecutor::HostResults& results, std::error_code errorCode
          ) {
          EXPECT_TRUE(!errorCode);
          callback(userdata, results);
        };
        ASSERT_TRUE(!bundle->GetExecutor().SetResultsCallback(hostCallback, &callbackData));
      }

      while (nva2x::GetNbReadyTracks(bundle->GetExecutor()) > 0) {
        std::size_t nbExecutedTracks = 0;
        ASSERT_TRUE(!bundle->GetExecutor().Execute(&nbExecutedTracks));
        ASSERT_EQ(nbExecutedTracks, nbTracks);
      }
      for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        ASSERT_TRUE(!bundle->GetExecutor().Wait(trackIndex));
      }

      for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        ASSERT_EQ(bundle->GetExecutor().GetTotalNbFrames(trackIndex), nbFrames);
        ASSERT_EQ(bundle->GetExecutor().GetNbAvailableExecutions(trackIndex), 0);
        ASSERT_EQ(callbackData.nbExecutedFrames[trackIndex], nbFrames);
      }

      expectedTimestamps = std::move(callbackData.timestamps);

      // Reset everything and add the emotion callback.
      for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        ASSERT_TRUE(!bundle->GetExecutor().Reset(trackIndex));
      }

      callbackData = {};
      callbackData.nbExecutedFrames.resize(nbTracks);
      callbackData.executor = &bundle->GetExecutor();
      callbackData.timestamps.resize(nbTracks);
      callbackData.expectedTimestamps = &expectedTimestamps;

      CallbackData emotionCallbackData;
      emotionCallbackData.nbExecutedFrames.resize(nbTracks);
      emotionCallbackData.executor = &bundle->GetExecutor();
      emotionCallbackData.timestamps.resize(nbTracks);
      emotionCallbackData.expectedTimestamps = &expectedTimestamps;
      ASSERT_TRUE(!nva2f::SetExecutorGeometryResultsCallback_INTERNAL(bundle->GetExecutor(), callback, &emotionCallbackData));

      while (nva2x::GetNbReadyTracks(bundle->GetExecutor()) > 0) {
        std::size_t nbExecutedTracks = 0;
        ASSERT_TRUE(!bundle->GetExecutor().Execute(&nbExecutedTracks));
        ASSERT_EQ(nbExecutedTracks, nbTracks);
      }
      for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        ASSERT_TRUE(!bundle->GetExecutor().Wait(trackIndex));
      }

      for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        ASSERT_EQ(bundle->GetExecutor().GetTotalNbFrames(trackIndex), nbFrames);
        ASSERT_EQ(bundle->GetExecutor().GetNbAvailableExecutions(trackIndex), 0);
        ASSERT_EQ(callbackData.nbExecutedFrames[trackIndex], nbFrames);
        ASSERT_EQ(emotionCallbackData.nbExecutedFrames[trackIndex], nbFrames);
      }
    }
  }
}

TEST(TestCoreExecutor, ExecuteErrors) {
  // Compute expected number of frames.
  const auto audio = GetAudio();

  for (const auto createFunc : {&CreateGeometryExecutorRegressionBundle, &CreateGeometryExecutorDiffusionBundle}) {
    auto bundle = createFunc(nva2f::IGeometryExecutor::ExecutionOption::All, true);
    ASSERT_TRUE(bundle);
    const auto nbTracks = bundle->GetExecutor().GetNbTracks();

    // Send empty callback.
    auto callback = [](void*, const nva2f::IGeometryExecutor::Results&) {
      return true;
    };
    ASSERT_TRUE(!bundle->GetExecutor().SetResultsCallback(callback, nullptr));

    // Reset the audio to have empty audio.
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
      bundle->GetAudioAccumulator(trackIndex).Reset();
    }

    auto runEverything = [executor = &bundle->GetExecutor(), nbTracks]() {
      while (nva2x::GetNbReadyTracks(*executor) > 0) {
        std::size_t nbExecutedTracks = 0;
        ASSERT_TRUE(!executor->Execute(&nbExecutedTracks));
        ASSERT_EQ(nbExecutedTracks, nbTracks);
      }
      EXPECT_EQ(nva2x::GetNbReadyTracks(*executor), 0);
    };

    // Make sure no execution is available.
    // Diffusion model will have available execution because of padding.
    runEverything();

    // Make sure running execution will fail.
    std::size_t nbExecutedTracks = 0;
    ASSERT_TRUE(bundle->GetExecutor().Execute(&nbExecutedTracks));
    ASSERT_EQ(nbExecutedTracks, 0);

    // Add the audio, run everything to be at the end.
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
      ASSERT_TRUE(!bundle->GetAudioAccumulator(trackIndex).Accumulate(nva2x::ToConstView(audio), bundle->GetCudaStream().Data()));
    }
    runEverything();

    // Make sure running execution will fail.
    ASSERT_TRUE(bundle->GetExecutor().Execute(&nbExecutedTracks));
    ASSERT_EQ(nbExecutedTracks, 0);
    EXPECT_EQ(nva2x::GetNbReadyTracks(bundle->GetExecutor()), 0);

    // Close and continue.
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
      ASSERT_TRUE(!bundle->GetAudioAccumulator(trackIndex).Close());
    }
    runEverything();
    // Make sure running execution will fail.
    ASSERT_TRUE(bundle->GetExecutor().Execute(&nbExecutedTracks));
    ASSERT_EQ(nbExecutedTracks, 0);
    EXPECT_EQ(nva2x::GetNbReadyTracks(bundle->GetExecutor()), 0);
  }
}

TEST(TestCoreExecutor, LoadError) {
  const auto regressionParams = GetRegressionParameters();
  const auto diffusionParams = GetDiffusionParameters();
  auto test_creation = [&regressionParams, &diffusionParams](
    const nva2f::GeometryExecutorCreationParameters& params
    ) {
    auto regressionExecutor = nva2x::ToUniquePtr(
      nva2f::CreateRegressionGeometryExecutor_INTERNAL(params, regressionParams.data)
    );
    const bool regressionSuccess = (regressionExecutor != nullptr);

    auto diffusionExecutor = nva2x::ToUniquePtr(
      nva2f::CreateDiffusionGeometryExecutor_INTERNAL(params, diffusionParams.data)
    );
    const bool diffusionSuccess = (diffusionExecutor != nullptr);

    EXPECT_EQ(regressionSuccess, diffusionSuccess);
    return regressionSuccess;
  };

  // Test various combinations of parameters and how creation handles it.
  using test_func_t = std::function<void(nva2f::GeometryExecutorCreationParameters&)>;
  std::vector<test_func_t> test_funcs = {
    [&](nva2f::GeometryExecutorCreationParameters& params) {
      // No change, should work.
      EXPECT_TRUE(test_creation(params));
    },
    [&](nva2f::GeometryExecutorCreationParameters& params) {
      // Default cuda stream, should work.
      params.cudaStream = nullptr;
      EXPECT_TRUE(test_creation(params));
    },
    [&](nva2f::GeometryExecutorCreationParameters& params) {
      // 0 track, should fail.
      params.nbTracks = 0;
      EXPECT_FALSE(test_creation(params));
    },
    [&](nva2f::GeometryExecutorCreationParameters& params) {
      // Too many tracks, should fail.
      params.nbTracks = 2048;
      EXPECT_FALSE(test_creation(params));
    },
    [&](nva2f::GeometryExecutorCreationParameters& params) {
      // No audio accumulators, should fail.
      params.sharedAudioAccumulators = nullptr;
      EXPECT_FALSE(test_creation(params));
    },
    [&](nva2f::GeometryExecutorCreationParameters& params) {
      // Missing audio accumulators, should fail.
      const_cast<const nva2x::IAudioAccumulator*&>(params.sharedAudioAccumulators[1]) = nullptr;
      EXPECT_FALSE(test_creation(params));
    },
    [&](nva2f::GeometryExecutorCreationParameters& params) {
      // No emotion accumulators, should fail.
      params.sharedEmotionAccumulators = nullptr;
      EXPECT_FALSE(test_creation(params));
    },
    [&](nva2f::GeometryExecutorCreationParameters& params) {
      // Missing emotion accumulators, should fail.
      const_cast<const nva2x::IEmotionAccumulator*&>(params.sharedEmotionAccumulators[1]) = nullptr;
      EXPECT_FALSE(test_creation(params));
    },
    [&](nva2f::GeometryExecutorCreationParameters& params) {
      // Emotion accumulator of the wrong size, should fail.
      auto accumulator = nva2x::ToUniquePtr(nva2x::internal::CreateEmotionAccumulator(11, 300, 0));
      EXPECT_TRUE(accumulator);
      const_cast<const nva2x::IEmotionAccumulator*&>(params.sharedEmotionAccumulators[1]) = accumulator.get();
      EXPECT_FALSE(test_creation(params));
    },
  };

  for (std::size_t i = 0; i < test_funcs.size(); ++i) {
    const auto& test_func = test_funcs[i];
    auto params = GetParameters(kEmotionSize);
    test_func(params.data);
  }
}

TEST(TestCoreExecutor, LoadErrorRegression) {
  const auto defaultParams = GetParameters(kEmotionSize);
  auto test_creation = [&defaultParams](
    const nva2f::IRegressionModel::GeometryExecutorCreationParameters& regressionParams,
    const nva2f::GeometryExecutorCreationParameters* params = nullptr
    ) {
    if (!params) {
      params = &defaultParams.data;
    }

    auto regressionExecutor = nva2x::ToUniquePtr(
      nva2f::CreateRegressionGeometryExecutor_INTERNAL(*params, regressionParams)
    );
    const bool regressionSuccess = (regressionExecutor != nullptr);
    return regressionSuccess;
  };

  // Test various combinations of parameters and how creation handles it.
  using test_func_t = std::function<void(nva2f::IRegressionModel::GeometryExecutorCreationParameters&)>;
  std::vector<test_func_t> test_funcs = {
    [&](auto& params) {
      // No change, should work.
      EXPECT_TRUE(test_creation(params));
    },
    // Mess with the network info data.
    [&](auto& params) {
      params.networkInfo.implicitEmotionLength *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.explicitEmotionLength *= 2;
      const auto adjustedParams = GetParameters(params.networkInfo.explicitEmotionLength);
      EXPECT_FALSE(test_creation(params, &adjustedParams.data));
    },
    [&](auto& params) {
      params.networkInfo.numShapesSkin *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.numShapesTongue *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.resultSkinSize *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.resultTongueSize *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.resultJawSize *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.resultEyesSize *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.bufferLength *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    // Mess with the rest of the initialization parameters.
    [&](auto& params) {
      params.networkData = nullptr;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      // CUDA engine deserialization throws an execption that reaches a noexcept function
      // when the network data size is half the right size or 1.
      // Don't test that case here.
      // Expected to be fixed in a future release of TensorRT.
      params.networkDataSize = 0;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.emotionDatabase = nullptr;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.sourceShot = nullptr;
      EXPECT_FALSE(test_creation(params));
      params.sourceShot = "fake_shot_that_does_not_exist";
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.sourceFrame = 2048;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.frameRateNumerator = 0;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.frameRateDenominator = 0;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.initializationSkinParams = nullptr;
      EXPECT_TRUE(test_creation(params));
    },
    [&](auto& params) {
      params.initializationTongueParams = nullptr;
      EXPECT_TRUE(test_creation(params));
    },
    [&](auto& params) {
      params.initializationTeethParams = nullptr;
      EXPECT_TRUE(test_creation(params));
    },
    [&](auto& params) {
      params.initializationEyesParams = nullptr;
      EXPECT_TRUE(test_creation(params));
    },
    [&](auto& params) {
      auto skinParams = *params.initializationSkinParams;
      params.initializationSkinParams = &skinParams;
      skinParams.data.neutralPose = skinParams.data.neutralPose.View(0, skinParams.data.neutralPose.Size() / 2);
      EXPECT_FALSE(test_creation(params));
      skinParams.data.neutralPose = skinParams.data.neutralPose.View(0, 1);
      EXPECT_FALSE(test_creation(params));
      skinParams.data.neutralPose = {};
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      auto skinParams = *params.initializationSkinParams;
      params.initializationSkinParams = &skinParams;
      skinParams.data.lipOpenPoseDelta = skinParams.data.lipOpenPoseDelta.View(0, skinParams.data.lipOpenPoseDelta.Size() / 2);
      EXPECT_FALSE(test_creation(params));
      skinParams.data.lipOpenPoseDelta = skinParams.data.lipOpenPoseDelta.View(0, 1);
      EXPECT_FALSE(test_creation(params));
      skinParams.data.lipOpenPoseDelta = {};
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      auto skinParams = *params.initializationSkinParams;
      params.initializationSkinParams = &skinParams;
      skinParams.data.eyeClosePoseDelta = skinParams.data.eyeClosePoseDelta.View(0, skinParams.data.eyeClosePoseDelta.Size() / 2);
      EXPECT_FALSE(test_creation(params));
      skinParams.data.eyeClosePoseDelta = skinParams.data.eyeClosePoseDelta.View(0, 1);
      EXPECT_FALSE(test_creation(params));
      skinParams.data.eyeClosePoseDelta = {};
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      auto tongueParams = *params.initializationTongueParams;
      params.initializationTongueParams = &tongueParams;
      tongueParams.data.neutralPose = tongueParams.data.neutralPose.View(0, tongueParams.data.neutralPose.Size() / 2);
      EXPECT_FALSE(test_creation(params));
      tongueParams.data.neutralPose = tongueParams.data.neutralPose.View(0, 1);
      EXPECT_FALSE(test_creation(params));
      tongueParams.data.neutralPose = {};
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      auto skinParams = *params.initializationSkinParams;
      params.initializationSkinParams = &skinParams;
      skinParams.pcaData.shapesMatrix = skinParams.pcaData.shapesMatrix.View(0, skinParams.pcaData.shapesMatrix.Size() / 2);
      EXPECT_FALSE(test_creation(params));
      skinParams.pcaData.shapesMatrix = skinParams.pcaData.shapesMatrix.View(0, 1);
      EXPECT_FALSE(test_creation(params));
      skinParams.pcaData.shapesMatrix = {};
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      auto tongueParams = *params.initializationTongueParams;
      params.initializationTongueParams = &tongueParams;
      tongueParams.pcaData.shapesMatrix = tongueParams.pcaData.shapesMatrix.View(0, tongueParams.pcaData.shapesMatrix.Size() / 2);
      EXPECT_FALSE(test_creation(params));
      tongueParams.pcaData.shapesMatrix = tongueParams.pcaData.shapesMatrix.View(0, 1);
      EXPECT_FALSE(test_creation(params));
      tongueParams.pcaData.shapesMatrix = {};
      EXPECT_FALSE(test_creation(params));
    },
  };

  const auto regressionParams = GetRegressionParameters();
  for (std::size_t i = 0; i < test_funcs.size(); ++i) {
    const auto& test_func = test_funcs[i];
    auto paramsToTest = regressionParams.data;
    test_func(paramsToTest);
  }
}

TEST(TestCoreExecutor, LoadErrorDiffusion) {
  const auto defaultParams = GetParameters(kEmotionSize);
  auto test_creation = [&defaultParams](
    const nva2f::IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams,
    const nva2f::GeometryExecutorCreationParameters* params = nullptr
    ) {
    if (!params) {
      params = &defaultParams.data;
    }

    auto diffusionExecutor = nva2x::ToUniquePtr(
      nva2f::CreateDiffusionGeometryExecutor_INTERNAL(*params, diffusionParams)
    );
    const bool diffusionSuccess = (diffusionExecutor != nullptr);
    return diffusionSuccess;
  };

  // Test various combinations of parameters and how creation handles it.
  using test_func_t = std::function<void(nva2f::IDiffusionModel::GeometryExecutorCreationParameters&)>;
  std::vector<test_func_t> test_funcs = {
    [&](auto& params) {
      // No change, should work.
      EXPECT_TRUE(test_creation(params));
    },
    // Mess with the network info data.
    [&](auto& params) {
      params.networkInfo.emotionLength *= 2;
      const auto adjustedParams = GetParameters(params.networkInfo.emotionLength);
      EXPECT_FALSE(test_creation(params, &adjustedParams.data));
    },
    [&](auto& params) {
      params.networkInfo.identityLength *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.skinDim *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.tongueDim *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.jawDim *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.eyesDim *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.numDiffusionSteps *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.numGruLayers *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.gruLatentDim *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.numFramesLeftTruncate *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.numFramesRightTruncate *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.numFramesCenter *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.bufferLength *= 2;
      EXPECT_FALSE(test_creation(params));
    },
    // Mess with the rest of the initialization parameters.
    [&](auto& params) {
      params.networkData = nullptr;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      // CUDA engine deserialization throws an execption that reaches a noexcept function
      // when the network data size is half the right size or 1.
      // Don't test that case here.
      // Expected to be fixed in a future release of TensorRT.
      params.networkDataSize = 0;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.identityIndex = 2048;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.constantNoise = true;
      EXPECT_TRUE(test_creation(params));
    },
    [&](auto& params) {
      params.constantNoise = false;
      EXPECT_TRUE(test_creation(params));
    },
    [&](auto& params) {
      params.initializationSkinParams = nullptr;
      EXPECT_TRUE(test_creation(params));
    },
    [&](auto& params) {
      params.initializationTongueParams = nullptr;
      EXPECT_TRUE(test_creation(params));
    },
    [&](auto& params) {
      params.initializationTeethParams = nullptr;
      EXPECT_TRUE(test_creation(params));
    },
    [&](auto& params) {
      params.initializationEyesParams = nullptr;
      EXPECT_TRUE(test_creation(params));
    },
    [&](auto& params) {
      auto skinParams = *params.initializationSkinParams;
      params.initializationSkinParams = &skinParams;
      skinParams.data.neutralPose = skinParams.data.neutralPose.View(0, skinParams.data.neutralPose.Size() / 2);
      EXPECT_FALSE(test_creation(params));
      skinParams.data.neutralPose = skinParams.data.neutralPose.View(0, 1);
      EXPECT_FALSE(test_creation(params));
      skinParams.data.neutralPose = {};
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      auto skinParams = *params.initializationSkinParams;
      params.initializationSkinParams = &skinParams;
      skinParams.data.lipOpenPoseDelta = skinParams.data.lipOpenPoseDelta.View(0, skinParams.data.lipOpenPoseDelta.Size() / 2);
      EXPECT_FALSE(test_creation(params));
      skinParams.data.lipOpenPoseDelta = skinParams.data.lipOpenPoseDelta.View(0, 1);
      EXPECT_FALSE(test_creation(params));
      skinParams.data.lipOpenPoseDelta = {};
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      auto skinParams = *params.initializationSkinParams;
      params.initializationSkinParams = &skinParams;
      skinParams.data.eyeClosePoseDelta = skinParams.data.eyeClosePoseDelta.View(0, skinParams.data.eyeClosePoseDelta.Size() / 2);
      EXPECT_FALSE(test_creation(params));
      skinParams.data.eyeClosePoseDelta = skinParams.data.eyeClosePoseDelta.View(0, 1);
      EXPECT_FALSE(test_creation(params));
      skinParams.data.eyeClosePoseDelta = {};
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      auto tongueParams = *params.initializationTongueParams;
      params.initializationTongueParams = &tongueParams;
      tongueParams.data.neutralPose = tongueParams.data.neutralPose.View(0, tongueParams.data.neutralPose.Size() / 2);
      EXPECT_FALSE(test_creation(params));
      tongueParams.data.neutralPose = tongueParams.data.neutralPose.View(0, 1);
      EXPECT_FALSE(test_creation(params));
      tongueParams.data.neutralPose = {};
      EXPECT_FALSE(test_creation(params));
    },
  };

  const auto diffusionParams = GetDiffusionParameters();
  for (std::size_t i = 0; i < test_funcs.size(); ++i) {
    const auto& test_func = test_funcs[i];
    auto paramsToTest = diffusionParams.data;
    test_func(paramsToTest);
  }
}

TEST(TestCoreExecutor, Queries) {
  // Compute expected number of frames.
  const auto audio = GetAudio();
  const auto nbFrames = (audio.size() * 60 + 16000 - 1) / 16000;

  for (const auto createFunc : {&CreateGeometryExecutorRegressionBundle, &CreateGeometryExecutorDiffusionBundle}) {
    const bool isRegression = createFunc == &CreateGeometryExecutorRegressionBundle;
    auto bundle = createFunc(nva2f::IGeometryExecutor::ExecutionOption::All, false);
    ASSERT_TRUE(bundle);

    auto callback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
      return true;
    };
    ASSERT_TRUE(!bundle->GetExecutor().SetResultsCallback(callback, nullptr));

    const auto& executor = bundle->GetExecutor();
    const auto& cudaStream = bundle->GetCudaStream();

    const auto trackIndex = kNbTracks / 2;

    // Add default emotion.
    const std::vector<float> defaultEmotion(kEmotionSize, 0.0f);
    for (std::size_t i = 0; i < kNbTracks; i++) {
      // For diffusion, only add on the "active" track so that the others are not ready.
      if (!isRegression && i != trackIndex) {
        continue;
      }

      ASSERT_TRUE(!bundle->GetEmotionAccumulator(i).Accumulate(0, nva2x::ToConstView(defaultEmotion), cudaStream.Data()));
      ASSERT_TRUE(!bundle->GetEmotionAccumulator(i).Close());
    }

    ASSERT_EQ(executor.GetNbTracks(), kNbTracks);
    ASSERT_EQ(executor.GetSamplingRate(), 16000);
    std::size_t frameRateNumerator, frameRateDenominator;
    executor.GetFrameRate(frameRateNumerator, frameRateDenominator);
    ASSERT_EQ(frameRateNumerator, 60);
    ASSERT_EQ(frameRateDenominator, 1);

    ASSERT_EQ(executor.GetSkinGeometrySize(), isRegression ? 61520*3 : 72006);
    ASSERT_EQ(executor.GetTongueGeometrySize(), isRegression ? 5602*3 : 16806);
    ASSERT_EQ(executor.GetJawTransformSize(), 16);
    ASSERT_EQ(executor.GetEyesRotationSize(), 6);

    for (std::size_t i = 0; i < 1000; ++i) {
      const auto timestamp = executor.GetFrameTimestamp(i);
      const auto expectedTimestamp = i * 16000 * frameRateDenominator / frameRateNumerator;
      ASSERT_EQ(timestamp, static_cast<nva2x::IExecutor::timestamp_t>(expectedTimestamp));
    }

    // Test initial track state.
    ASSERT_FALSE(nva2x::HasExecutionStarted(executor));
    ASSERT_EQ(nva2x::GetNbReadyTracks(executor), isRegression ? 0 : 1);
    for (std::size_t i = 0; i < kNbTracks; i++) {
      ASSERT_FALSE(executor.HasExecutionStarted(i));
      ASSERT_EQ(executor.GetNbAvailableExecutions(i), (!isRegression && i == trackIndex) ? 1 : 0);
      ASSERT_EQ(executor.GetTotalNbFrames(i), 0);
    }

    // Add some audio and check the state again.
    ASSERT_TRUE(!bundle->GetAudioAccumulator(trackIndex).Accumulate(nva2x::ToConstView(audio), cudaStream.Data()));

    ASSERT_FALSE(nva2x::HasExecutionStarted(executor));
    ASSERT_EQ(nva2x::GetNbReadyTracks(executor), 1);
    for (std::size_t i = 0; i < kNbTracks; i++) {
      ASSERT_FALSE(executor.HasExecutionStarted(i));
      ASSERT_EQ(executor.GetTotalNbFrames(i), 0);
      if (i == trackIndex) {
        ASSERT_GT(executor.GetNbAvailableExecutions(i), 0);
      }
      else {
        ASSERT_EQ(executor.GetNbAvailableExecutions(i), 0);
      }
    }
    auto previousNbAvailableExecutions = executor.GetNbAvailableExecutions(trackIndex);

    // Close the audio accumulator and check the state again.
    ASSERT_TRUE(!bundle->GetAudioAccumulator(trackIndex).Close());

    ASSERT_FALSE(nva2x::HasExecutionStarted(executor));
    ASSERT_EQ(nva2x::GetNbReadyTracks(executor), 1);
    for (std::size_t i = 0; i < kNbTracks; i++) {
      ASSERT_FALSE(executor.HasExecutionStarted(i));
      if (i == trackIndex) {
        ASSERT_EQ(executor.GetTotalNbFrames(i), nbFrames);
        ASSERT_GT(executor.GetNbAvailableExecutions(i), previousNbAvailableExecutions);
      }
      else {
        ASSERT_EQ(executor.GetTotalNbFrames(i), 0);
        ASSERT_EQ(executor.GetNbAvailableExecutions(i), 0);
      }
    }
    previousNbAvailableExecutions = executor.GetNbAvailableExecutions(trackIndex);

    // Do one callback to check only the active track is executed.
    struct CallbackData {
      std::size_t nbExecutedFrames;
      std::size_t trackIndex;
      const nva2f::IGeometryExecutor& executor;
    };
    CallbackData callbackData{0, trackIndex, executor};
    auto resultsCallback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
      auto& callbackData = *static_cast<CallbackData*>(userdata);
      EXPECT_EQ(results.trackIndex, callbackData.trackIndex);
      EXPECT_EQ(results.timeStampCurrentFrame, callbackData.executor.GetFrameTimestamp(callbackData.nbExecutedFrames));
      EXPECT_EQ(results.timeStampNextFrame, callbackData.executor.GetFrameTimestamp(callbackData.nbExecutedFrames + 1));
      ++callbackData.nbExecutedFrames;
      return true;
    };
    std::size_t nbExecutedTracks = 0;
    // All other calls are const except these ones.
    // We should be able to set a null callback.
    ASSERT_TRUE(!bundle->GetExecutor().SetResultsCallback(nullptr, nullptr));
    // We shouldn't be able to set a null callback with non-null userdata.
    ASSERT_TRUE(bundle->GetExecutor().SetResultsCallback(nullptr, &callbackData));
    ASSERT_TRUE(!bundle->GetExecutor().SetResultsCallback(resultsCallback, &callbackData));

    ASSERT_TRUE(!bundle->GetExecutor().Execute(&nbExecutedTracks));
    ASSERT_EQ(nbExecutedTracks, 1);
    // The first diffusion frames are before the audio starts.
    ASSERT_EQ(callbackData.nbExecutedFrames, isRegression ? 1 : 0);

    nbExecutedTracks = 0;
    ASSERT_TRUE(!bundle->GetExecutor().Execute(&nbExecutedTracks));
    ASSERT_EQ(nbExecutedTracks, 1);
    ASSERT_EQ(callbackData.nbExecutedFrames, isRegression ? 2 : 15);

    // Check the state again.
    ASSERT_TRUE(nva2x::HasExecutionStarted(executor));
    ASSERT_EQ(nva2x::GetNbReadyTracks(executor), 1);
    for (std::size_t i = 0; i < kNbTracks; i++) {
      if (i == trackIndex) {
        ASSERT_TRUE(executor.HasExecutionStarted(i));
        ASSERT_EQ(executor.GetTotalNbFrames(i), nbFrames);
        ASSERT_EQ(executor.GetNbAvailableExecutions(i), previousNbAvailableExecutions - 2);
      }
      else {
        ASSERT_FALSE(executor.HasExecutionStarted(i));
        ASSERT_EQ(executor.GetTotalNbFrames(i), 0);
        ASSERT_EQ(executor.GetNbAvailableExecutions(i), 0);
      }
    }

    // Test the state after reset.
    ASSERT_TRUE(!bundle->GetExecutor().Reset(trackIndex));

    ASSERT_FALSE(nva2x::HasExecutionStarted(executor));
    ASSERT_EQ(nva2x::GetNbReadyTracks(executor), 1);
    for (std::size_t i = 0; i < kNbTracks; i++) {
      ASSERT_FALSE(executor.HasExecutionStarted(i));
      if (i == trackIndex) {
        ASSERT_EQ(executor.GetTotalNbFrames(i), nbFrames);
        ASSERT_EQ(executor.GetNbAvailableExecutions(i), previousNbAvailableExecutions);
      }
      else {
        ASSERT_EQ(executor.GetTotalNbFrames(i), 0);
        ASSERT_EQ(executor.GetNbAvailableExecutions(i), 0);
      }
    }

    ASSERT_TRUE(!bundle->GetAudioAccumulator(trackIndex).Reset());

    ASSERT_FALSE(nva2x::HasExecutionStarted(executor));
    ASSERT_EQ(nva2x::GetNbReadyTracks(executor), isRegression ? 0 : 1);
    for (std::size_t i = 0; i < kNbTracks; i++) {
      ASSERT_FALSE(executor.HasExecutionStarted(i));
      ASSERT_EQ(executor.GetNbAvailableExecutions(i), (!isRegression && i == trackIndex) ? 1 : 0);
      ASSERT_EQ(executor.GetTotalNbFrames(i), 0);
    }

    ASSERT_TRUE(!bundle->GetEmotionAccumulator(trackIndex).Reset());

    ASSERT_FALSE(nva2x::HasExecutionStarted(executor));
    ASSERT_EQ(nva2x::GetNbReadyTracks(executor), 0);
    for (std::size_t i = 0; i < kNbTracks; i++) {
      ASSERT_FALSE(executor.HasExecutionStarted(i));
      ASSERT_EQ(executor.GetNbAvailableExecutions(i), 0);
      ASSERT_EQ(executor.GetTotalNbFrames(i), 0);
    }
  }
}

TEST(TestCoreExecutor, EmotionCallback) {
  // Compute expected number of frames.
  const auto audio = GetAudio();
  const auto nbFrames = (audio.size() * 60 + 16000 - 1) / 16000;

  for (const auto createFunc : {&CreateGeometryExecutorRegressionBundle, &CreateGeometryExecutorDiffusionBundle}) {
    const bool isRegression = createFunc == &CreateGeometryExecutorRegressionBundle;
    auto bundle = createFunc(nva2f::IGeometryExecutor::ExecutionOption::All, false);
    ASSERT_TRUE(bundle);

    auto& executor = bundle->GetExecutor();
    const auto& cudaStream = bundle->GetCudaStream();

    const auto trackIndex = kNbTracks / 2;

    // Add emotion values.
    std::vector<float> emotionValues(nbFrames * kEmotionSize, 0.0f);
    for (std::size_t frameIndex = 0; frameIndex < nbFrames; ++frameIndex) {
      for (std::size_t emotionIndex = 0; emotionIndex < kEmotionSize; ++emotionIndex) {
        const auto index = frameIndex * kEmotionSize + emotionIndex;
        emotionValues[frameIndex * kEmotionSize + emotionIndex] = index / static_cast<float>(nbFrames * kEmotionSize);
      }
    }
    for (std::size_t i = 0; i < kNbTracks; i++) {
      // For diffusion, only add on the "active" track so that the others are not ready.
      if (!isRegression && i != trackIndex) {
        continue;
      }

      for (std::size_t j = 0; j < nbFrames; ++j) {
        const auto timestamp = j * 16000 / 60;
        const auto emotionValuesView = nva2x::ToConstView(emotionValues).View(j * kEmotionSize, kEmotionSize);
        ASSERT_TRUE(!bundle->GetEmotionAccumulator(i).Accumulate(timestamp, emotionValuesView, cudaStream.Data()));
      }
      ASSERT_TRUE(!bundle->GetEmotionAccumulator(i).Close());
    }

    // Add audio.
    ASSERT_TRUE(!bundle->GetAudioAccumulator(trackIndex).Accumulate((nva2x::ToConstView(audio)), cudaStream.Data()));
    ASSERT_TRUE(!bundle->GetAudioAccumulator(trackIndex).Close());

    struct EmotionCallbackData {
      const std::size_t trackIndex;
      std::vector<float> emotionValues;
      std::vector<nva2f::IGeometryExecutor::timestamp_t> timestamps;
    };
    EmotionCallbackData emotionCallbackData{trackIndex, {}, {}};
    auto emotionCallback = [](void* userdata, const nva2f::IGeometryExecutor::Emotions& emotions) {
      auto& callbackData = *static_cast<EmotionCallbackData*>(userdata);
      EXPECT_EQ(emotions.trackIndex, callbackData.trackIndex);
      callbackData.timestamps.emplace_back(emotions.timeStampCurrentFrame);
      callbackData.emotionValues.resize(callbackData.emotionValues.size() + emotions.emotions.Size());
      const auto writeBegin = callbackData.emotionValues.data() + callbackData.emotionValues.size() - emotions.emotions.Size();
      EXPECT_TRUE(
        !nva2x::CopyDeviceToHost(
          {writeBegin, emotions.emotions.Size()},
          emotions.emotions,
          emotions.cudaStream
          )
        );
      EXPECT_TRUE(!cudaStreamSynchronize(emotions.cudaStream));
    };

    // It's ok to set a null callback for emotions.
    ASSERT_TRUE(!executor.SetEmotionsCallback(nullptr, &emotionCallbackData));
    ASSERT_TRUE(!executor.SetEmotionsCallback(emotionCallback, &emotionCallbackData));

    struct ResultsCallbackData {
      const std::size_t trackIndex;
      std::vector<nva2f::IGeometryExecutor::timestamp_t> timestamps;
    };
    ResultsCallbackData resultCallbackData{trackIndex, {}};
    auto resultsCallback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
      auto& callbackData = *static_cast<ResultsCallbackData*>(userdata);
      EXPECT_EQ(results.trackIndex, callbackData.trackIndex);
      callbackData.timestamps.emplace_back(results.timeStampCurrentFrame);
      return true;
    };
    ASSERT_TRUE(!executor.SetResultsCallback(resultsCallback, &resultCallbackData));

    const auto nbAvailableExecutions = executor.GetNbAvailableExecutions(trackIndex);
    for (std::size_t i = 0; i < nbAvailableExecutions; ++i) {
      std::size_t nbExecutedTracks = 0;
      ASSERT_TRUE(!executor.Execute(&nbExecutedTracks));
      ASSERT_EQ(nbExecutedTracks, 1);
    }
    ASSERT_EQ(nbFrames, emotionCallbackData.timestamps.size());
    ASSERT_EQ(nbFrames * kEmotionSize, emotionCallbackData.emotionValues.size());
    ASSERT_EQ(nbFrames, resultCallbackData.timestamps.size());
    ASSERT_EQ(emotionCallbackData.timestamps, resultCallbackData.timestamps);
    ASSERT_EQ(emotionCallbackData.emotionValues, emotionValues);
  }
}

TEST(TestCoreExecutor, ExecutionOptionBinaryOps) {
  using ExecutionOption = nva2f::IGeometryExecutor::ExecutionOption;
  constexpr auto None = ExecutionOption::None;
  constexpr auto Skin = ExecutionOption::Skin;
  constexpr auto Tongue = ExecutionOption::Tongue;
  constexpr auto Jaw = ExecutionOption::Jaw;
  constexpr auto Eyes = ExecutionOption::Eyes;
  constexpr auto All = ExecutionOption::All;

  // Test bitwise OR operations
  ASSERT_EQ(Skin | None, Skin);
  ASSERT_EQ(None | Skin, Skin);
  ASSERT_EQ(
    Skin | Tongue,
    ExecutionOption(static_cast<std::uint32_t>(Skin) | static_cast<std::uint32_t>(Tongue))
    );
  ASSERT_EQ(
    Skin | Jaw | Eyes,
    ExecutionOption(
      static_cast<std::uint32_t>(Skin) | static_cast<std::uint32_t>(Jaw) | static_cast<std::uint32_t>(Eyes)
      )
    );

  // Test bitwise AND operations
  ASSERT_EQ(Skin & None, None);
  ASSERT_EQ(None & Skin, None);
  ASSERT_EQ(All & Skin, Skin);
  ASSERT_EQ(All & Tongue, Tongue);
  ASSERT_EQ((Skin | Tongue) & Skin, Skin);
  ASSERT_EQ((Skin | Tongue) & Tongue, Tongue);

  // Test compound assignment operators
  auto option = None;
  option |= Skin;
  ASSERT_EQ(option, Skin);
  option |= Tongue;
  ASSERT_EQ(option, ExecutionOption(static_cast<std::uint32_t>(Skin) | static_cast<std::uint32_t>(Tongue)));

  option = All;
  option &= Skin;
  ASSERT_EQ(option, Skin);
  option = All;
  option &= (Skin | Tongue);
  ASSERT_EQ(option, ExecutionOption(static_cast<std::uint32_t>(Skin) | static_cast<std::uint32_t>(Tongue)));

  // Test combinations of all options
  ASSERT_EQ(All, Skin | Tongue | Jaw | Eyes);

  // Test IsAnySet functionality with two parameters
  // Test None against various flags
  ASSERT_FALSE(IsAnySet(None, None));
  ASSERT_FALSE(IsAnySet(None, Skin));
  ASSERT_FALSE(IsAnySet(None, All));

  // Test single flags against various combinations
  ASSERT_TRUE(IsAnySet(Skin, Skin));
  ASSERT_FALSE(IsAnySet(Skin, Tongue));
  ASSERT_TRUE(IsAnySet(Skin, All));
  ASSERT_TRUE(IsAnySet(Skin, Skin | Tongue));
  ASSERT_FALSE(IsAnySet(Skin, Tongue | Jaw));

  // Test multiple flags against various combinations
  ASSERT_TRUE(IsAnySet(Skin | Tongue, Skin));
  ASSERT_TRUE(IsAnySet(Skin | Tongue, Tongue));
  ASSERT_FALSE(IsAnySet(Skin | Tongue, Jaw | Eyes));
  ASSERT_TRUE(IsAnySet(Skin | Tongue, All));
  ASSERT_TRUE(IsAnySet(Skin | Jaw, Skin | Jaw));

  // Test All against various combinations
  ASSERT_TRUE(IsAnySet(All, Skin));
  ASSERT_TRUE(IsAnySet(All, Tongue | Jaw));
  ASSERT_TRUE(IsAnySet(All, All));
  ASSERT_FALSE(IsAnySet(All, None));
}

TEST(TestCoreExecutor, ExecutionOption) {
  const auto audio = GetAudio();

  using ExecutionOption = nva2f::IGeometryExecutor::ExecutionOption;
  for (const auto createFunc : {&CreateGeometryExecutorRegressionBundle, &CreateGeometryExecutorDiffusionBundle}) {
    // Get the default parameters.
    nva2f::AnimatorSkinParams skinDefaultParams;
    nva2f::AnimatorTongueParams tongueDefaultParams;
    nva2f::AnimatorTeethParams jawDefaultParams;
    nva2f::AnimatorEyesParams eyesDefaultParams;
    {
      auto bundle = createFunc(ExecutionOption::All, false);
      ASSERT_TRUE(bundle);
      ASSERT_TRUE(!nva2f::GetExecutorSkinParameters_INTERNAL(bundle->GetExecutor(), 0, skinDefaultParams));
      ASSERT_TRUE(!nva2f::GetExecutorTongueParameters_INTERNAL(bundle->GetExecutor(), 0, tongueDefaultParams));
      ASSERT_TRUE(!nva2f::GetExecutorTeethParameters_INTERNAL(bundle->GetExecutor(), 0, jawDefaultParams));
      ASSERT_TRUE(!nva2f::GetExecutorEyesParameters_INTERNAL(bundle->GetExecutor(), 0, eyesDefaultParams));
    }

    for (std::uint32_t optionToTest = 0; optionToTest <= 0b1111; ++optionToTest) {
      const auto bundleOption = static_cast<ExecutionOption>(optionToTest);
      auto bundle = createFunc(bundleOption, true);
      ASSERT_TRUE(bundle);

      const auto& cudaStream = bundle->GetCudaStream();
      auto& executor = bundle->GetExecutor();

      for (std::uint32_t subOptionToTest = 0; subOptionToTest <= 0b1111; ++subOptionToTest) {
        const auto option = static_cast<ExecutionOption>(subOptionToTest);

        const bool isSubOptionValid = (optionToTest & subOptionToTest) == subOptionToTest;
        if (!isSubOptionValid) {
          // Setting this option should fail, because it's not a subset of the
          // option to test.
          ASSERT_TRUE(executor.SetExecutionOption(option));
          continue;
        }

        ASSERT_TRUE(!executor.SetExecutionOption(option));

        // Test settings parameters.
        for (std::size_t i = 0; i < kNbTracks; ++i) {
          const bool expectedSkin = IsAnySet(bundleOption, ExecutionOption::Skin);
          nva2f::AnimatorSkinParams skinParams;
          ASSERT_EQ(expectedSkin, !nva2f::GetExecutorSkinParameters_INTERNAL(executor, i, skinParams));
          EXPECT_EQ(expectedSkin, !nva2f::SetExecutorSkinParameters_INTERNAL(executor, i, skinDefaultParams));

          const bool expectedTongue = IsAnySet(bundleOption, ExecutionOption::Tongue);
          nva2f::AnimatorTongueParams tongueParams;
          ASSERT_EQ(expectedTongue, !nva2f::GetExecutorTongueParameters_INTERNAL(executor, i, tongueParams));
          EXPECT_EQ(expectedTongue, !nva2f::SetExecutorTongueParameters_INTERNAL(executor, i, tongueDefaultParams));

          const bool expectedJaw = IsAnySet(bundleOption, ExecutionOption::Jaw);
          nva2f::AnimatorTeethParams jawParams;
          ASSERT_EQ(expectedJaw, !nva2f::GetExecutorTeethParameters_INTERNAL(executor, i, jawParams));
          EXPECT_EQ(expectedJaw, !nva2f::SetExecutorTeethParameters_INTERNAL(executor, i, jawDefaultParams));

          const bool expectedEyes = IsAnySet(bundleOption, ExecutionOption::Eyes);
          nva2f::AnimatorEyesParams eyesParams;
          ASSERT_EQ(expectedEyes, !nva2f::GetExecutorEyesParameters_INTERNAL(executor, i, eyesParams));
          EXPECT_EQ(expectedEyes, !nva2f::SetExecutorEyesParameters_INTERNAL(executor, i, eyesDefaultParams));
        }

        // Test results callback.
        struct CallbackData {
          ExecutionOption expectedOption;
          std::size_t nbExecutedCallbacks;
        };
        CallbackData callbackData{option, 0};
        auto callback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
          auto& callbackData = *static_cast<CallbackData*>(userdata);

          const bool expectedSkin = IsAnySet(callbackData.expectedOption, ExecutionOption::Skin);
          EXPECT_EQ(expectedSkin, results.skinGeometry.Data() != nullptr);
          EXPECT_EQ(expectedSkin, results.skinGeometry.Size() > 0);
          EXPECT_EQ(expectedSkin, results.skinCudaStream != nullptr);

          const bool expectedTongue = IsAnySet(callbackData.expectedOption, ExecutionOption::Tongue);
          EXPECT_EQ(expectedTongue, results.tongueGeometry.Data() != nullptr);
          EXPECT_EQ(expectedTongue, results.tongueGeometry.Size() > 0);
          EXPECT_EQ(expectedTongue, results.tongueCudaStream != nullptr);

          const bool expectedJaw = IsAnySet(callbackData.expectedOption, ExecutionOption::Jaw);
          EXPECT_EQ(expectedJaw, results.jawTransform.Data() != nullptr);
          EXPECT_EQ(expectedJaw, results.jawTransform.Size() > 0);
          EXPECT_EQ(expectedJaw, results.jawCudaStream != nullptr);

          const bool expectedEyes = IsAnySet(callbackData.expectedOption, ExecutionOption::Eyes);
          EXPECT_EQ(expectedEyes, results.eyesRotation.Data() != nullptr);
          EXPECT_EQ(expectedEyes, results.eyesRotation.Size() > 0);
          EXPECT_EQ(expectedEyes, results.eyesCudaStream != nullptr);

          ++callbackData.nbExecutedCallbacks;

          return true;
        };
        ASSERT_TRUE(!executor.SetResultsCallback(callback, &callbackData));

        while (callbackData.nbExecutedCallbacks == 0) {
          ASSERT_TRUE(!executor.Execute(nullptr));
        }

        // We shouldn't be able to set things.
        ASSERT_TRUE(executor.SetResultsCallback(callback, &callbackData));
        ASSERT_TRUE(executor.SetExecutionOption(option));
        EXPECT_TRUE(nva2f::SetExecutorSkinParameters_INTERNAL(executor, 0, skinDefaultParams));
        EXPECT_TRUE(nva2f::SetExecutorTongueParameters_INTERNAL(executor, 0, tongueDefaultParams));
        EXPECT_TRUE(nva2f::SetExecutorTeethParameters_INTERNAL(executor, 0, jawDefaultParams));
        EXPECT_TRUE(nva2f::SetExecutorEyesParameters_INTERNAL(executor, 0, eyesDefaultParams));

        // Reset all tracks to clear any previous state
        for (std::size_t i = 0; i < kNbTracks; i++) {
          ASSERT_TRUE(!executor.Reset(i));
        }

        // Now we should be able to set things.
        ASSERT_TRUE(!executor.SetResultsCallback(callback, &callbackData));
        ASSERT_TRUE(!executor.SetExecutionOption(option));
      }
    }
  }
}

TEST(TestCoreExecutor, Execution) {
  const auto audio = GetAudio();
  const auto nbFrames = (audio.size() * 60 + 16000 - 1) / 16000;

  // Add random cases.
  const int seed = static_cast<unsigned int>(time(NULL));
  std::cout << "Current srand seed: " << seed << std::endl;
  std::srand(seed); // make random inputs reproducible

  for (const auto createFunc : {&CreateGeometryExecutorRegressionBundle, &CreateGeometryExecutorDiffusionBundle}) {
    auto bundle = createFunc(nva2f::IGeometryExecutor::ExecutionOption::All, true);
    ASSERT_TRUE(bundle);

    const auto& cudaStream = bundle->GetCudaStream();
    auto& executor = bundle->GetExecutor();

    // Collect all the tracks.
    struct CallbackData {
      std::vector<std::vector<std::vector<float>>> geometryData;
    };
    CallbackData callbackData;
    callbackData.geometryData.resize(kNbTracks);
    auto callback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
      auto& callbackData = *static_cast<CallbackData*>(userdata);

      std::vector<float> frameData(
        results.skinGeometry.Size() +
        results.tongueGeometry.Size() +
        results.jawTransform.Size() +
        results.eyesRotation.Size()
      );
      float* destination = frameData.data();
      EXPECT_TRUE(
        !nva2x::CopyDeviceToHost(
          {destination, results.skinGeometry.Size()},
          results.skinGeometry,
          results.skinCudaStream
        )
      );
      destination += results.skinGeometry.Size();
      EXPECT_TRUE(
        !nva2x::CopyDeviceToHost(
          {destination, results.tongueGeometry.Size()},
          results.tongueGeometry,
          results.tongueCudaStream
        )
      );
      destination += results.tongueGeometry.Size();
      EXPECT_TRUE(
        !nva2x::CopyDeviceToHost(
          {destination, results.jawTransform.Size()},
          results.jawTransform,
          results.jawCudaStream
        )
      );
      destination += results.jawTransform.Size();
      EXPECT_TRUE(
        !nva2x::CopyDeviceToHost(
          {destination, results.eyesRotation.Size()},
          results.eyesRotation,
          results.eyesCudaStream
        )
      );
      destination += results.eyesRotation.Size();

      EXPECT_TRUE(!cudaStreamSynchronize(results.skinCudaStream));
      EXPECT_TRUE(!cudaStreamSynchronize(results.tongueCudaStream));
      EXPECT_TRUE(!cudaStreamSynchronize(results.jawCudaStream));
      EXPECT_TRUE(!cudaStreamSynchronize(results.eyesCudaStream));

      callbackData.geometryData[results.trackIndex].emplace_back(std::move(frameData));
      return true;
    };
    ASSERT_TRUE(!executor.SetResultsCallback(callback, &callbackData));

    // Execute everything.
    while (nva2x::internal::GetNbReadyTracks(executor) > 0) {
      std::size_t nbExecutedTracks = 0;
      ASSERT_TRUE(!executor.Execute(&nbExecutedTracks));
      ASSERT_EQ(nbExecutedTracks, kNbTracks);
    }

    // Everything should be the same.
    {
      const auto& referenceData = callbackData.geometryData[0];
      ASSERT_EQ(nbFrames, referenceData.size());
      for (std::size_t trackIndex = 1; trackIndex < kNbTracks; ++trackIndex) {
        const auto& dataToTest = callbackData.geometryData[trackIndex];
        ASSERT_EQ(referenceData, dataToTest);
      }
    }

    // Reset everything and re-run with a variable amount of audio data.
    for (std::size_t trackIndex = 0; trackIndex < kNbTracks; ++trackIndex) {
      callbackData.geometryData[trackIndex].clear();
      ASSERT_TRUE(!executor.Reset(trackIndex));
      ASSERT_TRUE(!bundle->GetAudioAccumulator(trackIndex).Reset());
    }

    std::vector<std::vector<float>> audioData(kNbTracks, audio);
    while (true) {
      bool empty = true;
      for (std::size_t trackIndex = 0; trackIndex < kNbTracks; ++trackIndex) {
        auto& audioToAccumulate = audioData[trackIndex];
        if (audioToAccumulate.empty()) {
          continue;
        }
        empty = false;
        auto& audioAccumulator = bundle->GetAudioAccumulator(trackIndex);
        std::size_t sizeToAccumulate = (rand() % 10) * 1600;
        sizeToAccumulate = std::min(sizeToAccumulate, audioToAccumulate.size());
        if (sizeToAccumulate > 0) {
          ASSERT_TRUE(!audioAccumulator.Accumulate(nva2x::HostTensorFloatConstView{audioToAccumulate.data(), sizeToAccumulate}, cudaStream.Data()));
          ASSERT_TRUE(!cudaStream.Synchronize());
          audioToAccumulate.erase(audioToAccumulate.begin(), audioToAccumulate.begin() + sizeToAccumulate);
          if (audioToAccumulate.empty()) {
            ASSERT_TRUE(!audioAccumulator.Close());
          }
        }
      }
      if (empty) {
        // We are done.
        break;
      }

      // Execute everything.
      while (nva2x::internal::GetNbReadyTracks(executor) > 0) {
        ASSERT_TRUE(!executor.Execute(nullptr));
      }
    }

    // Everything should be the same.
    {
      const auto& referenceData = callbackData.geometryData[0];
      ASSERT_EQ(nbFrames, referenceData.size());
      for (std::size_t trackIndex = 1; trackIndex < kNbTracks; ++trackIndex) {
        const auto& dataToTest = callbackData.geometryData[trackIndex];
        ASSERT_EQ(referenceData, dataToTest);
      }
    }

    // Reset everything and re-run with a variable amount of emotion data.
    for (std::size_t trackIndex = 0; trackIndex < kNbTracks; ++trackIndex) {
      callbackData.geometryData[trackIndex].clear();
      ASSERT_TRUE(!executor.Reset(trackIndex));
      ASSERT_TRUE(!bundle->GetEmotionAccumulator(trackIndex).Reset());
    }

    std::vector<float> emotions(nbFrames);
    for (std::size_t i = 0; i < nbFrames; ++i) {
      emotions[i] = static_cast<float>(i + 1) / (nbFrames + 1);
    }
    std::vector<std::vector<float>> emotionData(kNbTracks, emotions);
    std::vector<std::size_t> emotionAccumulated(kNbTracks, 0);
    std::vector<float> emotionToAdd;
    while (true) {
      bool empty = true;
      for (std::size_t trackIndex = 0; trackIndex < kNbTracks; ++trackIndex) {
        auto& emotionToAccumulate = emotionData[trackIndex];
        if (emotionToAccumulate.empty()) {
          continue;
        }
        empty = false;
        auto& emotionAccumulator = bundle->GetEmotionAccumulator(trackIndex);
        std::size_t sizeToAccumulate = (rand() % 10) * 1;
        sizeToAccumulate = std::min(sizeToAccumulate, emotionToAccumulate.size());
        if (sizeToAccumulate > 0) {
          for (std::size_t i = 0; i < sizeToAccumulate; ++i) {
            const float emotionToAccumulate = emotionData[trackIndex][i];
            const auto timestamp = executor.GetFrameTimestamp(emotionAccumulated[trackIndex] + i);
            emotionToAdd.clear();
            emotionToAdd.resize(emotionAccumulator.GetEmotionSize(), emotionToAccumulate);
            ASSERT_TRUE(!emotionAccumulator.Accumulate(timestamp, nva2x::ToConstView(emotionToAdd), cudaStream.Data()));
          }
          ASSERT_TRUE(!cudaStream.Synchronize());
          emotionToAccumulate.erase(emotionToAccumulate.begin(), emotionToAccumulate.begin() + sizeToAccumulate);
          emotionAccumulated[trackIndex] += sizeToAccumulate;
          if (emotionToAccumulate.empty()) {
            ASSERT_TRUE(!emotionAccumulator.Close());
          }
        }
      }
      if (empty) {
        // We are done.
        break;
      }

      // Execute everything.
      while (nva2x::internal::GetNbReadyTracks(executor) > 0) {
        ASSERT_TRUE(!executor.Execute(nullptr));
      }
    }

    // Everything should be the same.
    {
      const auto& referenceData = callbackData.geometryData[0];
      ASSERT_EQ(nbFrames, referenceData.size());
      for (std::size_t trackIndex = 1; trackIndex < kNbTracks; ++trackIndex) {
        const auto& dataToTest = callbackData.geometryData[trackIndex];
        ASSERT_EQ(referenceData, dataToTest);
      }
    }

    // Reset everything and re-run with a random reset in the middle.
    const auto previousReferenceData = callbackData.geometryData[0];
    for (std::size_t trackIndex = 0; trackIndex < kNbTracks; ++trackIndex) {
      callbackData.geometryData[trackIndex].clear();
      ASSERT_TRUE(!executor.Reset(trackIndex));
    }

    std::vector<std::size_t> resetIndices(kNbTracks);
    for (std::size_t i = 0; i < kNbTracks; ++i) {
      const auto nbAvailableExecutions = executor.GetNbAvailableExecutions(i);
      ASSERT_LT(2, nbAvailableExecutions);
      resetIndices[i] = rand() % (nbAvailableExecutions - 1) + 1;
    }

    std::size_t nbExecutions = 0;
    while (nva2x::internal::GetNbReadyTracks(executor) > 0) {
      ASSERT_TRUE(!executor.Execute(nullptr));
      ++nbExecutions;

      for (std::size_t trackIndex = 0; trackIndex < kNbTracks; ++trackIndex) {
        if (resetIndices[trackIndex] == nbExecutions) {
          ASSERT_TRUE(!executor.Reset(trackIndex));
          callbackData.geometryData[trackIndex].clear();
          resetIndices[trackIndex] = 0;
        }
      }
    }

    // Everything should be the same.
    {
      const auto& referenceData = previousReferenceData;
      ASSERT_EQ(nbFrames, referenceData.size());
      for (std::size_t trackIndex = 1; trackIndex < kNbTracks; ++trackIndex) {
        const auto& dataToTest = callbackData.geometryData[trackIndex];
        ASSERT_EQ(referenceData, dataToTest);
      }
    }
  }
}

TEST(TestCoreExecutor, ExecutionLimitedByEmotions) {
  const auto audio = GetAudio();
  const auto nbFrames = (audio.size() * 60 + 16000 - 1) / 16000;

  using func_t = nva2x::UniquePtr<nva2f::IGeometryExecutorBundle>(*)(nva2f::IGeometryExecutor::ExecutionOption, bool);
  struct TestData {
    func_t createFunc;
    std::size_t nbFramesPerExecution;
    std::size_t nbFramesBeforeStart;
  };
  constexpr std::array<TestData, 2> testData = {{
    {CreateGeometryExecutorRegressionBundle, 1, 0},
    {CreateGeometryExecutorDiffusionBundle, 30, 45},
  }};
  for (const auto& test : testData) {
    auto bundle = test.createFunc(nva2f::IGeometryExecutor::ExecutionOption::All, false);
    ASSERT_TRUE(bundle);

    const auto cudaStream = bundle->GetCudaStream().Data();

    auto& audioAccumulator = bundle->GetAudioAccumulator(0);
    ASSERT_TRUE(!audioAccumulator.Accumulate(nva2x::ToConstView(audio), bundle->GetCudaStream().Data()));
    ASSERT_TRUE(!audioAccumulator.Close());

    auto& executor = bundle->GetExecutor();
    auto callback = [](void*, const nva2f::IGeometryExecutor::Results&) { return true; };
    ASSERT_TRUE(!executor.SetResultsCallback(callback, nullptr));

    // Check when no emotions are available.
    ASSERT_EQ(executor.GetNbAvailableExecutions(0), 0);

    auto& emotionAccumulator = bundle->GetEmotionAccumulator(0);
    const std::vector<float> emotionData(emotionAccumulator.GetEmotionSize(), 0.0f);
    const auto emotion = nva2x::ToConstView(emotionData);

    const std::size_t nbFramesPerExecution = test.nbFramesPerExecution;
    const std::size_t nbFramesBeforeStart = test.nbFramesBeforeStart;

    // Add a single emotion, just before the last frame of the first execution which produces actual frames.
    const auto nbExecutionsToGetFrames = 1 + nbFramesBeforeStart / nbFramesPerExecution;
    const auto lastFirstOfFirstVisibleExecution = nbFramesPerExecution * nbExecutionsToGetFrames - nbFramesBeforeStart;
    {
      const auto timestamp = executor.GetFrameTimestamp(lastFirstOfFirstVisibleExecution - 1);
      ASSERT_TRUE(!emotionAccumulator.Accumulate(timestamp - 1, emotion, cudaStream));
      ASSERT_EQ(executor.GetNbAvailableExecutions(0), nbExecutionsToGetFrames - 1);

      ASSERT_TRUE(!emotionAccumulator.Accumulate(timestamp, emotion, cudaStream));
      ASSERT_EQ(executor.GetNbAvailableExecutions(0), nbExecutionsToGetFrames);
      for (std::size_t i = 0; i < nbExecutionsToGetFrames; ++i) {
        ASSERT_TRUE(!executor.Execute(nullptr));
      }
      ASSERT_EQ(executor.GetNbAvailableExecutions(0), 0);
    }

    const std::size_t kNbIterations = 3;
    for (std::size_t i = 0; i < kNbIterations; ++i) {
      ASSERT_EQ(executor.GetNbAvailableExecutions(0), i);

      const auto frame = (nbExecutionsToGetFrames + i + 1) * nbFramesPerExecution - nbFramesBeforeStart - 1;
      const auto timestamp = executor.GetFrameTimestamp(frame);
      ASSERT_TRUE(!emotionAccumulator.Accumulate(timestamp - 1, emotion, cudaStream));
      ASSERT_EQ(executor.GetNbAvailableExecutions(0), i);

      ASSERT_TRUE(!emotionAccumulator.Accumulate(timestamp, emotion, cudaStream));
      ASSERT_EQ(executor.GetNbAvailableExecutions(0), i + 1);

      ASSERT_TRUE(!emotionAccumulator.Accumulate(timestamp + 1, emotion, cudaStream));
      ASSERT_EQ(executor.GetNbAvailableExecutions(0), i + 1);
    }

    // Test that closing works.
    {
      ASSERT_EQ(executor.GetNbAvailableExecutions(0), kNbIterations);
      ASSERT_TRUE(!emotionAccumulator.Close());
      ASSERT_GT(executor.GetNbAvailableExecutions(0), kNbIterations);
    }
  }
}
