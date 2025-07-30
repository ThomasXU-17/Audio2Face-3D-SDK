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
#include "audio2x/internal/audio_utils.h"
#include "audio2x/internal/audio2x.h"
#include "audio2x/internal/unique_ptr.h"
#include "audio2emotion/internal/executor_classifier.h"
#include "audio2emotion/internal/executor_postprocess.h"

#include <gtest/gtest.h>

#include <any>

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
  constexpr std::size_t kInferencesToSkip = 29;

  TestData<nva2e::EmotionExecutorCreationParameters> GetParameters() {

    auto cudaStream = nva2x::ToSharedPtr(nva2x::internal::CreateCudaStream());
    EXPECT_TRUE(cudaStream);

    auto audioAccumulators = std::make_shared<std::vector<nva2x::UniquePtr<nva2x::IAudioAccumulator>>>(kNbTracks);
    auto sharedAudioAccumulators = std::make_shared<std::vector<const nva2x::IAudioAccumulator*>>(kNbTracks);
    for (std::size_t i = 0; i < kNbTracks; ++i) {
      (*audioAccumulators)[i] = nva2x::ToUniquePtr(nva2x::internal::CreateAudioAccumulator(16000, 0));
      EXPECT_TRUE((*audioAccumulators)[i]);
      (*sharedAudioAccumulators)[i] = (*audioAccumulators)[i].get();
    }

    TestData<nva2e::EmotionExecutorCreationParameters> params;
    params.data.cudaStream = cudaStream->Data();
    params.data.nbTracks = kNbTracks;
    params.data.sharedAudioAccumulators = sharedAudioAccumulators->data();

    params.ownedData = std::vector<std::any>{
      std::move(cudaStream),
      std::move(audioAccumulators),
      std::move(sharedAudioAccumulators),
    };

    return params;
  }

  TestData<nva2e::IClassifierModel::EmotionExecutorCreationParameters> GetClassifierParameters() {
    constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json";
    auto modelInfo = nva2x::ToSharedPtr(nva2e::ReadClassifierModelInfo_INTERNAL(modelPath));
    EXPECT_TRUE(modelInfo);

    auto classifierParams = modelInfo->GetExecutorCreationParameters(
      60000, 60, 1, kInferencesToSkip
      );

    return {classifierParams, std::move(modelInfo)};
  }

  TestData<nva2e::IPostProcessModel::EmotionExecutorCreationParameters> GetPostProcessParameters() {
    constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json";
    auto modelInfo = nva2x::ToSharedPtr(nva2e::ReadClassifierModelInfo_INTERNAL(modelPath));
    EXPECT_TRUE(modelInfo);

    auto classifierParams = modelInfo->GetExecutorCreationParameters(
      60000, 60, 1, kInferencesToSkip
      );
    // FIXME: Read properly.
    nva2e::IPostProcessModel::EmotionExecutorCreationParameters postProcessParams;
    postProcessParams.samplingRate = classifierParams.networkInfo.bufferSamplerate;
    postProcessParams.inputStrength = classifierParams.inputStrength;
    postProcessParams.frameRateNumerator = classifierParams.frameRateNumerator;
    postProcessParams.frameRateDenominator = classifierParams.frameRateDenominator;
    postProcessParams.postProcessData = classifierParams.postProcessData;
    postProcessParams.postProcessParams = classifierParams.postProcessParams;
    postProcessParams.sharedPreferredEmotionAccumulators = classifierParams.sharedPreferredEmotionAccumulators;

    return {postProcessParams, std::move(modelInfo)};
  }

  void InitBundle(nva2e::IEmotionExecutorBundle& bundle) {
    const auto nbTracks = bundle.GetExecutor().GetNbTracks();
    const auto audio = GetAudio();
    const auto emotionSize = bundle.GetExecutor().GetEmotionsSize();
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
      auto& emotionAccumulator = bundle.GetPreferredEmotionAccumulator(trackIndex);
      std::vector<float> emotion(emotionAccumulator.GetEmotionSize(), 0.0f);
      EXPECT_EQ(emotionAccumulator.GetEmotionSize(), emotionSize);
      EXPECT_TRUE(!emotionAccumulator.Reset());
      EXPECT_TRUE(!emotionAccumulator.Accumulate(0, nva2x::ToConstView(emotion), bundle.GetCudaStream().Data()));
      EXPECT_TRUE(!emotionAccumulator.Close());

      auto& audioAccumulator = bundle.GetAudioAccumulator(trackIndex);
      EXPECT_TRUE(!audioAccumulator.Reset());
      EXPECT_TRUE(!audioAccumulator.Accumulate(nva2x::ToConstView(audio), bundle.GetCudaStream().Data()));
      EXPECT_TRUE(!audioAccumulator.Close());
    }
  }

  nva2x::UniquePtr<nva2e::IEmotionExecutorBundle> CreateEmotionExecutorClassifierBundle(bool init) {
    constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json";
    auto bundle = nva2x::ToUniquePtr(
      nva2e::ReadClassifierEmotionExecutorBundle_INTERNAL(
        kNbTracks,
        modelPath,
        60000, 60, 1, kInferencesToSkip,
        nullptr
        )
      );
    EXPECT_TRUE(bundle);

    if (init) {
      InitBundle(*bundle);
    }

    return bundle;
  }

  nva2x::UniquePtr<nva2e::IEmotionExecutorBundle> CreateEmotionExecutorPostProcessBundle(bool init) {
    constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json";
    auto bundle = nva2x::ToUniquePtr(
      nva2e::ReadPostProcessEmotionExecutorBundle_INTERNAL(
        kNbTracks,
        modelPath,
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

}


TEST(TestCoreExecutor, SanityEmotionExecutor) {
  // Compute expected number of frames.
  const auto audio = GetAudio();
  const auto nbFrames = (audio.size() * 60 + 16000 - 1) / 16000;

  // Record timestamps to compare executors.
  std::vector<std::vector<nva2e::IEmotionExecutor::timestamp_t>> expectedTimestamps;

  for (const auto createFunc : {&CreateEmotionExecutorClassifierBundle, &CreateEmotionExecutorPostProcessBundle}) {
    auto bundle = createFunc(true);
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
      nva2e::IEmotionExecutor* executor;
      const std::vector<std::vector<nva2e::IEmotionExecutor::timestamp_t>>* expectedTimestamps;
      std::vector<std::vector<nva2e::IEmotionExecutor::timestamp_t>> timestamps;
    };
    CallbackData callbackData;
    callbackData.nbExecutedFrames.resize(nbTracks);
    callbackData.executor = &bundle->GetExecutor();
    callbackData.timestamps.resize(nbTracks);
    callbackData.expectedTimestamps = &expectedTimestamps;
    auto callback = [](void* userdata, const nva2e::IEmotionExecutor::Results& results) {
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

TEST(TestCoreExecutor, ExecuteErrors) {
  // Compute expected number of frames.
  const auto audio = GetAudio();

  for (const auto createFunc : {&CreateEmotionExecutorClassifierBundle, &CreateEmotionExecutorPostProcessBundle}) {
    auto bundle = createFunc(true);
    ASSERT_TRUE(bundle);
    const auto nbTracks = bundle->GetExecutor().GetNbTracks();

    // Send empty callback.
    auto callback = [](void*, const nva2e::IEmotionExecutor::Results&) {
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
  const auto classifierParams = GetClassifierParameters();
  auto test_creation_classifier = [&classifierParams](
    const nva2e::EmotionExecutorCreationParameters& params
    ) {
    auto classifierExecutor = nva2x::ToUniquePtr(
      nva2e::CreateClassifierEmotionExecutor_INTERNAL(params, classifierParams.data)
    );
    const bool classifierSuccess = (classifierExecutor != nullptr);

    return classifierSuccess;
  };

  const auto postProcessParams = GetPostProcessParameters();
  auto test_creation_postprocess = [&postProcessParams](
    const nva2e::EmotionExecutorCreationParameters& params
    ) {
    auto postProcessExecutor = nva2x::ToUniquePtr(
      nva2e::CreatePostProcessEmotionExecutor_INTERNAL(params, postProcessParams.data)
    );
    const bool postProcessSuccess = (postProcessExecutor != nullptr);

    return postProcessSuccess;
  };

  // Test various combinations of parameters and how creation handles it.
  using test_func_t = std::function<void(nva2e::EmotionExecutorCreationParameters&)>;
  std::vector<test_func_t> test_funcs = {
    [&](nva2e::EmotionExecutorCreationParameters& params) {
      // No change, should work.
      EXPECT_TRUE(test_creation_classifier(params));
      EXPECT_TRUE(test_creation_postprocess(params));
    },
    [&](nva2e::EmotionExecutorCreationParameters& params) {
      // Default cuda stream, should work.
      params.cudaStream = nullptr;
      EXPECT_TRUE(test_creation_classifier(params));
      EXPECT_TRUE(test_creation_postprocess(params));
    },
    [&](nva2e::EmotionExecutorCreationParameters& params) {
      // 0 track, should fail.
      params.nbTracks = 0;
      EXPECT_FALSE(test_creation_classifier(params));
      EXPECT_FALSE(test_creation_postprocess(params));
    },
    [&](nva2e::EmotionExecutorCreationParameters& params) {
      params.nbTracks = 2048;
      const std::vector<const nva2x::IAudioAccumulator*> accumulators(params.nbTracks, params.sharedAudioAccumulators[0]);
      params.sharedAudioAccumulators = accumulators.data();
      // Too many tracks, should fail.
      EXPECT_FALSE(test_creation_classifier(params));
      // Post process can still work.
      EXPECT_TRUE(test_creation_postprocess(params));
    },
    [&](nva2e::EmotionExecutorCreationParameters& params) {
      // No audio accumulators, should fail.
      params.sharedAudioAccumulators = nullptr;
      EXPECT_FALSE(test_creation_classifier(params));
      EXPECT_FALSE(test_creation_postprocess(params));
    },
    [&](nva2e::EmotionExecutorCreationParameters& params) {
      // Missing audio accumulators, should fail.
      const_cast<const nva2x::IAudioAccumulator*&>(params.sharedAudioAccumulators[1]) = nullptr;
      EXPECT_FALSE(test_creation_classifier(params));
      EXPECT_FALSE(test_creation_postprocess(params));
    },
  };

  for (std::size_t i = 0; i < test_funcs.size(); ++i) {
    const auto& test_func = test_funcs[i];
    auto params = GetParameters();
    test_func(params.data);
  }
}

namespace {

  template <typename Params, typename CreationFunc>
  void AddLoadErrorTestCases(
    std::vector<std::function<void(Params&)>>& test_funcs,
    CreationFunc&& test_creation
  ) {
    using test_func_t = std::function<void(Params&)>;
    std::vector<test_func_t> new_test_funcs = {
      [&](auto& params) {
        // No change, should work.
        EXPECT_TRUE(test_creation(params));
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
        auto& data = params.postProcessData;
        data.inferenceEmotionLength /= 2;
        EXPECT_FALSE(test_creation(params));
      },
      [&](auto& params) {
        auto& data = params.postProcessData;
        data.emotionCorrespondence = nullptr;
        EXPECT_FALSE(test_creation(params));
      },
      [&](auto& params) {
        auto& data = params.postProcessData;
        data.emotionCorrespondenceSize /= 2;
        EXPECT_FALSE(test_creation(params));
      },
      [&](auto& params) {
        auto& p = params.postProcessParams;
        p.beginningEmotion = p.beginningEmotion.View(0, p.beginningEmotion.Size() / 2);
        EXPECT_FALSE(test_creation(params));
        p.beginningEmotion = p.beginningEmotion.View(0, 1);
        EXPECT_FALSE(test_creation(params));
        p.beginningEmotion = {};
        EXPECT_FALSE(test_creation(params));
      },
      [&](auto& params) {
        auto& p = params.postProcessParams;
        p.preferredEmotion = p.preferredEmotion.View(0, p.preferredEmotion.Size() / 2);
        EXPECT_FALSE(test_creation(params));
        p.preferredEmotion = p.preferredEmotion.View(0, 1);
        EXPECT_FALSE(test_creation(params));
        p.preferredEmotion = {};
        EXPECT_FALSE(test_creation(params));
      },
      [&](auto& params) {
        auto& p = params.postProcessParams;
        {
          // Emotion accumulator of the wrong size, should fail.
          auto accumulator = nva2x::ToUniquePtr(nva2x::internal::CreateEmotionAccumulator(11, 300, 0));
          EXPECT_TRUE(accumulator);
          std::vector<const nva2x::IEmotionAccumulator*> accumulators(kNbTracks, nullptr);
          accumulators[0] = accumulator.get();
          params.sharedPreferredEmotionAccumulators = accumulators.data();
          EXPECT_FALSE(test_creation(params));
        }
        {
          // Emotion accumulator of the right size, should succeed.
          auto accumulator = nva2x::ToUniquePtr(nva2x::internal::CreateEmotionAccumulator(10, 300, 0));
          EXPECT_TRUE(accumulator);
          std::vector<const nva2x::IEmotionAccumulator*> accumulators(kNbTracks, nullptr);
          accumulators[0] = accumulator.get();
          params.sharedPreferredEmotionAccumulators = accumulators.data();
          EXPECT_TRUE(test_creation(params));
        }
      },
      [&](auto& params) {
        auto& p = params.postProcessParams;
        p.preferredEmotion = {};
        EXPECT_FALSE(test_creation(params));
        {
          // Emotion accumulators added, should succeed.
          auto accumulator = nva2x::ToUniquePtr(nva2x::internal::CreateEmotionAccumulator(10, 300, 0));
          EXPECT_TRUE(accumulator);
          std::vector<const nva2x::IEmotionAccumulator*> accumulators(kNbTracks, accumulator.get());
          params.sharedPreferredEmotionAccumulators = accumulators.data();
          EXPECT_TRUE(test_creation(params));
        }
      },
    };
    test_funcs.insert(test_funcs.end(), new_test_funcs.begin(), new_test_funcs.end());
  }

} // Anonymous namespace

TEST(TestCoreExecutor, LoadErrorClassifier) {
  const auto defaultParams = GetParameters();
  auto test_creation = [&defaultParams](
    const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    const nva2e::EmotionExecutorCreationParameters* params = nullptr
    ) {
    if (!params) {
      params = &defaultParams.data;
    }

    auto classifierExecutor = nva2x::ToUniquePtr(
      nva2e::CreateClassifierEmotionExecutor_INTERNAL(*params, classifierParams)
    );
    const bool classifierSuccess = (classifierExecutor != nullptr);
    return classifierSuccess;
  };

  // Test various combinations of parameters and how creation handles it.
  using test_func_t = std::function<void(nva2e::IClassifierModel::EmotionExecutorCreationParameters&)>;
  std::vector<test_func_t> test_funcs;
  AddLoadErrorTestCases(test_funcs, test_creation);
  decltype(test_funcs) specific_test_funcs = {
    // Mess with the network info data.
    [&](auto& params) {
      params.networkInfo.bufferLength = 0;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.bufferSamplerate = 0;
      EXPECT_FALSE(test_creation(params));
    },
    [&](auto& params) {
      params.networkInfo.emotionLength *= 2;
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
  };
  test_funcs.insert(test_funcs.end(), specific_test_funcs.begin(), specific_test_funcs.end());

  const auto classifierParams = GetClassifierParameters();
  for (std::size_t i = 0; i < test_funcs.size(); ++i) {
    const auto& test_func = test_funcs[i];
    auto paramsToTest = classifierParams.data;
    test_func(paramsToTest);
  }
}

TEST(TestCoreExecutor, LoadErrorPostProcess) {
  const auto defaultParams = GetParameters();
  auto test_creation = [&defaultParams](
    const nva2e::IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams,
    const nva2e::EmotionExecutorCreationParameters* params = nullptr
    ) {
    if (!params) {
      params = &defaultParams.data;
    }

    auto postProcessExecutor = nva2x::ToUniquePtr(
      nva2e::CreatePostProcessEmotionExecutor_INTERNAL(*params, postProcessParams)
    );
    const bool postProcessSuccess = (postProcessExecutor != nullptr);
    return postProcessSuccess;
  };

  // Test various combinations of parameters and how creation handles it.
  using test_func_t = std::function<void(nva2e::IPostProcessModel::EmotionExecutorCreationParameters&)>;
  std::vector<test_func_t> test_funcs;
  AddLoadErrorTestCases(test_funcs, test_creation);
  decltype(test_funcs) specific_test_funcs = {
    [&](auto& params) {
      params.samplingRate = 0;
      EXPECT_FALSE(test_creation(params));
    },
  };
  test_funcs.insert(test_funcs.end(), specific_test_funcs.begin(), specific_test_funcs.end());

  const auto postProcessParams = GetPostProcessParameters();
  for (std::size_t i = 0; i < test_funcs.size(); ++i) {
    const auto& test_func = test_funcs[i];
    auto paramsToTest = postProcessParams.data;
    test_func(paramsToTest);
  }
}

TEST(TestCoreExecutor, Queries) {
  // Compute expected number of frames.
  const auto audio = GetAudio();
  const auto nbFrames = (audio.size() * 60 + 16000 - 1) / 16000;

  using func_t = nva2x::UniquePtr<nva2e::IEmotionExecutorBundle>(*)(bool);
  struct TestData {
    func_t createFunc;
    std::size_t nbFramesPerExecution;
    bool closeAddsFrames;
  };
  constexpr std::array<TestData, 2> testData = {{
    {CreateEmotionExecutorClassifierBundle, 1 + kInferencesToSkip, true},
    {CreateEmotionExecutorPostProcessBundle, 1, false},
  }};
  for (const auto& test : testData) {
    auto bundle = test.createFunc(false);
    ASSERT_TRUE(bundle);

    const auto& executor = bundle->GetExecutor();
    const auto& cudaStream = bundle->GetCudaStream();

    const auto trackIndex = kNbTracks / 2;

    ASSERT_EQ(executor.GetNbTracks(), kNbTracks);
    ASSERT_EQ(executor.GetSamplingRate(), 16000);
    std::size_t frameRateNumerator, frameRateDenominator;
    executor.GetFrameRate(frameRateNumerator, frameRateDenominator);
    ASSERT_EQ(frameRateNumerator, 60);
    ASSERT_EQ(frameRateDenominator, 1);

    ASSERT_EQ(executor.GetEmotionsSize(), 10);

    for (std::size_t i = 0; i < 1000; ++i) {
      const auto timestamp = executor.GetFrameTimestamp(i);
      const auto expectedTimestamp = i * 16000 * frameRateDenominator / frameRateNumerator;
      ASSERT_EQ(timestamp, static_cast<nva2x::IExecutor::timestamp_t>(expectedTimestamp));
    }

    // Test initial track state.
    ASSERT_FALSE(nva2x::HasExecutionStarted(executor));
    ASSERT_EQ(nva2x::GetNbReadyTracks(executor), 0);
    for (std::size_t i = 0; i < kNbTracks; i++) {
      ASSERT_FALSE(executor.HasExecutionStarted(i));
      ASSERT_EQ(executor.GetNbAvailableExecutions(i), 0);
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
        if (test.closeAddsFrames) {
          ASSERT_GT(executor.GetNbAvailableExecutions(i), previousNbAvailableExecutions);
        }
        else {
          ASSERT_GE(executor.GetNbAvailableExecutions(i), previousNbAvailableExecutions);
        }
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
      const nva2e::IEmotionExecutor& executor;
    };
    CallbackData callbackData{0, trackIndex, executor};
    auto resultsCallback = [](void* userdata, const nva2e::IEmotionExecutor::Results& results) {
      auto& callbackData = *static_cast<CallbackData*>(userdata);
      EXPECT_EQ(results.trackIndex, callbackData.trackIndex);
      EXPECT_EQ(results.timeStampCurrentFrame, callbackData.executor.GetFrameTimestamp(callbackData.nbExecutedFrames));
      EXPECT_EQ(results.timeStampNextFrame, callbackData.executor.GetFrameTimestamp(callbackData.nbExecutedFrames + 1));
      ++callbackData.nbExecutedFrames;
      return true;
    };
    std::size_t nbExecutedTracks = 0;
    // We should be able to set a null callback.
    ASSERT_TRUE(!bundle->GetExecutor().SetResultsCallback(nullptr, nullptr));
    // We shouldn't be able to set a null callback with non-null userdata.
    ASSERT_TRUE(bundle->GetExecutor().SetResultsCallback(nullptr, &callbackData));
    ASSERT_TRUE(!bundle->GetExecutor().SetResultsCallback(resultsCallback, &callbackData));

    ASSERT_TRUE(!bundle->GetExecutor().Execute(&nbExecutedTracks));
    ASSERT_EQ(nbExecutedTracks, 1);
    ASSERT_EQ(callbackData.nbExecutedFrames, 1*test.nbFramesPerExecution);

    nbExecutedTracks = 0;
    ASSERT_TRUE(!bundle->GetExecutor().Execute(&nbExecutedTracks));
    ASSERT_EQ(nbExecutedTracks, 1);
    ASSERT_EQ(callbackData.nbExecutedFrames, 2*test.nbFramesPerExecution);

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
    ASSERT_EQ(nva2x::GetNbReadyTracks(executor), 0);
    for (std::size_t i = 0; i < kNbTracks; i++) {
      ASSERT_FALSE(executor.HasExecutionStarted(i));
      ASSERT_EQ(executor.GetNbAvailableExecutions(i), 0);
      ASSERT_EQ(executor.GetTotalNbFrames(i), 0);
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

  for (const auto createFunc : {&CreateEmotionExecutorClassifierBundle, &CreateEmotionExecutorPostProcessBundle}) {
    auto bundle = createFunc(true);
    ASSERT_TRUE(bundle);

    const auto& cudaStream = bundle->GetCudaStream();
    auto& executor = bundle->GetExecutor();

    // Collect all the tracks.
    struct CallbackData {
      std::vector<std::vector<std::vector<float>>> emotionsData;
    };
    CallbackData callbackData;
    callbackData.emotionsData.resize(kNbTracks);
    auto callback = [](void* userdata, const nva2e::IEmotionExecutor::Results& results) {
      auto& callbackData = *static_cast<CallbackData*>(userdata);

      std::vector<float> frameData(results.emotions.Size());
      float* destination = frameData.data();
      EXPECT_TRUE(
        !nva2x::CopyDeviceToHost(
          {destination, results.emotions.Size()},
          results.emotions,
          results.cudaStream
        )
      );

      EXPECT_TRUE(!cudaStreamSynchronize(results.cudaStream));

      callbackData.emotionsData[results.trackIndex].emplace_back(std::move(frameData));
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
      const auto& referenceData = callbackData.emotionsData[0];
      ASSERT_EQ(nbFrames, referenceData.size());
      for (std::size_t trackIndex = 1; trackIndex < kNbTracks; ++trackIndex) {
        const auto& dataToTest = callbackData.emotionsData[trackIndex];
        ASSERT_EQ(referenceData, dataToTest);
      }
    }

    // Reset everything and re-run with a variable amount of audio data.
    for (std::size_t trackIndex = 0; trackIndex < kNbTracks; ++trackIndex) {
      callbackData.emotionsData[trackIndex].clear();
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
      const auto& referenceData = callbackData.emotionsData[0];
      ASSERT_EQ(nbFrames, referenceData.size());
      for (std::size_t trackIndex = 1; trackIndex < kNbTracks; ++trackIndex) {
        const auto& dataToTest = callbackData.emotionsData[trackIndex];
        ASSERT_EQ(referenceData, dataToTest);
      }
    }

    // Reset everything and re-run with a variable amount of emotion data.
    for (std::size_t trackIndex = 0; trackIndex < kNbTracks; ++trackIndex) {
      callbackData.emotionsData[trackIndex].clear();
      ASSERT_TRUE(!executor.Reset(trackIndex));
      ASSERT_TRUE(!bundle->GetPreferredEmotionAccumulator(trackIndex).Reset());
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
        auto& emotionAccumulator = bundle->GetPreferredEmotionAccumulator(trackIndex);
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
      const auto& referenceData = callbackData.emotionsData[0];
      ASSERT_EQ(nbFrames, referenceData.size());
      for (std::size_t trackIndex = 1; trackIndex < kNbTracks; ++trackIndex) {
        const auto& dataToTest = callbackData.emotionsData[trackIndex];
        ASSERT_EQ(referenceData, dataToTest);
      }
    }

    // Reset everything and re-run with a random reset in the middle.
    const auto previousReferenceData = callbackData.emotionsData[0];
    for (std::size_t trackIndex = 0; trackIndex < kNbTracks; ++trackIndex) {
      callbackData.emotionsData[trackIndex].clear();
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
          callbackData.emotionsData[trackIndex].clear();
          resetIndices[trackIndex] = 0;
        }
      }
    }

    // Everything should be the same.
    {
      const auto& referenceData = previousReferenceData;
      ASSERT_EQ(nbFrames, referenceData.size());
      for (std::size_t trackIndex = 1; trackIndex < kNbTracks; ++trackIndex) {
        const auto& dataToTest = callbackData.emotionsData[trackIndex];
        ASSERT_EQ(referenceData, dataToTest);
      }
    }
  }
}

TEST(TestCoreExecutor, ExecutionLimitedByEmotions) {
  const auto audio = GetAudio();
  const auto nbFrames = (audio.size() * 60 + 16000 - 1) / 16000;

  using func_t = nva2x::UniquePtr<nva2e::IEmotionExecutorBundle>(*)(bool);
  struct TestData {
    func_t createFunc;
    std::size_t nbFramesPerExecution;
    std::size_t nbFramesBeforeStart;
  };
  constexpr std::array<TestData, 2> testData = {{
    {CreateEmotionExecutorClassifierBundle, kInferencesToSkip + 1, 0},
    {CreateEmotionExecutorPostProcessBundle, 1, 0},
  }};
  for (const auto& test : testData) {
    auto bundle = test.createFunc(false);
    ASSERT_TRUE(bundle);

    const auto cudaStream = bundle->GetCudaStream().Data();

    auto& audioAccumulator = bundle->GetAudioAccumulator(0);
    ASSERT_TRUE(!audioAccumulator.Accumulate(nva2x::ToConstView(audio), bundle->GetCudaStream().Data()));
    ASSERT_TRUE(!audioAccumulator.Close());

    auto& executor = bundle->GetExecutor();
    auto callback = [](void*, const nva2e::IEmotionExecutor::Results&) { return true; };
    ASSERT_TRUE(!executor.SetResultsCallback(callback, nullptr));

    // Enable preferred emotion so they are taken into account.
    nva2e::PostProcessParams postProcessParams;
    ASSERT_TRUE(!nva2e::GetExecutorPostProcessParameters_INTERNAL(executor, 0, postProcessParams));
    postProcessParams.enablePreferredEmotion = true;
    ASSERT_TRUE(!nva2e::SetExecutorPostProcessParameters_INTERNAL(executor, 0, postProcessParams));

    // Check when no emotions are available.
    ASSERT_EQ(executor.GetNbAvailableExecutions(0), 0);

    auto& emotionAccumulator = bundle->GetPreferredEmotionAccumulator(0);
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

TEST(TestCoreExecutor, PreferredEmotion) {
  const auto audio = GetAudio();
  const auto nbFrames = (audio.size() * 60 + 16000 - 1) / 16000;

  // Create random values for the emotions.
  const int seed = static_cast<unsigned int>(time(NULL));
  std::cout << "Current srand seed: " << seed << std::endl;
  std::srand(seed); // make random inputs reproducible

  static constexpr const std::size_t kNbEmotions = 10;
  std::vector<std::vector<std::vector<float>>> emotionsToPass(kNbTracks);
  for (auto& trackEmotions : emotionsToPass) {
    trackEmotions.resize(nbFrames);
    for (auto& frameEmotions : trackEmotions) {
      frameEmotions.resize(kNbEmotions);
      for (auto& emotion : frameEmotions) {
        emotion = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      }
    }
  }

  for (const auto createFunc : {&CreateEmotionExecutorClassifierBundle, &CreateEmotionExecutorPostProcessBundle}) {
    auto bundle = createFunc(true);
    ASSERT_TRUE(bundle);

    const auto cudaStream = bundle->GetCudaStream().Data();
    auto& executor = bundle->GetExecutor();

    // Collect all the tracks.
    struct CallbackData {
      std::vector<std::vector<std::vector<float>>> emotionsData;
    };
    CallbackData callbackData;
    callbackData.emotionsData.resize(kNbTracks);
    auto callback = [](void* userdata, const nva2e::IEmotionExecutor::Results& results) {
      auto& callbackData = *static_cast<CallbackData*>(userdata);

      std::vector<float> frameData(results.emotions.Size());
      float* destination = frameData.data();
      EXPECT_TRUE(
        !nva2x::CopyDeviceToHost(
          {destination, results.emotions.Size()},
          results.emotions,
          results.cudaStream
        )
      );

      EXPECT_TRUE(!cudaStreamSynchronize(results.cudaStream));

      callbackData.emotionsData[results.trackIndex].emplace_back(std::move(frameData));
      return true;
    };
    ASSERT_TRUE(!executor.SetResultsCallback(callback, &callbackData));

    // Pass the emotions to the executor.
    for (std::size_t trackIndex = 0; trackIndex < kNbTracks; ++trackIndex) {
      auto& emotionAccumulator = bundle->GetPreferredEmotionAccumulator(trackIndex);
      ASSERT_TRUE(!emotionAccumulator.Reset());
      for (std::size_t frameIndex = 0; frameIndex < nbFrames; ++frameIndex) {
        const auto timestamp = executor.GetFrameTimestamp(frameIndex);
        const auto emotions = nva2x::ToConstView(emotionsToPass[trackIndex][frameIndex]);
        ASSERT_TRUE(!emotionAccumulator.Accumulate(timestamp, emotions, cudaStream));
      }
      ASSERT_TRUE(!emotionAccumulator.Close());
    }

    // Set the right parameters.
    nva2e::PostProcessParams postProcessParams;
    for (std::size_t trackIndex = 0; trackIndex < kNbTracks; ++trackIndex) {
      ASSERT_TRUE(!nva2e::GetExecutorPostProcessParameters_INTERNAL(executor, trackIndex, postProcessParams));
      postProcessParams.enablePreferredEmotion = true;
      postProcessParams.preferredEmotionStrength = 1.0f;
      postProcessParams.emotionStrength = 1.0f;
      postProcessParams.liveTransitionTime = 0.0f;
      ASSERT_TRUE(!nva2e::SetExecutorPostProcessParameters_INTERNAL(executor, trackIndex, postProcessParams));
    }

    // Execute everything.
    while (nva2x::internal::GetNbReadyTracks(executor) > 0) {
      std::size_t nbExecutedTracks = 0;
      ASSERT_TRUE(!executor.Execute(&nbExecutedTracks));
      ASSERT_EQ(nbExecutedTracks, kNbTracks);
    }

    // Everything should be the same.
    {
      for (std::size_t trackIndex = 0; trackIndex < kNbTracks; ++trackIndex) {
        const auto& referenceData = emotionsToPass[trackIndex];
        ASSERT_EQ(nbFrames, referenceData.size());
        const auto& dataToTest = callbackData.emotionsData[trackIndex];
        ASSERT_EQ(referenceData, dataToTest);
      }
    }
  }
}
