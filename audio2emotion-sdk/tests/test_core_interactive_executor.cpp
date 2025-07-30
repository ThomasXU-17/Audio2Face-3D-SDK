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
#include "audio2emotion/internal/interactive_executor_classifier.h"
#include "audio2emotion/internal/interactive_executor_postprocess.h"

#include <gtest/gtest.h>

#include <any>

#define USE_EXPLICIT_CHECKS 1

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

  nva2x::UniquePtr<nva2e::IEmotionExecutorBundle> CreateEmotionExecutorBundle(bool classifier, std::size_t inferencesToSkip, bool init) {
    nva2x::UniquePtr<nva2e::IEmotionExecutorBundle> bundle;
    if (classifier) {
      constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json";
      bundle = nva2x::ToUniquePtr(
        nva2e::ReadClassifierEmotionExecutorBundle_INTERNAL(
          1,
          modelPath,
          60000, 60, 1, inferencesToSkip,
          nullptr
          )
        );
    }
    else {
      // We also read the model info, but don't actually need it.
      // It just makes initialization simpler and closer to the classifier case.
      constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json";
      bundle = nva2x::ToUniquePtr(
        nva2e::ReadPostProcessEmotionExecutorBundle_INTERNAL(
          1,
          modelPath,
          60, 1,
          nullptr
          )
        );
    }
    EXPECT_TRUE(bundle);

    if (init) {
      InitBundle(*bundle);
    }

    return bundle;
  }

  struct InteractiveExecutorBundle {
    nva2x::CudaStream cudaStream;
    nva2x::AudioAccumulator audioAccumulator;
    nva2x::EmotionAccumulator emotionAccumulator;
    nva2x::UniquePtr<nva2e::IEmotionInteractiveExecutor> interactiveExecutor;

    inline nva2x::CudaStream& GetCudaStream() { return cudaStream; }
    inline nva2x::AudioAccumulator& GetAudioAccumulator() { return audioAccumulator; }
    inline nva2x::EmotionAccumulator& GetPreferredEmotionAccumulator() { return emotionAccumulator; }
    inline nva2e::IEmotionInteractiveExecutor& GetExecutor() { return *interactiveExecutor; }
  };

  std::unique_ptr<InteractiveExecutorBundle> CreateEmotionInteractiveExecutorBundle(
    bool classifier, std::size_t inferencesToSkip, std::size_t batchSize, bool init
    ) {
    auto bundle = std::make_unique<InteractiveExecutorBundle>();
    EXPECT_TRUE(!bundle->cudaStream.Init());
    EXPECT_TRUE(!bundle->audioAccumulator.Allocate(16000, 0));
    EXPECT_TRUE(!bundle->emotionAccumulator.Allocate(10, 300, 0));

    nva2e::EmotionExecutorCreationParameters params;
    params.cudaStream = bundle->cudaStream.Data();
    params.nbTracks = 1;
    const nva2x::IAudioAccumulator* audioAccumulator = &bundle->audioAccumulator;
    params.sharedAudioAccumulators = &audioAccumulator;

    constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json";
    auto modelInfo = nva2x::ToUniquePtr(nva2e::ReadClassifierModelInfo_INTERNAL(modelPath));
    EXPECT_TRUE(modelInfo);

    const auto classifierParams = modelInfo->GetExecutorCreationParameters(
      60000, 60, 1, inferencesToSkip
      );

    if (classifier) {
      bundle->interactiveExecutor = nva2x::ToUniquePtr(
        nva2e::CreateClassifierEmotionInteractiveExecutor_INTERNAL(
          params,
          classifierParams,
          batchSize
          )
        );
    }
    else {
      bundle->interactiveExecutor = nva2x::ToUniquePtr(
        nva2e::CreatePostProcessEmotionInteractiveExecutor_INTERNAL(
          params,
          classifierParams,
          batchSize
          )
        );
    }
    EXPECT_TRUE(bundle->interactiveExecutor);

    if (init) {
      const auto audio = GetAudio();
      const auto emotionSize = bundle->GetExecutor().GetEmotionsSize();

      auto& emotionAccumulator = bundle->GetPreferredEmotionAccumulator();
      std::vector<float> emotion(emotionAccumulator.GetEmotionSize(), 0.0f);
      EXPECT_EQ(emotionAccumulator.GetEmotionSize(), emotionSize);
      EXPECT_TRUE(!emotionAccumulator.Reset());
      EXPECT_TRUE(!emotionAccumulator.Accumulate(0, nva2x::ToConstView(emotion), bundle->GetCudaStream().Data()));
      EXPECT_TRUE(!emotionAccumulator.Close());

      auto& audioAccumulator = bundle->GetAudioAccumulator();
      EXPECT_TRUE(!audioAccumulator.Reset());
      EXPECT_TRUE(!audioAccumulator.Accumulate(nva2x::ToConstView(audio), bundle->GetCudaStream().Data()));
      EXPECT_TRUE(!audioAccumulator.Close());
    }

    return bundle;
  }

  struct frame_t {
    std::int64_t timestamp;
    std::array<float, 10> emotions;

    bool operator==(const frame_t& other) const {
      return timestamp == other.timestamp && emotions == other.emotions;
    }
  };
  using frames_t = std::vector<frame_t>;

  bool callback(void* userdata, const nva2e::IEmotionExecutor::Results& results) {
    auto* frames = static_cast<frames_t*>(userdata);
    frame_t frame;
    frame.timestamp = results.timeStampCurrentFrame;
    EXPECT_TRUE(!nva2x::CopyDeviceToHost(
      {frame.emotions.data(), frame.emotions.size()}, results.emotions, results.cudaStream
      ));
    frames->emplace_back(std::move(frame));
    return true;
  };

}


TEST(TestCoreInteractiveExecutor, Correctness) {
  const std::size_t kInferencesToSkip = 10;
  for (const auto classifier : {true, false}) {
    auto executorBundle = CreateEmotionExecutorBundle(classifier, kInferencesToSkip, true);
    ASSERT_TRUE(executorBundle);

    std::vector<nva2e::PostProcessParams> postProcessParams(1);
    ASSERT_TRUE(!nva2e::GetExecutorPostProcessParameters_INTERNAL(
      executorBundle->GetExecutor(), 0, postProcessParams[0]
      ));
    // Do deep copies because the executor will be deleted.
    std::vector<float> beginningEmotion(
      postProcessParams[0].beginningEmotion.Data(),
      postProcessParams[0].beginningEmotion.Data() + postProcessParams[0].beginningEmotion.Size()
      );
    postProcessParams[0].beginningEmotion = nva2x::ToConstView(beginningEmotion);
    std::vector<float> preferredEmotion(
      postProcessParams[0].preferredEmotion.Data(),
      postProcessParams[0].preferredEmotion.Data() + postProcessParams[0].preferredEmotion.Size()
      );
    postProcessParams[0].preferredEmotion = nva2x::ToConstView(preferredEmotion);
    // Add another variation.
    postProcessParams.emplace_back(postProcessParams.back());
    postProcessParams.back().emotionContrast *= 2.0f;

    // Compute ground truth data from regular executor.
    std::vector<frames_t> groundTruth;
    for (const auto& params : postProcessParams) {
      ASSERT_TRUE(!executorBundle->GetExecutor().Reset(0));

      groundTruth.emplace_back();
      ASSERT_TRUE(!executorBundle->GetExecutor().SetResultsCallback(callback, &groundTruth.back()));

      ASSERT_TRUE(!nva2e::SetExecutorPostProcessParameters_INTERNAL(
        executorBundle->GetExecutor(), 0, params
        ));

      // Run the executor.
      while (nva2x::GetNbReadyTracks(executorBundle->GetExecutor()) > 0) {
        ASSERT_TRUE(!executorBundle->GetExecutor().Execute(nullptr));
      }
    }

    executorBundle.reset();

    // Create interactive executor.
    auto interactiveExecutorBundle = CreateEmotionInteractiveExecutorBundle(classifier, kInferencesToSkip, 32, true);
    ASSERT_TRUE(interactiveExecutorBundle);

    // Not having a callback should trigger an error, even if there is available data.
    ASSERT_LT(0, interactiveExecutorBundle->GetExecutor().GetTotalNbFrames());
    ASSERT_TRUE(interactiveExecutorBundle->GetExecutor().ComputeAllFrames());
    ASSERT_TRUE(interactiveExecutorBundle->GetExecutor().ComputeFrame(0));

    const int seed = static_cast<unsigned int>(time(NULL));
    std::cout << "Current srand seed: " << seed << std::endl;
    std::srand(seed); // make random inputs reproducible
    for (std::size_t i = 0; i < postProcessParams.size(); ++i) {
      frames_t interactiveFrames;
      ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().SetResultsCallback(callback, &interactiveFrames));

      // Set the parameters.
      ASSERT_TRUE(!nva2e::SetInteractiveExecutorPostProcessParameters_INTERNAL(
        interactiveExecutorBundle->GetExecutor(), postProcessParams[i]
        ));
      ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().Invalidate(nva2e::IEmotionInteractiveExecutor::kLayerAll));

      // Run the executor.
      ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().ComputeAllFrames());

      #if USE_EXPLICIT_CHECKS
      ASSERT_EQ(groundTruth[i].size(), interactiveFrames.size()) << "Index " << i;
      for (std::size_t j = 0; j < groundTruth[i].size(); ++j) {
        ASSERT_EQ(groundTruth[i][j].timestamp, interactiveFrames[j].timestamp) << "Index " << i << " frame " << j;
        for (std::size_t k = 0; k < groundTruth[i][j].emotions.size(); ++k) {
          ASSERT_EQ(groundTruth[i][j].emotions[k], interactiveFrames[j].emotions[k]) << "Index " << i << " frame " << j << " emotion " << k;
        }
      }
      #endif
      ASSERT_EQ(groundTruth[i], interactiveFrames) << "Index " << i;

      // Test a few random frames.
      for (int k = 0; k < 10; ++k) {
        const int paramIndex = std::rand() % postProcessParams.size();
        const int frameIndex = std::rand() % groundTruth[i].size();
        interactiveFrames.clear();

        ASSERT_TRUE(!nva2e::SetInteractiveExecutorPostProcessParameters_INTERNAL(
          interactiveExecutorBundle->GetExecutor(), postProcessParams[paramIndex]
          ));
        ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().ComputeFrame(frameIndex));

        ASSERT_EQ(1, interactiveFrames.size());
        #if USE_EXPLICIT_CHECKS
        ASSERT_EQ(groundTruth[paramIndex][frameIndex].timestamp, interactiveFrames[0].timestamp) << "Index " << i << " attempt " << k << " frame " << frameIndex;
        for (std::size_t k = 0; k < groundTruth[paramIndex][frameIndex].emotions.size(); ++k) {
          ASSERT_EQ(groundTruth[paramIndex][frameIndex].emotions[k], interactiveFrames[0].emotions[k]) << "Index " << i << " attempt " << k << " frame " << frameIndex << " emotion " << k;
        }
        #endif
        ASSERT_EQ(groundTruth[paramIndex][frameIndex], interactiveFrames[0]) << "Index " << i << " attempt " << k << " frame " << frameIndex;
      }
    }
  }
}


TEST(TestCoreInteractiveExecutor, BatchSize) {
  for (const auto classifier : {true, false}) {
    std::vector<frames_t> groundTruth;
    for (const auto batchSize : {1, 0, 32}) {
      // Create interactive executor.
      auto interactiveExecutorBundle = CreateEmotionInteractiveExecutorBundle(classifier, 0, batchSize, true);
      ASSERT_TRUE(interactiveExecutorBundle);

      frames_t interactiveFrames;
      ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().SetResultsCallback(callback, &interactiveFrames));

      // Run the executor.
      ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().ComputeAllFrames());

      if (groundTruth.empty()) {
        groundTruth.emplace_back(std::move(interactiveFrames));
      } else {
        ASSERT_EQ(groundTruth[0], interactiveFrames) << "batch size " << batchSize;
      }
    }
  }
}


TEST(TestCoreInteractiveExecutor, InferencesToSkip) {
  // Only test the classifier case, since inference to skip is irrelevant for post-process executor.
  for (const auto classifier : {true}) {
    constexpr const std::array<std::size_t, 4> kInferencesToSkip = {0, 1, 30, 15};

    // Compute ground truth data from regular executor.
    std::vector<frames_t> groundTruth;
    for (const auto inferencesToSkip : kInferencesToSkip) {
      auto executorBundle = CreateEmotionExecutorBundle(classifier, inferencesToSkip, true);
      ASSERT_TRUE(executorBundle);

      groundTruth.emplace_back();
      ASSERT_TRUE(!executorBundle->GetExecutor().SetResultsCallback(callback, &groundTruth.back()));

      // Run the executor.
      while (nva2x::GetNbReadyTracks(executorBundle->GetExecutor()) > 0) {
        ASSERT_TRUE(!executorBundle->GetExecutor().Execute(nullptr));
      }
    }

    // Create interactive executor with batch size 0 (i.e. max batch size) for faster execution.
    auto interactiveExecutorBundle = CreateEmotionInteractiveExecutorBundle(classifier, kInferencesToSkip[0], 0, true);
    ASSERT_TRUE(interactiveExecutorBundle);

    const int seed = static_cast<unsigned int>(time(NULL));
    std::cout << "Current srand seed: " << seed << std::endl;
    std::srand(seed); // make random inputs reproducible
    for (std::size_t i = 0; i < kInferencesToSkip.size(); ++i) {
      const auto inferencesToSkip = kInferencesToSkip[i];
      frames_t interactiveFrames;
      ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().SetResultsCallback(callback, &interactiveFrames));

      // Set the inferences to skip.
      ASSERT_TRUE(!nva2e::SetInteractiveExecutorInferencesToSkip_INTERNAL(
        interactiveExecutorBundle->GetExecutor(), inferencesToSkip
        ));
      ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().Invalidate(nva2e::IEmotionInteractiveExecutor::kLayerAll));

      // Run the executor.
      ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().ComputeAllFrames());

      #if USE_EXPLICIT_CHECKS
      ASSERT_EQ(groundTruth[i].size(), interactiveFrames.size()) << "inferences to skip " << inferencesToSkip;
      for (std::size_t j = 0; j < groundTruth[i].size(); ++j) {
        ASSERT_EQ(groundTruth[i][j].timestamp, interactiveFrames[j].timestamp) << "inferences to skip " << inferencesToSkip << " frame " << j;
        for (std::size_t k = 0; k < groundTruth[i][j].emotions.size(); ++k) {
          ASSERT_EQ(groundTruth[i][j].emotions[k], interactiveFrames[j].emotions[k]) << "inferences to skip " << inferencesToSkip << " frame " << j << " emotion " << k;
        }
      }
      #endif
      ASSERT_EQ(groundTruth[i], interactiveFrames) << "inferences to skip " << inferencesToSkip;

      // Evaluate a frame with a different inference to skip.
      const int inferencesToSkipIndex = (i + std::rand() % (kInferencesToSkip.size() - 1)) % kInferencesToSkip.size();
      const int frameIndex = std::rand() % groundTruth[inferencesToSkipIndex].size();
      interactiveFrames.clear();

      ASSERT_TRUE(!nva2e::SetInteractiveExecutorInferencesToSkip_INTERNAL(
        interactiveExecutorBundle->GetExecutor(), kInferencesToSkip[inferencesToSkipIndex]
        ));
      ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().ComputeFrame(frameIndex));

      #if USE_EXPLICIT_CHECKS
      ASSERT_EQ(groundTruth[inferencesToSkipIndex][frameIndex].timestamp, interactiveFrames[0].timestamp) << "inferences to skip " << inferencesToSkipIndex << " frame " << frameIndex;
      for (std::size_t k = 0; k < groundTruth[inferencesToSkipIndex][frameIndex].emotions.size(); ++k) {
        ASSERT_EQ(groundTruth[inferencesToSkipIndex][frameIndex].emotions[k], interactiveFrames[0].emotions[k]) << "inferences to skip " << inferencesToSkipIndex << " frame " << frameIndex << " emotion " << k;
      }
      #endif
      ASSERT_EQ(groundTruth[inferencesToSkipIndex][frameIndex], interactiveFrames[0])
        << "i " << i << " inferences to skip " << inferencesToSkipIndex << " frame " << frameIndex;
    }
  }
}


TEST(TestCoreInteractiveExecutor, Invalidation) {
  for (const auto classifier : {true, false}) {
    // Create interactive executor with batch size 0 (i.e. max batch size) for faster execution.
    auto interactiveExecutorBundle = CreateEmotionInteractiveExecutorBundle(classifier, 59, 0, true);
    ASSERT_TRUE(interactiveExecutorBundle);

    auto callback = [](void* userdata, const nva2e::IEmotionExecutor::Results& results) {
      return true;
    };
    ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().SetResultsCallback(callback, nullptr));

    using test_func_t = std::function<void(nva2e::IEmotionInteractiveExecutor&)>;
    std::vector<test_func_t> test_funcs = {
      [](auto& executor) {
        // Do nothing.

        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerAll));
        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerInference));
        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing));
      },
      [](auto& executor) {
        // Invalidate the audio accumulator.
        ASSERT_TRUE(!executor.Invalidate(nva2e::IEmotionInteractiveExecutor::kLayerAudioAccumulator));

        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerAll));
        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerInference));
        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing));
      },
      [](auto& executor) {
        // Invalidate the preferred emotion accumulator.
        ASSERT_TRUE(!executor.Invalidate(nva2e::IEmotionInteractiveExecutor::kLayerPreferredEmotionAccumulator));

        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerAll));
        ASSERT_TRUE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerInference));
        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing));
      },
      [](auto& executor) {
        ASSERT_TRUE(!nva2e::SetInteractiveExecutorInferencesToSkip_INTERNAL(executor, 119));

        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerAll));
        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerInference));
        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing));
      },
      [classifier](auto& executor) {
        // Same value as before.
        ASSERT_TRUE(!nva2e::SetInteractiveExecutorInferencesToSkip_INTERNAL(executor, 119));

        // Post-processing executor always invalidates.
        const bool valid = classifier;
        ASSERT_EQ(valid, executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerAll));
        ASSERT_EQ(valid, executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerInference));
        ASSERT_EQ(valid, executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing));
      },
      [](auto& executor) {
        ASSERT_TRUE(!nva2e::SetInteractiveExecutorInputStrength_INTERNAL(executor, 1.23f));

        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerAll));
        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerInference));
        ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing));
      },
      [](auto& executor) {
        // Same value as before.
        ASSERT_TRUE(!nva2e::SetInteractiveExecutorInputStrength_INTERNAL(executor, 1.23f));

        ASSERT_TRUE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerAll));
        ASSERT_TRUE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerInference));
        ASSERT_TRUE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing));
      },
    };

    // Add test cases for tweaking parameters.
    using test_param_func_t = std::function<bool(nva2e::PostProcessParams&)>;
    std::vector<test_param_func_t> test_param_funcs = {
      [](auto& params) {
        params.emotionContrast = 1.23f;
        return true;
      },
      [](auto& params) {
        params.emotionContrast = 1.23f;
        return false;
      },
      [](auto& params) {
        params.maxEmotions = 5;
        return true;
      },
      [](auto& params) {
        params.maxEmotions = 5;
        return false;
      },
      [](auto& params) {
        static std::vector<float> beginningEmotion(
          params.beginningEmotion.Data(), params.beginningEmotion.Data() + params.beginningEmotion.Size()
          );
        params.beginningEmotion = nva2x::ToConstView(beginningEmotion);
        return false;
      },
      [](auto& params) {
        static std::vector<float> beginningEmotion(
          params.beginningEmotion.Data(), params.beginningEmotion.Data() + params.beginningEmotion.Size()
          );
        beginningEmotion[4] = 0.987f;
        params.beginningEmotion = nva2x::ToConstView(beginningEmotion);
        return true;
      },
      [](auto& params) {
        static std::vector<float> beginningEmotion(
          params.beginningEmotion.Data(), params.beginningEmotion.Data() + params.beginningEmotion.Size()
          );
        beginningEmotion[4] = 0.987f;
        params.beginningEmotion = nva2x::ToConstView(beginningEmotion);
        return false;
      },
      [](auto& params) {
        static std::vector<float> preferredEmotion(
          params.preferredEmotion.Data(), params.preferredEmotion.Data() + params.preferredEmotion.Size()
          );
        params.preferredEmotion = nva2x::ToConstView(preferredEmotion);
        return false;
      },
      [](auto& params) {
        static std::vector<float> preferredEmotion(
          params.preferredEmotion.Data(), params.preferredEmotion.Data() + params.preferredEmotion.Size()
          );
        preferredEmotion[4] = 0.987f;
        params.preferredEmotion = nva2x::ToConstView(preferredEmotion);
        return true;
      },
      [](auto& params) {
        static std::vector<float> preferredEmotion(
          params.preferredEmotion.Data(), params.preferredEmotion.Data() + params.preferredEmotion.Size()
          );
        preferredEmotion[4] = 0.987f;
        params.preferredEmotion = nva2x::ToConstView(preferredEmotion);
        return false;
      },
      [](auto& params) {
        params.liveBlendCoef = 0.456f;
        return true;
      },
      [](auto& params) {
        params.liveBlendCoef = 0.456f;
        return false;
      },
      [](auto& params) {
        params.enablePreferredEmotion = true;
        return true;
      },
      [](auto& params) {
        params.enablePreferredEmotion = true;
        return false;
      },
      [](auto& params) {
        params.preferredEmotionStrength = 0.123f;
        return true;
      },
      [](auto& params) {
        params.preferredEmotionStrength = 0.123f;
        return false;
      },
      [](auto& params) {
        params.liveTransitionTime = 0.234f;
        return true;
      },
      [](auto& params) {
        params.liveTransitionTime = 0.234f;
        return false;
      },
      [](auto& params) {
        params.fixedDt = 0.345f;
        return true;
      },
      [](auto& params) {
        params.fixedDt = 0.345f;
        return false;
      },
      [](auto& params) {
        params.emotionStrength = 0.456f;
        return true;
      },
      [](auto& params) {
        params.emotionStrength = 0.456f;
        return false;
      },
    };
    for (const auto& test_param_func : test_param_funcs) {
      test_funcs.emplace_back([&](auto& executor) {
        nva2e::PostProcessParams params;
        ASSERT_TRUE(!nva2e::GetInteractiveExecutorPostProcessParameters_INTERNAL(executor, params));
        const bool changed = test_param_func(params);
        ASSERT_TRUE(!nva2e::SetInteractiveExecutorPostProcessParameters_INTERNAL(executor, params));

        ASSERT_EQ(!changed, executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerAll));
        ASSERT_EQ(true, executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerInference));
        ASSERT_EQ(!changed, executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing));
      });
    };

    for (const auto& test_func : test_funcs) {
      auto& executor = interactiveExecutorBundle->GetExecutor();
      test_func(executor);

      static_assert(nva2e::IEmotionInteractiveExecutor::kLayerAudioAccumulator == nva2e::IEmotionInteractiveExecutor::kLayerInference);
      static_assert(nva2e::IEmotionInteractiveExecutor::kLayerPreferredEmotionAccumulator == nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing);

      // Reset the invalidation state.
      ASSERT_TRUE(!executor.ComputeAllFrames());

      ASSERT_TRUE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerNone));
      ASSERT_TRUE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerAll));
      ASSERT_TRUE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerInference));
      ASSERT_TRUE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing));
      ASSERT_FALSE(executor.IsValid(nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing + 1));
    }
  }
}
