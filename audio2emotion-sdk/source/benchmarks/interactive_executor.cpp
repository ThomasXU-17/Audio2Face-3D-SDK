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
#include "audio2x/cuda_stream.h"

#include <benchmark/benchmark.h>

#include "utils.h"


namespace {

  std::vector<float> GetAudio() {
    return loadAudio();
  }

  struct InteractiveExecutorBundle {
    UniquePtr<nva2x::ICudaStream> cudaStream;
    UniquePtr<nva2x::IAudioAccumulator> audioAccumulator;
    UniquePtr<nva2x::IEmotionAccumulator> emotionAccumulator;
    UniquePtr<nva2e::IEmotionInteractiveExecutor> interactiveExecutor;

    inline nva2x::ICudaStream& GetCudaStream() { return *cudaStream; }
    inline nva2x::IAudioAccumulator& GetAudioAccumulator() { return *audioAccumulator; }
    inline nva2x::IEmotionAccumulator& GetPreferredEmotionAccumulator() { return *emotionAccumulator; }
    inline nva2e::IEmotionInteractiveExecutor& GetExecutor() { return *interactiveExecutor; }
  };

  std::unique_ptr<InteractiveExecutorBundle> CreateEmotionInteractiveExecutorBundle(
    bool classifier, std::size_t inferencesToSkip, std::size_t batchSize
    ) {
    auto bundle = std::make_unique<InteractiveExecutorBundle>();

    bundle->cudaStream = ToUniquePtr(nva2x::CreateCudaStream());
    CHECK_TRUE(bundle->cudaStream);

    bundle->audioAccumulator = ToUniquePtr(nva2x::CreateAudioAccumulator(16000, 0));
    CHECK_TRUE(bundle->audioAccumulator);

    bundle->emotionAccumulator = ToUniquePtr(nva2x::CreateEmotionAccumulator(10, 300, 0));
    CHECK_TRUE(bundle->emotionAccumulator);

    nva2e::EmotionExecutorCreationParameters params;
    params.cudaStream = bundle->cudaStream->Data();
    params.nbTracks = 1;
    const nva2x::IAudioAccumulator* audioAccumulatorPtr = bundle->audioAccumulator.get();
    params.sharedAudioAccumulators = &audioAccumulatorPtr;

    constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json";
    auto modelInfo = ToUniquePtr(nva2e::ReadClassifierModelInfo(modelPath));
    CHECK_TRUE(modelInfo);

    auto classifierParams = modelInfo->GetExecutorCreationParameters(
      60000, 60, 1, inferencesToSkip
      );
    const nva2x::IEmotionAccumulator* emotionAccumulatorPtr = bundle->emotionAccumulator.get();
    classifierParams.sharedPreferredEmotionAccumulators = &emotionAccumulatorPtr;

    if (classifier) {
      bundle->interactiveExecutor = ToUniquePtr(
        nva2e::CreateClassifierEmotionInteractiveExecutor(
          params,
          classifierParams,
          batchSize
          )
        );
    }
    else {
      bundle->interactiveExecutor = ToUniquePtr(
        nva2e::CreatePostProcessEmotionInteractiveExecutor(
          params,
          classifierParams,
          batchSize
          )
        );
    }
    CHECK_TRUE(bundle->interactiveExecutor);

    // Initialize the bundle.
    auto audio = GetAudio();
    static constexpr const std::size_t kTargetAudioSize = 16000 * 5;
    while (audio.size() < kTargetAudioSize) {
      audio.insert(audio.end(), audio.begin(), audio.end());
    }
    audio.resize(kTargetAudioSize);

    const auto emotionSize = bundle->GetExecutor().GetEmotionsSize();

    auto& emotionAccumulator = bundle->GetPreferredEmotionAccumulator();
    std::vector<float> emotion(emotionAccumulator.GetEmotionSize(), 0.0f);
    CHECK_TRUE(emotionAccumulator.GetEmotionSize() == emotionSize);
    CHECK_SUCCESS(emotionAccumulator.Reset());
    CHECK_SUCCESS(emotionAccumulator.Accumulate(
        0, nva2x::HostTensorFloatConstView{emotion.data(), emotion.size()}, bundle->GetCudaStream().Data()
        ));
    CHECK_SUCCESS(emotionAccumulator.Close());

    auto& audioAccumulator = bundle->GetAudioAccumulator();
    CHECK_SUCCESS(audioAccumulator.Reset());
    CHECK_SUCCESS(audioAccumulator.Accumulate(nva2x::HostTensorFloatConstView{audio.data(), audio.size()}, bundle->GetCudaStream().Data()));
    CHECK_SUCCESS(audioAccumulator.Close());

    auto callback = [](void* userdata, const nva2e::IEmotionExecutor::Results& results) {
      return true;
    };
    CHECK_SUCCESS(bundle->GetExecutor().SetResultsCallback(callback, nullptr));

    // Pre-compute everything.
    CHECK_SUCCESS(bundle->GetExecutor().ComputeAllFrames());

    return bundle;
  }

}


static void BM_InteractiveExecutorBatch(benchmark::State& state) {
    const auto batchSize = state.range(0);
    const auto inferencesToSkip = state.range(1);
    const bool classifier = state.range(2);

    auto bundle = CreateEmotionInteractiveExecutorBundle(classifier, inferencesToSkip, batchSize);
    CHECK_SUCCESS(bundle->GetCudaStream().Synchronize());

    std::size_t totalExecutions = 0;
    for (auto _ : state) {
        CHECK_SUCCESS(bundle->GetExecutor().Invalidate(nva2x::IInteractiveExecutor::kLayerAll));
        CHECK_SUCCESS(bundle->GetExecutor().ComputeAllFrames());
        CHECK_SUCCESS(bundle->GetCudaStream().Synchronize());
        ++totalExecutions;
    }
    state.SetItemsProcessed(totalExecutions);
}

BENCHMARK(BM_InteractiveExecutorBatch)->Apply([](benchmark::internal::Benchmark* b) {
    b->UseRealTime();
    // Assign meaningful names
    b->ArgNames({"BatchSize", "InferencesToSkip", "Classifier"});
    // Define for all the combinations
    b->ArgsProduct({
        {0, 1, 2, 4, 32, 128},
        {0, 1, 10, 30},
        {1},
    });
    // For post-process executor, batch size and inferences to skip are irrelevant.
    b->ArgsProduct({
        {0},
        {0},
        {0},
    });
});


static void BM_InteractiveExecutorLayer(benchmark::State& state) {
    const auto layer = state.range(0);
    const bool classifier = state.range(1);

    const char* layerName;
    switch (layer) {
        case nva2e::IEmotionInteractiveExecutor::kLayerNone:
            layerName = "None";
            break;
        case nva2e::IEmotionInteractiveExecutor::kLayerAll:
            layerName = "All";
            break;
        case nva2e::IEmotionInteractiveExecutor::kLayerInference:
            layerName = "Inference";
            break;
        case nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing:
            layerName = "PostProcessing";
            break;
        default:
            layerName = "Unknown";
            break;
    };
    state.SetLabel(layerName);

    auto bundle = CreateEmotionInteractiveExecutorBundle(classifier, 0, 0);
    CHECK_SUCCESS(bundle->GetCudaStream().Synchronize());

    // Target the middle frame for an average case.
    const auto targetFrame = bundle->GetExecutor().GetTotalNbFrames() / 2;
    std::size_t totalExecutions = 0;
    for (auto _ : state) {
        CHECK_SUCCESS(bundle->GetExecutor().Invalidate(layer));
        CHECK_SUCCESS(bundle->GetExecutor().ComputeFrame(targetFrame));
        CHECK_SUCCESS(bundle->GetCudaStream().Synchronize());
        ++totalExecutions;
    }
    state.SetItemsProcessed(totalExecutions);
}

BENCHMARK(BM_InteractiveExecutorLayer)->Apply([](benchmark::internal::Benchmark* b) {
    b->UseRealTime();
      // Assign meaningful names
    b->ArgNames({"Layer", "Classifier"});
    // Define for all the combinations
    static_assert(nva2e::IEmotionInteractiveExecutor::kLayerNone == 0);
    static_assert(nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing == 3);
    b->ArgsProduct({
        {0, 1, 2, 3},
        {1, 0},
    });
});
