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

  UniquePtr<nva2e::IEmotionExecutorBundle> CreateEmotionExecutorBundle(
    std::size_t inferencesToSkip, std::size_t batchSize
    ) {
    constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json";
    auto bundle = ToUniquePtr(
      nva2e::ReadClassifierEmotionExecutorBundle(
        batchSize,
        modelPath,
        30000,
        60, 1,
        inferencesToSkip,
        nullptr
        )
      );
    CHECK_TRUE(bundle);

    // Initialize the bundle.
    auto audio = GetAudio();
    static constexpr const std::size_t kTargetAudioSize = 16000 * 5;
    while (audio.size() < kTargetAudioSize) {
      audio.insert(audio.end(), audio.begin(), audio.end());
    }
    audio.resize(kTargetAudioSize);

    const auto emotionSize = bundle->GetExecutor().GetEmotionsSize();

    for (std::size_t i = 0; i < batchSize; ++i) {
      auto& emotionAccumulator = bundle->GetPreferredEmotionAccumulator(i);
      std::vector<float> emotion(emotionAccumulator.GetEmotionSize(), 0.0f);
      CHECK_TRUE(emotionAccumulator.GetEmotionSize() == emotionSize);
      CHECK_SUCCESS(emotionAccumulator.Reset());
      CHECK_SUCCESS(emotionAccumulator.Accumulate(
          0, nva2x::HostTensorFloatConstView{emotion.data(), emotion.size()}, bundle->GetCudaStream().Data()
          ));
      CHECK_SUCCESS(emotionAccumulator.Close());

      auto& audioAccumulator = bundle->GetAudioAccumulator(i);
      CHECK_SUCCESS(audioAccumulator.Reset());
      CHECK_SUCCESS(audioAccumulator.Accumulate(nva2x::HostTensorFloatConstView{audio.data(), audio.size()}, bundle->GetCudaStream().Data()));
      CHECK_SUCCESS(audioAccumulator.Close());
    }

    return bundle;
  }

}


static void BM_ExecutorPartial(benchmark::State& state) {
  constexpr const std::size_t batchSize = 32;
  constexpr const std::size_t inferencesToSkip = 10;
  const std::size_t activeTracks = state.range(0);

  auto bundle = CreateEmotionExecutorBundle(inferencesToSkip, batchSize);
  CHECK_SUCCESS(bundle->GetCudaStream().Synchronize());

  auto callback = [](void* userdata, const nva2e::IEmotionExecutor::Results& results) {
    return true;
  };
  CHECK_SUCCESS(bundle->GetExecutor().SetResultsCallback(callback, nullptr));

  for (std::size_t i = 0; i < batchSize - activeTracks; ++i) {
    CHECK_SUCCESS(bundle->GetAudioAccumulator(i).Reset());
    CHECK_SUCCESS(bundle->GetPreferredEmotionAccumulator(i).Reset());
  }

  // Warm-up.
  for (std::size_t i = 0; i < 10; ++i) {
    CHECK_SUCCESS(bundle->GetExecutor().Execute(nullptr));
  }

  std::size_t totalExecutions = 0;
  for (auto _ : state) {
    for (std::size_t i = 0; i < batchSize; ++i) {
      CHECK_SUCCESS(bundle->GetExecutor().Reset(i));
    }

    while (nva2x::GetNbReadyTracks(bundle->GetExecutor()) > 0) {
      CHECK_SUCCESS(bundle->GetExecutor().Execute(nullptr));
    }
    CHECK_SUCCESS(bundle->GetCudaStream().Synchronize());
    ++totalExecutions;
  }
  state.counters["inferences_per_second"] = benchmark::Counter(
    static_cast<double>(totalExecutions), benchmark::Counter::kIsRate
    );
  state.counters["tracks_per_second"] = benchmark::Counter(
    static_cast<double>(totalExecutions * activeTracks), benchmark::Counter::kIsRate
    );
}

BENCHMARK(BM_ExecutorPartial)->Apply([](benchmark::internal::Benchmark* b) {
    b->UseRealTime();
    // Assign meaningful names
    b->ArgName("ActiveTracks");
    // Define for all the combinations
    b->ArgsProduct({
        {1, 2, 4, 8, 16, 31, 32},
    });
    // Add 10 iterations
    b->Iterations(10);

});
