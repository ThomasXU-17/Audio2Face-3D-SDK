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
#include "audio2face/audio2face.h"
#include "audio2x/cuda_stream.h"

#include <benchmark/benchmark.h>

#include <iostream>
#include <sstream>

#include "utils.h"

#define CHECK_SUCCESS(func)                                                    \
  {                                                                            \
    std::error_code error = func;                                              \
    if (error) {                                                               \
      std::cout << "Error: Failed to execute: " << #func;                      \
      std::cout << ", Reason: " << error.message() << std::endl;               \
      exit(error.value());                                                     \
    }                                                                          \
  }

#define CHECK_TRUE(expression)                                                 \
  {                                                                            \
    if (!(expression)) {                                                       \
      std::cout << "Error: " << #expression << " is false" << std::endl;       \
      exit(1);                                                                 \
    }                                                                          \
  }


namespace {

  std::vector<float> GetAudio() {
    return loadAudio();
  }

  struct InteractiveExecutorBundle {
    UniquePtr<nva2x::ICudaStream> cudaStream;
    UniquePtr<nva2x::IAudioAccumulator> audioAccumulator;
    UniquePtr<nva2x::IEmotionAccumulator> emotionAccumulator;
    UniquePtr<nva2f::IFaceInteractiveExecutor> interactiveExecutor;

    inline nva2x::ICudaStream& GetCudaStream() { return *cudaStream; }
    inline nva2x::IAudioAccumulator& GetAudioAccumulator() { return *audioAccumulator; }
    inline nva2x::IEmotionAccumulator& GetEmotionAccumulator() { return *emotionAccumulator; }
    inline nva2f::IFaceInteractiveExecutor& GetExecutor() { return *interactiveExecutor; }
  };

  std::unique_ptr<InteractiveExecutorBundle> CreateGeometryInteractiveExecutorBundle(
    bool regression, std::size_t batchSize
    ) {
    auto bundle = std::make_unique<InteractiveExecutorBundle>();

    bundle->cudaStream = ToUniquePtr(nva2x::CreateCudaStream());
    CHECK_TRUE(bundle->cudaStream);

    bundle->audioAccumulator = ToUniquePtr(nva2x::CreateAudioAccumulator(16000, 0));
    CHECK_TRUE(bundle->audioAccumulator);

    bundle->emotionAccumulator = ToUniquePtr(nva2x::CreateEmotionAccumulator(10, 300, 0));
    CHECK_TRUE(bundle->emotionAccumulator);

    nva2f::GeometryExecutorCreationParameters params;
    params.cudaStream = bundle->cudaStream->Data();
    params.nbTracks = 1;
    const nva2x::IAudioAccumulator* audioAccumulatorPtr = bundle->audioAccumulator.get();
    params.sharedAudioAccumulators = &audioAccumulatorPtr;
    const nva2x::IEmotionAccumulator* emotionAccumulatorPtr = bundle->emotionAccumulator.get();
    params.sharedEmotionAccumulators = &emotionAccumulatorPtr;

    std::size_t emotionSize = 0;
    UniquePtr<nva2f::IGeometryInteractiveExecutor> interactiveExecutor;
    if (regression) {
      constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
      auto modelInfo = ToUniquePtr(nva2f::ReadRegressionModelInfo(modelPath));
      CHECK_TRUE(modelInfo);

      emotionSize = modelInfo->GetNetworkInfo().GetEmotionsCount();

      const auto regressionParams = modelInfo->GetExecutorCreationParameters(
        nva2f::IGeometryExecutor::ExecutionOption::All, 60, 1
        );

      interactiveExecutor = ToUniquePtr(
        nva2f::CreateRegressionGeometryInteractiveExecutor(
          params,
          regressionParams,
          batchSize
          )
        );
    }
    else {
      constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json";
      auto modelInfo = ToUniquePtr(nva2f::ReadDiffusionModelInfo(modelPath));
      CHECK_TRUE(modelInfo);

      emotionSize = modelInfo->GetNetworkInfo().GetEmotionsCount();

      const auto diffusionParams = modelInfo->GetExecutorCreationParameters(
        nva2f::IGeometryExecutor::ExecutionOption::All, 0, true
        );

      interactiveExecutor = ToUniquePtr(
        nva2f::CreateDiffusionGeometryInteractiveExecutor(
          params,
          diffusionParams,
          // This is not actually batch size, but reuse the same parameter.
          batchSize
          )
        );
    }
    CHECK_TRUE(interactiveExecutor);

    auto callback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
      return true;
    };
    CHECK_SUCCESS(interactiveExecutor->SetResultsCallback(callback, nullptr));

    bundle->interactiveExecutor = std::move(interactiveExecutor);

    // Initialize the bundle.
    auto audio = GetAudio();
    static constexpr const std::size_t kTargetAudioSize = 16000 * 5;
    while (audio.size() < kTargetAudioSize) {
      audio.insert(audio.end(), audio.begin(), audio.end());
    }
    audio.resize(kTargetAudioSize);

    auto& emotionAccumulator = bundle->GetEmotionAccumulator();
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

    // Pre-compute everything.
    CHECK_SUCCESS(bundle->GetExecutor().ComputeAllFrames());

    return bundle;
  }

  std::unique_ptr<InteractiveExecutorBundle> CreateBlendshapeSolveInteractiveExecutorBundle(
    bool regression, std::size_t batchSize, bool useGpuSolver
    ) {
    auto geometryBundle = CreateGeometryInteractiveExecutorBundle(regression, batchSize);
    CHECK_TRUE(geometryBundle);
    auto geometryExecutor = std::move(geometryBundle->interactiveExecutor);
    CHECK_TRUE(geometryExecutor);

    constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
    auto modelInfo = ToUniquePtr(nva2f::ReadRegressionBlendshapeSolveModelInfo(modelPath));
    CHECK_TRUE(modelInfo);

    const auto creationParams = modelInfo->GetExecutorCreationParameters(
      nva2f::IGeometryExecutor::ExecutionOption::All
      );

    if (useGpuSolver) {
        nva2f::DeviceBlendshapeSolveExecutorCreationParameters params;

        params.initializationSkinParams = creationParams.initializationSkinParams;
        params.initializationTongueParams = creationParams.initializationTongueParams;

        auto executor = ToUniquePtr(
            nva2f::CreateDeviceBlendshapeSolveInteractiveExecutor(
                static_cast<nva2f::IGeometryInteractiveExecutor*>(geometryExecutor.release()), params
                )
            );

        auto callback = [](void* userdata, const nva2f::IBlendshapeExecutor::DeviceResults& results) {
            return true;
        };
        CHECK_SUCCESS(executor->SetResultsCallback(callback, nullptr));

        geometryBundle->interactiveExecutor = std::move(executor);
    }
    else {
        nva2f::HostBlendshapeSolveExecutorCreationParameters params;

        params.initializationSkinParams = creationParams.initializationSkinParams;
        params.initializationTongueParams = creationParams.initializationTongueParams;

        params.sharedJobRunner = nullptr;

        auto executor = ToUniquePtr(
            nva2f::CreateHostBlendshapeSolveInteractiveExecutor(
                static_cast<nva2f::IGeometryInteractiveExecutor*>(geometryExecutor.release()), params
                )
            );

        auto callback = [](void* userdata, const nva2f::IBlendshapeExecutor::HostResults& results, std::error_code errorCode) {
            CHECK_SUCCESS(errorCode);
        };
        CHECK_SUCCESS(executor->SetResultsCallback(callback, nullptr));

        geometryBundle->interactiveExecutor = std::move(executor);
    }
    CHECK_TRUE(geometryBundle->interactiveExecutor);

    return geometryBundle;
  }

}


static void BM_InteractiveExecutorBatch(benchmark::State& state) {
    const auto batchSize = state.range(0);
    const auto regression = state.range(1);

    auto bundle = CreateGeometryInteractiveExecutorBundle(regression, batchSize);
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
    b->ArgNames({"BatchSize", "Regression"});
    // Define for all the combinations
    b->ArgsProduct({
        {0, 1, 4, 16, 32, 128},
        {1},
    });
    // For diffusion, "batch size" only affects previews.
    b->ArgsProduct({
        {0},
        {0},
    });
});


namespace {

    const char* GetLayerName(nva2f::IGeometryInteractiveExecutor::invalidation_layer_t layer) {
        switch (layer) {
            case nva2f::IGeometryInteractiveExecutor::kLayerNone:
                return "None";
            case nva2f::IGeometryInteractiveExecutor::kLayerAll:
                return "All";
            case nva2f::IGeometryInteractiveExecutor::kLayerInference:
                return "Inference";
            case nva2f::IGeometryInteractiveExecutor::kLayerSkin:
                return "Skin";
            case nva2f::IGeometryInteractiveExecutor::kLayerTongue:
                return "Tongue";
            case nva2f::IGeometryInteractiveExecutor::kLayerTeeth:
                return "Teeth";
            case nva2f::IGeometryInteractiveExecutor::kLayerEyes:
                return "Eyes";
            case nva2f::IBlendshapeInteractiveExecutor::kLayerSkinSolverPrepare:
                return "SkinSolverPrepare";
            case nva2f::IBlendshapeInteractiveExecutor::kLayerTongueSolverPrepare:
                return "TongueSolverPrepare";
            case nva2f::IBlendshapeInteractiveExecutor::kLayerBlendshapeWeights:
                return "BlendshapeWeights";
            default:
                return "Unknown";
        };
    }

}

static void BM_GeometryInteractiveExecutorLayer(benchmark::State& state) {
    const auto layer = state.range(0);
    const auto lookBack = state.range(1);
    const auto regression = state.range(2);

    const char* layerName = GetLayerName(layer);
    state.SetLabel(layerName);

    auto bundle = CreateGeometryInteractiveExecutorBundle(regression, lookBack);
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

BENCHMARK(BM_GeometryInteractiveExecutorLayer)->Apply([](benchmark::internal::Benchmark* b) {
    b->UseRealTime();
    // Assign meaningful names
    b->ArgNames({"Layer", "LookBack", "Regression"});
    // Define for all the combinations
    static_assert(nva2f::IGeometryInteractiveExecutor::kLayerNone == 0);
    static_assert(nva2f::IGeometryInteractiveExecutor::kLayerEyes == 6);
    // For regression, look back is not used.
    b->ArgsProduct({
        {0, 1, 2, 3, 4, 5, 6},
        {0},
        {1},
    });
    b->ArgsProduct({
        {0, 1, 2, 3, 4, 5, 6},
        {0, 1, 2, 4},
        {0},
    });
});

static void BM_BlendshapeInteractiveExecutorLayer(benchmark::State& state) {
    const auto layer = state.range(0);
    const auto useGpuSolver = state.range(1);

    const char* layerName = GetLayerName(layer);
    state.SetLabel(layerName);

    auto bundle = CreateBlendshapeSolveInteractiveExecutorBundle(true, 0, useGpuSolver);
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

BENCHMARK(BM_BlendshapeInteractiveExecutorLayer)->Apply([](benchmark::internal::Benchmark* b) {
    b->UseRealTime();
    // Assign meaningful name
    b->ArgNames({"Layer", "UseGpuSolver", "LookBack", "Regression"});
    // Define for all the combinations
    const std::vector<std::int64_t> kLayers = {
        nva2f::IGeometryInteractiveExecutor::kLayerNone,
        nva2f::IGeometryInteractiveExecutor::kLayerAll,
        nva2f::IGeometryInteractiveExecutor::kLayerInference,
        nva2f::IGeometryInteractiveExecutor::kLayerSkin,
        nva2f::IGeometryInteractiveExecutor::kLayerTongue,
        nva2f::IGeometryInteractiveExecutor::kLayerTeeth,
        nva2f::IGeometryInteractiveExecutor::kLayerEyes,
        nva2f::IBlendshapeInteractiveExecutor::kLayerSkinSolverPrepare,
        nva2f::IBlendshapeInteractiveExecutor::kLayerTongueSolverPrepare,
        nva2f::IBlendshapeInteractiveExecutor::kLayerBlendshapeWeights,
    };
    b->ArgsProduct({
        kLayers,
        {0, 1},
        {0},
        {1},
    });
    b->ArgsProduct({
        kLayers,
        {0, 1},
        {0, 1, 2, 4},
        {0},
    });
});
