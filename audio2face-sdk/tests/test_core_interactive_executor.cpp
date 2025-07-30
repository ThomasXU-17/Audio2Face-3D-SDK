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
#include "audio2x/internal/unique_ptr.h"
#include "audio2face/internal/executor_regression.h"
#include "audio2face/internal/executor_diffusion.h"
#include "audio2face/internal/interactive_executor_regression.h"
#include "audio2face/internal/interactive_executor_diffusion.h"
#include "audio2face/internal/interactive_executor_blendshapesolve.h"

#include <gtest/gtest.h>

#include <any>
#include <algorithm>

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

  constexpr std::size_t kEmotionSize = 10;
  constexpr bool kConstantNoise = false;

  template <typename Bundle>
  void InitBundle(Bundle& bundle) {
    const auto nbTracks = bundle.GetExecutor().GetNbTracks();
    const auto audio = GetAudio();
    const auto emotionSize = kEmotionSize;
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
      auto& emotionAccumulator = bundle.GetEmotionAccumulator(trackIndex);
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

  nva2x::UniquePtr<nva2f::IGeometryExecutorBundle> CreateGeometryExecutorBundle(bool regression, bool init) {
    nva2x::UniquePtr<nva2f::IGeometryExecutorBundle> bundle;
    if (regression) {
      constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
      bundle = nva2x::ToUniquePtr(
        nva2f::ReadRegressionGeometryExecutorBundle_INTERNAL(
          1,
          modelPath,
          nva2f::IGeometryExecutor::ExecutionOption::All, 60, 1,
          nullptr
          )
        );
    }
    else {
      constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json";
      bundle = nva2x::ToUniquePtr(
        nva2f::ReadDiffusionGeometryExecutorBundle_INTERNAL(
          1,
          modelPath,
          nva2f::IGeometryExecutor::ExecutionOption::All, 0, kConstantNoise,
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

  nva2x::UniquePtr<nva2f::IBlendshapeExecutorBundle> CreateBlendshapeSolveExecutorBundle(bool regression, bool init, bool useGpuSolver) {
    nva2x::UniquePtr<nva2f::IBlendshapeExecutorBundle> bundle;
    if (regression) {
      constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
      bundle = nva2x::ToUniquePtr(
        nva2f::ReadRegressionBlendshapeSolveExecutorBundle_INTERNAL(
          1,
          modelPath,
          nva2f::IGeometryExecutor::ExecutionOption::All, useGpuSolver, 60, 1,
          nullptr,
          nullptr
          )
        );
    }
    else {
      constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json";
      bundle = nva2x::ToUniquePtr(
        nva2f::ReadDiffusionBlendshapeSolveExecutorBundle_INTERNAL(
          1,
          modelPath,
          nva2f::IGeometryExecutor::ExecutionOption::All, useGpuSolver, 0, kConstantNoise,
          nullptr,
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
    nva2x::UniquePtr<nva2f::IFaceInteractiveExecutor> interactiveExecutor;

    inline nva2x::CudaStream& GetCudaStream() { return cudaStream; }
    inline nva2x::AudioAccumulator& GetAudioAccumulator() { return audioAccumulator; }
    inline nva2x::EmotionAccumulator& GetEmotionAccumulator() { return emotionAccumulator; }
    inline nva2f::IFaceInteractiveExecutor& GetExecutor() { return *interactiveExecutor; }
  };

  std::unique_ptr<InteractiveExecutorBundle> CreateGeometryInteractiveExecutorBundle(
    bool regression, std::size_t batchSize, bool init
    ) {
    auto bundle = std::make_unique<InteractiveExecutorBundle>();
    EXPECT_TRUE(!bundle->cudaStream.Init());
    EXPECT_TRUE(!bundle->audioAccumulator.Allocate(16000, 0));
    EXPECT_TRUE(!bundle->emotionAccumulator.Allocate(10, 300, 0));

    nva2f::GeometryExecutorCreationParameters params;
    params.cudaStream = bundle->cudaStream.Data();
    params.nbTracks = 1;
    const nva2x::IAudioAccumulator* audioAccumulator = &bundle->audioAccumulator;
    params.sharedAudioAccumulators = &audioAccumulator;
    const nva2x::IEmotionAccumulator* emotionAccumulator = &bundle->emotionAccumulator;
    params.sharedEmotionAccumulators = &emotionAccumulator;

    if (regression) {
      constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
      auto modelInfo = nva2x::ToUniquePtr(nva2f::ReadRegressionModelInfo_INTERNAL(modelPath));
      EXPECT_TRUE(modelInfo);

      const auto regressionParams = modelInfo->GetExecutorCreationParameters(
        nva2f::IGeometryExecutor::ExecutionOption::All, 60, 1
        );

      bundle->interactiveExecutor = nva2x::ToUniquePtr(
        nva2f::CreateRegressionGeometryInteractiveExecutor_INTERNAL(
          params,
          regressionParams,
          batchSize
          )
        );
    }
    else {
      constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json";
      auto modelInfo = nva2x::ToUniquePtr(nva2f::ReadDiffusionModelInfo_INTERNAL(modelPath));
      EXPECT_TRUE(modelInfo);

      const auto diffusionParams = modelInfo->GetExecutorCreationParameters(
        nva2f::IGeometryExecutor::ExecutionOption::All, 0, kConstantNoise
        );

      bundle->interactiveExecutor = nva2x::ToUniquePtr(
        nva2f::CreateDiffusionGeometryInteractiveExecutor_INTERNAL(
          params,
          diffusionParams,
          // This is not actually batch size, but reuse the same parameter.
          batchSize
          )
        );
    }
    EXPECT_TRUE(bundle->interactiveExecutor);

    if (init) {
      const auto audio = GetAudio();
      const auto emotionSize = kEmotionSize;

      auto& emotionAccumulator = bundle->GetEmotionAccumulator();
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

  nva2f::BlendshapeSolveExecutorCreationParameters GetBlendshapeSolveExecutorCreationParameters(
    bool regression, std::any& modelInfoHolder
  ) {
    if (regression) {
      constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
      auto modelInfo = nva2x::ToSharedPtr(nva2f::ReadRegressionBlendshapeSolveModelInfo_INTERNAL(modelPath));
      EXPECT_TRUE(modelInfo);

      modelInfoHolder = modelInfo;

      return modelInfo->GetExecutorCreationParameters(
        nva2f::IGeometryExecutor::ExecutionOption::All
        );
    }
    else {
      constexpr const char modelPath[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json";
      auto modelInfo = nva2x::ToSharedPtr(nva2f::ReadDiffusionBlendshapeSolveModelInfo_INTERNAL(modelPath));
      EXPECT_TRUE(modelInfo);

      modelInfoHolder = modelInfo;

      return modelInfo->GetExecutorCreationParameters(
        nva2f::IGeometryExecutor::ExecutionOption::All, 0
        );
    }
  }

  std::unique_ptr<InteractiveExecutorBundle> CreateBlendshapeSolveInteractiveExecutorBundle(
    bool regression, std::size_t batchSize, bool useGpuSolver, bool init
    ) {
    auto geometryBundle = CreateGeometryInteractiveExecutorBundle(regression, batchSize, init);
    EXPECT_TRUE(geometryBundle);
    auto geometryExecutor = std::move(geometryBundle->interactiveExecutor);
    EXPECT_TRUE(geometryExecutor);

    std::any modelInfoHolder;
    const auto creationParams = GetBlendshapeSolveExecutorCreationParameters(regression, modelInfoHolder);

    if (useGpuSolver) {
        nva2f::DeviceBlendshapeSolveExecutorCreationParameters params;

        params.initializationSkinParams = creationParams.initializationSkinParams;
        params.initializationTongueParams = creationParams.initializationTongueParams;

        geometryBundle->interactiveExecutor.reset(
            nva2f::CreateDeviceBlendshapeSolveInteractiveExecutor_INTERNAL(
                static_cast<nva2f::IGeometryInteractiveExecutor*>(geometryExecutor.release()), params
                )
            );
    }
    else {
        nva2f::HostBlendshapeSolveExecutorCreationParameters params;

        params.initializationSkinParams = creationParams.initializationSkinParams;
        params.initializationTongueParams = creationParams.initializationTongueParams;

        params.sharedJobRunner = nullptr;

        geometryBundle->interactiveExecutor.reset(
            nva2f::CreateHostBlendshapeSolveInteractiveExecutor_INTERNAL(
                static_cast<nva2f::IGeometryInteractiveExecutor*>(geometryExecutor.release()), params
                )
            );
    }
    EXPECT_TRUE(geometryBundle->interactiveExecutor);

    return geometryBundle;
  }

  struct frame_t {
    std::int64_t timestamp;
    std::vector<float> results;

    bool operator==(const frame_t& other) const {
      return timestamp == other.timestamp && results == other.results;
    }
  };
  using frames_t = std::vector<frame_t>;

  bool callbackGeometry(void* userdata, const nva2f::IGeometryExecutor::Results& results) {
    auto* frames = static_cast<frames_t*>(userdata);
    frame_t frame;
    frame.timestamp = results.timeStampCurrentFrame;
    frame.results.resize(
      results.skinGeometry.Size() +
      results.tongueGeometry.Size() +
      results.jawTransform.Size() +
      results.eyesRotation.Size()
      );
    auto destination = nva2x::ToView(frame.results);
    auto getDestination = [&destination](std::size_t size) {
      auto result = destination.View(0, size);
      destination = destination.View(size, destination.Size() - size);
      return result;
    };
    EXPECT_TRUE(!nva2x::CopyDeviceToHost(
      getDestination(results.skinGeometry.Size()), results.skinGeometry, results.skinCudaStream
      ));
    EXPECT_TRUE(!nva2x::CopyDeviceToHost(
      getDestination(results.tongueGeometry.Size()), results.tongueGeometry, results.tongueCudaStream
      ));
    EXPECT_TRUE(!nva2x::CopyDeviceToHost(
      getDestination(results.jawTransform.Size()), results.jawTransform, results.jawCudaStream
      ));
    EXPECT_TRUE(!nva2x::CopyDeviceToHost(
      getDestination(results.eyesRotation.Size()), results.eyesRotation, results.eyesCudaStream
      ));
    EXPECT_TRUE(destination.Size() == 0);
    frames->emplace_back(std::move(frame));
    return true;
  };

  void callbackBlendshapeHost(
    void* userdata, const nva2f::IBlendshapeExecutor::HostResults& results, std::error_code errorCode
    ) {
    assert(!errorCode);
    auto* frames = static_cast<frames_t*>(userdata);
    frame_t frame;
    frame.timestamp = results.timeStampCurrentFrame;
    frame.results.resize(results.weights.Size());
    auto destination = nva2x::ToView(frame.results);
    EXPECT_TRUE(!nva2x::CopyHostToHost(destination, results.weights));
    frames->emplace_back(std::move(frame));
  };

  bool callbackBlendshapeDevice(
    void* userdata, const nva2f::IBlendshapeExecutor::DeviceResults& results
    ) {
    auto* frames = static_cast<frames_t*>(userdata);
    frame_t frame;
    frame.timestamp = results.timeStampCurrentFrame;
    frame.results.resize(results.weights.Size());
    auto destination = nva2x::ToView(frame.results);
    EXPECT_TRUE(!nva2x::CopyDeviceToHost(destination, results.weights, results.cudaStream));
    frames->emplace_back(std::move(frame));
    return true;
  };

}


TEST(TestCoreInteractiveExecutor, Correctness) {
  const int seed = static_cast<unsigned int>(time(NULL));
  std::cout << "Current srand seed: " << seed << std::endl;
  std::srand(seed); // make random inputs reproducible

  for (const auto regression : {true, false}) {
    auto executorBundle = CreateGeometryExecutorBundle(regression, true);
    ASSERT_TRUE(executorBundle);

    std::vector<nva2f::AnimatorSkinParams> skinParams(1);
    ASSERT_TRUE(!nva2f::GetExecutorSkinParameters_INTERNAL(
      executorBundle->GetExecutor(), 0, skinParams[0]
      ));
    // Disable smoothing so that interactive frames gives the same results.
    skinParams[0].lowerFaceSmoothing = 0.0f;
    skinParams[0].upperFaceSmoothing = 0.0f;
    // Add another variation.
    skinParams.emplace_back(skinParams.back());
    skinParams.back().upperFaceStrength *= 2.0f;

    // Compute ground truth data from regular executor.
    std::vector<frames_t> groundTruth;
    for (const auto& params : skinParams) {
      ASSERT_TRUE(!executorBundle->GetExecutor().Reset(0));

      groundTruth.emplace_back();
      ASSERT_TRUE(!executorBundle->GetExecutor().SetResultsCallback(callbackGeometry, &groundTruth.back()));

      ASSERT_TRUE(!nva2f::SetExecutorSkinParameters_INTERNAL(
        executorBundle->GetExecutor(), 0, params
        ));

      // Run the executor.
      while (nva2x::GetNbReadyTracks(executorBundle->GetExecutor()) > 0) {
        ASSERT_TRUE(!executorBundle->GetExecutor().Execute(nullptr));
      }

      ASSERT_EQ(groundTruth.back().size(), executorBundle->GetExecutor().GetTotalNbFrames(0));
    }

    executorBundle.reset();

    frames_t interactiveFrames;
    frames_t interactiveFramesBlendshape;

    enum class SolverType { Geometry, BlendshapeHost, BlendshapeDevice };
    for (const auto solverType : {SolverType::Geometry, SolverType::BlendshapeHost, SolverType::BlendshapeDevice}) {
      // Create interactive executor.
      static constexpr std::size_t kBatchSize = 32;
      static constexpr std::size_t kNbInferencesForPreview = 0;
      const std::size_t batchSize = regression ? kBatchSize : kNbInferencesForPreview;
      std::unique_ptr<InteractiveExecutorBundle> interactiveExecutorBundle;
      if (solverType == SolverType::Geometry) {
        interactiveExecutorBundle = CreateGeometryInteractiveExecutorBundle(regression, batchSize, true);

        // Not having a callback should trigger an error, even if there is available data.
        ASSERT_LT(0, interactiveExecutorBundle->GetExecutor().GetTotalNbFrames());
        ASSERT_TRUE(interactiveExecutorBundle->GetExecutor().ComputeAllFrames());
        ASSERT_TRUE(interactiveExecutorBundle->GetExecutor().ComputeFrame(0));
      }
      else if (solverType == SolverType::BlendshapeHost) {
        interactiveExecutorBundle = CreateBlendshapeSolveInteractiveExecutorBundle(regression, batchSize, false, true);

        // Not having a callback should trigger an error, even if there is available data.
        ASSERT_LT(0, interactiveExecutorBundle->GetExecutor().GetTotalNbFrames());
        ASSERT_TRUE(interactiveExecutorBundle->GetExecutor().ComputeAllFrames());
        ASSERT_TRUE(interactiveExecutorBundle->GetExecutor().ComputeFrame(0));

        nva2f::IBlendshapeInteractiveExecutor& blendshapeExecutor = static_cast<nva2f::IBlendshapeInteractiveExecutor&>(
          interactiveExecutorBundle->GetExecutor()
          );
        ASSERT_TRUE(!blendshapeExecutor.SetResultsCallback(callbackBlendshapeHost, &interactiveFramesBlendshape));
      }
      else if (solverType == SolverType::BlendshapeDevice) {
        interactiveExecutorBundle = CreateBlendshapeSolveInteractiveExecutorBundle(regression, batchSize, true, true);

        // Not having a callback should trigger an error, even if there is available data.
        ASSERT_LT(0, interactiveExecutorBundle->GetExecutor().GetTotalNbFrames());
        ASSERT_TRUE(interactiveExecutorBundle->GetExecutor().ComputeAllFrames());
        ASSERT_TRUE(interactiveExecutorBundle->GetExecutor().ComputeFrame(0));

        nva2f::IBlendshapeInteractiveExecutor& blendshapeExecutor = static_cast<nva2f::IBlendshapeInteractiveExecutor&>(
          interactiveExecutorBundle->GetExecutor()
          );
        ASSERT_TRUE(!blendshapeExecutor.SetResultsCallback(callbackBlendshapeDevice, &interactiveFramesBlendshape));
      }

      ASSERT_TRUE(
        !nva2f::SetInteractiveExecutorGeometryResultsCallback_INTERNAL(
          interactiveExecutorBundle->GetExecutor(), callbackGeometry, &interactiveFrames
          )
        );

      for (std::size_t i = 0; i < skinParams.size(); ++i) {
        interactiveFrames.clear();
        interactiveFramesBlendshape.clear();

        // Set the parameters.
        ASSERT_TRUE(!nva2f::SetInteractiveExecutorSkinParameters_INTERNAL(
          interactiveExecutorBundle->GetExecutor(), skinParams[i]
          ));
        ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().Invalidate(nva2f::IGeometryInteractiveExecutor::kLayerAll));

        // Run the executor.
        ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().ComputeAllFrames());

        ASSERT_EQ(interactiveFrames.size(), interactiveExecutorBundle->GetExecutor().GetTotalNbFrames());
        if (solverType != SolverType::Geometry) {
          ASSERT_EQ(interactiveFramesBlendshape.size(), interactiveExecutorBundle->GetExecutor().GetTotalNbFrames());
        }

        if (regression) {
          ASSERT_EQ(groundTruth[i].size(), interactiveFrames.size()) << "Index " << i;
          for (std::size_t j = 0; j < groundTruth[i].size(); ++j) {
            ASSERT_EQ(groundTruth[i][j].timestamp, interactiveFrames[j].timestamp) << "Index " << i << " frame " << j;
            ASSERT_EQ(groundTruth[i][j].results.size(), interactiveFrames[j].results.size()) << "Index " << i << " frame " << j;
            for (std::size_t k = 0; k < groundTruth[i][j].results.size(); ++k) {
              // Batched PCA reconstruction gives slightly different results, see test_core_batch_pca_animator.cpp
              ASSERT_NEAR(groundTruth[i][j].results[k], interactiveFrames[j].results[k], 1e-4f) << "Index " << i << " frame " << j << " result " << k;
            }
          }
        }
        else {
          // Diffusion gives exactly the same results.
          ASSERT_EQ(groundTruth[i], interactiveFrames) << "Index " << i;
        }

        // Test a few random frames.
        int paramIndex = 0;
        int frameIndex = 0;
        for (int k = 0; k < 10; ++k) {
          paramIndex = std::rand() % skinParams.size();
          frameIndex = std::rand() % groundTruth[i].size();
          interactiveFrames.clear();

          ASSERT_TRUE(!nva2f::SetInteractiveExecutorSkinParameters_INTERNAL(
            interactiveExecutorBundle->GetExecutor(), skinParams[paramIndex]
            ));
          ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().ComputeFrame(frameIndex));

          ASSERT_EQ(1, interactiveFrames.size());
          ASSERT_EQ(groundTruth[paramIndex][frameIndex], interactiveFrames[0]) << "Index " << i << " attempt " << k << " frame " << frameIndex;
        }

        // Try running the same as the previous frame, but with a different parameter.
        paramIndex = (paramIndex + 1) % skinParams.size();
        interactiveFrames.clear();

        ASSERT_TRUE(!nva2f::SetInteractiveExecutorSkinParameters_INTERNAL(
          interactiveExecutorBundle->GetExecutor(), skinParams[paramIndex]
          ));
        ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().ComputeFrame(frameIndex));

        ASSERT_EQ(1, interactiveFrames.size());
        ASSERT_EQ(groundTruth[paramIndex][frameIndex], interactiveFrames[0]) << "Index " << i << " frame " << frameIndex;
      }
    }
  }
}


TEST(TestCoreInteractiveExecutor, CorrectnessBlendshape) {
  const int seed = static_cast<unsigned int>(time(NULL));
  std::cout << "Current srand seed: " << seed << std::endl;
  std::srand(seed); // make random inputs reproducible

  for (const auto& [regression, useGpuSolver] : {
    std::make_pair(true, false),
    std::make_pair(true, true),
    std::make_pair(false, false),
    std::make_pair(false, true),
    }) {
    auto executorBundle = CreateBlendshapeSolveExecutorBundle(regression, true, useGpuSolver);
    ASSERT_TRUE(executorBundle);

    nva2f::AnimatorSkinParams skinParams;
    ASSERT_TRUE(!nva2f::GetExecutorSkinParameters_INTERNAL(
      executorBundle->GetExecutor(), 0, skinParams
      ));

    nva2f::IBlendshapeSolver* skinSolver = nullptr;
    ASSERT_TRUE(!nva2f::GetExecutorSkinSolver_INTERNAL(
      executorBundle->GetExecutor(), 0, &skinSolver
      ));
    ASSERT_TRUE(skinSolver);
    std::vector<nva2f::BlendshapeSolverParams> blendshapeSkinParams(1);
    blendshapeSkinParams[0] = skinSolver->GetParameters();

    nva2f::IBlendshapeSolver* tongueSolver = nullptr;
    ASSERT_TRUE(!nva2f::GetExecutorTongueSolver_INTERNAL(
      executorBundle->GetExecutor(), 0, &tongueSolver
      ));
    ASSERT_TRUE(tongueSolver);
    std::vector<nva2f::BlendshapeSolverParams> blendshapeTongueParams(1);
    blendshapeTongueParams[0] = tongueSolver->GetParameters();

    // Disable smoothing so that interactive frames gives the same results.
    skinParams.lowerFaceSmoothing = 0.0f;
    skinParams.upperFaceSmoothing = 0.0f;
    blendshapeSkinParams[0].TemporalReg = 0.0f;
    blendshapeTongueParams[0].TemporalReg = 0.0f;
    // Add another variation.
    blendshapeSkinParams.emplace_back(blendshapeSkinParams.back());
    blendshapeSkinParams.back().L1Reg *= 10.0f;
    blendshapeTongueParams.emplace_back(blendshapeTongueParams.back());
    blendshapeTongueParams.back().L1Reg *= 10.0f;

    // Compute ground truth data from regular executor.
    std::vector<frames_t> groundTruth;
    for (std::size_t i = 0; i < blendshapeSkinParams.size(); ++i) {
      ASSERT_TRUE(!executorBundle->GetExecutor().Reset(0));

      groundTruth.emplace_back();
      if (useGpuSolver) {
        ASSERT_TRUE(!executorBundle->GetExecutor().SetResultsCallback(callbackBlendshapeDevice, &groundTruth.back()));
      } else {
        ASSERT_TRUE(!executorBundle->GetExecutor().SetResultsCallback(callbackBlendshapeHost, &groundTruth.back()));
      }

      ASSERT_TRUE(!nva2f::SetExecutorSkinParameters_INTERNAL(
        executorBundle->GetExecutor(), 0, skinParams
        ));
      ASSERT_TRUE(!skinSolver->SetParameters(blendshapeSkinParams[i]));
      ASSERT_TRUE(!skinSolver->Prepare());
      ASSERT_TRUE(!tongueSolver->SetParameters(blendshapeTongueParams[i]));
      ASSERT_TRUE(!tongueSolver->Prepare());

      // Run the executor.
      while (nva2x::GetNbReadyTracks(executorBundle->GetExecutor()) > 0) {
        ASSERT_TRUE(!executorBundle->GetExecutor().Execute(nullptr));
      }
      ASSERT_TRUE(!executorBundle->GetExecutor().Wait(0));
    }

    executorBundle.reset();

    frames_t interactiveFrames;

    // Create interactive executor.
    static constexpr std::size_t kBatchSize = 1;
    static constexpr std::size_t kNbInferencesForPreview = 1;
    const std::size_t batchSize = regression ? kBatchSize : kNbInferencesForPreview;
    auto interactiveExecutorBundle = CreateBlendshapeSolveInteractiveExecutorBundle(regression, kBatchSize, useGpuSolver, true);
    ASSERT_TRUE(interactiveExecutorBundle);

    // Not having a callback should trigger an error, even if there is available data.
    ASSERT_LT(0, interactiveExecutorBundle->GetExecutor().GetTotalNbFrames());
    ASSERT_TRUE(interactiveExecutorBundle->GetExecutor().ComputeAllFrames());
    ASSERT_TRUE(interactiveExecutorBundle->GetExecutor().ComputeFrame(0));

    nva2f::IBlendshapeInteractiveExecutor& blendshapeExecutor = static_cast<nva2f::IBlendshapeInteractiveExecutor&>(
      interactiveExecutorBundle->GetExecutor()
      );
    if (useGpuSolver) {
      ASSERT_TRUE(!blendshapeExecutor.SetResultsCallback(callbackBlendshapeDevice, &interactiveFrames));
    } else {
      ASSERT_TRUE(!blendshapeExecutor.SetResultsCallback(callbackBlendshapeHost, &interactiveFrames));
    }

    ASSERT_TRUE(!nva2f::SetInteractiveExecutorSkinParameters_INTERNAL(
      interactiveExecutorBundle->GetExecutor(), skinParams
      ));

    for (std::size_t i = 0; i < blendshapeSkinParams.size(); ++i) {
      interactiveFrames.clear();

      // Set the parameters.
      ASSERT_TRUE(!nva2f::SetInteractiveExecutorBlendshapeSkinParameters_INTERNAL(
        interactiveExecutorBundle->GetExecutor(), blendshapeSkinParams[i]
        ));
      ASSERT_TRUE(!nva2f::SetInteractiveExecutorBlendshapeTongueParameters_INTERNAL(
        interactiveExecutorBundle->GetExecutor(), blendshapeTongueParams[i]
        ));
      ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().Invalidate(nva2f::IGeometryInteractiveExecutor::kLayerAll));

      // Run the executor.
      ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().ComputeAllFrames());

      ASSERT_EQ(groundTruth[i], interactiveFrames) << "Index " << i;

      // Test a few random frames.
      int paramIndex = 0;
      int frameIndex = 0;
      for (int k = 0; k < 10; ++k) {
        paramIndex = std::rand() % blendshapeSkinParams.size();
        frameIndex = std::rand() % groundTruth[i].size();
        interactiveFrames.clear();

        ASSERT_TRUE(!nva2f::SetInteractiveExecutorBlendshapeSkinParameters_INTERNAL(
          interactiveExecutorBundle->GetExecutor(), blendshapeSkinParams[paramIndex]
          ));
        ASSERT_TRUE(!nva2f::SetInteractiveExecutorBlendshapeTongueParameters_INTERNAL(
          interactiveExecutorBundle->GetExecutor(), blendshapeTongueParams[paramIndex]
          ));
        ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().ComputeFrame(frameIndex));

        ASSERT_EQ(1, interactiveFrames.size());
        ASSERT_EQ(groundTruth[paramIndex][frameIndex], interactiveFrames[0]) << "Index " << i << " attempt " << k << " frame " << frameIndex;
      }

      // Try running the same as the previous frame, but with a different parameter.
      paramIndex = (paramIndex + 1) % blendshapeSkinParams.size();
      interactiveFrames.clear();

      ASSERT_TRUE(!nva2f::SetInteractiveExecutorBlendshapeSkinParameters_INTERNAL(
        interactiveExecutorBundle->GetExecutor(), blendshapeSkinParams[paramIndex]
        ));
      ASSERT_TRUE(!nva2f::SetInteractiveExecutorBlendshapeTongueParameters_INTERNAL(
        interactiveExecutorBundle->GetExecutor(), blendshapeTongueParams[paramIndex]
        ));
      ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().ComputeFrame(frameIndex));

      ASSERT_EQ(1, interactiveFrames.size());
      ASSERT_EQ(groundTruth[paramIndex][frameIndex], interactiveFrames[0]) << "Index " << i << " frame " << frameIndex;
    }
  }
}


TEST(TestCoreInteractiveExecutor, BatchSize) {
  std::vector<frames_t> groundTruth;
  for (const auto batchSize : {1, 0, 32}) {
    // Create interactive executor.
    auto interactiveExecutorBundle = CreateGeometryInteractiveExecutorBundle(true, batchSize, true);
    ASSERT_TRUE(interactiveExecutorBundle);

    frames_t interactiveFrames;
    ASSERT_TRUE(
      !nva2f::SetInteractiveExecutorGeometryResultsCallback_INTERNAL(
        interactiveExecutorBundle->GetExecutor(), callbackGeometry, &interactiveFrames
        )
      );

    // Run the executor.
    ASSERT_TRUE(!interactiveExecutorBundle->GetExecutor().ComputeAllFrames());

    if (groundTruth.empty()) {
      groundTruth.emplace_back(std::move(interactiveFrames));
    } else {
#if 0
      ASSERT_EQ(groundTruth[0], interactiveFrames) << "batch size " << batchSize;
#else
      ASSERT_EQ(groundTruth[0].size(), interactiveFrames.size()) << "batch size " << batchSize;
      for (std::size_t j = 0; j < groundTruth[0].size(); ++j) {
        ASSERT_EQ(groundTruth[0][j].timestamp, interactiveFrames[j].timestamp) << "batch size " << batchSize << " frame " << j;
        ASSERT_EQ(groundTruth[0][j].results.size(), interactiveFrames[j].results.size()) << "batch size " << batchSize << " frame " << j;
        for (std::size_t k = 0; k < groundTruth[0][j].results.size(); ++k) {
          // Batched PCA reconstruction gives slightly different results, see test_core_batch_pca_animator.cpp
          ASSERT_NEAR(groundTruth[0][j].results[k], interactiveFrames[j].results[k], 1e-4f) << "batch size " << batchSize << " frame " << j << " result " << k;
        }
      }
#endif
    }
  }
}


namespace {

  // A bunch of helpers for the invalidation tests.
  constexpr auto kInference = nva2f::IGeometryInteractiveExecutor::kLayerInference;
  constexpr auto kSkin = nva2f::IGeometryInteractiveExecutor::kLayerSkin;
  constexpr auto kTongue = nva2f::IGeometryInteractiveExecutor::kLayerTongue;
  constexpr auto kTeeth = nva2f::IGeometryInteractiveExecutor::kLayerTeeth;
  constexpr auto kEyes = nva2f::IGeometryInteractiveExecutor::kLayerEyes;

  using layers_t = std::vector<nva2x::IInteractiveExecutor::invalidation_layer_t>;
  bool validate(
    const nva2f::IFaceInteractiveExecutor& executor, const layers_t& allLayers, const layers_t& invalidLayers
    ) {
    if (!executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerNone)) {
      return false;
    }

    const auto isValidAll = std::size(invalidLayers) == 0;
    if (isValidAll != executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerAll)) {
      return false;
    }

    for (const auto layer : allLayers) {
      const bool isValid = std::find(invalidLayers.begin(), invalidLayers.end(), layer) == invalidLayers.end();
      if (isValid != executor.IsValid(layer)) {
        return false;
      }
    }

    return true;
  };

  using test_func_t = std::function<layers_t(nva2f::IFaceInteractiveExecutor&)>;

  template <typename Params>
  void add_params_test_case(
    std::vector<test_func_t>& test_funcs,
    const std::vector<std::function<bool(Params&)>>& test_param_funcs,
    std::error_code (*setter)(nva2f::IFaceInteractiveExecutor&, const Params&),
    std::error_code (*getter)(const nva2f::IFaceInteractiveExecutor&, Params&),
    layers_t invalidLayers) {
    for (const auto& test_param_func : test_param_funcs) {
      test_funcs.emplace_back([=](auto& executor) {
        Params params;
        EXPECT_TRUE(!getter(executor, params));
        const bool changed = test_param_func(params);
        EXPECT_TRUE(!setter(executor, params));

        layers_t invalidLayersToReturn ;
        if (changed) {
          invalidLayersToReturn = invalidLayers;
        }
        return invalidLayersToReturn;
      });
    }
  }

  static const auto kValidateTestFuncs = []() {
    // Basic test cases for things invalidating inference.
    std::vector<test_func_t> test_funcs = {
      [](auto& executor) -> layers_t {
        // Do nothing, everything still invalid since it's the first one.
        return {kInference, kSkin, kTongue, kTeeth, kEyes};
      },
      [](auto& executor) -> layers_t {
        // Invalidate the audio accumulator.
        EXPECT_TRUE(!executor.Invalidate(nva2f::IGeometryInteractiveExecutor::kLayerAudioAccumulator));
        return {kInference, kSkin, kTongue, kTeeth, kEyes};
      },
      [](auto& executor) -> layers_t {
        // Invalidate the emotion accumulator.
        EXPECT_TRUE(!executor.Invalidate(nva2f::IGeometryInteractiveExecutor::kLayerEmotionAccumulator));
        return {kInference, kSkin, kTongue, kTeeth, kEyes};
      },
      [](auto& executor) -> layers_t {
        EXPECT_TRUE(!nva2f::SetInteractiveExecutorInputStrength_INTERNAL(executor, 1.23f));
        return {kInference, kSkin, kTongue, kTeeth, kEyes};
      },
      [](auto& executor) -> layers_t {
        // Same value as before.
        EXPECT_TRUE(!nva2f::SetInteractiveExecutorInputStrength_INTERNAL(executor, 1.23f));
        return {};
      },
    };

    // Add test cases for tweaking parameters.

    // Skin
    using test_param_func_skin_t = std::function<bool(nva2f::AnimatorSkinParams&)>;
    const std::vector<test_param_func_skin_t> test_param_funcs_skin = {
      [](auto& params) {
        params.lowerFaceSmoothing = 0.023f;
        return true;
      },
      [](auto& params) {
        params.lowerFaceSmoothing = 0.023f;
        return false;
      },
      [](auto& params) {
        params.upperFaceSmoothing = 0.024f;
        return true;
      },
      [](auto& params) {
        params.upperFaceSmoothing = 0.024f;
        return false;
      },
      [](auto& params) {
        params.lowerFaceStrength = 1.25f;
        return true;
      },
      [](auto& params) {
        params.lowerFaceStrength = 1.25f;
        return false;
      },
      [](auto& params) {
        params.upperFaceStrength = 1.26f;
        return true;
      },
      [](auto& params) {
        params.upperFaceStrength = 1.26f;
        return false;
      },
      [](auto& params) {
        params.faceMaskLevel = 0.27f;
        return true;
      },
      [](auto& params) {
        params.faceMaskLevel = 0.27f;
        return false;
      },
      [](auto& params) {
        params.faceMaskSoftness = 0.28f;
        return true;
      },
      [](auto& params) {
        params.faceMaskSoftness = 0.28f;
        return false;
      },
      [](auto& params) {
        params.skinStrength = 1.29f;
        return true;
      },
      [](auto& params) {
        params.skinStrength = 1.29f;
        return false;
      },
      [](auto& params) {
        params.blinkStrength = 1.30f;
        return true;
      },
      [](auto& params) {
        params.blinkStrength = 1.30f;
        return false;
      },
      [](auto& params) {
        params.eyelidOpenOffset = 0.31f;
        return true;
      },
      [](auto& params) {
        params.eyelidOpenOffset = 0.31f;
        return false;
      },
      [](auto& params) {
        params.lipOpenOffset = 0.032f;
        return true;
      },
      [](auto& params) {
        params.lipOpenOffset = 0.032f;
        return false;
      },
      [](auto& params) {
        params.blinkOffset = 0.33f;
        return true;
      },
      [](auto& params) {
        params.blinkOffset = 0.33f;
        return false;
      },
    };
    add_params_test_case(
      test_funcs,
      test_param_funcs_skin,
      nva2f::SetInteractiveExecutorSkinParameters_INTERNAL,
      nva2f::GetInteractiveExecutorSkinParameters_INTERNAL,
      {kSkin}
      );

    // Tongue
    using test_param_func_tongue_t = std::function<bool(nva2f::AnimatorTongueParams&)>;
    const std::vector<test_param_func_tongue_t> test_param_funcs_tongue = {
      [](auto& params) {
        params.tongueStrength = 1.23f;
        return true;
      },
      [](auto& params) {
        params.tongueStrength = 1.23f;
        return false;
      },
      [](auto& params) {
        params.tongueHeightOffset = 1.24f;
        return true;
      },
      [](auto& params) {
        params.tongueHeightOffset = 1.24f;
        return false;
      },
      [](auto& params) {
        params.tongueDepthOffset = 1.25f;
        return true;
      },
      [](auto& params) {
        params.tongueDepthOffset = 1.25f;
        return false;
      },
    };
    add_params_test_case(
      test_funcs,
      test_param_funcs_tongue,
      nva2f::SetInteractiveExecutorTongueParameters_INTERNAL,
      nva2f::GetInteractiveExecutorTongueParameters_INTERNAL,
      {kTongue}
      );

    // Teeth
    using test_param_func_teeth_t = std::function<bool(nva2f::AnimatorTeethParams&)>;
    const std::vector<test_param_func_teeth_t> test_param_funcs_teeth = {
      [](auto& params) {
        params.lowerTeethStrength = 1.23f;
        return true;
      },
      [](auto& params) {
        params.lowerTeethStrength = 1.23f;
        return false;
      },
      [](auto& params) {
        params.lowerTeethHeightOffset = 1.24f;
        return true;
      },
      [](auto& params) {
        params.lowerTeethHeightOffset = 1.24f;
        return false;
      },
      [](auto& params) {
        params.lowerTeethDepthOffset = 1.25f;
        return true;
      },
      [](auto& params) {
        params.lowerTeethDepthOffset = 1.25f;
        return false;
      },
    };
    add_params_test_case(
      test_funcs,
      test_param_funcs_teeth,
      nva2f::SetInteractiveExecutorTeethParameters_INTERNAL,
      nva2f::GetInteractiveExecutorTeethParameters_INTERNAL,
      {kTeeth}
      );

    // Eyes
    using test_param_func_eyes_t = std::function<bool(nva2f::AnimatorEyesParams&)>;
    const std::vector<test_param_func_eyes_t> test_param_funcs_eyes = {
      [](auto& params) {
        params.eyeballsStrength = 1.23f;
        return true;
      },
      [](auto& params) {
        params.eyeballsStrength = 1.23f;
        return false;
      },
      [](auto& params) {
        params.saccadeStrength = 1.24f;
        return true;
      },
      [](auto& params) {
        params.saccadeStrength = 1.24f;
        return false;
      },
      [](auto& params) {
        params.rightEyeballRotationOffsetX = 1.25f;
        return true;
      },
      [](auto& params) {
        params.rightEyeballRotationOffsetX = 1.25f;
        return false;
      },
      [](auto& params) {
        params.rightEyeballRotationOffsetY = 1.26f;
        return true;
      },
      [](auto& params) {
        params.rightEyeballRotationOffsetY = 1.26f;
        return false;
      },
      [](auto& params) {
        params.leftEyeballRotationOffsetX = 1.27f;
        return true;
      },
      [](auto& params) {
        params.leftEyeballRotationOffsetX = 1.27f;
        return false;
      },
      [](auto& params) {
        params.leftEyeballRotationOffsetY = 1.28f;
        return true;
      },
      [](auto& params) {
        params.leftEyeballRotationOffsetY = 1.28f;
        return false;
      },
      [](auto& params) {
        params.saccadeSeed = 129.0f;
        return true;
      },
      [](auto& params) {
        params.saccadeSeed = 129.0f;
        return false;
      },
    };
    add_params_test_case(
      test_funcs,
      test_param_funcs_eyes,
      nva2f::SetInteractiveExecutorEyesParameters_INTERNAL,
      nva2f::GetInteractiveExecutorEyesParameters_INTERNAL,
      {kEyes}
      );

    return test_funcs;
  }();

}

TEST(TestCoreInteractiveExecutor, Invalidation) {
  for (const auto regression : {true, false}) {
    // Create interactive executor with batch size 0 (i.e. max batch size) for faster execution.
    auto interactiveExecutorBundle = CreateGeometryInteractiveExecutorBundle(regression, 0, true);
    ASSERT_TRUE(interactiveExecutorBundle);

    {
      auto& geometryExecutor = static_cast<nva2f::IGeometryInteractiveExecutor&>(interactiveExecutorBundle->GetExecutor());
      auto callback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
        return true;
      };
      ASSERT_TRUE(!geometryExecutor.SetResultsCallback(callback, nullptr));
    }

    // Put less audio for faster exection.
    ASSERT_TRUE(!interactiveExecutorBundle->GetAudioAccumulator().Reset());
    const std::vector<float> audio(1000, 0.0f);
    ASSERT_TRUE(
      !interactiveExecutorBundle->GetAudioAccumulator().Accumulate(
        nva2x::ToConstView(audio), interactiveExecutorBundle->GetCudaStream().Data()
        )
      );
    ASSERT_TRUE(!interactiveExecutorBundle->GetAudioAccumulator().Close());

    for (std::size_t testIndex = 0; testIndex < kValidateTestFuncs.size(); ++testIndex) {
      const auto& test_func = kValidateTestFuncs[testIndex];
      auto& executor = interactiveExecutorBundle->GetExecutor();
      const auto invalidLayers = test_func(executor);
      ASSERT_TRUE(validate(executor, {kInference, kSkin, kTongue, kTeeth, kEyes}, invalidLayers)) << "Test " << testIndex;

      static_assert(nva2f::IGeometryInteractiveExecutor::kLayerAudioAccumulator == nva2f::IGeometryInteractiveExecutor::kLayerInference);
      static_assert(nva2f::IGeometryInteractiveExecutor::kLayerEmotionAccumulator == nva2f::IGeometryInteractiveExecutor::kLayerInference);

      // Reset the invalidation state.
      ASSERT_TRUE(!executor.ComputeAllFrames());

      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerNone));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerAll));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerInference));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerSkin));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerTongue));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerTeeth));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerEyes));
      ASSERT_FALSE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerEyes + 1));
    }
  }
}

namespace {

  constexpr auto kSkinSolverPrepare = nva2f::IBlendshapeInteractiveExecutor::kLayerSkinSolverPrepare;
  constexpr auto kTongueSolverPrepare = nva2f::IBlendshapeInteractiveExecutor::kLayerTongueSolverPrepare;
  constexpr auto kBlendshapeWeights = nva2f::IBlendshapeInteractiveExecutor::kLayerBlendshapeWeights;

  static const auto kValidateTestFuncsBlendshape = []() {
    // Add test cases for tweaking parameters.
    std::vector<test_func_t> test_funcs;

    // Config
    using test_param_func_config_t = std::function<bool(nva2f::BlendshapeSolverConfig&)>;
    const std::vector<test_param_func_config_t> test_param_funcs_config = {
      [](auto& params) {
        static std::vector<int> activePoses{params.activePoses, params.activePoses + params.numBlendshapes};
        const auto index = activePoses.size() / 2;
        activePoses[index] = 1 - activePoses[index];
        params.activePoses = activePoses.data();
        return true;
      },
      [](auto& params) {
        return false;
      },
    };
    add_params_test_case(
      test_funcs,
      test_param_funcs_config,
      nva2f::SetInteractiveExecutorBlendshapeSkinConfig_INTERNAL,
      nva2f::GetInteractiveExecutorBlendshapeSkinConfig_INTERNAL,
      {kSkinSolverPrepare, kBlendshapeWeights}
      );
    add_params_test_case(
      test_funcs,
      test_param_funcs_config,
      nva2f::SetInteractiveExecutorBlendshapeTongueConfig_INTERNAL,
      nva2f::GetInteractiveExecutorBlendshapeTongueConfig_INTERNAL,
      {kTongueSolverPrepare, kBlendshapeWeights}
      );

    // Params
    using test_param_func_params_t = std::function<bool(nva2f::BlendshapeSolverParams&)>;
    const std::vector<test_param_func_params_t> test_param_funcs_params = {
      [](auto& params) {
        params.L1Reg = 1.23f;
        return true;
      },
      [](auto& params) {
        params.L1Reg = 1.23f;
        return false;
      },
      [](auto& params) {
        params.L2Reg = 1.24f;
        return true;
      },
      [](auto& params) {
        params.L2Reg = 1.24f;
        return false;
      },
      [](auto& params) {
        params.SymmetryReg = 1.25f;
        return true;
      },
      [](auto& params) {
        params.SymmetryReg = 1.25f;
        return false;
      },
      [](auto& params) {
        params.TemporalReg = 1.26f;
        return true;
      },
      [](auto& params) {
        params.TemporalReg = 1.26f;
        return false;
      },
      [](auto& params) {
        params.templateBBSize = 1.27f;
        return true;
      },
      [](auto& params) {
        params.templateBBSize = 1.27f;
        return false;
      },
      [](auto& params) {
        params.tolerance = 1.28f;
        return true;
      },
      [](auto& params) {
        params.tolerance = 1.28f;
        return false;
      },
    };
    add_params_test_case(
      test_funcs,
      test_param_funcs_params,
      nva2f::SetInteractiveExecutorBlendshapeSkinParameters_INTERNAL,
      nva2f::GetInteractiveExecutorBlendshapeSkinParameters_INTERNAL,
      {kSkinSolverPrepare, kBlendshapeWeights}
      );
    add_params_test_case(
      test_funcs,
      test_param_funcs_params,
      nva2f::SetInteractiveExecutorBlendshapeTongueParameters_INTERNAL,
      nva2f::GetInteractiveExecutorBlendshapeTongueParameters_INTERNAL,
      {kTongueSolverPrepare, kBlendshapeWeights}
      );

    return test_funcs;
  }();

}

TEST(TestCoreInteractiveExecutor, InvalidationBlendshape) {
  for (const auto& [regression, useGpuSolver] : {
    std::make_pair(true, false),
    std::make_pair(true, true),
    std::make_pair(false, false),
    std::make_pair(false, true),
    }) {
    // Create interactive executor with batch size 0 (i.e. max batch size) for faster execution.
    auto interactiveExecutorBundle = CreateBlendshapeSolveInteractiveExecutorBundle(regression, 0, useGpuSolver, true);
    ASSERT_TRUE(interactiveExecutorBundle);

    // Put less audio for faster exection.
    ASSERT_TRUE(!interactiveExecutorBundle->GetAudioAccumulator().Reset());
    const std::vector<float> audio(1000, 0.0f);
    ASSERT_TRUE(
      !interactiveExecutorBundle->GetAudioAccumulator().Accumulate(
        nva2x::ToConstView(audio), interactiveExecutorBundle->GetCudaStream().Data()
        )
      );
    ASSERT_TRUE(!interactiveExecutorBundle->GetAudioAccumulator().Close());

    // Set a callback but ignore the results.
    auto& blendshapeExecutor = static_cast<nva2f::IBlendshapeInteractiveExecutor&>(interactiveExecutorBundle->GetExecutor());
    frames_t interactiveFrames;
    if (blendshapeExecutor.GetResultType() == nva2f::IBlendshapeInteractiveExecutor::ResultsType::HOST) {
      ASSERT_TRUE(!blendshapeExecutor.SetResultsCallback(callbackBlendshapeHost, &interactiveFrames));
    }
    else {
      ASSERT_TRUE(!blendshapeExecutor.SetResultsCallback(callbackBlendshapeDevice, &interactiveFrames));
    }

    for (std::size_t testIndex = 0; testIndex < kValidateTestFuncs.size(); ++testIndex) {
      const auto& test_func = kValidateTestFuncs[testIndex];
      auto& executor = interactiveExecutorBundle->GetExecutor();
      auto invalidLayers = test_func(executor);
      if (!invalidLayers.empty()) {
        invalidLayers.emplace_back(kBlendshapeWeights);
      }

      ASSERT_TRUE(
        validate(
          executor,
          {kInference, kSkin, kTongue, kTeeth, kEyes, kSkinSolverPrepare, kTongueSolverPrepare, kBlendshapeWeights},
          invalidLayers
          )
        ) << "Test " << testIndex;

      // Reset the invalidation state.
      interactiveFrames.clear();
      ASSERT_TRUE(!executor.ComputeAllFrames());

      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerNone));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerAll));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerInference));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerSkin));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerTongue));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerTeeth));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerEyes));
      ASSERT_TRUE(executor.IsValid(nva2f::IBlendshapeInteractiveExecutor::kLayerSkinSolverPrepare));
      ASSERT_TRUE(executor.IsValid(nva2f::IBlendshapeInteractiveExecutor::kLayerTongueSolverPrepare));
      ASSERT_TRUE(executor.IsValid(nva2f::IBlendshapeInteractiveExecutor::kLayerBlendshapeWeights));
    }

    for (std::size_t testIndex = 0; testIndex < kValidateTestFuncsBlendshape.size(); ++testIndex) {
      const auto& test_func = kValidateTestFuncsBlendshape[testIndex];
      auto& executor = interactiveExecutorBundle->GetExecutor();
      auto invalidLayers = test_func(executor);

      ASSERT_TRUE(
        validate(
          executor,
          {kInference, kSkin, kTongue, kTeeth, kEyes, kSkinSolverPrepare, kTongueSolverPrepare, kBlendshapeWeights},
          invalidLayers
          )
        ) << "Test " << testIndex;

      // Reset the invalidation state.
      interactiveFrames.clear();
      ASSERT_TRUE(!executor.ComputeAllFrames());

      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerNone));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerAll));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerInference));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerSkin));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerTongue));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerTeeth));
      ASSERT_TRUE(executor.IsValid(nva2f::IGeometryInteractiveExecutor::kLayerEyes));
      ASSERT_TRUE(executor.IsValid(nva2f::IBlendshapeInteractiveExecutor::kLayerSkinSolverPrepare));
      ASSERT_TRUE(executor.IsValid(nva2f::IBlendshapeInteractiveExecutor::kLayerTongueSolverPrepare));
      ASSERT_TRUE(executor.IsValid(nva2f::IBlendshapeInteractiveExecutor::kLayerBlendshapeWeights));
    }
  }
}
