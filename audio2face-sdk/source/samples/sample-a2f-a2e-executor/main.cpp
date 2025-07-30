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
#include "audio2emotion/audio2emotion.h"
#include "audio2x/cuda_utils.h"

#include <cstdint>
#include "AudioFile.h"

#include <any>
#include <iostream>
#include <memory>

//
// This example demonstrates how to use Audio2Face and Audio2Emotion executors.
// It shows several ways to create and run geometry executors along with
// emotion executors:
//
// 1. Creating executors for regression model vs. diffusion model
// 2. Creating executors using bundles vs. creating individual components
// 3. Running executors in offline mode (all audio is available upfront) vs.
//    streaming mode (audio is available by chunk as execution progresses)
// 4. Proper setup of callbacks and data flow between audio, emotion, and
//    geometry processing
//

//
// Boilerplate utilities for sample.
//

#define CHECK_RESULT(func)                                                     \
  {                                                                            \
    std::error_code error = (func);                                            \
    if (error) {                                                               \
      std::cout << "Error (" << __LINE__ << "): Failed to execute: " << #func; \
      std::cout << ", Reason: "<< error.message() << std::endl;                \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CHECK_ERROR(expression)                                                \
  {                                                                            \
    if (!(expression)) {                                                       \
      std::cout << "Error (" << __LINE__ << "): " << #expression;              \
      std::cout << " is NULL" << std::endl;                                    \
      exit(1);                                                                 \
    }                                                                          \
  }

struct Destroyer {
  template <typename T> void operator()(T *obj) const {
    obj->Destroy();
  }
};
template <typename T> using UniquePtr = std::unique_ptr<T, Destroyer>;
template <typename T> UniquePtr<T> ToUniquePtr(T* ptr) { return UniquePtr<T>(ptr); }

template <typename T>
std::shared_ptr<T> ToSharedPtr(T* ptr) {
  return std::shared_ptr<T>(ptr, [](T* p) { p->Destroy(); });
}

std::vector<float> readAudio(std::string_view audioFilePath) {
  AudioFile<float> audioFile;
  std::cout << "Loading audio file: " << audioFilePath << std::endl;
  CHECK_ERROR(audioFile.load(audioFilePath.data()));
  audioFile.printSummary();
  CHECK_ERROR(audioFile.getSampleRate() == 16000);

  return audioFile.samples[0];
}

std::vector<float> loadAudio() {
  constexpr std::string_view audioFilePath = TEST_DATA_DIR "sample-data/audio_4sec_16k_s16le.wav";
  return readAudio(audioFilePath);
}


//
// Helpers to create geometry executors.
//

struct GeometryExecutorData {
  nva2x::ICudaStream* cudaStream{nullptr};
  nva2x::IAudioAccumulator* audioAccumulator{nullptr};
  nva2x::IEmotionAccumulator* emotionAccumulator{nullptr};
  nva2f::IGeometryExecutor* executor{nullptr};

  // Use a std::any to store data that will be destroyed with GeometryExecutorData.
  std::any ownedData;
};

// Create a regression geometry executor using a bundle, which handles the
// creation of the various objects needed for the geometry executor.
GeometryExecutorData CreateRegressionGeometryExecutorBundle() {
  constexpr char filename[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
  auto bundle = ToSharedPtr(
    nva2f::ReadRegressionGeometryExecutorBundle(
      1,
      filename,
      nva2f::IGeometryExecutor::ExecutionOption::All,
      60, 1,
      nullptr
      )
    );
  CHECK_ERROR(bundle);

  GeometryExecutorData data;
  data.cudaStream = &bundle->GetCudaStream();
  data.audioAccumulator = &bundle->GetAudioAccumulator(0);
  data.emotionAccumulator = &bundle->GetEmotionAccumulator(0);
  data.executor = &bundle->GetExecutor();
  data.ownedData = std::move(bundle);

  return data;
}

// Create a diffusion geometry executor using a bundle, which handles the
// creation of the various objects needed for the geometry executor.
GeometryExecutorData CreateDiffusionGeometryExecutorBundle() {
  constexpr char filename[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json";
  auto bundle = ToSharedPtr(
    nva2f::ReadDiffusionGeometryExecutorBundle(
      1,
      filename,
      nva2f::IGeometryExecutor::ExecutionOption::All,
      0,
      false,
      nullptr
      )
    );
  CHECK_ERROR(bundle);

  GeometryExecutorData data;
  data.cudaStream = &bundle->GetCudaStream();
  data.audioAccumulator = &bundle->GetAudioAccumulator(0);
  data.emotionAccumulator = &bundle->GetEmotionAccumulator(0);
  data.executor = &bundle->GetExecutor();
  data.ownedData = std::move(bundle);

  return data;
}

// Create a regression geometry executor by building individual pieces.
GeometryExecutorData CreateRegressionGeometryExecutorPieces() {
  constexpr char filename[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json";
  const auto modelInfo = ToUniquePtr(nva2f::ReadRegressionModelInfo(filename));
  CHECK_ERROR(modelInfo);

  auto cudaStream = ToSharedPtr(nva2x::CreateCudaStream());
  CHECK_ERROR(cudaStream);

  // Audio accumulator will have buffers of 1 second each, no pre-allocation.
  auto audioAccumulator = ToSharedPtr(nva2x::CreateAudioAccumulator(16000, 0));
  CHECK_ERROR(audioAccumulator);

  // Emotion accumulator will have buffers of 300 frames each, no pre-allocation.
  auto emotionAccumulator = ToSharedPtr(
    nva2x::CreateEmotionAccumulator(modelInfo->GetNetworkInfo().GetEmotionsCount(), 300, 0)
    );
  CHECK_ERROR(emotionAccumulator);

  // Create geometry executor.
  nva2f::GeometryExecutorCreationParameters params;
  params.cudaStream = cudaStream->Data();
  params.nbTracks = 1;
  const auto sharedAudioAccumulator = audioAccumulator.get();
  params.sharedAudioAccumulators = &sharedAudioAccumulator;
  const auto sharedEmotionAccumulator = emotionAccumulator.get();
  params.sharedEmotionAccumulators = &sharedEmotionAccumulator;

  // Run at 60 FPS.
  const auto regressionParams = modelInfo->GetExecutorCreationParameters(
    nva2f::IGeometryExecutor::ExecutionOption::All, 60, 1
    );

  auto executor = ToSharedPtr(
    nva2f::CreateRegressionGeometryExecutor(params, regressionParams)
    );
  CHECK_ERROR(executor);

  GeometryExecutorData data;
  data.cudaStream = cudaStream.get();
  data.audioAccumulator = audioAccumulator.get();
  data.emotionAccumulator = emotionAccumulator.get();
  data.executor = executor.get();
  data.ownedData = std::vector<std::any>{
    std::move(cudaStream),
    std::move(audioAccumulator),
    std::move(emotionAccumulator),
    std::move(executor)
  };

  return data;
}

// Create a diffusion geometry executor by building individual pieces.
GeometryExecutorData CreateDiffusionGeometryExecutorPieces() {
  constexpr char filename[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json";
  const auto modelInfo = ToUniquePtr(nva2f::ReadDiffusionModelInfo(filename));
  CHECK_ERROR(modelInfo);

  auto cudaStream = ToSharedPtr(nva2x::CreateCudaStream());
  CHECK_ERROR(cudaStream);

  // Audio accumulator will have buffers of 1 second each, no pre-allocation.
  auto audioAccumulator = ToSharedPtr(nva2x::CreateAudioAccumulator(16000, 0));
  CHECK_ERROR(audioAccumulator);

  // Emotion accumulator will have buffers of 300 frames each, no pre-allocation.
  auto emotionAccumulator = ToSharedPtr(
    nva2x::CreateEmotionAccumulator(modelInfo->GetNetworkInfo().GetEmotionsCount(), 300, 0)
    );
  CHECK_ERROR(emotionAccumulator);

  // Create geometry executor.
  nva2f::GeometryExecutorCreationParameters params;
  params.cudaStream = cudaStream->Data();
  params.nbTracks = 1;
  const auto sharedAudioAccumulator = audioAccumulator.get();
  params.sharedAudioAccumulators = &sharedAudioAccumulator;
  const auto sharedEmotionAccumulator = emotionAccumulator.get();
  params.sharedEmotionAccumulators = &sharedEmotionAccumulator;

  // Choose identity 0 and non-constant noise.
  const auto diffusionParams = modelInfo->GetExecutorCreationParameters(
    nva2f::IGeometryExecutor::ExecutionOption::All, 0, false
    );

  auto executor = ToSharedPtr(
    nva2f::CreateDiffusionGeometryExecutor(params, diffusionParams)
    );
  CHECK_ERROR(executor);

  GeometryExecutorData data;
  data.cudaStream = cudaStream.get();
  data.audioAccumulator = audioAccumulator.get();
  data.emotionAccumulator = emotionAccumulator.get();
  data.executor = executor.get();
  data.ownedData = std::vector<std::any>{
    std::move(cudaStream),
    std::move(audioAccumulator),
    std::move(emotionAccumulator),
    std::move(executor)
  };

  return data;
}


//
// Helper to create emotion executors.
//

// Create a simple emotion executor.
UniquePtr<nva2e::IEmotionExecutor> CreateEmotionExecutor(
  cudaStream_t cudaStream, nva2x::IAudioAccumulator& audioAccumulator
  ) {
  auto modelInfo = ToUniquePtr(nva2e::ReadClassifierModelInfo(
      TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json"
      ));
  CHECK_ERROR(modelInfo);

  nva2e::EmotionExecutorCreationParameters params;
  params.cudaStream = cudaStream;
  params.nbTracks = 1;
  const auto sharedAudioAccumulator = &audioAccumulator;
  params.sharedAudioAccumulators = &sharedAudioAccumulator;

  // Notice that it produces frames at a different frame rate than the geometry executor.
  auto classifierParams = modelInfo->GetExecutorCreationParameters(
    60000, 30, 1, 30
    );

  auto executor = ToUniquePtr(nva2e::CreateClassifierEmotionExecutor(params, classifierParams));
  CHECK_ERROR(executor);

  return executor;
}


//
// Functions to run the executors.
//

// This function runs the executor offline, with the audio already accumulated.
void RunExecutorOffline(
  GeometryExecutorData geometryExecutorData,
  UniquePtr<nva2e::IEmotionExecutor> emotionExecutor
  ) {
  //
  // Setup
  //

  // Set a callback on the geometry executor to get the results.
  // In this example, it simply counts the number of frames processed.
  struct GeometryExecutorCallbackData {
    std::size_t frameIndex{0};
  };
  GeometryExecutorCallbackData callbackData;
  auto callback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
    auto& data = *static_cast<GeometryExecutorCallbackData*>(userdata);
    data.frameIndex++;
    return true;
  };
  CHECK_RESULT(geometryExecutorData.executor->SetResultsCallback(callback, &callbackData));

  // Connect the emotion executor to the emotion accumulator.
  // This will be the link to the geometry executor.
  // This is the manual way to do it, but only works for a single track.
  auto emotionExecutorCallback = [](void* userdata, const nva2e::IEmotionExecutor::Results& results) {
    assert(results.trackIndex == 0);
    auto& emotionAccumulator = *static_cast<nva2x::IEmotionAccumulator*>(userdata);
    CHECK_RESULT(emotionAccumulator.Accumulate(results.timeStampCurrentFrame, results.emotions, results.cudaStream));
    return true;
  };
  CHECK_RESULT(emotionExecutor->SetResultsCallback(emotionExecutorCallback, geometryExecutorData.emotionAccumulator));

  // Then, load all the audio and accumulate it.
  const auto audioBuffer = loadAudio();
  CHECK_ERROR(!audioBuffer.empty());
  CHECK_RESULT(
    geometryExecutorData.audioAccumulator->Accumulate(
      nva2x::HostTensorFloatConstView{audioBuffer.data(), audioBuffer.size()}, geometryExecutorData.cudaStream->Data()
      )
    );
  CHECK_RESULT(geometryExecutorData.audioAccumulator->Close());

  //
  // Execution.
  //

  // Process all emotion.
  while (nva2x::GetNbReadyTracks(*emotionExecutor) > 0) {
    CHECK_RESULT(emotionExecutor->Execute(nullptr));
  }
  CHECK_RESULT(geometryExecutorData.emotionAccumulator->Close());

  // Process all geometry.
  while (nva2x::GetNbReadyTracks(*geometryExecutorData.executor)) {
    CHECK_RESULT(geometryExecutorData.executor->Execute(nullptr));
  }

  std::cout << "Processed " << callbackData.frameIndex << " frames." << std::endl;
}

// This function runs the executor in a streaming mode, where the audio is
// accumulated in between executions.
void RunExecutorStreaming(
  GeometryExecutorData geometryExecutorData,
  UniquePtr<nva2e::IEmotionExecutor> emotionExecutor
  ) {
  //
  // Setup
  //

  // Set a callback on the geometry executor to get the results.
  // In this example, it simply counts the number of frames processed.
  struct GeometryExecutorCallbackData {
    std::size_t frameIndex{0};
  };
  GeometryExecutorCallbackData callbackData;
  auto callback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
    auto& data = *static_cast<GeometryExecutorCallbackData*>(userdata);
    data.frameIndex++;
    return true;
  };
  CHECK_RESULT(geometryExecutorData.executor->SetResultsCallback(callback, &callbackData));

  // Connect the emotion executor to the emotion accumulator.
  // This will be the link to the geometry executor.
  // This is the manual way to do it, and works for multiple tracks.
  std::vector<nva2x::IEmotionAccumulator*> emotionAccumulators = {
    geometryExecutorData.emotionAccumulator
  };
  auto emotionExecutorCallback = [](void* userdata, const nva2e::IEmotionExecutor::Results& results) {
    auto& emotionAccumulators = *static_cast<std::vector<nva2x::IEmotionAccumulator*>*>(userdata);
    CHECK_RESULT(emotionAccumulators[results.trackIndex]->Accumulate(results.timeStampCurrentFrame, results.emotions, results.cudaStream));
    return true;
  };
  CHECK_RESULT(emotionExecutor->SetResultsCallback(emotionExecutorCallback, &emotionAccumulators));


  // First, load all the audio, but don't accumulate it yet.
  const auto audioBuffer = loadAudio();
  CHECK_ERROR(!audioBuffer.empty());

  //
  // Execution.
  //

  // Processing tries to do as much emotion as possible first, because it might
  // unblock geometry processing which requires emotion results.
  auto processAvailableData = [&]() {
    // Note that in multi-track mode, we might want to only run when at least
    // a certain number of tracks are ready to maximize parallelism.

    // Process available emotion.
    while (nva2x::GetNbReadyTracks(*emotionExecutor) > 0) {
      CHECK_RESULT(emotionExecutor->Execute(nullptr));
    }
    if (geometryExecutorData.audioAccumulator->IsClosed()) {
      // We are done accumulating audio.  If there is no more emotion to process,
      // it means we are done accumulating emotions.
      if (emotionExecutor->GetNbAvailableExecutions(0) == 0) {
        CHECK_RESULT(geometryExecutorData.emotionAccumulator->Close());
      }
    }

    const auto frameIndexBefore = callbackData.frameIndex;
    // Process available geometry.
    while (nva2x::GetNbReadyTracks(*geometryExecutorData.executor)) {
      CHECK_RESULT(geometryExecutorData.executor->Execute(nullptr));
    }
    const auto frameIndexAfter = callbackData.frameIndex;
    std::cout << "   Processed " << (frameIndexAfter - frameIndexBefore)
              << " frames in chunk." << std::endl;
  };

  // Process the audio in chunks of 16000 samples.
  constexpr std::size_t kChunkSize = 16000;
  for (std::size_t i = 0; i < audioBuffer.size(); i += kChunkSize) {
    // Load an audio chunk.
    // This could come from another thread.
    const auto chunkData = audioBuffer.data() + i;
    const auto chunkSize = std::min(kChunkSize, audioBuffer.size() - i);
    CHECK_RESULT(
      geometryExecutorData.audioAccumulator->Accumulate(
        nva2x::HostTensorFloatConstView{chunkData, chunkSize}, geometryExecutorData.cudaStream->Data()
        )
      );

    // Process available data.
    processAvailableData();
  }
  CHECK_RESULT(geometryExecutorData.audioAccumulator->Close());
  // After closing the audio, we might be able to do more processing.
  processAvailableData();

  std::cout << "Processed " << callbackData.frameIndex << " frames." << std::endl;
}


// This function runs the executor in a streaming mode, where the audio is
// accumulated in between executions, but tries returning frames with the lowest
// possible latency.
void RunExecutorStreamingLowLatency(
  GeometryExecutorData geometryExecutorData,
  UniquePtr<nva2e::IEmotionExecutor> emotionExecutor
  ) {
  //
  // Setup
  //

  // Set a callback on the geometry executor to get the results.
  // In this example, it simply counts the number of frames processed.
  struct GeometryExecutorCallbackData {
    std::size_t frameIndex{0};
  };
  GeometryExecutorCallbackData callbackData;
  auto callback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
    auto& data = *static_cast<GeometryExecutorCallbackData*>(userdata);
    data.frameIndex++;
    return true;
  };
  CHECK_RESULT(geometryExecutorData.executor->SetResultsCallback(callback, &callbackData));

  // Connect the emotion executor to the emotion accumulator.
  // This will be the link to the geometry executor.
  // This is the way using a binder that handles it.
  auto binder = ToUniquePtr(nva2e::CreateEmotionBinder(
    *emotionExecutor, &geometryExecutorData.emotionAccumulator, 1
    ));
  CHECK_ERROR(binder);


  // First, load all the audio, but don't accumulate it yet.
  const auto audioBuffer = loadAudio();
  CHECK_ERROR(!audioBuffer.empty());

  //
  // Execution.
  //

  // Processing tries to do as little emotion processing to unblock geometry
  // processing which requires emotion results, so that emotion processing can
  // run as quickly as possible.
  auto processAvailableData = [&]() {
    // Note that in multi-track mode, we might want to only run when at least
    // a certain number of tracks are ready to maximize parallelism.

    const auto frameIndexBefore = callbackData.frameIndex;
    while (true) {
      // Process available geometry
      const auto nbGeometryTracks = nva2x::GetNbReadyTracks(*geometryExecutorData.executor);
      if (nbGeometryTracks > 0) {
        CHECK_RESULT(geometryExecutorData.executor->Execute(nullptr));
        continue;
      }

      // No geometry can be processed, process emotion which might enable further geometry processing.
      const auto nbEmotionTracks = nva2x::GetNbReadyTracks(*emotionExecutor);
      if (nbEmotionTracks > 0) {
        CHECK_RESULT(emotionExecutor->Execute(nullptr));
        continue;
      }

      if (geometryExecutorData.audioAccumulator->IsClosed()) {
        if (!geometryExecutorData.emotionAccumulator->IsClosed()) {
          // We are done accumulating audio.  If there is no more emotion to process,
          // it means we are done accumulating emotions.
          CHECK_RESULT(geometryExecutorData.emotionAccumulator->Close());
          continue;
        }
      }

      // No more emotion or geometry to process, exit.
      break;
    }

    const auto frameIndexAfter = callbackData.frameIndex;
    std::cout << "   Processed " << (frameIndexAfter - frameIndexBefore)
              << " frames in chunk." << std::endl;
  };

  auto dropUnusedData = [&]() {
    const auto nbTracks = geometryExecutorData.executor->GetNbTracks();
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
      if (!geometryExecutorData.emotionAccumulator->IsEmpty()) {
        const auto timestampToRead = geometryExecutorData.executor->GetNextEmotionTimestampToRead(trackIndex);
        const auto lastAccumulatedTimestamp = geometryExecutorData.emotionAccumulator->LastAccumulatedTimestamp();
        // It might be ok to drop past the last accumulated timestamp, if we are sure
        // to add emotions up to the timestampToRead after (which we will).  But the
        // emotion accumulator can't know that and therefore will return an error if
        // dropping past the last accumulated timestamp.
        const auto timestampToDrop = std::min(timestampToRead, lastAccumulatedTimestamp);
        CHECK_RESULT(geometryExecutorData.emotionAccumulator->DropEmotionsBefore(timestampToDrop));
      }

      // It might be ok to drop certain samples for the geometry executor, but the
        // emotion executor might still need them.  So we take the minimum of the two.
      const auto sampleGeometry = geometryExecutorData.executor->GetNextAudioSampleToRead(trackIndex);
      const auto sampleEmotion = emotionExecutor->GetNextAudioSampleToRead(trackIndex);
      const auto sample = std::min(sampleGeometry, sampleEmotion);
      CHECK_RESULT(geometryExecutorData.audioAccumulator->DropSamplesBefore(sample));
    }
  };

  // Process the audio in chunks of 16000 samples.
  constexpr std::size_t kChunkSize = 16000;
  for (std::size_t i = 0; i < audioBuffer.size(); i += kChunkSize) {
    // Load an audio chunk.
    // This could come from another thread.
    const auto chunkData = audioBuffer.data() + i;
    const auto chunkSize = std::min(kChunkSize, audioBuffer.size() - i);
    CHECK_RESULT(
      geometryExecutorData.audioAccumulator->Accumulate(
        nva2x::HostTensorFloatConstView{chunkData, chunkSize}, geometryExecutorData.cudaStream->Data()
        )
      );

    // Process available data.
    processAvailableData();
    // Drop data which will not be read again in the accumulators.
    dropUnusedData();
  }
  CHECK_RESULT(geometryExecutorData.audioAccumulator->Close());
  // After closing the audio, we might be able to do more processing.
  processAvailableData();

  std::cout << "Processed " << callbackData.frameIndex << " frames." << std::endl;
}


//
// Combinations of previous setup functions for complete workflows.
//

// This function creates a regression geometry executor using a bundle,
// and runs the executor offline, with the audio already accumulated.
void RunRegressionBundleOffline() {
  auto geometryExecutorData = CreateRegressionGeometryExecutorBundle();
  auto emotionExecutor = CreateEmotionExecutor(
    geometryExecutorData.cudaStream->Data(), *geometryExecutorData.audioAccumulator
    );
  RunExecutorOffline(geometryExecutorData, std::move(emotionExecutor));
}

// This function creates a diffusion geometry executor using a bundle,
// and runs the executor offline, with the audio already accumulated.
void RunDiffusionBundleOffline() {
  auto geometryExecutorData = CreateDiffusionGeometryExecutorBundle();
  auto emotionExecutor = CreateEmotionExecutor(
    geometryExecutorData.cudaStream->Data(), *geometryExecutorData.audioAccumulator
    );
  RunExecutorOffline(geometryExecutorData, std::move(emotionExecutor));
}

// This function creates a regression geometry executor using individually created pieces,
// and runs the executor offline, with the audio already accumulated.
void RunRegressionPiecesOffline() {
  auto geometryExecutorData = CreateRegressionGeometryExecutorPieces();
  auto emotionExecutor = CreateEmotionExecutor(
    geometryExecutorData.cudaStream->Data(), *geometryExecutorData.audioAccumulator
    );
  RunExecutorOffline(geometryExecutorData, std::move(emotionExecutor));
}

// This function creates a diffusion geometry executor using individually created pieces,
// and runs the executor offline, with the audio already accumulated.
void RunDiffusionPiecesOffline() {
  auto geometryExecutorData = CreateDiffusionGeometryExecutorPieces();
  auto emotionExecutor = CreateEmotionExecutor(
    geometryExecutorData.cudaStream->Data(), *geometryExecutorData.audioAccumulator
    );
  RunExecutorOffline(geometryExecutorData, std::move(emotionExecutor));
}

// This function creates a regression geometry executor using a bundle,
// and runs the executor in a streaming mode, where the audio is
// accumulated in between executions.
void RunRegressionBundleStreaming() {
  auto geometryExecutorData = CreateRegressionGeometryExecutorBundle();
  auto emotionExecutor = CreateEmotionExecutor(
    geometryExecutorData.cudaStream->Data(), *geometryExecutorData.audioAccumulator
    );
  RunExecutorStreaming(geometryExecutorData, std::move(emotionExecutor));
}

// This function creates a diffusion geometry executor using a bundle,
// and runs the executor in a streaming mode, where the audio is
// accumulated in between executions.
void RunDiffusionBundleStreaming() {
  auto geometryExecutorData = CreateDiffusionGeometryExecutorBundle();
  auto emotionExecutor = CreateEmotionExecutor(
    geometryExecutorData.cudaStream->Data(), *geometryExecutorData.audioAccumulator
    );
  RunExecutorStreaming(geometryExecutorData, std::move(emotionExecutor));
}

// This function creates a regression geometry executor using individually created pieces,
// and runs the executor in a streaming mode, where the audio is
// accumulated in between executions.
void RunRegressionPiecesStreaming() {
  auto geometryExecutorData = CreateRegressionGeometryExecutorPieces();
  auto emotionExecutor = CreateEmotionExecutor(
    geometryExecutorData.cudaStream->Data(), *geometryExecutorData.audioAccumulator
    );
  RunExecutorStreaming(geometryExecutorData, std::move(emotionExecutor));
}

// This function creates a diffusion geometry executor using individually created pieces,
// and runs the executor in a streaming mode, where the audio is
// accumulated in between executions.
void RunDiffusionPiecesStreaming() {
  auto geometryExecutorData = CreateDiffusionGeometryExecutorPieces();
  auto emotionExecutor = CreateEmotionExecutor(
    geometryExecutorData.cudaStream->Data(), *geometryExecutorData.audioAccumulator
    );
  RunExecutorStreaming(geometryExecutorData, std::move(emotionExecutor));
}

// This function creates a regression geometry executor using a bundle,
// and runs the executor in a streaming mode, where the audio is
// accumulated in between executions, but tries returning frames with the lowest
// possible latency.
void RunRegressionBundleStreamingLowLatency() {
  auto geometryExecutorData = CreateRegressionGeometryExecutorBundle();
  auto emotionExecutor = CreateEmotionExecutor(
    geometryExecutorData.cudaStream->Data(), *geometryExecutorData.audioAccumulator
    );
  RunExecutorStreamingLowLatency(geometryExecutorData, std::move(emotionExecutor));
}

// This function creates a diffusion geometry executor using a bundle,
// and runs the executor in a streaming mode, where the audio is
// accumulated in between executions, but tries returning frames with the lowest
// possible latency.
void RunDiffusionBundleStreamingLowLatency() {
  auto geometryExecutorData = CreateDiffusionGeometryExecutorBundle();
  auto emotionExecutor = CreateEmotionExecutor(
    geometryExecutorData.cudaStream->Data(), *geometryExecutorData.audioAccumulator
    );
  RunExecutorStreamingLowLatency(geometryExecutorData, std::move(emotionExecutor));
}

// This function creates a regression geometry executor using individually created pieces,
// and runs the executor in a streaming mode, where the audio is
// accumulated in between executions, but tries returning frames with the lowest
// possible latency.
void RunRegressionPiecesStreamingLowLatency() {
  auto geometryExecutorData = CreateRegressionGeometryExecutorPieces();
  auto emotionExecutor = CreateEmotionExecutor(
    geometryExecutorData.cudaStream->Data(), *geometryExecutorData.audioAccumulator
    );
  RunExecutorStreamingLowLatency(geometryExecutorData, std::move(emotionExecutor));
}

// This function creates a diffusion geometry executor using individually created pieces,
// and runs the executor in a streaming mode, where the audio is
// accumulated in between executions, but tries returning frames with the lowest
// possible latency.
void RunDiffusionPiecesStreamingLowLatency() {
  auto geometryExecutorData = CreateDiffusionGeometryExecutorPieces();
  auto emotionExecutor = CreateEmotionExecutor(
    geometryExecutorData.cudaStream->Data(), *geometryExecutorData.audioAccumulator
    );
  RunExecutorStreamingLowLatency(geometryExecutorData, std::move(emotionExecutor));
}


int main(void) {
  std::cout << "================================================" << std::endl;
  std::cout << "    Audio2Face SDK Example" << std::endl;
  std::cout << "    Audio2Face + Audio2Emotion Integration" << std::endl;
  std::cout << "================================================" << std::endl;

  constexpr int deviceID = 0;
  CHECK_RESULT(nva2x::SetCudaDeviceIfNeeded(deviceID));

  std::cout << "\nRunning regression bundle offline...\n" << std::endl;
  RunRegressionBundleOffline();
  std::cout << "\nDone.\n" << std::endl;

  std::cout << "\nRunning diffusion bundle offline...\n" << std::endl;
  RunDiffusionBundleOffline();
  std::cout << "\nDone.\n" << std::endl;

  std::cout << "\nRunning regression pieces offline...\n" << std::endl;
  RunRegressionPiecesOffline();
  std::cout << "\nDone.\n" << std::endl;

  std::cout << "\nRunning diffusion pieces offline...\n" << std::endl;
  RunDiffusionPiecesOffline();
  std::cout << "\nDone.\n" << std::endl;

  std::cout << "\nRunning regression bundle streaming...\n" << std::endl;
  RunRegressionBundleStreaming();
  std::cout << "\nDone.\n" << std::endl;

  std::cout << "\nRunning diffusion bundle streaming...\n" << std::endl;
  RunDiffusionBundleStreaming();
  std::cout << "\nDone.\n" << std::endl;

  std::cout << "\nRunning regression pieces streaming...\n" << std::endl;
  RunRegressionPiecesStreaming();
  std::cout << "\nDone.\n" << std::endl;

  std::cout << "\nRunning diffusion pieces streaming...\n" << std::endl;
  RunDiffusionPiecesStreaming();
  std::cout << "\nDone.\n" << std::endl;

  std::cout << "\nRunning regression bundle streaming low latency...\n" << std::endl;
  RunRegressionBundleStreamingLowLatency();
  std::cout << "\nDone.\n" << std::endl;

  std::cout << "\nRunning diffusion bundle streaming low latency...\n" << std::endl;
  RunDiffusionBundleStreamingLowLatency();
  std::cout << "\nDone.\n" << std::endl;

  std::cout << "\nRunning regression pieces streaming low latency...\n" << std::endl;
  RunRegressionPiecesStreamingLowLatency();
  std::cout << "\nDone.\n" << std::endl;

  std::cout << "\nRunning diffusion pieces streaming low latency...\n" << std::endl;
  RunDiffusionPiecesStreamingLowLatency();
  std::cout << "\nDone.\n" << std::endl;

  std::cout << "================================================" << std::endl;
  std::cout << "    Finished" << std::endl;
  std::cout << "================================================" << std::endl;

  return 0;
}
