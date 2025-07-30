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
#include "audio2x/cuda_utils.h"
#include "audio2x/cuda_stream.h"

#include <cstdint>
#include "AudioFile.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

static constexpr bool kUsePreferredEmotion = false;
static constexpr std::size_t kInferencesToSkip = 30;

#define CHECK_SUCCESS(func)                                                    \
  {                                                                            \
    std::error_code error = func;                                              \
    if (error) {                                                               \
      std::cout << "Error: Failed to execute: " << #func;                      \
      std::cout << ", Reason: " << error.message() << std::endl;               \
      exit(error.value());                                                     \
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

#define CHECK_NOT_NULL(expression)                                             \
  {                                                                            \
    if ((expression) == nullptr) {                                             \
      std::cout << "Error: " << #expression << " is NULL" << std::endl;        \
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

std::vector<float> readAudio(std::string_view audioFilePath) {
    AudioFile<float> audioFile;
    std::cout << "Loading audio file: " << audioFilePath << std::endl;
    audioFile.load(audioFilePath.data());
    audioFile.printSummary();
    CHECK_ERROR(audioFile.getSampleRate() == 16000);

    return audioFile.samples[0];
}

void writeToCSV(std::string_view filename, const std::vector<std::vector<float>>& data) {
    std::ofstream outputFile(filename.data());

    if (!outputFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    const auto emotions = {
        "amazement", "anger", "cheekiness", "disgust", "fear", "grief", "joy", "outofbreath", "pain", "sadness"
        };
    for (std::size_t i = 0; i < emotions.size(); ++i) {
        if (i > 0) {
            outputFile << ',';
        }
        outputFile << *(emotions.begin() + i);
    }
    outputFile << '\n';

    for (const auto& row : data) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            if (i > 0) {
                outputFile << ',';
            }
            outputFile << row[i];
        }
        outputFile << '\n';
    }

    outputFile.close();
}


std::vector<std::vector<float>> runFake(const std::vector<float>& audioBuffer) {
    constexpr std::size_t samplerate = 16000;
    constexpr std::size_t fps = 30;
    constexpr std::size_t nbEmotions = 10;
    const std::size_t nbFrames = (audioBuffer.size() * fps + samplerate - 1) / samplerate;
    std::vector<std::vector<float>> curves(nbFrames);
    for (std::size_t i = 0; i < nbFrames; ++i) {
        const float t = i * (1.0f / fps);
        const float v = (std::sin(t * 10) + 1) * (0.5f / nbEmotions);

        curves[i].resize(nbEmotions);
        for (std::size_t j = 0; j < nbEmotions; ++j) {
            curves[i][j] = v + (1.0f / nbEmotions) * j;
        }
    }
    return curves;
}

std::vector<std::vector<std::vector<float>>> runExecutor(const std::vector<float>& audioBuffer) {
    constexpr int deviceID = 0;
    CHECK_SUCCESS(nva2x::SetCudaDeviceIfNeeded(deviceID));
    auto cudaStream = ToUniquePtr(nva2x::CreateCudaStream());
    CHECK_NOT_NULL(cudaStream);
    auto audioAccumulator = ToUniquePtr(nva2x::CreateAudioAccumulator(16000, 0));
    CHECK_NOT_NULL(audioAccumulator);

    CHECK_SUCCESS(audioAccumulator->Accumulate(nva2x::HostTensorFloatConstView{audioBuffer.data(), audioBuffer.size()}, cudaStream->Data()));
    CHECK_SUCCESS(audioAccumulator->Close());

    auto modelInfo = ToUniquePtr(nva2e::ReadClassifierModelInfo(
        TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json"
        ));
    CHECK_NOT_NULL(modelInfo);

    constexpr std::size_t nbTracks = 8;
    std::array<nva2x::IAudioAccumulator*, nbTracks> sharedAudioAccumulators;
    std::fill(sharedAudioAccumulators.begin(), sharedAudioAccumulators.end(), audioAccumulator.get());

    nva2e::EmotionExecutorCreationParameters params;
    params.cudaStream = cudaStream->Data();
    params.nbTracks = nbTracks;
    params.sharedAudioAccumulators = sharedAudioAccumulators.data();

    auto classifierParams = modelInfo->GetExecutorCreationParameters(
        60000, 30, 1, 0
        );

    std::array<float, 10> preferredEmotions;
    if (kUsePreferredEmotion) {
        std::fill(preferredEmotions.begin(), preferredEmotions.end(), 0.0f);
        preferredEmotions[2] = 0.5f;
        classifierParams.postProcessParams.enablePreferredEmotion = true;
        classifierParams.postProcessParams.preferredEmotion = {preferredEmotions.data(), preferredEmotions.size()};
    }

    classifierParams.inferencesToSkip = kInferencesToSkip;

    auto executor = ToUniquePtr(nva2e::CreateClassifierEmotionExecutor(params, classifierParams));
    CHECK_NOT_NULL(executor);

    std::vector<std::vector<std::vector<float>>> curves(nbTracks);

    auto callback = [](void* userdata, const nva2e::IEmotionExecutor::Results& results) {
        auto& curves = (*static_cast<std::vector<std::vector<std::vector<float>>>*>(userdata))[results.trackIndex];
        std::vector<float> hostEmotions(results.emotions.Size());
        nva2x::CopyDeviceToHost({hostEmotions.data(), hostEmotions.size()}, results.emotions, results.cudaStream);
        curves.emplace_back(std::move(hostEmotions));
        return true;
    };
    CHECK_SUCCESS(executor->SetResultsCallback(callback, &curves));

    while (nva2x::GetNbReadyTracks(*executor) > 0) {
        CHECK_SUCCESS(executor->Execute(nullptr));
    }

    return curves;
}

std::vector<std::vector<float>> runInteractiveExecutor(const std::vector<float>& audioBuffer) {
    constexpr int deviceID = 0;
    CHECK_SUCCESS(nva2x::SetCudaDeviceIfNeeded(deviceID));
    auto cudaStream = ToUniquePtr(nva2x::CreateCudaStream());
    CHECK_NOT_NULL(cudaStream);
    auto audioAccumulator = ToUniquePtr(nva2x::CreateAudioAccumulator(16000, 0));
    CHECK_NOT_NULL(audioAccumulator);

    CHECK_SUCCESS(audioAccumulator->Accumulate(nva2x::HostTensorFloatConstView{audioBuffer.data(), audioBuffer.size()}, cudaStream->Data()));
    CHECK_SUCCESS(audioAccumulator->Close());

    auto modelInfo = ToUniquePtr(nva2e::ReadClassifierModelInfo(
        TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json"
        ));
    CHECK_NOT_NULL(modelInfo);

    constexpr std::size_t nbTracks = 1;
    std::array<nva2x::IAudioAccumulator*, nbTracks> sharedAudioAccumulators;
    std::fill(sharedAudioAccumulators.begin(), sharedAudioAccumulators.end(), audioAccumulator.get());

    nva2e::EmotionExecutorCreationParameters params;
    params.cudaStream = cudaStream->Data();
    params.nbTracks = nbTracks;
    params.sharedAudioAccumulators = sharedAudioAccumulators.data();

    auto classifierParams = modelInfo->GetExecutorCreationParameters(
        60000, 30, 1, 0
        );

    std::array<float, 10> preferredEmotions;
    if (kUsePreferredEmotion) {
        std::fill(preferredEmotions.begin(), preferredEmotions.end(), 0.0f);
        preferredEmotions[2] = 0.5f;
        classifierParams.postProcessParams.enablePreferredEmotion = true;
        classifierParams.postProcessParams.preferredEmotion = {preferredEmotions.data(), preferredEmotions.size()};
    }

    classifierParams.inferencesToSkip = kInferencesToSkip;

    auto executor = ToUniquePtr(nva2e::CreateClassifierEmotionInteractiveExecutor(params, classifierParams, 128));
    CHECK_NOT_NULL(executor);

    std::vector<std::vector<float>> curves(nbTracks);

    auto callback = [](void* userdata, const nva2e::IEmotionInteractiveExecutor::Results& results) {
        assert(results.trackIndex == 0);
        auto& curves = (*static_cast<std::vector<std::vector<float>>*>(userdata));
        std::vector<float> hostEmotions(results.emotions.Size());
        nva2x::CopyDeviceToHost({hostEmotions.data(), hostEmotions.size()}, results.emotions, results.cudaStream);
        curves.emplace_back(std::move(hostEmotions));
        return true;
    };
    CHECK_SUCCESS(executor->SetResultsCallback(callback, &curves));

    CHECK_SUCCESS(executor->ComputeAllFrames());

    return curves;
}


int main(void) {
    constexpr std::string_view audioFilePath = TEST_DATA_DIR "sample-data/audio_4sec_16k_s16le.wav";

    const std::vector<float> audioBuffer = readAudio(audioFilePath);

    {
        const auto curves = runFake(audioBuffer);
        writeToCSV("output_fake.csv", curves);
    }

    {
        const auto curves = runExecutor(audioBuffer);
        for (std::size_t i = 0; i < curves.size(); ++i) {
            writeToCSV("output_executor_" + std::to_string(i) + ".csv", curves[i]);
        }
    }

    {
        const auto curves = runInteractiveExecutor(audioBuffer);
        writeToCSV("output_interactive_executor.csv", curves);
    }

    return 0;
}
