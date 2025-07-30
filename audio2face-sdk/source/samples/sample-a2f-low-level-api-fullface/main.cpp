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
#include "audio2x/cuda_utils.h"
#include "audio2x/cuda_stream.h"
#include "audio2x/io.h"
#include "audio2x/inference_engine.h"

#include <cstdint>
#include "AudioFile.h"
#include "cnpy.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define CHECK_SUCCESS(func)                                                    \
  {                                                                            \
    std::error_code error = func;                                              \
    if (error) {                                                               \
      std::cout << "Error: Failed to execute: " << #func;                      \
      std::cout << ", Reason: "<< error.message() << std::endl;                \
      return false;                                                            \
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
      return false;                                                            \
    }                                                                          \
  }

struct Destroyer {
  template <typename T> void operator()(T *obj) const {
    obj->Destroy();
  }
};
template <typename T> using UniquePtr = std::unique_ptr<T, Destroyer>;
template <typename T> UniquePtr<T> ToUniquePtr(T* ptr) { return UniquePtr<T>(ptr); }

UniquePtr<nva2x::IHostTensorFloat> ReadFromNpyArray(cnpy::npz_t &inputData, const char *key) {
    if (inputData.count(key) == 0) {
        return nullptr;
    }
    cnpy::NpyArray nparr = inputData[key];
    size_t size = 1;
    for(auto s : nparr.shape) {
        size *= s;
    }
    auto tensor = ToUniquePtr(nva2x::CreateHostTensorFloat(size));
    memcpy(tensor->Data(), nparr.data<float>(), size * sizeof(float));
    return tensor;
}

bool sample(void) {
  std::cout << "================================================" << std::endl;
  std::cout << "    Audio2Face SDK Example" << std::endl;
  std::cout << "    Low-level API Full-Face" << std::endl;
  std::cout << "================================================" << std::endl;

  std::string inputDataFilePath = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model_data.npz";
  std::string networkPath = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/network.trt";
  std::string emotionDatabaseFilePath =
      TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/implicit_emo_db.npz";
  std::string emotionShot = "p1_neutral";
  unsigned int emotionFrame = 42;
  std::vector<float> explicitEmotion = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  unsigned int fps = 60;
  nva2f::AnimatorSkinParams animatorSkinParams = {
      0.1f, 0.1f, 1.0f, 1.0f, 0.5f, 0.1f, 1.0f, 1.0f, 0.0f, 0.0f};
  nva2f::AnimatorTongueParams animatorTongueParams = {1.0f, 0.0f, 0.0f};
  nva2f::AnimatorTeethParams animatorTeethParams = {1.0f, 0.0f, 0.0f};
  nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float dt = 1.f / float(fps);
  float inputStrength = 1.0;
  unsigned int trackSamplerate = 16000;
  nva2f::IRegressionModel::NetworkInfo networkInfo = {
      16, 10, 272, 10, 61520 * 3, 5602 * 3, 15, 4, 8320, 4160, 16000};

  constexpr int deviceID = 0;
  CHECK_SUCCESS(nva2x::SetCudaDeviceIfNeeded(deviceID));
  auto cudaStream = ToUniquePtr(nva2x::CreateCudaStream());
  CHECK_NOT_NULL(cudaStream);

  // =============================================================================
  // get the "audioBuffer" from the audio
  const char *audioFilePath = TEST_DATA_DIR "sample-data/audio_4sec_16k_s16le.wav";
  AudioFile<float> audioFile;
  std::cout << "Loading audio file: " << audioFilePath << std::endl;
  audioFile.load(audioFilePath);
  audioFile.printSummary();
  CHECK_ERROR(audioFile.getSampleRate() == 16000);

  std::vector<float> audioBuffer(audioFile.getNumSamplesPerChannel());
  std::memcpy(audioBuffer.data(), audioFile.samples[0].data(), sizeof(float) * audioFile.getNumSamplesPerChannel());

  cnpy::npz_t inputData = cnpy::npz_load(inputDataFilePath);
  auto shapesMatrixSkinHost = ReadFromNpyArray(inputData, "shapes_matrix_skin");
  CHECK_NOT_NULL(shapesMatrixSkinHost);
  auto shapesMeanSkinHost = ReadFromNpyArray(inputData, "shapes_mean_skin");
  CHECK_NOT_NULL(shapesMeanSkinHost);
  auto lipOpenPoseDeltaHost = ReadFromNpyArray(inputData, "lip_open_pose_delta");
  CHECK_NOT_NULL(lipOpenPoseDeltaHost);
  auto eyeClosePoseDeltaHost = ReadFromNpyArray(inputData, "eye_close_pose_delta");
  CHECK_NOT_NULL(eyeClosePoseDeltaHost);
  auto shapesMatrixTongueHost = ReadFromNpyArray(inputData, "shapes_matrix_tongue");
  CHECK_NOT_NULL(shapesMatrixTongueHost);
  auto shapesMeanTongueHost = ReadFromNpyArray(inputData, "shapes_mean_tongue");
  CHECK_NOT_NULL(shapesMeanTongueHost);
  auto neutralJawHost = ReadFromNpyArray(inputData, "neutral_jaw");
  CHECK_NOT_NULL(neutralJawHost);
  auto saccadeRotHost = ReadFromNpyArray(inputData, "saccade_rot_matrix");
  CHECK_NOT_NULL(saccadeRotHost);

  auto inferenceInputBuffers = ToUniquePtr(
    nva2f::CreateInferenceInputBuffersForRegressionModel(networkInfo));
  CHECK_NOT_NULL(inferenceInputBuffers);
  auto inferenceOutputBuffers = ToUniquePtr(
    nva2f::CreateInferenceOutputBuffersForRegressionModel(networkInfo));
  CHECK_NOT_NULL(inferenceOutputBuffers);

  auto accumulator = ToUniquePtr(nva2x::CreateAudioAccumulator(16000, 0));
  CHECK_SUCCESS(accumulator->Accumulate(
    nva2x::HostTensorFloatConstView{audioBuffer.data(), audioBuffer.size()}, cudaStream->Data()
    ));
  CHECK_SUCCESS(accumulator->Close());
  CHECK_SUCCESS(cudaStream->Synchronize());

  auto emotionDatabase = ToUniquePtr(nva2f::CreateEmotionDatabase());
  CHECK_SUCCESS(emotionDatabase->SetCudaStream(cudaStream->Data()));
  CHECK_SUCCESS(emotionDatabase->InitFromFile(emotionDatabaseFilePath.c_str()));
  CHECK_SUCCESS(cudaStream->Synchronize());

  auto networkDataBytes = ToUniquePtr(nva2x::CreateDataBytes());
  CHECK_SUCCESS(networkDataBytes->ReadFromFile(networkPath.c_str()));
  auto inference = ToUniquePtr(nva2x::CreateInferenceEngine());
  CHECK_SUCCESS(inference->Init(networkDataBytes->Data(), networkDataBytes->Size()));
  const auto maxBatchSize = inference->GetMaxBatchSize(nva2f::GetBindingsDescriptionForRegressionModel());
  std::cout << "Max batch size : " << maxBatchSize << std::endl;
  CHECK_SUCCESS(inference->CheckBindings(nva2f::GetBindingsDescriptionForRegressionModel()));

  // Buffer bindings
  auto bindings = ToUniquePtr(nva2f::CreateBindingsForRegressionModel());
  CHECK_NOT_NULL(bindings);

  CHECK_SUCCESS(bindings->SetInputBinding(nva2f::IRegressionModel::kEmotionTensorIndex, inferenceInputBuffers->GetEmotionTensor()));
  CHECK_SUCCESS(bindings->SetInputBinding(nva2f::IRegressionModel::kInputTensorIndex, inferenceInputBuffers->GetInputTensor()));
  CHECK_SUCCESS(bindings->SetOutputBinding(nva2f::IRegressionModel::kResultTensorIndex, inferenceOutputBuffers->GetResultTensor()));
  CHECK_SUCCESS(inference->BindBuffers(*bindings, 1));

  auto animatorPcaSkin = ToUniquePtr(nva2f::CreateAnimatorPcaReconstruction());
  CHECK_SUCCESS(animatorPcaSkin->SetCudaStream(cudaStream->Data()));
  CHECK_SUCCESS(animatorPcaSkin->Init());
  CHECK_SUCCESS(animatorPcaSkin->SetAnimatorData({*shapesMatrixSkinHost, shapesMeanSkinHost->Size()}));
  CHECK_SUCCESS(cudaStream->Synchronize());

  auto animatorSkin = ToUniquePtr(nva2f::CreateAnimatorSkin());
  CHECK_SUCCESS(animatorSkin->SetCudaStream(cudaStream->Data()));
  CHECK_SUCCESS(animatorSkin->Init(animatorSkinParams));
  CHECK_SUCCESS(animatorSkin->SetAnimatorData({*shapesMeanSkinHost, *lipOpenPoseDeltaHost, *eyeClosePoseDeltaHost}));
  CHECK_SUCCESS(cudaStream->Synchronize());

  auto animatorPcaTongue = ToUniquePtr(nva2f::CreateAnimatorPcaReconstruction());
  CHECK_SUCCESS(animatorPcaTongue->SetCudaStream(cudaStream->Data()));
  CHECK_SUCCESS(animatorPcaTongue->Init());
  CHECK_SUCCESS(animatorPcaTongue->SetAnimatorData({*shapesMatrixTongueHost, shapesMeanTongueHost->Size()}));
  CHECK_SUCCESS(cudaStream->Synchronize());

  auto animatorTongue = ToUniquePtr(nva2f::CreateAnimatorTongue());
  CHECK_SUCCESS(animatorTongue->SetCudaStream(cudaStream->Data()));
  CHECK_SUCCESS(animatorTongue->Init(animatorTongueParams));
  CHECK_SUCCESS(animatorTongue->SetAnimatorData({*shapesMeanTongueHost}));
  CHECK_SUCCESS(cudaStream->Synchronize());

  auto animatorTeeth = ToUniquePtr(nva2f::CreateAnimatorTeeth());
  CHECK_SUCCESS(animatorTeeth->Init(animatorTeethParams));
  CHECK_SUCCESS(animatorTeeth->SetAnimatorData({*neutralJawHost}));

  auto animatorEyes = ToUniquePtr(nva2f::CreateAnimatorEyes());
  CHECK_SUCCESS(animatorEyes->Init(animatorEyesParams));
  CHECK_SUCCESS(animatorEyes->SetAnimatorData({*saccadeRotHost}));

  CHECK_SUCCESS(emotionDatabase->GetEmotion(emotionShot.c_str(), emotionFrame,
                                         inferenceInputBuffers->GetImplicitEmotions()));
  CHECK_SUCCESS(nva2x::CopyHostToDevice(
      inferenceInputBuffers->GetExplicitEmotions(), {explicitEmotion.data(), explicitEmotion.size()}, cudaStream->Data()));
  CHECK_SUCCESS(cudaStream->Synchronize());

  auto transferBuffers = ToUniquePtr(
    nva2x::CreateHostPinnedTensorFloat(networkInfo.resultJawSize + networkInfo.resultEyesSize)
    );
  auto transferJaw = transferBuffers->View(0, networkInfo.resultJawSize);
  auto transferEyes = transferBuffers->View(networkInfo.resultJawSize, networkInfo.resultEyesSize);

  size_t numFrames = size_t(
      std::ceil((float(audioBuffer.size()) / float(trackSamplerate)) / dt));
  auto resultBuffers = ToUniquePtr(nva2f::CreateResultBuffersForRegressionModel(networkInfo, numFrames));
  CHECK_NOT_NULL(resultBuffers);
  float jawTransform[16];
  float rightEyeRotation[3];
  float leftEyeRotation[3];

  CHECK_SUCCESS(animatorPcaSkin->Reset());
  CHECK_SUCCESS(animatorSkin->Reset());
  CHECK_SUCCESS(animatorPcaTongue->Reset());
  CHECK_SUCCESS(animatorTongue->Reset());
  CHECK_SUCCESS(animatorTeeth->Reset());
  CHECK_SUCCESS(animatorEyes->Reset());

  for (unsigned int i = 0; i < numFrames; ++i) {
    using timestamp_t = nva2x::IAudioAccumulator::timestamp_t;
    const auto startIndex = static_cast<timestamp_t>((i * networkInfo.bufferSamplerate) / fps)
        - static_cast<timestamp_t>(networkInfo.bufferOffset);
    CHECK_SUCCESS(accumulator->Read(inferenceInputBuffers->GetInput(), startIndex, inputStrength, cudaStream->Data()));
    CHECK_SUCCESS(inference->Run(cudaStream->Data()));
    CHECK_SUCCESS(animatorPcaSkin->Animate(inferenceOutputBuffers->GetInferenceResultSkin(),
                                     resultBuffers->GetResultSkinGeometry(i)));
    CHECK_SUCCESS(animatorSkin->Animate(resultBuffers->GetResultSkinGeometry(i), dt,
                                     resultBuffers->GetResultSkinGeometry(i)));
    CHECK_SUCCESS(animatorPcaTongue->Animate(inferenceOutputBuffers->GetInferenceResultTongue(),
                                       resultBuffers->GetResultTongueGeometry(i)));
    CHECK_SUCCESS(animatorTongue->Animate(resultBuffers->GetResultTongueGeometry(i), dt,
                                       resultBuffers->GetResultTongueGeometry(i)));
    CHECK_SUCCESS(nva2x::CopyDeviceToHost(transferJaw,
                                       inferenceOutputBuffers->GetInferenceResultJaw()));
    CHECK_SUCCESS(nva2x::CopyDeviceToHost(transferEyes,
                                       inferenceOutputBuffers->GetInferenceResultEyes()));
    CHECK_SUCCESS(cudaStream->Synchronize());
    CHECK_SUCCESS(animatorTeeth->ComputeJawTransform(
        {jawTransform, 16}, transferJaw));
    CHECK_SUCCESS(animatorEyes->ComputeEyesRotation(
        {rightEyeRotation, 3},
        {leftEyeRotation, 3},
        transferEyes));
  }

  std::cout << "================================================" << std::endl;
  std::cout << "    Finished" << std::endl;
  std::cout << "================================================" << std::endl;

  return true;
}

int main(void) {
  if (!sample()) {
    return 1;
  }
  return 0;
}
