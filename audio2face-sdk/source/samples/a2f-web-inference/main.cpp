// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Audio2Face Web Inference Tool
// This program runs offline inference with command-line arguments for model selection
// and audio input, outputting results in JSON format.

#include "audio2face/audio2face.h"
#include "audio2emotion/audio2emotion.h"
#include "audio2x/cuda_utils.h"

#include <cuda_runtime.h>

#include <cstdint>
#include "AudioFile.h"
#include <cxxopts.hpp>
#include <nlohmann/json.hpp>

#include <any>
#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <filesystem>

using json = nlohmann::json;

//
// Boilerplate utilities
//

#define CHECK_RESULT(func)                                                     \
  {                                                                            \
    std::error_code error = (func);                                            \
    if (error) {                                                               \
      std::cerr << "Error (" << __LINE__ << "): Failed to execute: " << #func; \
      std::cerr << ", Reason: "<< error.message() << std::endl;                \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CHECK_ERROR(expression)                                                \
  {                                                                            \
    if (!(expression)) {                                                       \
      std::cerr << "Error (" << __LINE__ << "): " << #expression;              \
      std::cerr << " is NULL" << std::endl;                                    \
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


//
// Audio loading
//

std::vector<float> readAudio(const std::string& audioFilePath, int& sampleRate, double& duration) {
  AudioFile<float> audioFile;
  std::cerr << "Loading audio file: " << audioFilePath << std::endl;
  
  if (!audioFile.load(audioFilePath)) {
    std::cerr << "Error: Failed to load audio file: " << audioFilePath << std::endl;
    exit(1);
  }
  
  sampleRate = audioFile.getSampleRate();
  duration = audioFile.getLengthInSeconds();
  
  std::cerr << "Audio info: " << sampleRate << " Hz, " 
            << duration << " seconds, "
            << audioFile.getNumChannels() << " channels" << std::endl;
  
  if (sampleRate != 16000) {
    std::cerr << "Warning: Audio sample rate is " << sampleRate 
              << " Hz, expected 16000 Hz. Results may be affected." << std::endl;
  }
  
  return audioFile.samples[0];
}


//
// Model configuration
//

struct ModelConfig {
  std::string name;
  std::string type;  // "regression" or "diffusion"
  std::string modelPath;
  std::string emotionModelPath;
};

std::map<std::string, ModelConfig> getModelConfigs(const std::string& dataDir) {
  std::map<std::string, ModelConfig> configs;
  
  configs["mark"] = {
    "Mark (Regression v2.3)",
    "regression",
    dataDir + "/generated/audio2face-sdk/samples/data/mark/model.json",
    dataDir + "/generated/audio2emotion-sdk/samples/model/model.json"
  };
  
  configs["claire"] = {
    "Claire (Regression v2.3.1)",
    "regression",
    dataDir + "/generated/audio2face-sdk/samples/data/claire/model.json",
    dataDir + "/generated/audio2emotion-sdk/samples/model/model.json"
  };
  
  configs["james"] = {
    "James (Regression v2.3.1)",
    "regression",
    dataDir + "/generated/audio2face-sdk/samples/data/james/model.json",
    dataDir + "/generated/audio2emotion-sdk/samples/model/model.json"
  };
  
  configs["multi-diffusion"] = {
    "Multi-Diffusion (v3.0)",
    "diffusion",
    dataDir + "/generated/audio2face-sdk/samples/data/multi-diffusion/model.json",
    dataDir + "/generated/audio2emotion-sdk/samples/model/model.json"
  };
  
  return configs;
}


//
// Geometry executor data structure
//

struct GeometryExecutorData {
  nva2x::ICudaStream* cudaStream{nullptr};
  nva2x::IAudioAccumulator* audioAccumulator{nullptr};
  nva2x::IEmotionAccumulator* emotionAccumulator{nullptr};
  nva2f::IGeometryExecutor* executor{nullptr};
  std::any ownedData;
};


//
// Create geometry executors
//

GeometryExecutorData CreateRegressionGeometryExecutor(const std::string& modelPath, int fps = 60) {
  auto bundle = ToSharedPtr(
    nva2f::ReadRegressionGeometryExecutorBundle(
      1,
      modelPath.c_str(),
      nva2f::IGeometryExecutor::ExecutionOption::All,
      fps, 1,
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

GeometryExecutorData CreateDiffusionGeometryExecutor(const std::string& modelPath, int identityIndex = 0) {
  auto bundle = ToSharedPtr(
    nva2f::ReadDiffusionGeometryExecutorBundle(
      1,
      modelPath.c_str(),
      nva2f::IGeometryExecutor::ExecutionOption::All,
      identityIndex,
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


//
// Create emotion executor
//

UniquePtr<nva2e::IEmotionExecutor> CreateEmotionExecutor(
  const std::string& emotionModelPath,
  cudaStream_t cudaStream, 
  nva2x::IAudioAccumulator& audioAccumulator
) {
  auto modelInfo = ToUniquePtr(nva2e::ReadClassifierModelInfo(emotionModelPath.c_str()));
  CHECK_ERROR(modelInfo);

  nva2e::EmotionExecutorCreationParameters params;
  params.cudaStream = cudaStream;
  params.nbTracks = 1;
  const auto sharedAudioAccumulator = &audioAccumulator;
  params.sharedAudioAccumulators = &sharedAudioAccumulator;

  auto classifierParams = modelInfo->GetExecutorCreationParameters(60000, 30, 1, 30);
  auto executor = ToUniquePtr(nva2e::CreateClassifierEmotionExecutor(params, classifierParams));
  CHECK_ERROR(executor);

  return executor;
}


//
// Result storage
//

struct FrameResult {
  std::size_t frameIndex;
  double timestamp;
  std::vector<float> skinGeometry;
  std::vector<float> tongueGeometry;
  std::vector<float> jawTransform;
  std::vector<float> eyesRotation;
};

struct InferenceResults {
  std::string modelId;
  std::string modelType;
  std::string audioFile;
  int sampleRate;
  int fps;
  double duration;
  std::vector<FrameResult> frames;
};


//
// Run offline inference
//

InferenceResults RunOfflineInference(
  const std::string& modelId,
  const ModelConfig& config,
  const std::string& audioPath,
  int fps = 60
) {
  InferenceResults results;
  results.modelId = modelId;
  results.modelType = config.type;
  results.audioFile = audioPath;
  results.fps = fps;

  // Load audio
  int sampleRate;
  double duration;
  auto audioBuffer = readAudio(audioPath, sampleRate, duration);
  results.sampleRate = sampleRate;
  results.duration = duration;
  
  if (audioBuffer.empty()) {
    std::cerr << "Error: Audio buffer is empty" << std::endl;
    exit(1);
  }

  // Create geometry executor based on model type
  GeometryExecutorData geometryData;
  if (config.type == "regression") {
    geometryData = CreateRegressionGeometryExecutor(config.modelPath, fps);
  } else {
    geometryData = CreateDiffusionGeometryExecutor(config.modelPath);
  }

  // Create emotion executor
  auto emotionExecutor = CreateEmotionExecutor(
    config.emotionModelPath,
    geometryData.cudaStream->Data(),
    *geometryData.audioAccumulator
  );

  // Get geometry sizes for pre-allocation
  const auto skinSize = geometryData.executor->GetSkinGeometrySize();
  const auto tongueSize = geometryData.executor->GetTongueGeometrySize();
  const auto jawSize = geometryData.executor->GetJawTransformSize();
  const auto eyesSize = geometryData.executor->GetEyesRotationSize();

  std::cerr << "Geometry sizes - Skin: " << skinSize 
            << ", Tongue: " << tongueSize 
            << ", Jaw: " << jawSize 
            << ", Eyes: " << eyesSize << std::endl;

  // Set up geometry callback to collect results
  struct CallbackData {
    InferenceResults* results;
    std::size_t skinSize;
    std::size_t tongueSize;
    std::size_t jawSize;
    std::size_t eyesSize;
    cudaStream_t stream;
  };
  
  CallbackData callbackData{&results, skinSize, tongueSize, jawSize, eyesSize, geometryData.cudaStream->Data()};
  
  auto geometryCallback = [](void* userdata, const nva2f::IGeometryExecutor::Results& res) {
    auto& data = *static_cast<CallbackData*>(userdata);
    
    FrameResult frame;
    frame.frameIndex = data.results->frames.size();
    frame.timestamp = static_cast<double>(res.timeStampCurrentFrame) / 1000000.0;  // Convert to seconds
    
    // Copy skin geometry from GPU to CPU
    if (res.skinGeometry.Data() && data.skinSize > 0) {
      frame.skinGeometry.resize(data.skinSize);
      cudaMemcpyAsync(frame.skinGeometry.data(), res.skinGeometry.Data(), 
                      data.skinSize * sizeof(float), cudaMemcpyDeviceToHost, res.skinCudaStream);
      cudaStreamSynchronize(res.skinCudaStream);
    }
    
    // Copy tongue geometry
    if (res.tongueGeometry.Data() && data.tongueSize > 0) {
      frame.tongueGeometry.resize(data.tongueSize);
      cudaMemcpyAsync(frame.tongueGeometry.data(), res.tongueGeometry.Data(),
                      data.tongueSize * sizeof(float), cudaMemcpyDeviceToHost, res.tongueCudaStream);
      cudaStreamSynchronize(res.tongueCudaStream);
    }
    
    // Copy jaw transform
    if (res.jawTransform.Data() && data.jawSize > 0) {
      frame.jawTransform.resize(data.jawSize);
      cudaMemcpyAsync(frame.jawTransform.data(), res.jawTransform.Data(),
                      data.jawSize * sizeof(float), cudaMemcpyDeviceToHost, res.jawCudaStream);
      cudaStreamSynchronize(res.jawCudaStream);
    }
    
    // Copy eyes rotation
    if (res.eyesRotation.Data() && data.eyesSize > 0) {
      frame.eyesRotation.resize(data.eyesSize);
      cudaMemcpyAsync(frame.eyesRotation.data(), res.eyesRotation.Data(),
                      data.eyesSize * sizeof(float), cudaMemcpyDeviceToHost, res.eyesCudaStream);
      cudaStreamSynchronize(res.eyesCudaStream);
    }
    
    data.results->frames.push_back(std::move(frame));
    return true;
  };
  
  CHECK_RESULT(geometryData.executor->SetResultsCallback(geometryCallback, &callbackData));

  // Connect emotion executor to emotion accumulator
  auto emotionCallback = [](void* userdata, const nva2e::IEmotionExecutor::Results& res) {
    auto& emotionAccumulator = *static_cast<nva2x::IEmotionAccumulator*>(userdata);
    CHECK_RESULT(emotionAccumulator.Accumulate(res.timeStampCurrentFrame, res.emotions, res.cudaStream));
    return true;
  };
  CHECK_RESULT(emotionExecutor->SetResultsCallback(emotionCallback, geometryData.emotionAccumulator));

  // Accumulate all audio
  std::cerr << "Accumulating audio data..." << std::endl;
  CHECK_RESULT(
    geometryData.audioAccumulator->Accumulate(
      nva2x::HostTensorFloatConstView{audioBuffer.data(), audioBuffer.size()},
      geometryData.cudaStream->Data()
    )
  );
  CHECK_RESULT(geometryData.audioAccumulator->Close());

  // Process all emotion
  std::cerr << "Processing emotions..." << std::endl;
  while (nva2x::GetNbReadyTracks(*emotionExecutor) > 0) {
    CHECK_RESULT(emotionExecutor->Execute(nullptr));
  }
  CHECK_RESULT(geometryData.emotionAccumulator->Close());

  // Process all geometry
  std::cerr << "Processing geometry..." << std::endl;
  while (nva2x::GetNbReadyTracks(*geometryData.executor)) {
    CHECK_RESULT(geometryData.executor->Execute(nullptr));
  }

  std::cerr << "Processed " << results.frames.size() << " frames." << std::endl;
  return results;
}


//
// JSON output
//

json ResultsToJson(const InferenceResults& results) {
  json j;
  
  j["model_id"] = results.modelId;
  j["model_type"] = results.modelType;
  j["audio_file"] = results.audioFile;
  j["sample_rate"] = results.sampleRate;
  j["fps"] = results.fps;
  j["duration_seconds"] = results.duration;
  j["total_frames"] = results.frames.size();
  
  j["metadata"] = {
    {"skin_geometry_size", results.frames.empty() ? 0 : results.frames[0].skinGeometry.size()},
    {"tongue_geometry_size", results.frames.empty() ? 0 : results.frames[0].tongueGeometry.size()},
    {"jaw_transform_size", results.frames.empty() ? 0 : results.frames[0].jawTransform.size()},
    {"eyes_rotation_size", results.frames.empty() ? 0 : results.frames[0].eyesRotation.size()}
  };
  
  j["frames"] = json::array();
  for (const auto& frame : results.frames) {
    json frameJson;
    frameJson["frame_index"] = frame.frameIndex;
    frameJson["timestamp"] = frame.timestamp;
    
    if (!frame.skinGeometry.empty()) {
      frameJson["skin_geometry"] = frame.skinGeometry;
    }
    if (!frame.tongueGeometry.empty()) {
      frameJson["tongue_geometry"] = frame.tongueGeometry;
    }
    if (!frame.jawTransform.empty()) {
      frameJson["jaw_transform"] = frame.jawTransform;
    }
    if (!frame.eyesRotation.empty()) {
      frameJson["eyes_rotation"] = frame.eyesRotation;
    }
    
    j["frames"].push_back(frameJson);
  }
  
  return j;
}


//
// Main
//

int main(int argc, char* argv[]) {
  // Default data directory
  std::string defaultDataDir = TEST_DATA_DIR "_data";
  
  // Parse command line arguments
  cxxopts::Options options("a2f-web-inference", "Audio2Face Web Inference Tool");
  
  options.add_options()
    ("m,model", "Model ID (mark, claire, james, multi-diffusion)", cxxopts::value<std::string>()->default_value("mark"))
    ("a,audio", "Input audio file path (16kHz WAV recommended)", cxxopts::value<std::string>())
    ("o,output", "Output JSON file path", cxxopts::value<std::string>()->default_value("-"))
    ("d,data-dir", "Data directory path", cxxopts::value<std::string>()->default_value(defaultDataDir))
    ("f,fps", "Output frame rate", cxxopts::value<int>()->default_value("60"))
    ("i,identity", "Identity index for diffusion models", cxxopts::value<int>()->default_value("0"))
    ("l,list", "List available models")
    ("h,help", "Print usage");
  
  try {
    auto result = options.parse(argc, argv);
    
    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
    }
    
    std::string dataDir = result["data-dir"].as<std::string>();
    auto modelConfigs = getModelConfigs(dataDir);
    
    if (result.count("list")) {
      std::cout << "Available models:" << std::endl;
      for (const auto& [id, config] : modelConfigs) {
        std::cout << "  " << id << " - " << config.name << " (" << config.type << ")" << std::endl;
        std::cout << "    Path: " << config.modelPath << std::endl;
        bool exists = std::filesystem::exists(config.modelPath);
        std::cout << "    Status: " << (exists ? "Available" : "Not found") << std::endl;
      }
      return 0;
    }
    
    if (!result.count("audio")) {
      std::cerr << "Error: Audio file path is required. Use -a or --audio option." << std::endl;
      std::cerr << options.help() << std::endl;
      return 1;
    }
    
    std::string modelId = result["model"].as<std::string>();
    std::string audioPath = result["audio"].as<std::string>();
    std::string outputPath = result["output"].as<std::string>();
    int fps = result["fps"].as<int>();
    
    // Check if model exists
    if (modelConfigs.find(modelId) == modelConfigs.end()) {
      std::cerr << "Error: Unknown model ID: " << modelId << std::endl;
      std::cerr << "Available models: ";
      for (const auto& [id, _] : modelConfigs) {
        std::cerr << id << " ";
      }
      std::cerr << std::endl;
      return 1;
    }
    
    const auto& modelConfig = modelConfigs[modelId];
    
    // Check if model file exists
    if (!std::filesystem::exists(modelConfig.modelPath)) {
      std::cerr << "Error: Model file not found: " << modelConfig.modelPath << std::endl;
      std::cerr << "Please run gen_testdata.sh first to generate the model files." << std::endl;
      return 1;
    }
    
    // Check if audio file exists
    if (!std::filesystem::exists(audioPath)) {
      std::cerr << "Error: Audio file not found: " << audioPath << std::endl;
      return 1;
    }
    
    // Initialize CUDA
    std::cerr << "Initializing CUDA..." << std::endl;
    CHECK_RESULT(nva2x::SetCudaDeviceIfNeeded(0));
    
    // Run inference
    std::cerr << "Starting inference with model: " << modelConfig.name << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    auto results = RunOfflineInference(modelId, modelConfig, audioPath, fps);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cerr << "Inference completed in " << duration.count() << " ms" << std::endl;
    
    // Convert to JSON
    json outputJson = ResultsToJson(results);
    outputJson["inference_time_ms"] = duration.count();
    
    // Output JSON
    if (outputPath == "-") {
      std::cout << outputJson.dump(2) << std::endl;
    } else {
      std::ofstream outFile(outputPath);
      if (!outFile.is_open()) {
        std::cerr << "Error: Failed to open output file: " << outputPath << std::endl;
        return 1;
      }
      outFile << outputJson.dump(2) << std::endl;
      std::cerr << "Results written to: " << outputPath << std::endl;
    }
    
    return 0;
    
  } catch (const cxxopts::exceptions::exception& e) {
    std::cerr << "Error parsing options: " << e.what() << std::endl;
    return 1;
  }
}
