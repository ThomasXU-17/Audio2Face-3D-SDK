// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Audio2Face Web Inference Tool
// This program runs offline inference with command-line arguments for model selection
// and audio input, outputting results in JSON format compatible with a2f_export format.

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
// Blendshape executor data structure
//

struct BlendshapeExecutorData {
  nva2x::ICudaStream* cudaStream{nullptr};
  nva2x::IAudioAccumulator* audioAccumulator{nullptr};
  nva2x::IEmotionAccumulator* emotionAccumulator{nullptr};
  nva2f::IBlendshapeExecutor* executor{nullptr};
  nva2f::IBlendshapeSolver* skinSolver{nullptr};
  nva2f::IBlendshapeSolver* tongueSolver{nullptr};
  std::any ownedData;
};


//
// Create blendshape executors
//

BlendshapeExecutorData CreateRegressionBlendshapeExecutor(const std::string& modelPath, int fps = 60) {
  auto bundle = ToSharedPtr(
    nva2f::ReadRegressionBlendshapeSolveExecutorBundle(
      1,
      modelPath.c_str(),
      nva2f::IGeometryExecutor::ExecutionOption::All,
      false,  // useGpuSolver = false for host results
      fps, 1,
      nullptr,
      nullptr
    )
  );
  CHECK_ERROR(bundle);

  BlendshapeExecutorData data;
  data.cudaStream = &bundle->GetCudaStream();
  data.audioAccumulator = &bundle->GetAudioAccumulator(0);
  data.emotionAccumulator = &bundle->GetEmotionAccumulator(0);
  data.executor = &bundle->GetExecutor();
  
  // Get skin and tongue solvers for pose names
  CHECK_RESULT(nva2f::GetExecutorSkinSolver(*data.executor, 0, &data.skinSolver));
  CHECK_RESULT(nva2f::GetExecutorTongueSolver(*data.executor, 0, &data.tongueSolver));
  
  data.ownedData = std::move(bundle);

  return data;
}

BlendshapeExecutorData CreateDiffusionBlendshapeExecutor(const std::string& modelPath, int identityIndex = 0) {
  auto bundle = ToSharedPtr(
    nva2f::ReadDiffusionBlendshapeSolveExecutorBundle(
      1,
      modelPath.c_str(),
      nva2f::IGeometryExecutor::ExecutionOption::All,
      false,  // useGpuSolver = false for host results
      identityIndex,
      false,
      nullptr,
      nullptr
    )
  );
  CHECK_ERROR(bundle);

  BlendshapeExecutorData data;
  data.cudaStream = &bundle->GetCudaStream();
  data.audioAccumulator = &bundle->GetAudioAccumulator(0);
  data.emotionAccumulator = &bundle->GetEmotionAccumulator(0);
  data.executor = &bundle->GetExecutor();
  
  // Get skin and tongue solvers for pose names
  CHECK_RESULT(nva2f::GetExecutorSkinSolver(*data.executor, 0, &data.skinSolver));
  CHECK_RESULT(nva2f::GetExecutorTongueSolver(*data.executor, 0, &data.tongueSolver));
  
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

struct InferenceResults {
  std::string audioFile;
  int fps;
  double duration;
  std::vector<std::string> facsNames;
  std::vector<std::vector<float>> weightMat;  // [numFrames][numPoses]
  std::vector<std::string> joints;
  std::vector<std::vector<float>> rotations;  // Currently empty
  std::vector<std::vector<float>> translations;  // Currently empty
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
  results.audioFile = audioPath;
  results.fps = fps;

  // Load audio
  int sampleRate;
  double duration;
  auto audioBuffer = readAudio(audioPath, sampleRate, duration);
  results.duration = duration;
  
  if (audioBuffer.empty()) {
    std::cerr << "Error: Audio buffer is empty" << std::endl;
    exit(1);
  }

  // Create blendshape executor based on model type
  BlendshapeExecutorData blendshapeData;
  if (config.type == "regression") {
    blendshapeData = CreateRegressionBlendshapeExecutor(config.modelPath, fps);
  } else {
    blendshapeData = CreateDiffusionBlendshapeExecutor(config.modelPath);
  }

  // Create emotion executor
  auto emotionExecutor = CreateEmotionExecutor(
    config.emotionModelPath,
    blendshapeData.cudaStream->Data(),
    *blendshapeData.audioAccumulator
  );

  // Get pose names from skin solver
  CHECK_ERROR(blendshapeData.skinSolver);
  const int numSkinPoses = blendshapeData.skinSolver->NumBlendshapePoses();
  std::cerr << "Skin blendshapes: " << numSkinPoses << std::endl;
  
  for (int i = 0; i < numSkinPoses; ++i) {
    const char* poseName = blendshapeData.skinSolver->GetPoseName(i);
    if (poseName) {
      results.facsNames.push_back(poseName);
    }
  }
  
  // Add tongue pose if available
  if (blendshapeData.tongueSolver) {
    const int numTonguePoses = blendshapeData.tongueSolver->NumBlendshapePoses();
    std::cerr << "Tongue blendshapes: " << numTonguePoses << std::endl;
    for (int i = 0; i < numTonguePoses; ++i) {
      const char* poseName = blendshapeData.tongueSolver->GetPoseName(i);
      if (poseName) {
        results.facsNames.push_back(poseName);
      }
    }
  }

  // Set up default joints
  results.joints = {"jaw", "eye_L", "eye_R"};

  // Set up callback to collect results
  struct CallbackData {
    InferenceResults* results;
    std::size_t weightCount;
  };
  
  CallbackData callbackData{&results, blendshapeData.executor->GetWeightCount()};
  
  auto hostCallback = [](
    void* userdata, 
    const nva2f::IBlendshapeExecutor::HostResults& res,
    std::error_code errorCode
  ) {
    if (errorCode) {
      std::cerr << "Error in callback: " << errorCode.message() << std::endl;
      return;
    }
    
    auto& data = *static_cast<CallbackData*>(userdata);
    
    // Copy weights to result
    std::vector<float> frameWeights(res.weights.Size());
    for (std::size_t i = 0; i < res.weights.Size(); ++i) {
      frameWeights[i] = res.weights.Data()[i];
    }
    
    data.results->weightMat.push_back(std::move(frameWeights));
  };
  
  CHECK_RESULT(blendshapeData.executor->SetResultsCallback(hostCallback, &callbackData));

  // Connect emotion executor to emotion accumulator
  auto emotionCallback = [](void* userdata, const nva2e::IEmotionExecutor::Results& res) {
    auto& emotionAccumulator = *static_cast<nva2x::IEmotionAccumulator*>(userdata);
    CHECK_RESULT(emotionAccumulator.Accumulate(res.timeStampCurrentFrame, res.emotions, res.cudaStream));
    return true;
  };
  CHECK_RESULT(emotionExecutor->SetResultsCallback(emotionCallback, blendshapeData.emotionAccumulator));

  // Accumulate all audio
  std::cerr << "Accumulating audio data..." << std::endl;
  CHECK_RESULT(
    blendshapeData.audioAccumulator->Accumulate(
      nva2x::HostTensorFloatConstView{audioBuffer.data(), audioBuffer.size()},
      blendshapeData.cudaStream->Data()
    )
  );
  CHECK_RESULT(blendshapeData.audioAccumulator->Close());

  // Process all emotion
  std::cerr << "Processing emotions..." << std::endl;
  while (nva2x::GetNbReadyTracks(*emotionExecutor) > 0) {
    CHECK_RESULT(emotionExecutor->Execute(nullptr));
  }
  CHECK_RESULT(blendshapeData.emotionAccumulator->Close());

  // Process all blendshapes
  std::cerr << "Processing blendshapes..." << std::endl;
  while (nva2x::GetNbReadyTracks(*blendshapeData.executor)) {
    CHECK_RESULT(blendshapeData.executor->Execute(nullptr));
  }
  
  // Wait for async operations to complete
  CHECK_RESULT(blendshapeData.executor->Wait(0));

  std::cerr << "Processed " << results.weightMat.size() << " frames." << std::endl;
  return results;
}


//
// JSON output in a2f_export format
//

json ResultsToJson(const InferenceResults& results) {
  json j;
  
  j["exportFps"] = static_cast<double>(results.fps);
  j["trackPath"] = results.audioFile;
  j["numPoses"] = results.facsNames.size();
  j["numFrames"] = results.weightMat.size();
  j["facsNames"] = results.facsNames;
  j["weightMat"] = results.weightMat;
  j["joints"] = results.joints;
  j["rotations"] = results.rotations;
  j["translations"] = results.translations;
  
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
    auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cerr << "Inference completed in " << durationMs.count() << " ms" << std::endl;
    
    // Convert to JSON
    json outputJson = ResultsToJson(results);
    
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
