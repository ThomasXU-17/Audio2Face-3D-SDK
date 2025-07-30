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
#include "audio2emotion/internal/logger.h"
#include "audio2emotion/internal/macros.h"
#include "audio2emotion/internal/executor_classifier.h"
#include "audio2emotion/internal/executor_postprocess.h"
#include "audio2x/error.h"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>

namespace {

    nlohmann::json openJsonFile(const char* path) {
        auto u8Path = std::filesystem::u8path(path);
        std::ifstream f(u8Path);
        if (!(f && f.good() && f.is_open())) {
            std::system_error err{errno,std::generic_category()};
            throw err;
        }
        return nlohmann::json::parse(f);
    }

    template <typename Func>
    std::error_code checkError(const char* path, Func&& func) {
        A2E_CHECK_ERROR_WITH_MSG(path, "Path cannot be null", nva2x::ErrorCode::eNullPointer);
        try {
            return func(openJsonFile(path));
        }
        catch (std::exception& e)  {
            LOG_ERROR("Unable to parse file: " << path << " ; Message: " << e.what());
            return nva2x::ErrorCode::eReadFileFailed;
        }
        catch (...)  {
            LOG_ERROR("Unable to parse file: " << path);
            return nva2x::ErrorCode::eReadFileFailed;
        }
    }

}


namespace nva2e::IClassifierModel {

INetworkInfo::~INetworkInfo() = default;

NetworkInfo NetworkInfoOwner::GetNetworkInfo(std::size_t bufferLength) const {
    NetworkInfo networkInfo = _networkInfo;
    networkInfo.bufferLength = bufferLength;
    return networkInfo;
}

std::size_t NetworkInfoOwner::GetEmotionsCount() const {
    return _emotions.size();
}

const char* NetworkInfoOwner::GetEmotionName(std::size_t index) const {
    if (index >= _emotions.size()) {
        return nullptr;
    }
    return _emotions[index].c_str();
}

void NetworkInfoOwner::Destroy() {
    delete this;
}

std::error_code NetworkInfoOwner::Init(const char* path) {
    return checkError(path, [this](nlohmann::json data) {
        NetworkInfo networkInfo;
        // Buffer length is controlled by the user.
        networkInfo.bufferLength = 0;
        networkInfo.bufferSamplerate = data.at("audio_params").at("samplerate").get<std::size_t>();
        _networkInfo = std::move(networkInfo);

        auto emotions = data.at("emotions").get<std::vector<std::string>>();
        _emotions = std::move(emotions);

        return nva2x::ErrorCode::eSuccess;
    });
}


IConfigInfo::~IConfigInfo() = default;

const PostProcessData& ConfigInfoOwner::GetPostProcessData() const {
    return _postProcessData;
}

const PostProcessParams& ConfigInfoOwner::GetPostProcessParams() const {
    return _postProcessParams;
}

float ConfigInfoOwner::GetInputStrength() const {
    return _inputStrength;
}

void ConfigInfoOwner::Destroy() {
    delete this;
}

std::error_code ConfigInfoOwner::Init(const char* path, std::size_t emotionCount, const char* const emotionNames[]) {
    A2E_CHECK_ERROR_WITH_MSG(emotionCount > 0, "Emotion count must be greater than 0", nva2x::ErrorCode::eInvalidValue);
    A2E_CHECK_ERROR_WITH_MSG(emotionNames, "Emotion names cannot be null", nva2x::ErrorCode::eNullPointer);
    return checkError(path, [this, emotionCount, emotionNames](nlohmann::json data) {
        float readInputStrength;
        PostProcessData readData;
        PostProcessParams readParams;

        auto config = data.at("post_processing_config");

        // For now, input strength is not present in the config file.
        readInputStrength = 1.0f;

        readData.inferenceEmotionLength = emotionCount;
        readData.outputEmotionLength = config.at("output_emotion_length").get<std::size_t>();
        readParams.emotionContrast = config.at("emotion_contrast").get<float>();
        readParams.maxEmotions = config.at("max_emotions").get<std::size_t>();
        // Don't touch beginning emotion from the default one.
        readParams.liveBlendCoef = config.at("live_blend_coef").get<float>();
        readParams.enablePreferredEmotion = config.at("enable_preferred_emotion").get<bool>();
        readParams.preferredEmotionStrength = config.at("preferred_emotion_strength").get<float>();
        readParams.liveTransitionTime = config.at("transition_smoothing").get<float>();
        readParams.fixedDt = config.at("fixed_dt").get<float>();
        readParams.emotionStrength = config.at("emotion_strength").get<float>();

        auto preferredEmotion = config.at("preferred_emotion").get<std::vector<float>>();
        A2E_CHECK_ERROR_WITH_MSG(
            preferredEmotion.size() == readData.outputEmotionLength,
            "Preferred emotion must have " << readData.outputEmotionLength << " elements",
            nva2x::ErrorCode::eReadFileFailed
        );

        auto jsonEmotionCorrespondence = config.at("emotion_correspondence");
        std::vector<int> emotionCorrespondence(emotionCount);
        for (std::size_t i = 0; i < emotionCount; ++i) {
            A2E_CHECK_ERROR_WITH_MSG(emotionNames[i], "Emotion name cannot be null", nva2x::ErrorCode::eNullPointer);
            emotionCorrespondence[i] = jsonEmotionCorrespondence.at(emotionNames[i]).get<int>();
        }

        _inputStrength = readInputStrength;
        _preferredEmotion = std::move(preferredEmotion);
        readParams.preferredEmotion = nva2x::HostTensorFloatConstView(_preferredEmotion.data(), _preferredEmotion.size());

        _beginningEmotion = std::vector<float>(readData.outputEmotionLength, 0.0f);
        readParams.beginningEmotion = nva2x::HostTensorFloatConstView(_beginningEmotion.data(), _beginningEmotion.size());

        _emotionCorrespondence = std::move(emotionCorrespondence);
        readData.emotionCorrespondence = _emotionCorrespondence.data();
        readData.emotionCorrespondenceSize = _emotionCorrespondence.size();

        _postProcessData = std::move(readData);
        _postProcessParams = std::move(readParams);

        return nva2x::ErrorCode::eSuccess;
    });
}


IEmotionModelInfo::~IEmotionModelInfo() = default;

const INetworkInfo& ModelInfoOwner::GetNetworkInfo() const {
    return _networkInfo;
}

const IConfigInfo& ModelInfoOwner::GetConfigInfo() const {
    return _configInfo;
}

EmotionExecutorCreationParameters ModelInfoOwner::GetExecutorCreationParameters(
    std::size_t bufferLength,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    std::size_t inferencesToSkip
    ) const {
    EmotionExecutorCreationParameters params;

    params.networkInfo = _networkInfo.GetNetworkInfo(bufferLength);
    params.networkInfo.bufferLength = bufferLength;
    params.networkInfo.emotionLength = _networkInfo.GetEmotionsCount();
    params.networkData = _networkData.Data();
    params.networkDataSize = _networkData.Size();
    params.inputStrength = _configInfo.GetInputStrength();

    params.frameRateNumerator = frameRateNumerator;
    params.frameRateDenominator = frameRateDenominator;

    params.postProcessData = _configInfo.GetPostProcessData();
    params.postProcessParams = _configInfo.GetPostProcessParams();

    params.inferencesToSkip = inferencesToSkip;

    return params;
}

void ModelInfoOwner::Destroy() {
    delete this;
}

std::error_code ModelInfoOwner::Init(const char* path) {
    A2E_CHECK_ERROR_WITH_MSG(path, "Model path cannot be null", nva2x::ErrorCode::eNullPointer);
    return checkError(path, [&path, this](nlohmann::json data) -> std::error_code {
        auto convert_path = [&path](const std::string& jsonPath) -> std::string {
            const auto u8Path = std::filesystem::u8path(jsonPath);
            if (u8Path.is_relative()) {
                return (std::filesystem::u8path(path).parent_path() / u8Path).string();
            }
            else {
                return u8Path.string();
            }
        };

        const auto networkInfoPath = convert_path(data.at("networkInfoPath").get<std::string>());
        const auto networkPath = convert_path(data.at("networkPath").get<std::string>());
        const auto modelConfigPath = convert_path(data.at("modelConfigPath").get<std::string>());

        return Init(
            networkInfoPath.c_str(),
            networkPath.c_str(),
            modelConfigPath.c_str()
            );
    });
}

std::error_code ModelInfoOwner::Init(
    const char* networkInfoPath,
    const char* networkPath,
    const char* modelConfigPath) {

    A2E_CHECK_ERROR_WITH_MSG(networkInfoPath, "Network info path cannot be null", nva2x::ErrorCode::eNullPointer);
    NetworkInfoOwner networkInfo;
    A2E_CHECK_RESULT_WITH_MSG(networkInfo.Init(networkInfoPath), "Unable to read network info");

    A2E_CHECK_ERROR_WITH_MSG(networkPath, "Network path cannot be null", nva2x::ErrorCode::eNullPointer);
    nva2x::DataBytes networkDataBytes;
    A2E_CHECK_RESULT_WITH_MSG(networkDataBytes.ReadFromFile(networkPath), "Unable to read TRT file");

    A2E_CHECK_ERROR_WITH_MSG(modelConfigPath, "Model config path cannot be null", nva2x::ErrorCode::eNullPointer);
    ConfigInfoOwner configInfo;
    std::vector<const char*> emotionNames(networkInfo.GetEmotionsCount());
    for (std::size_t i = 0; i < networkInfo.GetEmotionsCount(); ++i) {
        emotionNames[i] = networkInfo.GetEmotionName(i);
    }
    A2E_CHECK_RESULT_WITH_MSG(
        configInfo.Init(modelConfigPath, emotionNames.size(), emotionNames.data()),
        "Unable to read post-processing params"
        );

    _networkInfo = std::move(networkInfo);
    _networkData = std::move(networkDataBytes);
    _configInfo = std::move(configInfo);

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2e::IClassifierModel



namespace nva2e::IPostProcessModel {

IEmotionModelInfo::~IEmotionModelInfo() = default;

const INetworkInfo& ModelInfoOwner::GetNetworkInfo() const {
    return _networkInfo;
}

const IConfigInfo& ModelInfoOwner::GetConfigInfo() const {
    return _configInfo;
}

EmotionExecutorCreationParameters ModelInfoOwner::GetExecutorCreationParameters(
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator
    ) const {
    EmotionExecutorCreationParameters params;

    params.samplingRate = _networkInfo.GetNetworkInfo(0).bufferSamplerate;
    params.inputStrength = _configInfo.GetInputStrength();

    params.frameRateNumerator = frameRateNumerator;
    params.frameRateDenominator = frameRateDenominator;

    params.postProcessData = _configInfo.GetPostProcessData();
    params.postProcessParams = _configInfo.GetPostProcessParams();

    return params;
}

void ModelInfoOwner::Destroy() {
    delete this;
}

std::error_code ModelInfoOwner::Init(const char* path) {
    A2E_CHECK_ERROR_WITH_MSG(path, "Model path cannot be null", nva2x::ErrorCode::eNullPointer);
    return checkError(path, [&path, this](nlohmann::json data) -> std::error_code {
        auto convert_path = [&path](const std::string& jsonPath) -> std::string {
            const auto u8Path = std::filesystem::u8path(jsonPath);
            if (u8Path.is_relative()) {
                return (std::filesystem::u8path(path).parent_path() / u8Path).string();
            }
            else {
                return u8Path.string();
            }
        };

        const auto networkInfoPath = convert_path(data.at("networkInfoPath").get<std::string>());
        const auto modelConfigPath = convert_path(data.at("modelConfigPath").get<std::string>());

        return Init(
            networkInfoPath.c_str(),
            modelConfigPath.c_str()
            );
    });
}

std::error_code ModelInfoOwner::Init(
    const char* networkInfoPath,
    const char* modelConfigPath) {

    A2E_CHECK_ERROR_WITH_MSG(networkInfoPath, "Network info path cannot be null", nva2x::ErrorCode::eNullPointer);
    NetworkInfoOwner networkInfo;
    A2E_CHECK_RESULT_WITH_MSG(networkInfo.Init(networkInfoPath), "Unable to read network info");

    A2E_CHECK_ERROR_WITH_MSG(modelConfigPath, "Model config path cannot be null", nva2x::ErrorCode::eNullPointer);
    ConfigInfoOwner configInfo;
    std::vector<const char*> emotionNames(networkInfo.GetEmotionsCount());
    for (std::size_t i = 0; i < networkInfo.GetEmotionsCount(); ++i) {
        emotionNames[i] = networkInfo.GetEmotionName(i);
    }
    A2E_CHECK_RESULT_WITH_MSG(
        configInfo.Init(modelConfigPath, emotionNames.size(), emotionNames.data()),
        "Unable to read post-processing params"
        );

    _networkInfo = std::move(networkInfo);
    _configInfo = std::move(configInfo);

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2e::IPostProcessModel


namespace nva2e {


IEmotionExecutorBundle::~IEmotionExecutorBundle() = default;

nva2x::ICudaStream& EmotionExecutorBundle::GetCudaStream() {
    return _cudaStream;
}

const nva2x::ICudaStream& EmotionExecutorBundle::GetCudaStream() const {
    return _cudaStream;
}

nva2x::IAudioAccumulator& EmotionExecutorBundle::GetAudioAccumulator(std::size_t trackIndex) {
    return *_audioAccumulators[trackIndex];
}

const nva2x::IAudioAccumulator& EmotionExecutorBundle::GetAudioAccumulator(std::size_t trackIndex) const {
    return *_audioAccumulators[trackIndex];
}

nva2x::IEmotionAccumulator& EmotionExecutorBundle::GetPreferredEmotionAccumulator(std::size_t trackIndex) {
    return *_preferredEmotionAccumulators[trackIndex];
}

const nva2x::IEmotionAccumulator& EmotionExecutorBundle::GetPreferredEmotionAccumulator(std::size_t trackIndex) const {
    return *_preferredEmotionAccumulators[trackIndex];
}

IEmotionExecutor& EmotionExecutorBundle::GetExecutor() {
    return *_emotionExecutor;
}

const IEmotionExecutor& EmotionExecutorBundle::GetExecutor() const {
    return *_emotionExecutor;
}

void EmotionExecutorBundle::Destroy() {
    delete this;
}

std::error_code EmotionExecutorBundle::InitClassifier(
    std::size_t nbTracks,
    const char* path,
    std::size_t bufferLength,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    std::size_t inferencesToSkip,
    IClassifierModel::IEmotionModelInfo** outModelInfo
    ) {
    // Read model info.
    IClassifierModel::ModelInfoOwner modelInfo;
    A2E_CHECK_RESULT_WITH_MSG(modelInfo.Init(path), "Unable to read classifier model info");

    A2E_CHECK_RESULT_WITH_MSG(_cudaStream.Init(), "Unable to create cuda stream");

    _audioAccumulators.resize(nbTracks);
    _preferredEmotionAccumulators.resize(nbTracks);

    const auto emotionsCount = modelInfo.GetConfigInfo().GetPostProcessData().outputEmotionLength;
    for (std::size_t i = 0; i < nbTracks; ++i) {
        // Hard-coded values of 1 second per buffer, no pre-allocation.
        _audioAccumulators[i] = std::make_unique<nva2x::AudioAccumulator>();
        A2E_CHECK_RESULT_WITH_MSG(_audioAccumulators[i]->Allocate(16000, 0), "Unable to create audio accumulator");

        // Hard-coded values of 300 emotions per buffer (30 FPS * 10 seconds), no pre-allocation.
        _preferredEmotionAccumulators[i] = std::make_unique<nva2x::EmotionAccumulator>();
        A2E_CHECK_RESULT_WITH_MSG(
            _preferredEmotionAccumulators[i]->Allocate(emotionsCount, 300, 0),
            "Unable to create emotion accumulator"
            );
    }

    // Create emotion executor.
    EmotionExecutorCreationParameters params;
    params.cudaStream = _cudaStream.Data();
    params.nbTracks = nbTracks;

    std::vector<const nva2x::IAudioAccumulator*> audioAccumulators(nbTracks);
    std::vector<const nva2x::IEmotionAccumulator*> preferredEmotionAccumulators(nbTracks);
    for (std::size_t i = 0; i < nbTracks; ++i) {
        audioAccumulators[i] = _audioAccumulators[i].get();
        preferredEmotionAccumulators[i] = _preferredEmotionAccumulators[i].get();
    }
    params.sharedAudioAccumulators = audioAccumulators.data();

    auto classifierParams = modelInfo.GetExecutorCreationParameters(
        bufferLength, frameRateNumerator, frameRateDenominator, inferencesToSkip
        );
    classifierParams.sharedPreferredEmotionAccumulators = preferredEmotionAccumulators.data();

    _emotionExecutor.reset(
        nva2e::CreateClassifierEmotionExecutor_INTERNAL(params, classifierParams)
        );
    A2E_CHECK_ERROR_WITH_MSG(_emotionExecutor, "Unable to create emotion executor", nva2x::ErrorCode::eInvalidValue);

    if (outModelInfo) {
        *outModelInfo = new IClassifierModel::ModelInfoOwner(std::move(modelInfo));
    }

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutorBundle::InitPostProcess(
    std::size_t nbTracks,
    const char* path,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IPostProcessModel::IEmotionModelInfo** outModelInfo
    ) {
    IPostProcessModel::ModelInfoOwner modelInfo;
    A2E_CHECK_RESULT_WITH_MSG(modelInfo.Init(path), "Unable to read classifier model info");

    A2E_CHECK_RESULT_WITH_MSG(_cudaStream.Init(), "Unable to create cuda stream");

    _audioAccumulators.resize(nbTracks);
    _preferredEmotionAccumulators.resize(nbTracks);

    const auto emotionsCount = modelInfo.GetConfigInfo().GetPostProcessData().outputEmotionLength;
    for (std::size_t i = 0; i < nbTracks; ++i) {
        // Hard-coded values of 1 second per buffer, no pre-allocation.
        _audioAccumulators[i] = std::make_unique<nva2x::AudioAccumulator>();
        A2E_CHECK_RESULT_WITH_MSG(_audioAccumulators[i]->Allocate(16000, 0), "Unable to create audio accumulator");

        // Hard-coded values of 300 emotions per buffer (30 FPS * 10 seconds), no pre-allocation.
        _preferredEmotionAccumulators[i] = std::make_unique<nva2x::EmotionAccumulator>();
        A2E_CHECK_RESULT_WITH_MSG(
            _preferredEmotionAccumulators[i]->Allocate(emotionsCount, 300, 0),
            "Unable to create emotion accumulator"
            );
    }

    // Create emotion executor.
    EmotionExecutorCreationParameters params;
    params.cudaStream = _cudaStream.Data();
    params.nbTracks = nbTracks;

    std::vector<const nva2x::IAudioAccumulator*> audioAccumulators(nbTracks);
    std::vector<const nva2x::IEmotionAccumulator*> preferredEmotionAccumulators(nbTracks);
    for (std::size_t i = 0; i < nbTracks; ++i) {
        audioAccumulators[i] = _audioAccumulators[i].get();
        preferredEmotionAccumulators[i] = _preferredEmotionAccumulators[i].get();
    }
    params.sharedAudioAccumulators = audioAccumulators.data();

    auto postProcessParams = modelInfo.GetExecutorCreationParameters(
        frameRateNumerator, frameRateDenominator
        );
    postProcessParams.sharedPreferredEmotionAccumulators = preferredEmotionAccumulators.data();

    _emotionExecutor.reset(
        nva2e::CreatePostProcessEmotionExecutor_INTERNAL(params, postProcessParams)
        );
    A2E_CHECK_ERROR_WITH_MSG(_emotionExecutor, "Unable to create emotion executor", nva2x::ErrorCode::eInvalidValue);

    if (outModelInfo) {
        *outModelInfo = new IPostProcessModel::ModelInfoOwner(std::move(modelInfo));
    }

    return nva2x::ErrorCode::eSuccess;
}


IClassifierModel::INetworkInfo* ReadClassifierNetworkInfo_INTERNAL(const char* path) {
    LOG_DEBUG("ReadClassifierNetworkInfo()");
    auto networkInfo = std::make_unique<IClassifierModel::NetworkInfoOwner>();
    if (networkInfo->Init(path)) {
        LOG_ERROR("Unable to read classifier model network info: " << path);
        return nullptr;
    }
    return networkInfo.release();
}

IClassifierModel::IConfigInfo* ReadClassifierConfigInfo_INTERNAL(
    const char* path, std::size_t emotionCount, const char* const emotionNames[]
    ) {
    LOG_DEBUG("ReadClassifierConfigInfo()");
    auto configInfo = std::make_unique<IClassifierModel::ConfigInfoOwner>();
    if (configInfo->Init(path, emotionCount, emotionNames)) {
        LOG_ERROR("Unable to read classifier model config info: " << path);
        return nullptr;
    }
    return configInfo.release();
}

IClassifierModel::IEmotionModelInfo* ReadClassifierModelInfo_INTERNAL(const char* path) {
    LOG_DEBUG("ReadClassifierModelInfo()");
    auto modelInfo = std::make_unique<IClassifierModel::ModelInfoOwner>();
    if (modelInfo->Init(path)) {
        LOG_ERROR("Unable to read classifier model info: " << path);
        return nullptr;
    }
    return modelInfo.release();
}

IClassifierModel::IEmotionModelInfo* ReadClassifierModelInfo_INTERNAL(
    const char* networkInfoPath,
    const char* networkPath,
    const char* modelConfigPath
    ) {
    LOG_DEBUG("ReadClassifierModelInfo()");
    auto modelInfo = std::make_unique<IClassifierModel::ModelInfoOwner>();
    if (modelInfo->Init(networkInfoPath, networkPath, modelConfigPath)) {
        LOG_ERROR("Unable to read classifier model info");
        return nullptr;
    }
    return modelInfo.release();
}

IPostProcessModel::INetworkInfo* ReadPostProcessNetworkInfo_INTERNAL(const char* path) {
    LOG_DEBUG("ReadPostProcessNetworkInfo()");
    auto networkInfo = std::make_unique<IPostProcessModel::NetworkInfoOwner>();
    if (networkInfo->Init(path)) {
        LOG_ERROR("Unable to read post-process model network info: " << path);
        return nullptr;
    }
    return networkInfo.release();
}

IPostProcessModel::IConfigInfo* ReadPostProcessConfigInfo_INTERNAL(
    const char* path, std::size_t emotionCount, const char* const emotionNames[]
    ) {
    LOG_DEBUG("ReadPostProcessConfigInfo()");
    auto configInfo = std::make_unique<IPostProcessModel::ConfigInfoOwner>();
    if (configInfo->Init(path, emotionCount, emotionNames)) {
        LOG_ERROR("Unable to read post-process model config info: " << path);
        return nullptr;
    }
    return configInfo.release();
}

IPostProcessModel::IEmotionModelInfo* ReadPostProcessModelInfo_INTERNAL(const char* path) {
    LOG_DEBUG("ReadPostProcessModelInfo()");
    auto modelInfo = std::make_unique<IPostProcessModel::ModelInfoOwner>();
    if (modelInfo->Init(path)) {
        LOG_ERROR("Unable to read post-process model info: " << path);
        return nullptr;
    }
    return modelInfo.release();
}

IPostProcessModel::IEmotionModelInfo* ReadPostProcessModelInfo_INTERNAL(
    const char* networkInfoPath,
    const char* modelConfigPath
    ) {
    LOG_DEBUG("ReadPostProcessModelInfo()");
    auto modelInfo = std::make_unique<IPostProcessModel::ModelInfoOwner>();
    if (modelInfo->Init(networkInfoPath, modelConfigPath)) {
        LOG_ERROR("Unable to read post-process model info");
        return nullptr;
    }
    return modelInfo.release();
}

IEmotionExecutorBundle* ReadClassifierEmotionExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    std::size_t bufferLength, std::size_t frameRateNumerator, std::size_t frameRateDenominator,
    std::size_t inferencesToSkip,
    IClassifierModel::IEmotionModelInfo** outModelInfo
    ) {
    LOG_DEBUG("ReadClassifierEmotionExecutorBundle()");
    auto bundle = std::make_unique<EmotionExecutorBundle>();
    if (bundle->InitClassifier(
        nbTracks,
        path,
        bufferLength, frameRateNumerator, frameRateDenominator,
        inferencesToSkip,
        outModelInfo
        )) {
        LOG_ERROR("Unable to create emotion executor bundle");
        return nullptr;
    }
    return bundle.release();
}

IEmotionExecutorBundle* ReadPostProcessEmotionExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    std::size_t frameRateNumerator, std::size_t frameRateDenominator,
    IPostProcessModel::IEmotionModelInfo** outModelInfo
    ) {
    LOG_DEBUG("ReadPostProcessEmotionExecutorBundle()");
    auto bundle = std::make_unique<EmotionExecutorBundle>();
    if (bundle->InitPostProcess(
        nbTracks,
        path,
        frameRateNumerator, frameRateDenominator,
        outModelInfo
        )) {
        LOG_ERROR("Unable to create emotion executor bundle");
        return nullptr;
    }
    return bundle.release();
}

} // namespace nva2e
