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
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2face/internal/executor_regression.h"
#include "audio2face/internal/executor_diffusion.h"
#include "audio2face/internal/executor_blendshapesolve.h"
#include "audio2x/internal/tensor_dict.h"
#include "audio2x/internal/npz_utils.h"
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
        A2F_CHECK_ERROR_WITH_MSG(path, "Path cannot be null", nva2x::ErrorCode::eNullPointer);
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


// For IGeometryExecutor::ExecutionOption operators.
using namespace ::nva2f::internal;

namespace nva2f::IRegressionModel {

INetworkInfo::~INetworkInfo() = default;

const NetworkInfo& NetworkInfoOwner::GetNetworkInfo() const {
    return _networkInfo;
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

nva2x::HostTensorFloatConstView NetworkInfoOwner::GetDefaultEmotion() const {
    return nva2x::ToConstView(_defaultEmotion);
}

const char* NetworkInfoOwner::GetIdentityName() const {
    return _identity.c_str();
}

void NetworkInfoOwner::Destroy() {
    delete this;
}

std::error_code NetworkInfoOwner::Init(const char* path) {
    return checkError(path, [this](nlohmann::json data) {
        A2F_CHECK_ERROR_WITH_MSG(
            data.at("id").at("type") == "regression" && data.at("id").at("output") == "geometry",
            "Invalid network type",
            nva2x::ErrorCode::eReadFileFailed
        );

        _identity = data.at("id").at("actor").get<std::string>();

        auto params = data.at("params");
        auto emotions = params.at("explicit_emotions").get<std::vector<std::string>>();
        auto defaultEmotion = params.at("default_emotion").get<std::vector<float>>();
        A2F_CHECK_ERROR_WITH_MSG(
            emotions.size() == defaultEmotion.size(),
            "Emotion size must match default emotion size",
            nva2x::ErrorCode::eReadFileFailed
        );

        NetworkInfo networkInfo;
        networkInfo.implicitEmotionLength = params.at("implicit_emotion_len").get<std::size_t>();
        networkInfo.explicitEmotionLength = emotions.size();
        networkInfo.numShapesSkin = params.at("num_shapes_skin").get<std::size_t>();
        networkInfo.numShapesTongue = params.at("num_shapes_tongue").get<std::size_t>();
        networkInfo.resultSkinSize = params.at("num_verts_skin").get<std::size_t>() * 3;
        networkInfo.resultTongueSize = params.at("num_verts_tongue").get<std::size_t>() * 3;
        networkInfo.resultJawSize = params.at("result_jaw_size").get<std::size_t>();
        networkInfo.resultEyesSize = params.at("result_eyes_size").get<std::size_t>();

        auto audio_params = data.at("audio_params");
        networkInfo.bufferLength = audio_params.at("buffer_len").get<std::size_t>();
        networkInfo.bufferOffset = audio_params.at("buffer_ofs").get<std::size_t>();
        networkInfo.bufferSamplerate = audio_params.at("samplerate").get<std::size_t>();

        _emotions = std::move(emotions);
        _defaultEmotion = std::move(defaultEmotion);
        _networkInfo = std::move(networkInfo);

        return nva2x::ErrorCode::eSuccess;
    });
}


IAnimatorData::~IAnimatorData() = default;

AnimatorDataView AnimatorDataOwner::GetAnimatorData() const {
    AnimatorDataView data;
    data.skin.neutralPose = _shapesMeanSkin;
    data.skin.lipOpenPoseDelta = _lipOpenPoseDelta;
    data.skin.eyeClosePoseDelta = _eyeClosePoseDelta;
    data.tongue.neutralPose = _shapesMeanTongue;
    data.teeth.neutralJaw = _neutralJaw;
    data.eyes.saccadeRot = _saccadeRot;
    return data;
}

IAnimatorPcaReconstruction::HostData AnimatorDataOwner::GetSkinPcaReconstructionData() const {
    return {_shapesMatrixSkin, _shapesMeanSkin.Size()};
}

IAnimatorPcaReconstruction::HostData AnimatorDataOwner::GetTonguePcaReconstructionData() const {
    return {_shapesMatrixTongue, _shapesMeanTongue.Size()};
}

void AnimatorDataOwner::Destroy() {
    delete this;
}

std::error_code AnimatorDataOwner::Init(const char* path) {
    nva2x::HostTensorDict data;
    A2F_CHECK_RESULT(data.ReadFromFile(path));

    const auto shapesMatrixSkinHost = data.At("shapes_matrix_skin");
    A2F_CHECK_ERROR_WITH_MSG(shapesMatrixSkinHost, "Unable to get shapes matrix for skin", nva2x::ErrorCode::eReadFileFailed);
    const auto shapesMeanSkinHost = data.At("shapes_mean_skin");
    A2F_CHECK_ERROR_WITH_MSG(shapesMeanSkinHost, "Unable to get shapes mean for skin", nva2x::ErrorCode::eReadFileFailed);
    const auto lipOpenPoseDeltaHost = data.At("lip_open_pose_delta");
    A2F_CHECK_ERROR_WITH_MSG(lipOpenPoseDeltaHost, "Unable to get lip open pose delta for skin", nva2x::ErrorCode::eReadFileFailed);
    const auto eyeClosePoseDeltaHost = data.At("eye_close_pose_delta");
    A2F_CHECK_ERROR_WITH_MSG(eyeClosePoseDeltaHost, "Unable to get eye close pose delta for skin", nva2x::ErrorCode::eReadFileFailed);

    const auto shapesMatrixTongueHost = data.At("shapes_matrix_tongue");
    A2F_CHECK_ERROR_WITH_MSG(shapesMatrixTongueHost, "Unable to get shapes matrix for tongue", nva2x::ErrorCode::eReadFileFailed);
    const auto shapesMeanTongueHost = data.At("shapes_mean_tongue");
    A2F_CHECK_ERROR_WITH_MSG(shapesMeanTongueHost, "Unable to get shapes mean for tongue", nva2x::ErrorCode::eReadFileFailed);

    const auto neutralJawHost = data.At("neutral_jaw");
    A2F_CHECK_ERROR_WITH_MSG(neutralJawHost, "Unable to get neutral jaw for teeth", nva2x::ErrorCode::eReadFileFailed);

    const auto saccadeRotHost = data.At("saccade_rot_matrix");
    A2F_CHECK_ERROR_WITH_MSG(saccadeRotHost, "Unable to get saccade rotations for eyes", nva2x::ErrorCode::eReadFileFailed);

    A2F_CHECK_RESULT(_shapesMatrixSkin.Init(*shapesMatrixSkinHost));
    A2F_CHECK_RESULT(_shapesMeanSkin.Init(*shapesMeanSkinHost));
    A2F_CHECK_RESULT(_lipOpenPoseDelta.Init(*lipOpenPoseDeltaHost));
    A2F_CHECK_RESULT(_eyeClosePoseDelta.Init(*eyeClosePoseDeltaHost));
    A2F_CHECK_RESULT(_shapesMatrixTongue.Init(*shapesMatrixTongueHost));
    A2F_CHECK_RESULT(_shapesMeanTongue.Init(*shapesMeanTongueHost));
    A2F_CHECK_RESULT(_neutralJaw.Init(*neutralJawHost));
    A2F_CHECK_RESULT(_saccadeRot.Init(*saccadeRotHost));

    return nva2x::ErrorCode::eSuccess;
}


IGeometryModelInfo::~IGeometryModelInfo() = default;

const INetworkInfo& ModelInfoOwner::GetNetworkInfo() const {
    return _networkInfo;
}

const IAnimatorData& ModelInfoOwner::GetAnimatorData() const {
    return _animatorData;
}

const AnimatorParams& ModelInfoOwner::GetAnimatorParams() const {
    return _animatorParams;
}

GeometryExecutorCreationParameters ModelInfoOwner::GetExecutorCreationParameters(
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator
    ) const {
    GeometryExecutorCreationParameters params;

    params.networkInfo = _networkInfo.GetNetworkInfo();
    params.networkData = _networkData.Data();
    params.networkDataSize = _networkData.Size();
    params.inputStrength = _inputStrength;

    params.emotionDatabase = &_emotionDatabase;
    params.sourceShot = _sourceShot.c_str();
    params.sourceFrame = _sourceFrame;

    params.frameRateNumerator = frameRateNumerator;
    params.frameRateDenominator = frameRateDenominator;

    if (IsAnySet(executionOption, IGeometryExecutor::ExecutionOption::Skin)) {
        params.initializationSkinParams = &_skinParams;
    }
    if (IsAnySet(executionOption, IGeometryExecutor::ExecutionOption::Tongue)) {
        params.initializationTongueParams = &_tongueParams;
    }
    if (IsAnySet(executionOption, IGeometryExecutor::ExecutionOption::Jaw)) {
        params.initializationTeethParams = &_teethParams;
    }
    if (IsAnySet(executionOption, IGeometryExecutor::ExecutionOption::Eyes)) {
        params.initializationEyesParams = &_eyesParams;
    }

    return params;
}

void ModelInfoOwner::Destroy() {
    delete this;
}

std::error_code ModelInfoOwner::Init(const char* path) {
    A2F_CHECK_ERROR_WITH_MSG(path, "Model path cannot be null", nva2x::ErrorCode::eNullPointer);
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
        const auto emotionDatabasePath = convert_path(data.at("emotionDatabasePath").get<std::string>());
        const auto modelConfigPath = convert_path(data.at("modelConfigPath").get<std::string>());
        const auto modelDataPath = convert_path(data.at("modelDataPath").get<std::string>());

        return Init(
            networkInfoPath.c_str(),
            networkPath.c_str(),
            emotionDatabasePath.c_str(),
            modelConfigPath.c_str(),
            modelDataPath.c_str()
            );
    });
}

std::error_code ModelInfoOwner::Init(
    const char* networkInfoPath,
    const char* networkPath,
    const char* emotionDatabasePath,
    const char* modelConfigPath,
    const char* modelDataPath) {

    A2F_CHECK_ERROR_WITH_MSG(networkInfoPath, "Network info path cannot be null", nva2x::ErrorCode::eNullPointer);
    NetworkInfoOwner networkInfo;
    A2F_CHECK_RESULT_WITH_MSG(networkInfo.Init(networkInfoPath), "Unable to read network info");

    A2F_CHECK_ERROR_WITH_MSG(networkPath, "Network path cannot be null", nva2x::ErrorCode::eNullPointer);
    nva2x::DataBytes networkDataBytes;
    A2F_CHECK_RESULT_WITH_MSG(networkDataBytes.ReadFromFile(networkPath), "Unable to read TRT file");

    A2F_CHECK_ERROR_WITH_MSG(emotionDatabasePath, "Emotion database path cannot be null", nva2x::ErrorCode::eNullPointer);
    EmotionDatabase emotionDatabase;
    A2F_CHECK_RESULT_WITH_MSG(emotionDatabase.InitFromFile(emotionDatabasePath), "Unable to read emotion database");

    A2F_CHECK_ERROR_WITH_MSG(modelConfigPath, "Model config path cannot be null", nva2x::ErrorCode::eNullPointer);
    float inputStrength;
    AnimatorParams animatorParams;
    A2F_CHECK_RESULT_WITH_MSG(
        ReadAnimatorParams_INTERNAL(
            modelConfigPath, inputStrength, animatorParams
            ),
        "Unable to read animator params"
        );

    A2F_CHECK_ERROR_WITH_MSG(modelDataPath, "Model data path cannot be null", nva2x::ErrorCode::eNullPointer);
    AnimatorDataOwner animatorData;
    A2F_CHECK_RESULT_WITH_MSG(animatorData.Init(modelDataPath), "Unable to read animator data");

    // OPTME: We re-open the json file just to get the source shot and source frame.
    // Maybe they should be stored somewhere else...?
    std::string sourceShot;
    std::size_t sourceFrame;
    auto success = checkError(modelConfigPath, [&sourceShot, &sourceFrame](nlohmann::json data) {
        auto config = data.at("config");
        sourceShot = config.at("source_shot").get<std::string>();
        sourceFrame = config.at("source_frame").get<std::size_t>();

        return nva2x::ErrorCode::eSuccess;
    });
    A2F_CHECK_RESULT_WITH_MSG(success, "Unable to read source shot and frame from model config");

    _networkInfo = std::move(networkInfo);
    _networkData = std::move(networkDataBytes);
    _emotionDatabase = std::move(emotionDatabase);
    _inputStrength = inputStrength;
    _animatorParams = std::move(animatorParams);
    _animatorData = std::move(animatorData);
    _sourceShot = std::move(sourceShot);
    _sourceFrame = sourceFrame;

    _skinParams.params = _animatorParams.skin;
    _skinParams.pcaData = _animatorData.GetSkinPcaReconstructionData();
    _skinParams.data = _animatorData.GetAnimatorData().skin;

    _tongueParams.params = _animatorParams.tongue;
    _tongueParams.pcaData = _animatorData.GetTonguePcaReconstructionData();
    _tongueParams.data = _animatorData.GetAnimatorData().tongue;

    _teethParams.params = _animatorParams.teeth;
    _teethParams.data = _animatorData.GetAnimatorData().teeth;

    _eyesParams.params = _animatorParams.eyes;
    _eyesParams.data = _animatorData.GetAnimatorData().eyes;

    return nva2x::ErrorCode::eSuccess;
}


IBlendshapeSolveModelInfo::~IBlendshapeSolveModelInfo() = default;

BlendshapeSolveExecutorCreationParameters BlendshapeSolveModelInfoOwner::GetExecutorCreationParameters(
    IGeometryExecutor::ExecutionOption executionOption
    ) const {
    BlendshapeSolveExecutorCreationParameters params;

    if (IsAnySet(executionOption, IGeometryExecutor::ExecutionOption::Skin)) {
        params.initializationSkinParams = &_skinParams;
    }
    if (IsAnySet(executionOption, IGeometryExecutor::ExecutionOption::Tongue)) {
        params.initializationTongueParams = &_tongueParams;
    }

    return params;
}

void BlendshapeSolveModelInfoOwner::Destroy() {
    delete this;
}

std::error_code BlendshapeSolveModelInfoOwner::Init(const char* path) {
    A2F_CHECK_ERROR_WITH_MSG(path, "Model path cannot be null", nva2x::ErrorCode::eNullPointer);
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

        const auto skinConfigPath = convert_path(data.at("blendshapePaths").at("skin").at("config").get<std::string>());
        const auto skinDataPath = convert_path(data.at("blendshapePaths").at("skin").at("data").get<std::string>());
        const auto tongueConfigPath = convert_path(data.at("blendshapePaths").at("tongue").at("config").get<std::string>());
        const auto tongueDataPath = convert_path(data.at("blendshapePaths").at("tongue").at("data").get<std::string>());

        return Init(
            skinConfigPath.c_str(),
            skinDataPath.c_str(),
            tongueConfigPath.c_str(),
            tongueDataPath.c_str()
            );
    });
}

std::error_code BlendshapeSolveModelInfoOwner::Init(
    const char* skinConfigPath,
    const char* skinDataPath,
    const char* tongueConfigPath,
    const char* tongueDataPath
    ) {
    A2F_CHECK_RESULT_WITH_MSG(_skinConfig.Init(skinConfigPath), "Unable to read skin config");
    A2F_CHECK_RESULT_WITH_MSG(_skinData.Init(skinDataPath), "Unable to read skin data");
    A2F_CHECK_RESULT_WITH_MSG(_tongueConfig.Init(tongueConfigPath), "Unable to read tongue config");
    A2F_CHECK_RESULT_WITH_MSG(_tongueData.Init(tongueDataPath), "Unable to read tongue data");

    _skinParams.params = _skinConfig.GetBlendshapeSolverParams();
    _skinParams.config = _skinConfig.GetBlendshapeSolverConfig();
    _skinParams.data = _skinData.GetBlendshapeSolverDataView();

    _tongueParams.params = _tongueConfig.GetBlendshapeSolverParams();
    _tongueParams.config = _tongueConfig.GetBlendshapeSolverConfig();
    _tongueParams.data = _tongueData.GetBlendshapeSolverDataView();

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f::IRegressionModel


namespace nva2f::IDiffusionModel {

INetworkInfo::~INetworkInfo() = default;

const NetworkInfo& NetworkInfoOwner::GetNetworkInfo() const {
    return _networkInfo;
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

nva2x::HostTensorFloatConstView NetworkInfoOwner::GetDefaultEmotion() const {
    return nva2x::ToConstView(_defaultEmotion);
}

std::size_t NetworkInfoOwner::GetIdentityLength() const {
    return _identities.size();
}

const char* NetworkInfoOwner::GetIdentityName(std::size_t index) const {
    return _identities[index].c_str();
}

void NetworkInfoOwner::Destroy() {
    delete this;
}

std::error_code NetworkInfoOwner::Init(const char* path) {
    return checkError(path, [this](nlohmann::json data) {
        A2F_CHECK_ERROR_WITH_MSG(
            data.at("id").at("type") == "diffusion" && data.at("id").at("output") == "geometry",
            "Invalid network type",
            nva2x::ErrorCode::eReadFileFailed
        );

        auto params = data.at("params");
        auto emotions = params.at("emotions").get<std::vector<std::string>>();
        auto defaultEmotion = params.at("default_emotion").get<std::vector<float>>();
        A2F_CHECK_ERROR_WITH_MSG(
            emotions.size() == defaultEmotion.size(),
            "Emotion size must match default emotion size",
            nva2x::ErrorCode::eReadFileFailed
        );
        auto identities = params.at("identities").get<std::vector<std::string>>();

        NetworkInfo networkInfo;
        networkInfo.emotionLength = emotions.size();
        networkInfo.identityLength = identities.size();
        networkInfo.skinDim = params.at("skin_size").get<std::size_t>();
        networkInfo.tongueDim = params.at("tongue_size").get<std::size_t>();
        networkInfo.jawDim = params.at("jaw_size").get<std::size_t>();
        networkInfo.eyesDim = params.at("eyes_size").get<std::size_t>();
        networkInfo.numDiffusionSteps = params.at("num_diffusion_steps").get<std::size_t>();
        networkInfo.numGruLayers = params.at("num_gru_layers").get<std::size_t>();
        networkInfo.gruLatentDim = params.at("gru_latent_dim").get<std::size_t>();
        networkInfo.numFramesLeftTruncate = params.at("num_frames_left_truncate").get<std::size_t>();
        networkInfo.numFramesRightTruncate = params.at("num_frames_right_truncate").get<std::size_t>();
        networkInfo.numFramesCenter = params.at("num_frames_center").get<std::size_t>();

        auto audio_params = data.at("audio_params");
        networkInfo.bufferLength = audio_params.at("buffer_len").get<std::size_t>();
        networkInfo.paddingLeft = audio_params.at("padding_left").get<std::size_t>();
        networkInfo.paddingRight = audio_params.at("padding_right").get<std::size_t>();
        networkInfo.bufferSamplerate = audio_params.at("samplerate").get<std::size_t>();

        _emotions = std::move(emotions);
        _defaultEmotion = std::move(defaultEmotion);
        _identities = std::move(identities);
        _networkInfo = std::move(networkInfo);

        return nva2x::ErrorCode::eSuccess;
    });
}


IAnimatorData::~IAnimatorData() = default;

AnimatorDataView AnimatorDataOwner::GetAnimatorData() const {
    AnimatorDataView data;
    data.skin.neutralPose = _neutralSkin;
    data.skin.lipOpenPoseDelta = _lipOpenPoseDelta;
    data.skin.eyeClosePoseDelta = _eyeClosePoseDelta;
    data.tongue.neutralPose = _neutralTongue;
    data.teeth.neutralJaw = _neutralJaw;
    data.eyes.saccadeRot = _saccadeRot;
    return data;
}

void AnimatorDataOwner::Destroy() {
    delete this;
}

std::error_code AnimatorDataOwner::Init(const char* path) {
    nva2x::HostTensorDict data;
    A2F_CHECK_RESULT(data.ReadFromFile(path));

    const auto neutralSkinHost = data.At("neutral_skin");
    A2F_CHECK_ERROR_WITH_MSG(neutralSkinHost, "Unable to get neutral pose for skin", nva2x::ErrorCode::eReadFileFailed);
    const auto lipOpenPoseDeltaHost = data.At("lip_open_pose_delta");
    A2F_CHECK_ERROR_WITH_MSG(lipOpenPoseDeltaHost, "Unable to get lip open pose delta for skin", nva2x::ErrorCode::eReadFileFailed);
    const auto eyeClosePoseDeltaHost = data.At("eye_close_pose_delta");
    A2F_CHECK_ERROR_WITH_MSG(eyeClosePoseDeltaHost, "Unable to get eye close pose delta for skin", nva2x::ErrorCode::eReadFileFailed);

    const auto neutralTongueHost = data.At("neutral_tongue");
    A2F_CHECK_ERROR_WITH_MSG(neutralTongueHost, "Unable to get neutral pose for tongue", nva2x::ErrorCode::eReadFileFailed);

    const auto neutralJawHost = data.At("neutral_jaw");
    A2F_CHECK_ERROR_WITH_MSG(neutralJawHost, "Unable to get neutral jaw for teeth", nva2x::ErrorCode::eReadFileFailed);

    const auto saccadeRotHost = data.At("saccade_rot_matrix");
    A2F_CHECK_ERROR_WITH_MSG(saccadeRotHost, "Unable to get saccade rotations for eyes", nva2x::ErrorCode::eReadFileFailed);

    A2F_CHECK_RESULT(_neutralSkin.Init(*neutralSkinHost));
    A2F_CHECK_RESULT(_lipOpenPoseDelta.Init(*lipOpenPoseDeltaHost));
    A2F_CHECK_RESULT(_eyeClosePoseDelta.Init(*eyeClosePoseDeltaHost));
    A2F_CHECK_RESULT(_neutralTongue.Init(*neutralTongueHost));
    A2F_CHECK_RESULT(_neutralJaw.Init(*neutralJawHost));
    A2F_CHECK_RESULT(_saccadeRot.Init(*saccadeRotHost));

    return nva2x::ErrorCode::eSuccess;
}


IGeometryModelInfo::~IGeometryModelInfo() = default;

const INetworkInfo& ModelInfoOwner::GetNetworkInfo() const {
    return _networkInfo;
}

const IAnimatorData* ModelInfoOwner::GetAnimatorData(std::size_t identityIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(identityIndex < _identityData.size(), "Identity index out of bounds", nullptr);
    return &_identityData[identityIndex].animatorData;
}

const AnimatorParams* ModelInfoOwner::GetAnimatorParams(std::size_t identityIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(identityIndex < _identityData.size(), "Identity index out of bounds", nullptr);
    return &_identityData[identityIndex].defaultAnimatorParams;
}

GeometryExecutorCreationParameters ModelInfoOwner::GetExecutorCreationParameters(
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t identityIndex,
    bool constantNoise
    ) const {
    GeometryExecutorCreationParameters params;

    params.networkInfo = _networkInfo.GetNetworkInfo();
    params.networkData = _networkData.Data();
    params.networkDataSize = _networkData.Size();

    if (IsAnySet(executionOption, IGeometryExecutor::ExecutionOption::Skin)) {
        params.initializationSkinParams = &_skinParams[identityIndex];
    }
    if (IsAnySet(executionOption, IGeometryExecutor::ExecutionOption::Tongue)) {
        params.initializationTongueParams = &_tongueParams[identityIndex];
    }
    if (IsAnySet(executionOption, IGeometryExecutor::ExecutionOption::Jaw)) {
        params.initializationTeethParams = &_teethParams[identityIndex];
    }
    if (IsAnySet(executionOption, IGeometryExecutor::ExecutionOption::Eyes)) {
        params.initializationEyesParams = &_eyesParams[identityIndex];
    }

    params.identityIndex = identityIndex;
    params.constantNoise = constantNoise;

    return params;
}

void ModelInfoOwner::Destroy() {
    delete this;
}

std::error_code ModelInfoOwner::Init(const char* path) {
    A2F_CHECK_ERROR_WITH_MSG(path, "Model path cannot be null", nva2x::ErrorCode::eNullPointer);
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
        auto modelConfigPaths = data.at("modelConfigPaths").get<std::vector<std::string>>();
        for (auto& path : modelConfigPaths) {
            path = convert_path(path);
        }
        auto modelDataPaths = data.at("modelDataPaths").get<std::vector<std::string>>();
        for (auto& path : modelDataPaths) {
            path = convert_path(path);
        }

        A2F_CHECK_ERROR_WITH_MSG(
            modelConfigPaths.size() == modelDataPaths.size(),
            "Model config and model data size mismatch",
            nva2x::ErrorCode::eMismatch
            );
        std::vector<const char*> modelConfigPathsPtr(modelConfigPaths.size());
        std::vector<const char*> modelDataPathsPtr(modelDataPaths.size());
        for (std::size_t i = 0; i < modelConfigPaths.size(); ++i) {
            modelConfigPathsPtr[i] = modelConfigPaths[i].c_str();
            modelDataPathsPtr[i] = modelDataPaths[i].c_str();
        }

        return Init(
            networkInfoPath.c_str(),
            networkPath.c_str(),
            modelConfigPathsPtr.size(),
            modelConfigPathsPtr.data(),
            modelDataPathsPtr.data()
            );
    });
}

std::error_code ModelInfoOwner::Init(
    const char* networkInfoPath,
    const char* networkPath,
    std::size_t identityCount,
    const char* const* modelConfigPaths,
    const char* const* modelDataPaths) {

    A2F_CHECK_ERROR_WITH_MSG(networkInfoPath, "Network info path cannot be null", nva2x::ErrorCode::eNullPointer);
    NetworkInfoOwner networkInfo;
    A2F_CHECK_RESULT_WITH_MSG(networkInfo.Init(networkInfoPath), "Unable to read network info");
    A2F_CHECK_ERROR_WITH_MSG(identityCount == networkInfo.GetIdentityLength(), "Identity count mismatch", nva2x::ErrorCode::eMismatch);

    A2F_CHECK_ERROR_WITH_MSG(networkPath, "Network path cannot be null", nva2x::ErrorCode::eNullPointer);
    nva2x::DataBytes networkDataBytes;
    A2F_CHECK_RESULT_WITH_MSG(networkDataBytes.ReadFromFile(networkPath), "Unable to read TRT file");

    std::vector<IdentityData> identityData(identityCount);
    for (std::size_t i = 0; i < identityCount; ++i) {
        A2F_CHECK_ERROR_WITH_MSG(modelConfigPaths[i], "Model config path cannot be null", nva2x::ErrorCode::eNullPointer);
        A2F_CHECK_RESULT_WITH_MSG(
            ReadAnimatorParams_INTERNAL(
                modelConfigPaths[i], identityData[i].inputStrength, identityData[i].defaultAnimatorParams
                ),
            "Unable to read animator params"
            );

        A2F_CHECK_ERROR_WITH_MSG(modelDataPaths[i], "Model data path cannot be null", nva2x::ErrorCode::eNullPointer);
        A2F_CHECK_RESULT_WITH_MSG(identityData[i].animatorData.Init(modelDataPaths[i]), "Unable to read animator data");
    }

    _networkInfo = std::move(networkInfo);
    _networkData = std::move(networkDataBytes);
    _identityData = std::move(identityData);

    _skinParams.resize(_identityData.size());
    _tongueParams.resize(_identityData.size());
    _teethParams.resize(_identityData.size());
    _eyesParams.resize(_identityData.size());
    for (std::size_t i = 0; i < _identityData.size(); ++i) {
        _skinParams[i].params = _identityData[i].defaultAnimatorParams.skin;
        _skinParams[i].data = _identityData[i].animatorData.GetAnimatorData().skin;

        _tongueParams[i].params = _identityData[i].defaultAnimatorParams.tongue;
        _tongueParams[i].data = _identityData[i].animatorData.GetAnimatorData().tongue;

        _teethParams[i].params = _identityData[i].defaultAnimatorParams.teeth;
        _teethParams[i].data = _identityData[i].animatorData.GetAnimatorData().teeth;

        _eyesParams[i].params = _identityData[i].defaultAnimatorParams.eyes;
        _eyesParams[i].data = _identityData[i].animatorData.GetAnimatorData().eyes;
    }

    return nva2x::ErrorCode::eSuccess;
}


IBlendshapeSolveModelInfo::~IBlendshapeSolveModelInfo() = default;

BlendshapeSolveExecutorCreationParameters BlendshapeSolveModelInfoOwner::GetExecutorCreationParameters(
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t identityIndex
    ) const {
    BlendshapeSolveExecutorCreationParameters params;

    if (IsAnySet(executionOption, IGeometryExecutor::ExecutionOption::Skin)) {
        params.initializationSkinParams = &_skinParams[identityIndex];
    }
    if (IsAnySet(executionOption, IGeometryExecutor::ExecutionOption::Tongue)) {
        params.initializationTongueParams = &_tongueParams[identityIndex];
    }

    return params;
}

void BlendshapeSolveModelInfoOwner::Destroy() {
    delete this;
}

std::error_code BlendshapeSolveModelInfoOwner::Init(const char* path) {
    A2F_CHECK_ERROR_WITH_MSG(path, "Model path cannot be null", nva2x::ErrorCode::eNullPointer);
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

        std::vector<std::string> skinConfigPaths;
        std::vector<std::string> skinDataPaths;
        std::vector<std::string> tongueConfigPaths;
        std::vector<std::string> tongueDataPaths;

        const auto blendshapePaths = data.at("blendshapePaths");
        for (const auto& identityPath : blendshapePaths) {
            skinConfigPaths.emplace_back(convert_path(identityPath.at("skin").at("config").get<std::string>()));
            skinDataPaths.emplace_back(convert_path(identityPath.at("skin").at("data").get<std::string>()));
            tongueConfigPaths.emplace_back(convert_path(identityPath.at("tongue").at("config").get<std::string>()));
            tongueDataPaths.emplace_back(convert_path(identityPath.at("tongue").at("data").get<std::string>()));
        }

        std::vector<const char*> skinConfigPathsPtr(skinConfigPaths.size());
        std::vector<const char*> skinDataPathsPtr(skinDataPaths.size());
        std::vector<const char*> tongueConfigPathsPtr(tongueConfigPaths.size());
        std::vector<const char*> tongueDataPathsPtr(tongueDataPaths.size());
        for (std::size_t i = 0; i < skinConfigPaths.size(); ++i) {
            skinConfigPathsPtr[i] = skinConfigPaths[i].c_str();
            skinDataPathsPtr[i] = skinDataPaths[i].c_str();
            tongueConfigPathsPtr[i] = tongueConfigPaths[i].c_str();
            tongueDataPathsPtr[i] = tongueDataPaths[i].c_str();
        }

        return Init(
            skinConfigPathsPtr.size(),
            skinConfigPathsPtr.data(),
            skinDataPathsPtr.data(),
            tongueConfigPathsPtr.data(),
            tongueDataPathsPtr.data()
            );
    });
}

std::error_code BlendshapeSolveModelInfoOwner::Init(
    std::size_t identityCount,
    const char* const* skinConfigPaths,
    const char* const* skinDataPaths,
    const char* const* tongueConfigPaths,
    const char* const* tongueDataPaths
    ) {
    A2F_CHECK_ERROR_WITH_MSG(identityCount > 0, "Identity count must be greater than 0", nva2x::ErrorCode::eInvalidValue);
    A2F_CHECK_ERROR_WITH_MSG(skinConfigPaths, "Skin config paths cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_ERROR_WITH_MSG(skinDataPaths, "Skin data paths cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_ERROR_WITH_MSG(tongueConfigPaths, "Tongue config paths cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_ERROR_WITH_MSG(tongueDataPaths, "Tongue data paths cannot be null", nva2x::ErrorCode::eNullPointer);

    std::vector<IdentityData> identityData(identityCount);
    for (std::size_t i = 0; i < identityCount; ++i) {
        A2F_CHECK_RESULT_WITH_MSG(
            identityData[i].skinConfig.Init(skinConfigPaths[i]),
            "Unable to read skin config"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            identityData[i].skinData.Init(skinDataPaths[i]),
            "Unable to read skin data"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            identityData[i].tongueConfig.Init(tongueConfigPaths[i]),
            "Unable to read tongue config"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            identityData[i].tongueData.Init(tongueDataPaths[i]),
            "Unable to read tongue data"
            );
    }

    _identityData = std::move(identityData);

    _skinParams.resize(_identityData.size());
    _tongueParams.resize(_identityData.size());
    for (std::size_t i = 0; i < _identityData.size(); ++i) {
        _skinParams[i].params = _identityData[i].skinConfig.GetBlendshapeSolverParams();
        _skinParams[i].config = _identityData[i].skinConfig.GetBlendshapeSolverConfig();
        _skinParams[i].data = _identityData[i].skinData.GetBlendshapeSolverDataView();

        _tongueParams[i].params = _identityData[i].tongueConfig.GetBlendshapeSolverParams();
        _tongueParams[i].config = _identityData[i].tongueConfig.GetBlendshapeSolverConfig();
        _tongueParams[i].data = _identityData[i].tongueData.GetBlendshapeSolverDataView();
    }

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f::IDiffusionModel


namespace nva2f {


IBlendshapeSolverConfig::~IBlendshapeSolverConfig() = default;

const BlendshapeSolverParams& BlendshapeSolverConfigOwner::GetBlendshapeSolverParams() const {
    return _blendshapeSolverParams;
}

BlendshapeSolverConfig BlendshapeSolverConfigOwner::GetBlendshapeSolverConfig() const {
    assert(_activePoses.size() == _cancelPoses.size());
    assert(_activePoses.size() == _symmetryPoses.size());
    assert(_activePoses.size() == _multipliers.size());
    assert(_activePoses.size() == _offsets.size());

    BlendshapeSolverConfig config;
    config.numBlendshapes = _activePoses.size();
    config.activePoses = _activePoses.data();
    config.cancelPoses = _cancelPoses.data();
    config.symmetryPoses = _symmetryPoses.data();
    config.multipliers = nva2x::ToConstView(_multipliers);
    config.offsets = nva2x::ToConstView(_offsets);
    return config;
}

void BlendshapeSolverConfigOwner::Destroy() {
    delete this;
}

std::error_code BlendshapeSolverConfigOwner::Init(const char* configPath) {
    return checkError(configPath, [this](nlohmann::json data) {
        BlendshapeSolverParams readParams;

        auto config = data.at("blendshape_params");

        readParams.L1Reg = config.at("strengthL1regularization").get<float>();
        readParams.L2Reg = config.at("strengthL2regularization").get<float>();
        readParams.TemporalReg = config.at("strengthTemporalSmoothing").get<float>();
        readParams.SymmetryReg = config.at("strengthSymmetry").get<float>();
        // the default value for templateBBSize is for the skin blendshape which the value was hard-coded in the sdk
        // we can make it a required field after updating the all the json files in the blendshape package
        readParams.templateBBSize = config.value("templateBBSize", kDefaultTemplateBBSize);

        auto numPoses = config.at("numPoses").get<std::size_t>();
        auto activePoses = config.at("bsSolveActivePoses").get<std::vector<int>>();
        A2F_CHECK_ERROR_WITH_MSG(activePoses.size() == numPoses, "Active poses size mismatch", nva2x::ErrorCode::eMismatch);
        auto cancelPoses = config.at("bsSolveCancelPoses").get<std::vector<int>>();
        A2F_CHECK_ERROR_WITH_MSG(cancelPoses.size() == numPoses, "Cancel poses size mismatch", nva2x::ErrorCode::eMismatch);
        auto symmetryPoses = config.at("bsSolveSymmetryPoses").get<std::vector<int>>();
        A2F_CHECK_ERROR_WITH_MSG(symmetryPoses.size() == numPoses, "Symmetry poses size mismatch", nva2x::ErrorCode::eMismatch);
        auto multipliers = config.at("bsWeightMultipliers").get<std::vector<float>>();
        A2F_CHECK_ERROR_WITH_MSG(multipliers.size() == numPoses, "Multipliers size mismatch", nva2x::ErrorCode::eMismatch);
        auto offsets = config.at("bsWeightOffsets").get<std::vector<float>>();
        A2F_CHECK_ERROR_WITH_MSG(offsets.size() == numPoses, "Offsets size mismatch", nva2x::ErrorCode::eMismatch);

        _blendshapeSolverParams = std::move(readParams);
        _activePoses = std::move(activePoses);
        _cancelPoses = std::move(cancelPoses);
        _symmetryPoses = std::move(symmetryPoses);
        _multipliers = std::move(multipliers);
        _offsets = std::move(offsets);

        return nva2x::ErrorCode::eSuccess;
    });
}


IBlendshapeSolverData::~IBlendshapeSolverData() = default;

BlendshapeSolverDataView BlendshapeSolverDataOwner::GetBlendshapeSolverDataView() const {
    assert(_neutralPose.size() > 0);
    assert(_deltaPoses.size() % _neutralPose.size() == 0);
    assert(_poseNames.size() == _poseNamesCstr.size());
    assert(_neutralPose.size() * _poseNames.size() == _deltaPoses.size());

    BlendshapeSolverDataView data;
    data.neutralPose = nva2x::ToConstView(_neutralPose);
    data.deltaPoses = nva2x::ToConstView(_deltaPoses);
    data.poseMask = _poseMask.data();
    data.poseMaskSize = _poseMask.size();
    data.poseNames = _poseNamesCstr.data();
    data.poseNamesSize = _poseNamesCstr.size();
    return data;
}

void BlendshapeSolverDataOwner::Destroy() {
    delete this;
}

std::error_code BlendshapeSolverDataOwner::Init(const char* dataPath) {
    auto errorData = [dataPath, this]() -> std::error_code {
        auto u8Path = std::filesystem::u8path(dataPath);
        if (u8Path.extension() != std::filesystem::u8path(".npz")) {
            LOG_ERROR("Unsupported file extension.");
            return nva2x::ErrorCode::eOpenFileFailed;
        }

        cnpy::npz_t npzInputData;
        try {
            npzInputData = nva2x::npz_load(dataPath);
        } catch(const std::exception& e [[maybe_unused]]) {
            LOG_ERROR("Unable to parse blendshape data file: " << dataPath << " ; Message: " << e.what());
            return nva2x::ErrorCode::eOpenFileFailed;
        }

        const auto itPoseNames = npzInputData.find("poseNames");
        if (itPoseNames == npzInputData.end()) {
            LOG_ERROR("poseNames is not found");
            return nva2x::ErrorCode::eInvalidValue;
        }
        const cnpy::NpyArray blendshapeNamesNpyArr = itPoseNames->second;
        std::vector<std::string> blendshapeNames = nva2x::parse_string_array_from_npy_array(blendshapeNamesNpyArr);
        if (blendshapeNames.empty()) {
            LOG_ERROR("poseNames is empty");
            return nva2x::ErrorCode::eInvalidValue;
        }
        if (blendshapeNames[0] != "neutral") {
            LOG_ERROR("poseNames[0] should be the neutral pose");
            return nva2x::ErrorCode::eInvalidValue;
        }

        const auto itNeutralPose = npzInputData.find(blendshapeNames[0]);
        if (itNeutralPose == npzInputData.end()) {
            LOG_ERROR("neutralPose (" << blendshapeNames[0] << ") is not found");
            return nva2x::ErrorCode::eInvalidValue;
        }
        std::vector<float> neutralPose = itNeutralPose->second.as_vec<float>();
        const std::size_t numVertexPositions = neutralPose.size();

        blendshapeNames.erase(blendshapeNames.begin());
        const std::size_t numBlendshapes = blendshapeNames.size();

        std::vector<float> deltaPoses(numVertexPositions * numBlendshapes);
        {
            for (std::size_t i = 0; i < numBlendshapes; ++i) {
                const auto itPose = npzInputData.find(blendshapeNames[i]);
                if (itPose == npzInputData.end()) {
                    LOG_ERROR("Delta pose (" << blendshapeNames[i] << ") is not found");
                    return nva2x::ErrorCode::eInvalidValue;
                }
                cnpy::NpyArray pose = itPose->second;
                A2F_CHECK_ERROR_WITH_MSG(
                    numVertexPositions * sizeof(float) == pose.num_bytes(),
                    "Delta poses size mismatch",
                    nva2x::ErrorCode::eMismatch
                    );
                const float* input = pose.data<float>();
                auto output = deltaPoses.begin() + i * numVertexPositions;
                std::copy(input, input + numVertexPositions, output);
            }
        }

        std::vector<int> poseMask;
        if (npzInputData.find("frontalMask") != npzInputData.end()) {
            poseMask = npzInputData["frontalMask"].as_vec<int>();
        }

        _neutralPose = std::move(neutralPose);
        _deltaPoses = std::move(deltaPoses);
        _poseMask = std::move(poseMask);
        _poseNames = std::move(blendshapeNames);
        _poseNamesCstr.resize(_poseNames.size());
        for (std::size_t i = 0; i < _poseNames.size(); ++i) {
            _poseNamesCstr[i] = _poseNames[i].c_str();
        }

        return nva2x::ErrorCode::eSuccess;
    }();
    if (errorData) {
        LOG_ERROR("Unable to parse blendshape data file: " << dataPath);
        return errorData;
    }

    return nva2x::ErrorCode::eSuccess;
}


IGeometryExecutorBundle::~IGeometryExecutorBundle() = default;

nva2x::ICudaStream& GeometryExecutorBundle::GetCudaStream() {
    return _cudaStream;
}

const nva2x::ICudaStream& GeometryExecutorBundle::GetCudaStream() const {
    return _cudaStream;
}

nva2x::IAudioAccumulator& GeometryExecutorBundle::GetAudioAccumulator(std::size_t trackIndex) {
    return *_audioAccumulators[trackIndex];
}

const nva2x::IAudioAccumulator& GeometryExecutorBundle::GetAudioAccumulator(std::size_t trackIndex) const {
    return *_audioAccumulators[trackIndex];
}

nva2x::IEmotionAccumulator& GeometryExecutorBundle::GetEmotionAccumulator(std::size_t trackIndex) {
    return *_emotionAccumulators[trackIndex];
}

const nva2x::IEmotionAccumulator& GeometryExecutorBundle::GetEmotionAccumulator(std::size_t trackIndex) const {
    return *_emotionAccumulators[trackIndex];
}

IGeometryExecutor& GeometryExecutorBundle::GetExecutor() {
    return *_geometryExecutor;
}

const IGeometryExecutor& GeometryExecutorBundle::GetExecutor() const {
    return *_geometryExecutor;
}

void GeometryExecutorBundle::Destroy() {
    delete this;
}

std::error_code GeometryExecutorBundle::InitRegression(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IRegressionModel::IGeometryModelInfo** outModelInfo
    ) {
    // Read model info.
    IRegressionModel::ModelInfoOwner modelInfo;
    A2F_CHECK_RESULT_WITH_MSG(modelInfo.Init(path), "Unable to read regression model info");

    const std::size_t emotionsCount = modelInfo.GetNetworkInfo().GetEmotionsCount();
    A2F_CHECK_RESULT_WITH_MSG(InitBase(nbTracks, emotionsCount), "Unable to initialize geometry executor bundle");

    // Create geometry executor.
    GeometryExecutorCreationParameters params;
    params.cudaStream = _cudaStream.Data();
    params.nbTracks = nbTracks;

    std::vector<const nva2x::IAudioAccumulator*> audioAccumulators(nbTracks);
    std::vector<const nva2x::IEmotionAccumulator*> emotionAccumulators(nbTracks);
    for (std::size_t i = 0; i < nbTracks; ++i) {
        audioAccumulators[i] = _audioAccumulators[i].get();
        emotionAccumulators[i] = _emotionAccumulators[i].get();
    }
    params.sharedAudioAccumulators = audioAccumulators.data();
    params.sharedEmotionAccumulators = emotionAccumulators.data();

    const auto regressionParams = modelInfo.GetExecutorCreationParameters(
        executionOption, frameRateNumerator, frameRateDenominator
        );

    _geometryExecutor.reset(
        nva2f::CreateRegressionGeometryExecutor_INTERNAL(params, regressionParams)
        );
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Unable to create geometry executor", nva2x::ErrorCode::eInvalidValue);

    if (outModelInfo) {
        *outModelInfo = new IRegressionModel::ModelInfoOwner(std::move(modelInfo));
    }

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorBundle::InitDiffusion(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t identityIndex,
    bool constantNoise,
    IDiffusionModel::IGeometryModelInfo** outModelInfo
    ) {
    // Read model info.
    IDiffusionModel::ModelInfoOwner modelInfo;
    A2F_CHECK_RESULT_WITH_MSG(modelInfo.Init(path), "Unable to read diffusion model info");

    const std::size_t emotionsCount = modelInfo.GetNetworkInfo().GetEmotionsCount();
    A2F_CHECK_RESULT_WITH_MSG(InitBase(nbTracks, emotionsCount), "Unable to initialize geometry executor bundle");

    // Create geometry executor.
    GeometryExecutorCreationParameters params;
    params.cudaStream = _cudaStream.Data();
    params.nbTracks = nbTracks;

    std::vector<const nva2x::IAudioAccumulator*> audioAccumulators(nbTracks);
    std::vector<const nva2x::IEmotionAccumulator*> emotionAccumulators(nbTracks);
    for (std::size_t i = 0; i < nbTracks; ++i) {
        audioAccumulators[i] = _audioAccumulators[i].get();
        emotionAccumulators[i] = _emotionAccumulators[i].get();
    }
    params.sharedAudioAccumulators = audioAccumulators.data();
    params.sharedEmotionAccumulators = emotionAccumulators.data();

    const auto diffusionParams = modelInfo.GetExecutorCreationParameters(
        executionOption, identityIndex, constantNoise
        );

    _geometryExecutor.reset(
        nva2f::CreateDiffusionGeometryExecutor_INTERNAL(params, diffusionParams)
        );
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Unable to create geometry executor", nva2x::ErrorCode::eInvalidValue);

    if (outModelInfo) {
        *outModelInfo = new IDiffusionModel::ModelInfoOwner(std::move(modelInfo));
    }

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorBundle::InitBase(std::size_t nbTracks, std::size_t emotionsCount) {
    A2F_CHECK_RESULT_WITH_MSG(_cudaStream.Init(), "Unable to create cuda stream");

    _audioAccumulators.resize(nbTracks);
    _emotionAccumulators.resize(nbTracks);

    for (std::size_t i = 0; i < nbTracks; ++i) {
        // Hard-coded values of 1 second per buffer, no pre-allocation.
        _audioAccumulators[i] = std::make_unique<nva2x::AudioAccumulator>();
        A2F_CHECK_RESULT_WITH_MSG(_audioAccumulators[i]->Allocate(16000, 0), "Unable to create audio accumulator");

        // Hard-coded values of 300 emotions per buffer (30 FPS * 10 seconds), no pre-allocation.
        _emotionAccumulators[i] = std::make_unique<nva2x::EmotionAccumulator>();
        A2F_CHECK_RESULT_WITH_MSG(
            _emotionAccumulators[i]->Allocate(emotionsCount, 300, 0),
            "Unable to create emotion accumulator"
            );
    }

    return nva2x::ErrorCode::eSuccess;
}


IBlendshapeExecutorBundle::~IBlendshapeExecutorBundle() = default;

nva2x::ICudaStream& BlendshapeSolveExecutorBundle::GetCudaStream() {
    return _cudaStream;
}

const nva2x::ICudaStream& BlendshapeSolveExecutorBundle::GetCudaStream() const {
    return _cudaStream;
}

nva2x::IAudioAccumulator& BlendshapeSolveExecutorBundle::GetAudioAccumulator(std::size_t trackIndex) {
    return *_audioAccumulators[trackIndex];
}

const nva2x::IAudioAccumulator& BlendshapeSolveExecutorBundle::GetAudioAccumulator(std::size_t trackIndex) const {
    return *_audioAccumulators[trackIndex];
}

nva2x::IEmotionAccumulator& BlendshapeSolveExecutorBundle::GetEmotionAccumulator(std::size_t trackIndex) {
    return *_emotionAccumulators[trackIndex];
}

const nva2x::IEmotionAccumulator& BlendshapeSolveExecutorBundle::GetEmotionAccumulator(std::size_t trackIndex) const {
    return *_emotionAccumulators[trackIndex];
}

IBlendshapeExecutor& BlendshapeSolveExecutorBundle::GetExecutor() {
    return *_blendshapeExecutor;
}

const IBlendshapeExecutor& BlendshapeSolveExecutorBundle::GetExecutor() const {
    return *_blendshapeExecutor;
}

void BlendshapeSolveExecutorBundle::Destroy() {
    delete this;
}

std::error_code BlendshapeSolveExecutorBundle::InitRegression(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    bool useGpuSolver,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IRegressionModel::IGeometryModelInfo** outModelInfo,
    IRegressionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
    ) {
    // Read model info.
    IRegressionModel::ModelInfoOwner modelInfo;
    A2F_CHECK_RESULT_WITH_MSG(modelInfo.Init(path), "Unable to read regression model info");

    const std::size_t emotionsCount = modelInfo.GetNetworkInfo().GetEmotionsCount();
    A2F_CHECK_RESULT_WITH_MSG(InitBase(nbTracks, emotionsCount), "Unable to initialize geometry executor bundle");

    // Create geometry executor.
    GeometryExecutorCreationParameters params;
    params.cudaStream = _cudaStream.Data();
    params.nbTracks = nbTracks;

    std::vector<nva2x::IAudioAccumulator*> audioAccumulators(nbTracks);
    std::vector<nva2x::IEmotionAccumulator*> emotionAccumulators(nbTracks);
    for (std::size_t i = 0; i < nbTracks; ++i) {
        audioAccumulators[i] = _audioAccumulators[i].get();
        emotionAccumulators[i] = _emotionAccumulators[i].get();
    }
    params.sharedAudioAccumulators = audioAccumulators.data();
    params.sharedEmotionAccumulators = emotionAccumulators.data();

    const auto regressionParams = modelInfo.GetExecutorCreationParameters(
        executionOption, frameRateNumerator, frameRateDenominator
        );

    auto geometryExecutor = nva2x::ToUniquePtr(
        nva2f::CreateRegressionGeometryExecutor_INTERNAL(params, regressionParams)
        );
    A2F_CHECK_ERROR_WITH_MSG(geometryExecutor, "Unable to create geometry executor", nva2x::ErrorCode::eInvalidValue);

    // Create blendshape executor.
    IRegressionModel::BlendshapeSolveModelInfoOwner blendshapeModelInfo;
    A2F_CHECK_RESULT_WITH_MSG(blendshapeModelInfo.Init(path), "Unable to read blendshape model info");

    const auto creationParams = blendshapeModelInfo.GetExecutorCreationParameters(
        executionOption
        );

    if (outModelInfo) {
        *outModelInfo = new IRegressionModel::ModelInfoOwner(std::move(modelInfo));
    }
    if (outBlendshapeSolveModelInfo) {
        *outBlendshapeSolveModelInfo = new IRegressionModel::BlendshapeSolveModelInfoOwner(std::move(blendshapeModelInfo));
    }

    return InitExecutor(std::move(geometryExecutor), creationParams, useGpuSolver);
}

std::error_code BlendshapeSolveExecutorBundle::InitDiffusion(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    bool useGpuSolver,
    std::size_t identityIndex,
    bool constantNoise,
    IDiffusionModel::IGeometryModelInfo** outModelInfo,
    IDiffusionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
    ) {
    // Read model info.
    IDiffusionModel::ModelInfoOwner modelInfo;
    A2F_CHECK_RESULT_WITH_MSG(modelInfo.Init(path), "Unable to read diffusion model info");

    const std::size_t emotionsCount = modelInfo.GetNetworkInfo().GetEmotionsCount();
    A2F_CHECK_RESULT_WITH_MSG(InitBase(nbTracks, emotionsCount), "Unable to initialize geometry executor bundle");

    // Create geometry executor.
    GeometryExecutorCreationParameters params;
    params.cudaStream = _cudaStream.Data();
    params.nbTracks = nbTracks;

    std::vector<nva2x::IAudioAccumulator*> audioAccumulators(nbTracks);
    std::vector<nva2x::IEmotionAccumulator*> emotionAccumulators(nbTracks);
    for (std::size_t i = 0; i < nbTracks; ++i) {
        audioAccumulators[i] = _audioAccumulators[i].get();
        emotionAccumulators[i] = _emotionAccumulators[i].get();
    }
    params.sharedAudioAccumulators = audioAccumulators.data();
    params.sharedEmotionAccumulators = emotionAccumulators.data();

    const auto diffusionParams = modelInfo.GetExecutorCreationParameters(
        executionOption, identityIndex, constantNoise
        );

    auto geometryExecutor = nva2x::ToUniquePtr(
        nva2f::CreateDiffusionGeometryExecutor_INTERNAL(params, diffusionParams)
        );
    A2F_CHECK_ERROR_WITH_MSG(geometryExecutor, "Unable to create geometry executor", nva2x::ErrorCode::eInvalidValue);

    // Create blendshape executor.
    IDiffusionModel::BlendshapeSolveModelInfoOwner blendshapeModelInfo;
    A2F_CHECK_RESULT_WITH_MSG(blendshapeModelInfo.Init(path), "Unable to read blendshape model info");

    const auto creationParams = blendshapeModelInfo.GetExecutorCreationParameters(
        executionOption, identityIndex
        );

    if (outModelInfo) {
        *outModelInfo = new IDiffusionModel::ModelInfoOwner(std::move(modelInfo));
    }
    if (outBlendshapeSolveModelInfo) {
        *outBlendshapeSolveModelInfo = new IDiffusionModel::BlendshapeSolveModelInfoOwner(std::move(blendshapeModelInfo));
    }

    return InitExecutor(std::move(geometryExecutor), creationParams, useGpuSolver);
}

std::error_code BlendshapeSolveExecutorBundle::InitBase(
    std::size_t nbTracks, std::size_t emotionsCount
    ) {
    A2F_CHECK_RESULT_WITH_MSG(_cudaStream.Init(), "Unable to create cuda stream");

    _audioAccumulators.resize(nbTracks);
    _emotionAccumulators.resize(nbTracks);

    for (std::size_t i = 0; i < nbTracks; ++i) {
        // Hard-coded values of 1 second per buffer, no pre-allocation.
        _audioAccumulators[i] = std::make_unique<nva2x::AudioAccumulator>();
        A2F_CHECK_RESULT_WITH_MSG(_audioAccumulators[i]->Allocate(16000, 0), "Unable to create audio accumulator");

        // Hard-coded values of 300 emotions per buffer (30 FPS * 10 seconds), no pre-allocation.
        _emotionAccumulators[i] = std::make_unique<nva2x::EmotionAccumulator>();
        A2F_CHECK_RESULT_WITH_MSG(
            _emotionAccumulators[i]->Allocate(emotionsCount, 300, 0),
            "Unable to create emotion accumulator"
            );
    }

    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveExecutorBundle::InitExecutor(
    nva2x::UniquePtr<IGeometryExecutor> geometryExecutor,
    const BlendshapeSolveExecutorCreationParameters& creationParams,
    bool useGpuSolver
    ) {
    // Create blendshape executor.
    if (useGpuSolver) {
        nva2f::DeviceBlendshapeSolveExecutorCreationParameters params;

        params.initializationSkinParams = creationParams.initializationSkinParams;
        params.initializationTongueParams = creationParams.initializationTongueParams;

        _blendshapeExecutor.reset(
            nva2f::CreateDeviceBlendshapeSolveExecutor_INTERNAL(
                geometryExecutor.release(), params
                )
            );
    }
    else {
        nva2f::HostBlendshapeSolveExecutorCreationParameters params;

        params.initializationSkinParams = creationParams.initializationSkinParams;
        params.initializationTongueParams = creationParams.initializationTongueParams;

        params.sharedJobRunner = nullptr;

        _blendshapeExecutor.reset(
            nva2f::CreateHostBlendshapeSolveExecutor_INTERNAL(
                geometryExecutor.release(), params
                )
            );
    }
    A2F_CHECK_ERROR_WITH_MSG(
        _blendshapeExecutor,
        "Unable to create blendshape executor",
        nva2x::ErrorCode::eInvalidValue
        );

    return nva2x::ErrorCode::eSuccess;
}


std::error_code ReadAnimatorParams_INTERNAL(const char* path, float& inputStrength, AnimatorParams& params) {
    LOG_DEBUG("ReadAnimatorParams()");
    return checkError(path, [&inputStrength, &params](nlohmann::json data) {
        float readInputStrength;
        AnimatorParams readParams;

        auto config = data.at("config");

        readInputStrength = config.at("input_strength").get<float>();

        readParams.skin.lowerFaceSmoothing = config.at("lower_face_smoothing").get<float>();
        readParams.skin.upperFaceSmoothing = config.at("upper_face_smoothing").get<float>();
        readParams.skin.lowerFaceStrength = config.at("lower_face_strength").get<float>();
        readParams.skin.upperFaceStrength = config.at("upper_face_strength").get<float>();
        readParams.skin.faceMaskLevel = config.at("face_mask_level").get<float>();
        readParams.skin.faceMaskSoftness = config.at("face_mask_softness").get<float>();
        readParams.skin.skinStrength = config.at("skin_strength").get<float>();
        readParams.skin.blinkStrength = config.at("blink_strength").get<float>();
        readParams.skin.eyelidOpenOffset = config.at("eyelid_open_offset").get<float>();
        readParams.skin.lipOpenOffset = config.at("lip_open_offset").get<float>();
        // blink offset is not present in the config file.
        readParams.skin.blinkOffset = 0.0f;

        readParams.tongue.tongueStrength = config.at("tongue_strength").get<float>();
        readParams.tongue.tongueHeightOffset = config.at("tongue_height_offset").get<float>();
        readParams.tongue.tongueDepthOffset = config.at("tongue_depth_offset").get<float>();

        readParams.teeth.lowerTeethStrength = config.at("lower_teeth_strength").get<float>();
        readParams.teeth.lowerTeethHeightOffset = config.at("lower_teeth_height_offset").get<float>();
        readParams.teeth.lowerTeethDepthOffset = config.at("lower_teeth_depth_offset").get<float>();

        readParams.eyes.eyeballsStrength = config.at("eyeballs_strength").get<float>();
        readParams.eyes.saccadeStrength = config.at("saccade_strength").get<float>();
        readParams.eyes.rightEyeballRotationOffsetX = config.at("right_eye_rot_x_offset").get<float>();
        readParams.eyes.rightEyeballRotationOffsetY = config.at("right_eye_rot_y_offset").get<float>();
        readParams.eyes.leftEyeballRotationOffsetX = config.at("left_eye_rot_x_offset").get<float>();
        readParams.eyes.leftEyeballRotationOffsetY = config.at("left_eye_rot_y_offset").get<float>();
        readParams.eyes.saccadeSeed = config.at("eye_saccade_seed").get<float>();

        inputStrength = readInputStrength;
        params = std::move(readParams);

        return nva2x::ErrorCode::eSuccess;
    });
}

IRegressionModel::INetworkInfo* ReadRegressionNetworkInfo_INTERNAL(const char* path) {
    LOG_DEBUG("IRegressionModel::ReadNetworkInfo()");
    auto networkInfo = std::make_unique<IRegressionModel::NetworkInfoOwner>();
    if (networkInfo->Init(path)) {
        LOG_ERROR("Unable to read regression model network info: " << path);
        return nullptr;
    }
    return networkInfo.release();
}

IRegressionModel::IAnimatorData* ReadRegressionAnimatorData_INTERNAL(const char* path) {
    LOG_DEBUG("IRegressionModel::ReadAnimatorData()");
    auto animatorData = std::make_unique<IRegressionModel::AnimatorDataOwner>();
    if (animatorData->Init(path)) {
        LOG_ERROR("Unable to read regression model animator data: " << path);
        return nullptr;
    }
    return animatorData.release();
}

IRegressionModel::IGeometryModelInfo* ReadRegressionModelInfo_INTERNAL(const char* path) {
    LOG_DEBUG("IRegressionModel::ReadModelInfo()");
    auto modelInfo = std::make_unique<IRegressionModel::ModelInfoOwner>();
    if (modelInfo->Init(path)) {
        LOG_ERROR("Unable to read regression model info: " << path);
        return nullptr;
    }
    return modelInfo.release();
}

IRegressionModel::IGeometryModelInfo* ReadRegressionModelInfo_INTERNAL(
    const char* networkInfoPath,
    const char* networkPath,
    const char* emotionDatabasePath,
    const char* modelConfigPath,
    const char* modelDataPath) {
    LOG_DEBUG("IRegressionModel::ReadModelInfo()");
    auto modelInfo = std::make_unique<IRegressionModel::ModelInfoOwner>();
    if (modelInfo->Init(networkInfoPath, networkPath, emotionDatabasePath, modelConfigPath, modelDataPath)) {
        LOG_ERROR("Unable to read regression model info");
        return nullptr;
    }
    return modelInfo.release();
}


IDiffusionModel::INetworkInfo* ReadDiffusionNetworkInfo_INTERNAL(const char* path) {
    LOG_DEBUG("IDiffusionModel::ReadNetworkInfo()");
    auto networkInfo = std::make_unique<IDiffusionModel::NetworkInfoOwner>();
    if (networkInfo->Init(path)) {
        LOG_ERROR("Unable to read diffusion model network info: " << path);
        return nullptr;
    }
    return networkInfo.release();
}

IDiffusionModel::IAnimatorData* ReadDiffusionAnimatorData_INTERNAL(const char* path) {
    LOG_DEBUG("IDiffusionModel::ReadAnimatorData()");
    auto animatorData = std::make_unique<IDiffusionModel::AnimatorDataOwner>();
    if (animatorData->Init(path)) {
        LOG_ERROR("Unable to read diffusion model animator data: " << path);
        return nullptr;
    }
    return animatorData.release();
}

IDiffusionModel::IGeometryModelInfo* ReadDiffusionModelInfo_INTERNAL(const char* path) {
    LOG_DEBUG("IDiffusionModel::ReadDiffusionModelInfo()");
    auto modelInfo = std::make_unique<IDiffusionModel::ModelInfoOwner>();
    if (modelInfo->Init(path)) {
        LOG_ERROR("Unable to read diffusion model info: " << path);
        return nullptr;
    }
    return modelInfo.release();
}

IDiffusionModel::IGeometryModelInfo* ReadDiffusionModelInfo_INTERNAL(
    const char* networkInfoPath,
    const char* networkPath,
    std::size_t identityCount,
    const char* const* modelConfigPaths,
    const char* const* modelDataPaths) {
    LOG_DEBUG("IDiffusionModel::ReadDiffusionModelInfo()");
    auto modelInfo = std::make_unique<IDiffusionModel::ModelInfoOwner>();
    if (modelInfo->Init(networkInfoPath, networkPath, identityCount, modelConfigPaths, modelDataPaths)) {
        LOG_ERROR("Unable to read diffusion model info");
        return nullptr;
    }
    return modelInfo.release();
}

IBlendshapeSolverConfig* ReadBlendshapeSolverConfig_INTERNAL(const char* configPath) {
    LOG_DEBUG("ReadBlendshapeSolverConfig()");
    auto blendshapeSolverConfig = std::make_unique<BlendshapeSolverConfigOwner>();
    if (blendshapeSolverConfig->Init(configPath)) {
        LOG_ERROR("Unable to read blendshape solver config");
        return nullptr;
    }
    return blendshapeSolverConfig.release();
}

IBlendshapeSolverData* ReadBlendshapeSolverData_INTERNAL(const char* dataPath) {
    LOG_DEBUG("ReadBlendshapeSolverData()");
    auto blendshapeSolverData = std::make_unique<BlendshapeSolverDataOwner>();
    if (blendshapeSolverData->Init(dataPath)) {
        LOG_ERROR("Unable to read blendshape solver data");
        return nullptr;
    }
    return blendshapeSolverData.release();
}

IRegressionModel::IBlendshapeSolveModelInfo* ReadRegressionBlendshapeSolveModelInfo_INTERNAL(const char* path) {
    LOG_DEBUG("IRegressionModel::ReadBlendshapeSolveModelInfo()");
    auto blendshapeSolveModelInfo = std::make_unique<IRegressionModel::BlendshapeSolveModelInfoOwner>();
    if (blendshapeSolveModelInfo->Init(path)) {
        LOG_ERROR("Unable to read regression blendshape solve model info");
        return nullptr;
    }
    return blendshapeSolveModelInfo.release();
}

IRegressionModel::IBlendshapeSolveModelInfo* ReadRegressionBlendshapeSolveModelInfo_INTERNAL(
    const char* skinConfigPath,
    const char* skinDataPath,
    const char* tongueConfigPath,
    const char* tongueDataPath
    ) {
    LOG_DEBUG("IRegressionModel::ReadBlendshapeSolveModelInfo()");
    auto blendshapeSolveModelInfo = std::make_unique<IRegressionModel::BlendshapeSolveModelInfoOwner>();
    if (blendshapeSolveModelInfo->Init(skinConfigPath, skinDataPath, tongueConfigPath, tongueDataPath)) {
        LOG_ERROR("Unable to read regression blendshape solve model info");
        return nullptr;
    }
    return blendshapeSolveModelInfo.release();
}

IDiffusionModel::IBlendshapeSolveModelInfo* ReadDiffusionBlendshapeSolveModelInfo_INTERNAL(const char* path) {
    LOG_DEBUG("IDiffusionModel::ReadBlendshapeSolveModelInfo()");
    auto blendshapeSolveModelInfo = std::make_unique<IDiffusionModel::BlendshapeSolveModelInfoOwner>();
    if (blendshapeSolveModelInfo->Init(path)) {
        LOG_ERROR("Unable to read diffusion blendshape solve model info");
        return nullptr;
    }
    return blendshapeSolveModelInfo.release();
}

IDiffusionModel::IBlendshapeSolveModelInfo* ReadDiffusionBlendshapeSolveModelInfo_INTERNAL(
    std::size_t identityCount,
    const char* const* skinConfigPaths,
    const char* const* skinDataPaths,
    const char* const* tongueConfigPaths,
    const char* const* tongueDataPaths
    ) {
    LOG_DEBUG("IDiffusionModel::ReadBlendshapeSolveModelInfo()");
    auto blendshapeSolveModelInfo = std::make_unique<IDiffusionModel::BlendshapeSolveModelInfoOwner>();
    if (blendshapeSolveModelInfo->Init(identityCount, skinConfigPaths, skinDataPaths, tongueConfigPaths, tongueDataPaths)) {
        LOG_ERROR("Unable to read diffusion blendshape solve model info");
        return nullptr;
    }
    return blendshapeSolveModelInfo.release();
}

IGeometryExecutorBundle* ReadRegressionGeometryExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IRegressionModel::IGeometryModelInfo** outModelInfo
    ) {
    LOG_DEBUG("ReadRegressionGeometryExecutorBundle()");
    auto geometryExecutorBundle = std::make_unique<GeometryExecutorBundle>();
    if (geometryExecutorBundle->InitRegression(
        nbTracks, path, executionOption, frameRateNumerator, frameRateDenominator, outModelInfo
        )) {
        LOG_ERROR("Unable to read regression geometry executor bundle");
        return nullptr;
    }
    return geometryExecutorBundle.release();
}

IGeometryExecutorBundle* ReadDiffusionGeometryExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t identityIndex,
    bool constantNoise,
    IDiffusionModel::IGeometryModelInfo** outModelInfo
    ) {
    LOG_DEBUG("ReadDiffusionGeometryExecutorBundle()");
    auto geometryExecutorBundle = std::make_unique<GeometryExecutorBundle>();
    if (geometryExecutorBundle->InitDiffusion(
        nbTracks, path, executionOption, identityIndex, constantNoise, outModelInfo
        )) {
        LOG_ERROR("Unable to read diffusion geometry executor bundle");
        return nullptr;
    }
    return geometryExecutorBundle.release();
}

IBlendshapeExecutorBundle* ReadRegressionBlendshapeSolveExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    bool useGpuSolver,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IRegressionModel::IGeometryModelInfo** outModelInfo,
    IRegressionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
    ) {
    LOG_DEBUG("ReadRegressionBlendshapeSolveExecutorBundle()");
    auto blendshapeSolveExecutorBundle = std::make_unique<BlendshapeSolveExecutorBundle>();
    if (blendshapeSolveExecutorBundle->InitRegression(
        nbTracks, path, executionOption, useGpuSolver, frameRateNumerator, frameRateDenominator, outModelInfo, outBlendshapeSolveModelInfo
        )) {
        LOG_ERROR("Unable to read regression blendshape executor bundle");
        return nullptr;
    }
    return blendshapeSolveExecutorBundle.release();
}

IBlendshapeExecutorBundle* ReadDiffusionBlendshapeSolveExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    bool useGpuSolver,
    std::size_t identityIndex,
    bool constantNoise,
    IDiffusionModel::IGeometryModelInfo** outModelInfo,
    IDiffusionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
    ) {
    LOG_DEBUG("ReadDiffusionBlendshapeSolveExecutorBundle()");
    auto blendshapeSolveExecutorBundle = std::make_unique<BlendshapeSolveExecutorBundle>();
    if (blendshapeSolveExecutorBundle->InitDiffusion(
        nbTracks, path, executionOption, useGpuSolver, identityIndex, constantNoise, outModelInfo, outBlendshapeSolveModelInfo
        )) {
        LOG_ERROR("Unable to read diffusion blendshape executor bundle");
        return nullptr;
    }
    return blendshapeSolveExecutorBundle.release();
}

} // namespace nva2f
