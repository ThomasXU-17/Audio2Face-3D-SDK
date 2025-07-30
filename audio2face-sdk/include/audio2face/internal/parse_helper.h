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
#pragma once

#include "audio2face/parse_helper.h"
#include "audio2face/internal/emotion.h"
#include "audio2x/internal/tensor.h"
#include "audio2x/internal/io.h"
#include "audio2x/internal/cuda_stream.h"
#include "audio2x/internal/audio_accumulator.h"
#include "audio2x/internal/emotion_accumulator.h"
#include "audio2x/internal/unique_ptr.h"

#include <string>
#include <vector>

namespace nva2f {

std::error_code ReadAnimatorParams_INTERNAL(const char* path, float& inputStrength, AnimatorParams& params);


namespace IRegressionModel {

class NetworkInfoOwner : public INetworkInfo {
public:
    const NetworkInfo& GetNetworkInfo() const override;
    std::size_t GetEmotionsCount() const override;
    const char* GetEmotionName(std::size_t index) const override;
    nva2x::HostTensorFloatConstView GetDefaultEmotion() const override;
    const char* GetIdentityName() const override;

    void Destroy() override;

    std::error_code Init(const char* path);

private:
    NetworkInfo _networkInfo;
    std::vector<std::string> _emotions;
    std::vector<float> _defaultEmotion;
    std::string _identity;
};

class AnimatorDataOwner : public IAnimatorData {
public:
    AnimatorDataView GetAnimatorData() const override;
    IAnimatorPcaReconstruction::HostData GetSkinPcaReconstructionData() const override;
    IAnimatorPcaReconstruction::HostData GetTonguePcaReconstructionData() const override;
    void Destroy() override;

    std::error_code Init(const char* path);

private:
    nva2x::HostTensorFloat _shapesMatrixSkin;
    nva2x::HostTensorFloat _shapesMeanSkin;
    nva2x::HostTensorFloat _lipOpenPoseDelta;
    nva2x::HostTensorFloat _eyeClosePoseDelta;
    nva2x::HostTensorFloat _shapesMatrixTongue;
    nva2x::HostTensorFloat _shapesMeanTongue;
    nva2x::HostTensorFloat _neutralJaw;
    nva2x::HostTensorFloat _saccadeRot;

};

class ModelInfoOwner : public IGeometryModelInfo {
public:
    const INetworkInfo& GetNetworkInfo() const override;
    const IAnimatorData& GetAnimatorData() const override;
    const AnimatorParams& GetAnimatorParams() const override;

    GeometryExecutorCreationParameters GetExecutorCreationParameters(
        IGeometryExecutor::ExecutionOption executionOption,
        std::size_t frameRateNumerator,
        std::size_t frameRateDenominator
        ) const override;
    void Destroy() override;

    std::error_code Init(const char* modelPath);
    std::error_code Init(
        const char* networkInfoPath,
        const char* networkPath,
        const char* emotionDatabasePath,
        const char* modelConfigPath,
        const char* modelDataPath
        );

private:
    NetworkInfoOwner _networkInfo;
    nva2x::DataBytes _networkData;
    EmotionDatabase _emotionDatabase;
    float _inputStrength;
    AnimatorParams _animatorParams;
    AnimatorDataOwner _animatorData;
    std::string _sourceShot;
    std::size_t _sourceFrame;

    GeometryExecutorCreationParameters::SkinParameters _skinParams;
    GeometryExecutorCreationParameters::TongueParameters _tongueParams;
    GeometryExecutorCreationParameters::TeethParameters _teethParams;
    GeometryExecutorCreationParameters::EyesParameters _eyesParams;
};

} // namespace IRegressionModel


namespace IDiffusionModel {

class NetworkInfoOwner : public INetworkInfo {
public:
    const NetworkInfo& GetNetworkInfo() const override;
    std::size_t GetEmotionsCount() const override;
    const char* GetEmotionName(std::size_t index) const override;
    nva2x::HostTensorFloatConstView GetDefaultEmotion() const override;
    std::size_t GetIdentityLength() const override;
    const char* GetIdentityName(std::size_t index) const override;

    void Destroy() override;

    std::error_code Init(const char* path);

private:
    NetworkInfo _networkInfo;
    std::vector<std::string> _emotions;
    std::vector<float> _defaultEmotion;
    std::vector<std::string> _identities;
};

class AnimatorDataOwner : public IAnimatorData {
public:
    AnimatorDataView GetAnimatorData() const override;
    void Destroy() override;

    std::error_code Init(const char* path);

private:
    nva2x::HostTensorFloat _neutralSkin;
    nva2x::HostTensorFloat _lipOpenPoseDelta;
    nva2x::HostTensorFloat _eyeClosePoseDelta;
    nva2x::HostTensorFloat _neutralTongue;
    nva2x::HostTensorFloat _neutralJaw;
    nva2x::HostTensorFloat _saccadeRot;
};

class ModelInfoOwner : public IGeometryModelInfo {
public:
    const INetworkInfo& GetNetworkInfo() const override;
    const IAnimatorData* GetAnimatorData(std::size_t identityIndex) const override;
    const AnimatorParams* GetAnimatorParams(std::size_t identityIndex) const override;

    GeometryExecutorCreationParameters GetExecutorCreationParameters(
        IGeometryExecutor::ExecutionOption executionOption,
        std::size_t identityIndex,
        bool constantNoise
        ) const override;
    void Destroy() override;

    std::error_code Init(const char* modelPath);
    std::error_code Init(
        const char* networkInfoPath,
        const char* networkPath,
        std::size_t identityCount,
        const char* const* modelConfigPaths,
        const char* const* modelDataPaths
        );

private:
    NetworkInfoOwner _networkInfo;
    nva2x::DataBytes _networkData;
    struct IdentityData {
        float inputStrength;
        AnimatorParams defaultAnimatorParams;
        AnimatorDataOwner animatorData;
    };
    std::vector<IdentityData> _identityData;

    std::vector<GeometryExecutorCreationParameters::SkinParameters> _skinParams;
    std::vector<GeometryExecutorCreationParameters::TongueParameters> _tongueParams;
    std::vector<GeometryExecutorCreationParameters::TeethParameters> _teethParams;
    std::vector<GeometryExecutorCreationParameters::EyesParameters> _eyesParams;
};

}  // namespace IDiffusionModel


class BlendshapeSolverConfigOwner : public IBlendshapeSolverConfig {
public:
    const BlendshapeSolverParams& GetBlendshapeSolverParams() const override;
    BlendshapeSolverConfig GetBlendshapeSolverConfig() const override;
    void Destroy() override;

    std::error_code Init(const char* configPath);

private:
    BlendshapeSolverParams _blendshapeSolverParams;

    std::vector<int> _activePoses;
    std::vector<int> _cancelPoses;
    std::vector<int> _symmetryPoses;
    std::vector<float> _multipliers;
    std::vector<float> _offsets;
};

class BlendshapeSolverDataOwner : public IBlendshapeSolverData {
public:
    BlendshapeSolverDataView GetBlendshapeSolverDataView() const override;
    void Destroy() override;

    std::error_code Init(const char* dataPath);

private:
    std::vector<float> _neutralPose;
    std::vector<float> _deltaPoses;
    std::vector<int> _poseMask;
    std::vector<std::string> _poseNames;
    std::vector<const char*> _poseNamesCstr;
};


namespace IRegressionModel {

class BlendshapeSolveModelInfoOwner : public IBlendshapeSolveModelInfo {
public:
  BlendshapeSolveExecutorCreationParameters GetExecutorCreationParameters(
      IGeometryExecutor::ExecutionOption executionOption
      ) const override;
  void Destroy() override;

    std::error_code Init(const char* modelPath);
    std::error_code Init(
        const char* skinConfigPath,
        const char* skinDataPath,
        const char* tongueConfigPath,
        const char* tongueDataPath
    );

private:
    BlendshapeSolverConfigOwner _skinConfig;
    BlendshapeSolverDataOwner _skinData;
    BlendshapeSolverConfigOwner _tongueConfig;
    BlendshapeSolverDataOwner _tongueData;

    BlendshapeSolveExecutorCreationParameters::BlendshapeParams _skinParams;
    BlendshapeSolveExecutorCreationParameters::BlendshapeParams _tongueParams;
};

} // namespace IRegressionModel


namespace IDiffusionModel {

class BlendshapeSolveModelInfoOwner : public IBlendshapeSolveModelInfo {
public:
  BlendshapeSolveExecutorCreationParameters GetExecutorCreationParameters(
      IGeometryExecutor::ExecutionOption executionOption,
      std::size_t identityIndex
      ) const override;
  void Destroy() override;

    std::error_code Init(const char* modelPath);
    std::error_code Init(
        std::size_t identityCount,
        const char* const* skinConfigPaths,
        const char* const* skinDataPaths,
        const char* const* tongueConfigPaths,
        const char* const* tongueDataPaths
    );

private:
    struct IdentityData {
        BlendshapeSolverConfigOwner skinConfig;
        BlendshapeSolverDataOwner skinData;
        BlendshapeSolverConfigOwner tongueConfig;
        BlendshapeSolverDataOwner tongueData;
    };
    std::vector<IdentityData> _identityData;

    std::vector<BlendshapeSolveExecutorCreationParameters::BlendshapeParams> _skinParams;
    std::vector<BlendshapeSolveExecutorCreationParameters::BlendshapeParams> _tongueParams;
};

} // namespace IDiffusionModel


class GeometryExecutorBundle : public IGeometryExecutorBundle {
public:
    nva2x::ICudaStream& GetCudaStream() override;
    const nva2x::ICudaStream& GetCudaStream() const override;

    nva2x::IAudioAccumulator& GetAudioAccumulator(std::size_t trackIndex) override;
    const nva2x::IAudioAccumulator& GetAudioAccumulator(std::size_t trackIndex) const override;

    nva2x::IEmotionAccumulator& GetEmotionAccumulator(std::size_t trackIndex) override;
    const nva2x::IEmotionAccumulator& GetEmotionAccumulator(std::size_t trackIndex) const override;

    IGeometryExecutor& GetExecutor() override;
    const IGeometryExecutor& GetExecutor() const override;

    void Destroy() override;

    std::error_code InitRegression(
        std::size_t nbTracks,
        const char* path,
        IGeometryExecutor::ExecutionOption executionOption,
        std::size_t frameRateNumerator,
        std::size_t frameRateDenominator,
        IRegressionModel::IGeometryModelInfo** outModelInfo
        );
    std::error_code InitDiffusion(
        std::size_t nbTracks,
        const char* path,
        IGeometryExecutor::ExecutionOption executionOption,
        std::size_t identityIndex,
        bool constantNoise,
        IDiffusionModel::IGeometryModelInfo** outModelInfo
        );

private:
    std::error_code InitBase(std::size_t nbTracks, std::size_t emotionsCount);

    nva2x::CudaStream _cudaStream;
    std::vector<std::unique_ptr<nva2x::AudioAccumulator>> _audioAccumulators;
    std::vector<std::unique_ptr<nva2x::EmotionAccumulator>> _emotionAccumulators;
    nva2x::UniquePtr<IGeometryExecutor> _geometryExecutor;
};

class BlendshapeSolveExecutorBundle : public IBlendshapeExecutorBundle {
public:
    nva2x::ICudaStream& GetCudaStream() override;
    const nva2x::ICudaStream& GetCudaStream() const override;

    nva2x::IAudioAccumulator& GetAudioAccumulator(std::size_t trackIndex) override;
    const nva2x::IAudioAccumulator& GetAudioAccumulator(std::size_t trackIndex) const override;

    nva2x::IEmotionAccumulator& GetEmotionAccumulator(std::size_t trackIndex) override;
    const nva2x::IEmotionAccumulator& GetEmotionAccumulator(std::size_t trackIndex) const override;

    IBlendshapeExecutor& GetExecutor() override;
    const IBlendshapeExecutor& GetExecutor() const override;

    void Destroy() override;

    std::error_code InitRegression(
        std::size_t nbTracks,
        const char* path,
        IGeometryExecutor::ExecutionOption executionOption,
        bool useGpuSolver,
        std::size_t frameRateNumerator,
        std::size_t frameRateDenominator,
        IRegressionModel::IGeometryModelInfo** outModelInfo,
        IRegressionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
        );
    std::error_code InitDiffusion(
        std::size_t nbTracks,
        const char* path,
        IGeometryExecutor::ExecutionOption executionOption,
        bool useGpuSolver,
        std::size_t identityIndex,
        bool constantNoise,
        IDiffusionModel::IGeometryModelInfo** outModelInfo,
        IDiffusionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
        );

private:
    std::error_code InitBase(std::size_t nbTracks, std::size_t emotionsCount);
    std::error_code InitExecutor(
        nva2x::UniquePtr<IGeometryExecutor> geometryExecutor,
        const BlendshapeSolveExecutorCreationParameters& creationParams,
        bool useGpuSolver
        );

    nva2x::CudaStream _cudaStream;
    std::vector<std::unique_ptr<nva2x::AudioAccumulator>> _audioAccumulators;
    std::vector<std::unique_ptr<nva2x::EmotionAccumulator>> _emotionAccumulators;
    nva2x::UniquePtr<IBlendshapeExecutor> _blendshapeExecutor;
};


IRegressionModel::INetworkInfo* ReadRegressionNetworkInfo_INTERNAL(const char* path);
IRegressionModel::IAnimatorData* ReadRegressionAnimatorData_INTERNAL(const char* path);

IRegressionModel::IGeometryModelInfo* ReadRegressionModelInfo_INTERNAL(const char* path);
IRegressionModel::IGeometryModelInfo* ReadRegressionModelInfo_INTERNAL(
    const char* networkInfoPath,
    const char* networkPath,
    const char* emotionDatabasePath,
    const char* modelConfigPath,
    const char* modelDataPath
    );

IDiffusionModel::INetworkInfo* ReadDiffusionNetworkInfo_INTERNAL(const char* path);
IDiffusionModel::IAnimatorData* ReadDiffusionAnimatorData_INTERNAL(const char* path);

IDiffusionModel::IGeometryModelInfo* ReadDiffusionModelInfo_INTERNAL(const char* path);
IDiffusionModel::IGeometryModelInfo* ReadDiffusionModelInfo_INTERNAL(
    const char* networkInfoPath,
    const char* networkPath,
    std::size_t identityCount,
    const char* const* modelConfigPaths,
    const char* const* modelDataPaths
    );

IBlendshapeSolverConfig* ReadBlendshapeSolverConfig_INTERNAL(const char* configPath);
IBlendshapeSolverData* ReadBlendshapeSolverData_INTERNAL(const char* dataPath);

IRegressionModel::IBlendshapeSolveModelInfo* ReadRegressionBlendshapeSolveModelInfo_INTERNAL(const char* path);
IRegressionModel::IBlendshapeSolveModelInfo* ReadRegressionBlendshapeSolveModelInfo_INTERNAL(
    const char* skinConfigPath,
    const char* skinDataPath,
    const char* tongueConfigPath,
    const char* tongueDataPath
    );

IDiffusionModel::IBlendshapeSolveModelInfo* ReadDiffusionBlendshapeSolveModelInfo_INTERNAL(const char* path);
IDiffusionModel::IBlendshapeSolveModelInfo* ReadDiffusionBlendshapeSolveModelInfo_INTERNAL(
    std::size_t identityCount,
    const char* const* skinConfigPaths,
    const char* const* skinDataPaths,
    const char* const* tongueConfigPaths,
    const char* const* tongueDataPaths
    );

IGeometryExecutorBundle* ReadRegressionGeometryExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IRegressionModel::IGeometryModelInfo** outModelInfo
    );
IGeometryExecutorBundle* ReadDiffusionGeometryExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t identityIndex,
    bool constantNoise,
    IDiffusionModel::IGeometryModelInfo** outModelInfo
    );

IBlendshapeExecutorBundle* ReadRegressionBlendshapeSolveExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    bool useGpuSolver,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IRegressionModel::IGeometryModelInfo** outModelInfo,
    IRegressionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
    );
IBlendshapeExecutorBundle* ReadDiffusionBlendshapeSolveExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    bool useGpuSolver,
    std::size_t identityIndex,
    bool constantNoise,
    IDiffusionModel::IGeometryModelInfo** outModelInfo,
    IDiffusionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
    );

} // namespace nva2f
