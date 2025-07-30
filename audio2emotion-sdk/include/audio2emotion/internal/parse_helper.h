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

#include "audio2emotion/parse_helper.h"
#include "audio2x/internal/io.h"
#include "audio2x/internal/cuda_stream.h"
#include "audio2x/internal/audio_accumulator.h"
#include "audio2x/internal/emotion_accumulator.h"
#include "audio2x/internal/unique_ptr.h"

#include <string>
#include <vector>

namespace nva2e {

namespace IClassifierModel {

class NetworkInfoOwner : public INetworkInfo {
public:
    NetworkInfo GetNetworkInfo(std::size_t bufferLength) const override;
    std::size_t GetEmotionsCount() const override;
    const char* GetEmotionName(std::size_t index) const override;

    void Destroy() override;

    std::error_code Init(const char* path);

private:
    NetworkInfo _networkInfo;
    std::vector<std::string> _emotions;
};

class ConfigInfoOwner : public IConfigInfo {
public:
    const PostProcessData& GetPostProcessData() const override;
    const PostProcessParams& GetPostProcessParams() const override;
    float GetInputStrength() const override;

    void Destroy() override;

    std::error_code Init(const char* path, std::size_t emotionCount, const char* const emotionNames[]);

private:
    PostProcessData _postProcessData;
    PostProcessParams _postProcessParams;
    float _inputStrength;

    std::vector<int> _emotionCorrespondence;
    std::vector<float> _preferredEmotion;
    std::vector<float> _beginningEmotion;
};

class ModelInfoOwner : public IEmotionModelInfo {
public:
    const INetworkInfo& GetNetworkInfo() const override;
    const IConfigInfo& GetConfigInfo() const override;
    EmotionExecutorCreationParameters GetExecutorCreationParameters(
        std::size_t bufferLength,
        std::size_t frameRateNumerator,
        std::size_t frameRateDenominator,
        std::size_t inferencesToSkip
        ) const override;
    void Destroy() override;

    std::error_code Init(const char* modelPath);
    std::error_code Init(
        const char* networkInfoPath,
        const char* networkPath,
        const char* modelConfigPath
        );

private:
    NetworkInfoOwner _networkInfo;
    nva2x::DataBytes _networkData;
    ConfigInfoOwner _configInfo;
};

} // namespace IClassifierModel


namespace IPostProcessModel {

using IClassifierModel::NetworkInfoOwner;
using IClassifierModel::ConfigInfoOwner;

class ModelInfoOwner : public IEmotionModelInfo {
public:
    const INetworkInfo& GetNetworkInfo() const override;
    const IConfigInfo& GetConfigInfo() const override;
    EmotionExecutorCreationParameters GetExecutorCreationParameters(
        std::size_t frameRateNumerator,
        std::size_t frameRateDenominator
        ) const override;
    void Destroy() override;

    std::error_code Init(const char* modelPath);
    std::error_code Init(
        const char* networkInfoPath,
        const char* modelConfigPath
        );

private:
    NetworkInfoOwner _networkInfo;
    ConfigInfoOwner _configInfo;
};

} // namespace IPostProcessModel


class EmotionExecutorBundle : public IEmotionExecutorBundle {
public:
    nva2x::ICudaStream& GetCudaStream() override;
    const nva2x::ICudaStream& GetCudaStream() const override;

    nva2x::IAudioAccumulator& GetAudioAccumulator(std::size_t trackIndex) override;
    const nva2x::IAudioAccumulator& GetAudioAccumulator(std::size_t trackIndex) const override;

    nva2x::IEmotionAccumulator& GetPreferredEmotionAccumulator(std::size_t trackIndex) override;
    const nva2x::IEmotionAccumulator& GetPreferredEmotionAccumulator(std::size_t trackIndex) const override;

    IEmotionExecutor& GetExecutor() override;
    const IEmotionExecutor& GetExecutor() const override;

    void Destroy() override;

    std::error_code InitClassifier(
        std::size_t nbTracks,
        const char* path,
        std::size_t bufferLength, std::size_t frameRateNumerator, std::size_t frameRateDenominator,
        std::size_t inferencesToSkip,
        IClassifierModel::IEmotionModelInfo** outModelInfo
        );
    std::error_code InitPostProcess(
        std::size_t nbTracks,
        const char* path,
        std::size_t frameRateNumerator, std::size_t frameRateDenominator,
        IPostProcessModel::IEmotionModelInfo** outModelInfo
        );

private:
    nva2x::CudaStream _cudaStream;
    std::vector<std::unique_ptr<nva2x::AudioAccumulator>> _audioAccumulators;
    std::vector<std::unique_ptr<nva2x::EmotionAccumulator>> _preferredEmotionAccumulators;
    nva2x::UniquePtr<IEmotionExecutor> _emotionExecutor;
};


IClassifierModel::INetworkInfo* ReadClassifierNetworkInfo_INTERNAL(const char* path);
IClassifierModel::IConfigInfo* ReadClassifierConfigInfo_INTERNAL(
    const char* path, std::size_t emotionCount, const char* const emotionNames[]
    );
IClassifierModel::IEmotionModelInfo* ReadClassifierModelInfo_INTERNAL(const char* path);
IClassifierModel::IEmotionModelInfo* ReadClassifierModelInfo_INTERNAL(
    const char* networkInfoPath, const char* networkPath, const char* modelConfigPath
    );

IEmotionExecutorBundle* ReadClassifierEmotionExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    std::size_t bufferLength, std::size_t frameRateNumerator, std::size_t frameRateDenominator,
    std::size_t inferencesToSkip,
    IClassifierModel::IEmotionModelInfo** outModelInfo
    );

IPostProcessModel::INetworkInfo* ReadPostProcessNetworkInfo_INTERNAL(const char* path);
IPostProcessModel::IConfigInfo* ReadPostProcessConfigInfo_INTERNAL(
    const char* path, std::size_t emotionCount, const char* const emotionNames[]
    );
IPostProcessModel::IEmotionModelInfo* ReadPostProcessModelInfo_INTERNAL(const char* path);
IPostProcessModel::IEmotionModelInfo* ReadPostProcessModelInfo_INTERNAL(
    const char* networkInfoPath, const char* modelConfigPath
    );

IEmotionExecutorBundle* ReadPostProcessEmotionExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    std::size_t frameRateNumerator, std::size_t frameRateDenominator,
    IPostProcessModel::IEmotionModelInfo** outModelInfo
    );

} // namespace nva2e
