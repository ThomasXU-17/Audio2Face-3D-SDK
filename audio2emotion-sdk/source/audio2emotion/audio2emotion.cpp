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

namespace nva2e {

// Implementations are inside audio2emotion-core.lib
const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForClassifierModel_INTERNAL();
nva2x::IBufferBindings* CreateBindingsForClassifierModel_INTERNAL();

IPostProcessor* CreatePostProcessor_INTERNAL();

std::error_code GetExecutorInputStrength_INTERNAL(
    const IEmotionExecutor& executor, float& inputStrength
    );
std::error_code SetExecutorInputStrength_INTERNAL(
    IEmotionExecutor& executor, float inputStrength
    );
std::error_code GetExecutorPostProcessParameters_INTERNAL(
    const IEmotionExecutor& executor, std::size_t trackIndex, PostProcessParams& params
    );
std::error_code SetExecutorPostProcessParameters_INTERNAL(
    IEmotionExecutor& executor, std::size_t trackIndex, const PostProcessParams& params
    );

bool AreEqual_INTERNAL(const PostProcessParams& a, const PostProcessParams& b);

IEmotionExecutor* CreateClassifierEmotionExecutor_INTERNAL(
    const EmotionExecutorCreationParameters& params,
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams
    );

IEmotionExecutor* CreatePostProcessEmotionExecutor_INTERNAL(
    const EmotionExecutorCreationParameters& params,
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams
    );
IEmotionExecutor* CreatePostProcessEmotionExecutor_INTERNAL(
    const EmotionExecutorCreationParameters& params,
    const IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
    );

IEmotionBinder* CreateEmotionBinder_INTERNAL(
    IEmotionExecutor& executor,
    nva2x::IEmotionAccumulator* const* emotionAccumulators,
    std::size_t nbEmotionAccumulators
    );

std::error_code GetInteractiveExecutorInferencesToSkip_INTERNAL(
    const IEmotionInteractiveExecutor& executor, std::size_t& inferencesToSkip
    );
std::error_code SetInteractiveExecutorInferencesToSkip_INTERNAL(
    IEmotionInteractiveExecutor& executor, std::size_t inferencesToSkip
    );
std::error_code GetInteractiveExecutorInputStrength_INTERNAL(
    const IEmotionInteractiveExecutor& executor, float& inputStrength
    );
std::error_code SetInteractiveExecutorInputStrength_INTERNAL(
    IEmotionInteractiveExecutor& executor, float inputStrength
    );
std::error_code GetInteractiveExecutorPostProcessParameters_INTERNAL(
    const IEmotionInteractiveExecutor& executor, PostProcessParams& params
    );
std::error_code SetInteractiveExecutorPostProcessParameters_INTERNAL(
    IEmotionInteractiveExecutor& executor, const PostProcessParams& params
    );

IEmotionInteractiveExecutor* CreateClassifierEmotionInteractiveExecutor_INTERNAL(
    const EmotionExecutorCreationParameters& params,
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    std::size_t batchSize
    );

IEmotionInteractiveExecutor* CreatePostProcessEmotionInteractiveExecutor_INTERNAL(
    const EmotionExecutorCreationParameters& params,
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    std::size_t batchSize
    );
IEmotionInteractiveExecutor* CreatePostProcessEmotionInteractiveExecutor_INTERNAL(
    const EmotionExecutorCreationParameters& params,
    const IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
    );

IClassifierModel::INetworkInfo* ReadClassifierNetworkInfo_INTERNAL(const char* path);
IClassifierModel::IConfigInfo* ReadClassifierConfigInfo_INTERNAL(
    const char* path, std::size_t emotionCount, const char* const emotionNames[]
    );
IClassifierModel::IEmotionModelInfo* ReadClassifierModelInfo_INTERNAL(const char* path);
IClassifierModel::IEmotionModelInfo* ReadClassifierModelInfo_INTERNAL(
    const char* networkInfoPath, const char* networkPath, const char* modelConfigPath
    );

IPostProcessModel::INetworkInfo* ReadPostProcessNetworkInfo_INTERNAL(const char* path);
IPostProcessModel::IConfigInfo* ReadPostProcessConfigInfo_INTERNAL(
    const char* path, std::size_t emotionCount, const char* const emotionNames[]
    );
IPostProcessModel::IEmotionModelInfo* ReadPostProcessModelInfo_INTERNAL(const char* path);
IPostProcessModel::IEmotionModelInfo* ReadPostProcessModelInfo_INTERNAL(
    const char* networkPath, const char* modelConfigPath
    );

IEmotionExecutorBundle* ReadClassifierEmotionExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    std::size_t bufferLength, std::size_t frameRateNumerator, std::size_t frameRateDenominator,
    std::size_t inferencesToSkip,
    IClassifierModel::IEmotionModelInfo** outModelInfo
    );

IEmotionExecutorBundle* ReadPostProcessEmotionExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    std::size_t frameRateNumerator, std::size_t frameRateDenominator,
    IPostProcessModel::IEmotionModelInfo** outModelInfo
    );

//////////////////////////////////////////////
const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForClassifierModel() {
    return GetBindingsDescriptionForClassifierModel_INTERNAL();
}

nva2x::IBufferBindings* CreateBindingsForClassifierModel() {
    return CreateBindingsForClassifierModel_INTERNAL();
}

IPostProcessor* CreatePostProcessor() {
    return CreatePostProcessor_INTERNAL();
}

std::error_code GetExecutorInputStrength(
    const IEmotionExecutor& executor, float& inputStrength
    ) {
    return GetExecutorInputStrength_INTERNAL(executor, inputStrength);
}

std::error_code SetExecutorInputStrength(
    IEmotionExecutor& executor, float inputStrength
    ) {
    return SetExecutorInputStrength_INTERNAL(executor, inputStrength);
}

std::error_code GetExecutorPostProcessParameters(
    const IEmotionExecutor& executor, std::size_t trackIndex, PostProcessParams& params
    ) {
    return GetExecutorPostProcessParameters_INTERNAL(executor, trackIndex, params);
}

std::error_code SetExecutorPostProcessParameters(
    IEmotionExecutor& executor, std::size_t trackIndex, const PostProcessParams& params
    ) {
    return SetExecutorPostProcessParameters_INTERNAL(executor, trackIndex, params);
}

bool AreEqual(const PostProcessParams& a, const PostProcessParams& b) {
    return AreEqual_INTERNAL(a, b);
}

IEmotionExecutor* CreateClassifierEmotionExecutor(
    const EmotionExecutorCreationParameters& params,
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams
    ) {
    return CreateClassifierEmotionExecutor_INTERNAL(params, classifierParams);
}

IEmotionExecutor* CreatePostProcessEmotionExecutor(
    const EmotionExecutorCreationParameters& params,
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams
    ) {
    return CreatePostProcessEmotionExecutor_INTERNAL(params, classifierParams);
}

IEmotionExecutor* CreatePostProcessEmotionExecutor(
    const EmotionExecutorCreationParameters& params,
    const IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
    ) {
    return CreatePostProcessEmotionExecutor_INTERNAL(params, postProcessParams);
}

IEmotionBinder* CreateEmotionBinder(
    IEmotionExecutor& executor,
    nva2x::IEmotionAccumulator* const* emotionAccumulators,
    std::size_t nbEmotionAccumulators
    ) {
    return CreateEmotionBinder_INTERNAL(executor, emotionAccumulators, nbEmotionAccumulators);
}

std::error_code GetInteractiveExecutorInferencesToSkip(
    const IEmotionInteractiveExecutor& executor, std::size_t& inferencesToSkip
    ) {
    return GetInteractiveExecutorInferencesToSkip_INTERNAL(executor, inferencesToSkip);
}

std::error_code SetInteractiveExecutorInferencesToSkip(
    IEmotionInteractiveExecutor& executor, std::size_t inferencesToSkip
    ) {
    return SetInteractiveExecutorInferencesToSkip_INTERNAL(executor, inferencesToSkip);
}

std::error_code GetInteractiveExecutorInputStrength(
    const IEmotionInteractiveExecutor& executor, float& inputStrength
    ) {
    return GetInteractiveExecutorInputStrength_INTERNAL(executor, inputStrength);
}

std::error_code SetInteractiveExecutorInputStrength(
    IEmotionInteractiveExecutor& executor, float inputStrength
    ) {
    return SetInteractiveExecutorInputStrength_INTERNAL(executor, inputStrength);
}

std::error_code GetInteractiveExecutorPostProcessParameters(
    const IEmotionInteractiveExecutor& executor, PostProcessParams& params
    ) {
    return GetInteractiveExecutorPostProcessParameters_INTERNAL(executor, params);
}

std::error_code SetInteractiveExecutorPostProcessParameters(
    IEmotionInteractiveExecutor& executor, const PostProcessParams& params
    ) {
    return SetInteractiveExecutorPostProcessParameters_INTERNAL(executor, params);
}

IEmotionInteractiveExecutor* CreateClassifierEmotionInteractiveExecutor(
    const EmotionExecutorCreationParameters& params,
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    std::size_t batchSize
    ) {
    return CreateClassifierEmotionInteractiveExecutor_INTERNAL(params, classifierParams, batchSize);
}

IEmotionInteractiveExecutor* CreatePostProcessEmotionInteractiveExecutor(
    const EmotionExecutorCreationParameters& params,
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    std::size_t batchSize
    ) {
    return CreatePostProcessEmotionInteractiveExecutor_INTERNAL(params, classifierParams, batchSize);
}

IEmotionInteractiveExecutor* CreatePostProcessEmotionInteractiveExecutor(
    const EmotionExecutorCreationParameters& params,
    const IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
    ) {
    return CreatePostProcessEmotionInteractiveExecutor_INTERNAL(params, postProcessParams);
}

IClassifierModel::INetworkInfo* ReadClassifierNetworkInfo(const char* path) {
    return ReadClassifierNetworkInfo_INTERNAL(path);
}

IClassifierModel::IConfigInfo* ReadClassifierConfigInfo(
    const char* path, std::size_t emotionCount, const char* const emotionNames[]
    ) {
    return ReadClassifierConfigInfo_INTERNAL(path, emotionCount, emotionNames);
}

IClassifierModel::IEmotionModelInfo* ReadClassifierModelInfo(const char* path) {
    return ReadClassifierModelInfo_INTERNAL(path);
}

IClassifierModel::IEmotionModelInfo* ReadClassifierModelInfo(
    const char* networkInfoPath, const char* networkPath, const char* modelConfigPath
    ) {
    return ReadClassifierModelInfo_INTERNAL(networkInfoPath, networkPath, modelConfigPath);
}

IPostProcessModel::INetworkInfo* ReadPostProcessNetworkInfo(const char* path) {
    return ReadPostProcessNetworkInfo_INTERNAL(path);
}

IPostProcessModel::IConfigInfo* ReadPostProcessConfigInfo(
    const char* path, std::size_t emotionCount, const char* const emotionNames[]
    ) {
    return ReadPostProcessConfigInfo_INTERNAL(path, emotionCount, emotionNames);
}

IPostProcessModel::IEmotionModelInfo* ReadPostProcessModelInfo(const char* path) {
    return ReadPostProcessModelInfo_INTERNAL(path);
}

IPostProcessModel::IEmotionModelInfo* ReadPostProcessModelInfo(
    const char* networkPath, const char* modelConfigPath
    ) {
    return ReadPostProcessModelInfo_INTERNAL(networkPath, modelConfigPath);
}

IEmotionExecutorBundle* ReadClassifierEmotionExecutorBundle(
    std::size_t nbTracks,
    const char* path,
    std::size_t bufferLength, std::size_t frameRateNumerator, std::size_t frameRateDenominator,
    std::size_t inferencesToSkip,
    IClassifierModel::IEmotionModelInfo** outModelInfo
    ) {
    return ReadClassifierEmotionExecutorBundle_INTERNAL(
        nbTracks,
        path,
        bufferLength, frameRateNumerator, frameRateDenominator,
        inferencesToSkip,
        outModelInfo
        );
}

IEmotionExecutorBundle* ReadPostProcessEmotionExecutorBundle(
    std::size_t nbTracks,
    const char* path,
    std::size_t frameRateNumerator, std::size_t frameRateDenominator,
    IPostProcessModel::IEmotionModelInfo** outModelInfo
    ) {
    return ReadPostProcessEmotionExecutorBundle_INTERNAL(
        nbTracks,
        path,
        frameRateNumerator, frameRateDenominator,
        outModelInfo
        );
}

} // namespace nva2e
