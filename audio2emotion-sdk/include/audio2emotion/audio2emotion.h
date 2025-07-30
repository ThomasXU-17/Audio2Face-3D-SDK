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

#include "audio2emotion/dll_export.h"
#include "audio2emotion/executor_classifier.h"
#include "audio2emotion/executor_postprocess.h"
#include "audio2emotion/executor.h"
#include "audio2emotion/interactive_executor.h"
#include "audio2emotion/model.h"
#include "audio2emotion/parse_helper.h"
#include "audio2emotion/postprocess.h"
#include "audio2x/inference_engine.h"
#include "audio2x/emotion_accumulator.h"

namespace nva2e {

// Return the buffer bindings description for the classifier model.
// This provides information about the input/output buffer requirements for the emotion classifier.
AUDIO2EMOTION_DLL_API const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForClassifierModel();

// Create buffer bindings for the classifier model.
// It returns a pointer to the created buffer bindings that can be used for inference.
AUDIO2EMOTION_DLL_API nva2x::IBufferBindings* CreateBindingsForClassifierModel();

// Create a post-processor for emotion data.
// It returns a pointer to the created post-processor that can be used to process emotion inference results.
AUDIO2EMOTION_DLL_API IPostProcessor* CreatePostProcessor();

// Get the input strength parameter from an emotion executor (common to all tracks).
AUDIO2EMOTION_DLL_API std::error_code GetExecutorInputStrength(
    const IEmotionExecutor& executor, float& inputStrength
    );

// Set the input strength parameter for an emotion executor (common to all tracks).
AUDIO2EMOTION_DLL_API std::error_code SetExecutorInputStrength(
    IEmotionExecutor& executor, float inputStrength
    );

// Get the post-process parameters for a specific track in an emotion executor.
AUDIO2EMOTION_DLL_API std::error_code GetExecutorPostProcessParameters(
    const IEmotionExecutor& executor, std::size_t trackIndex, PostProcessParams& params
    );

// Set the post-process parameters for a specific track in an emotion executor.
AUDIO2EMOTION_DLL_API std::error_code SetExecutorPostProcessParameters(
    IEmotionExecutor& executor, std::size_t trackIndex, const PostProcessParams& params
    );

// Compare two PostProcessParams objects for equality.
AUDIO2EMOTION_DLL_API bool AreEqual(const PostProcessParams& a, const PostProcessParams& b);

// Create a classifier-based emotion executor.
AUDIO2EMOTION_DLL_API IEmotionExecutor* CreateClassifierEmotionExecutor(
    const EmotionExecutorCreationParameters& params,
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams
    );

// Create a post-process emotion executor using classifier parameters.
// This executor does not run inference, it only runs post-processing.
AUDIO2EMOTION_DLL_API IEmotionExecutor* CreatePostProcessEmotionExecutor(
    const EmotionExecutorCreationParameters& params,
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams
    );

// Create a post-process emotion executor using post-process model parameters.
// This version of the function does not require a valid inference model.
// This executor does not run inference, it only runs post-processing.
AUDIO2EMOTION_DLL_API IEmotionExecutor* CreatePostProcessEmotionExecutor(
    const EmotionExecutorCreationParameters& params,
    const IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
    );

// Create an emotion binder that connects an executor to emotion accumulators.
AUDIO2EMOTION_DLL_API IEmotionBinder* CreateEmotionBinder(
    IEmotionExecutor& executor,
    nva2x::IEmotionAccumulator* const* emotionAccumulators,
    std::size_t nbEmotionAccumulators
    );

// Get the number of inferences to skip for an interactive executor.
AUDIO2EMOTION_DLL_API std::error_code GetInteractiveExecutorInferencesToSkip(
    const IEmotionInteractiveExecutor& executor, std::size_t& inferencesToSkip
    );

// Set the number of inferences to skip for an interactive executor.
AUDIO2EMOTION_DLL_API std::error_code SetInteractiveExecutorInferencesToSkip(
    IEmotionInteractiveExecutor& executor, std::size_t inferencesToSkip
    );

// Get the input strength parameter from an interactive emotion executor.
AUDIO2EMOTION_DLL_API std::error_code GetInteractiveExecutorInputStrength(
    const IEmotionInteractiveExecutor& executor, float& inputStrength
    );

// Set the input strength parameter for an interactive emotion executor.
AUDIO2EMOTION_DLL_API std::error_code SetInteractiveExecutorInputStrength(
    IEmotionInteractiveExecutor& executor, float inputStrength
    );

// Get the post-process parameters for an interactive emotion executor.
AUDIO2EMOTION_DLL_API std::error_code GetInteractiveExecutorPostProcessParameters(
    const IEmotionInteractiveExecutor& executor, PostProcessParams& params
    );

// Set the post-process parameters for an interactive emotion executor.
AUDIO2EMOTION_DLL_API std::error_code SetInteractiveExecutorPostProcessParameters(
    IEmotionInteractiveExecutor& executor, const PostProcessParams& params
    );

// Create a classifier-based interactive emotion executor.
AUDIO2EMOTION_DLL_API IEmotionInteractiveExecutor* CreateClassifierEmotionInteractiveExecutor(
    const EmotionExecutorCreationParameters& params,
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    std::size_t batchSize
    );

// Create a post-process interactive emotion executor using classifier parameters.
// This executor does not run inference, it only runs post-processing.
AUDIO2EMOTION_DLL_API IEmotionInteractiveExecutor* CreatePostProcessEmotionInteractiveExecutor(
    const EmotionExecutorCreationParameters& params,
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    std::size_t batchSize
    );

// Create a post-process interactive emotion executor using post-process model parameters.
// This version of the function does not require a valid inference model.
// This executor does not run inference, it only runs post-processing.
AUDIO2EMOTION_DLL_API IEmotionInteractiveExecutor* CreatePostProcessEmotionInteractiveExecutor(
    const EmotionExecutorCreationParameters& params,
    const IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams
    );

// Read classifier network information from a file.
AUDIO2EMOTION_DLL_API IClassifierModel::INetworkInfo* ReadClassifierNetworkInfo(const char* path);

// Read classifier configuration information from a file.
AUDIO2EMOTION_DLL_API IClassifierModel::IConfigInfo* ReadClassifierConfigInfo(
    const char* path, std::size_t emotionCount, const char* const emotionNames[]
    );

// Read classifier model information from a single model.json file.
AUDIO2EMOTION_DLL_API IClassifierModel::IEmotionModelInfo* ReadClassifierModelInfo(const char* path);

// Read classifier model information from individual files.
AUDIO2EMOTION_DLL_API IClassifierModel::IEmotionModelInfo* ReadClassifierModelInfo(
    const char* networkInfoPath, const char* networkPath, const char* modelConfigPath
    );

// Read post-process network information from a file.
AUDIO2EMOTION_DLL_API IPostProcessModel::INetworkInfo* ReadPostProcessNetworkInfo(const char* path);

// Read post-process configuration information from a file.
AUDIO2EMOTION_DLL_API IPostProcessModel::IConfigInfo* ReadPostProcessConfigInfo(
    const char* path, std::size_t emotionCount, const char* const emotionNames[]
    );

// Read post-process model information from a single model.json file.
AUDIO2EMOTION_DLL_API IPostProcessModel::IEmotionModelInfo* ReadPostProcessModelInfo(const char* path);

// Read post-process model information from individual files.
AUDIO2EMOTION_DLL_API IPostProcessModel::IEmotionModelInfo* ReadPostProcessModelInfo(
    const char* networkInfoPath, const char* modelConfigPath
    );

// Read a complete classifier emotion executor bundle from a single model.json file.
AUDIO2EMOTION_DLL_API IEmotionExecutorBundle* ReadClassifierEmotionExecutorBundle(
    std::size_t nbTracks,
    const char* path,
    std::size_t bufferLength, std::size_t frameRateNumerator, std::size_t frameRateDenominator,
    std::size_t inferencesToSkip,
    IClassifierModel::IEmotionModelInfo** outModelInfo
    );

// Read a complete post-process emotion executor bundle from a single model.json file.
AUDIO2EMOTION_DLL_API IEmotionExecutorBundle* ReadPostProcessEmotionExecutorBundle(
    std::size_t nbTracks,
    const char* path,
    std::size_t frameRateNumerator, std::size_t frameRateDenominator,
    IPostProcessModel::IEmotionModelInfo** outModelInfo
    );

} // namespace nva2e
