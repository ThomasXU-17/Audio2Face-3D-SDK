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
#include "audio2emotion/internal/executor_classifier_core.h"
#include "audio2emotion/internal/macros.h"
#include "audio2emotion/internal/model.h"
#include "audio2emotion/internal/postprocess.h"
#include "audio2x/error.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"

#include <cassert>
#include <numeric>

namespace nva2e::IClassifierModel {

std::error_code EmotionExecutorCore::ReadAudioBuffer(
    const nva2x::IAudioAccumulator& audioAccumulator, std::size_t batchIndex, std::size_t start
    ) {
    const auto audioBuffer = GetInferenceAudioBuffer(batchIndex, 1);
    A2E_CHECK_RESULT_WITH_MSG(
        audioAccumulator.Read(audioBuffer, start, _inputStrength, _cudaStream),
        "Unable to read audio buffer"
        );

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutorCore::BindBuffers(std::size_t batchSize) {
    A2E_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetInputBinding(kInputTensorIndex, GetInferenceAudioBuffer(0, batchSize)),
        "Unable to bind input buffer"
        );
    A2E_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetOutputBinding(kResultTensorIndex, GetInferenceOutputBuffer(0, batchSize)),
        "Unable to bind result buffer"
        );
    A2E_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetDynamicDimension(kInputTensorIndex, 1, _networkInfo.bufferLength),
        "Unable to set audio dynamic window size"
        );
    A2E_CHECK_RESULT_WITH_MSG(
        _inferenceEngine.BindBuffers(*_bufferBindings, batchSize),
        "Unable to bind buffers"
        );

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutorCore::RunInference() {
    A2E_CHECK_RESULT_WITH_MSG(_inferenceEngine.Run(_cudaStream), "Unable to run inference engine");
    return nva2x::ErrorCode::eSuccess;
}

nva2x::DeviceTensorFloatView EmotionExecutorCore::GetInferenceAudioBuffer(std::size_t start, std::size_t count) {
    const auto stride = _networkInfo.bufferLength;
    return _inferenceAudioBuffer.View(start * stride, count * stride);
}

nva2x::DeviceTensorFloatConstView EmotionExecutorCore::GetInferenceAudioBuffer(std::size_t start, std::size_t count) const {
    const auto stride = _networkInfo.bufferLength;
    return _inferenceAudioBuffer.View(start * stride, count * stride);
}

std::error_code EmotionExecutorCore::Init(
    std::size_t nbTracks, cudaStream_t cudaStream,
    const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
    std::size_t batchSize
    ) {
    if (batchSize != nbTracks) {
        A2E_CHECK_ERROR_WITH_MSG(nbTracks == 1, "Number of tracks must be 1", nva2x::ErrorCode::eInvalidValue);
    }
    else {
        A2E_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
    }

    _networkInfo = classifierParams.networkInfo;
    _inferencesToSkip = classifierParams.inferencesToSkip;

    A2E_CHECK_ERROR_WITH_MSG(classifierParams.networkData, "Network data cannot be null", nva2x::ErrorCode::eNullPointer);
    A2E_CHECK_ERROR_WITH_MSG(classifierParams.networkDataSize, "Network data size cannot be zero", nva2x::ErrorCode::eInvalidValue);
    A2E_CHECK_RESULT_WITH_MSG(
        _inferenceEngine.Init(classifierParams.networkData, classifierParams.networkDataSize),
        "Unable to initialize inference engine"
    );
    const auto& bindingsDescription = GetBindingsDescription();
    A2E_CHECK_RESULT_WITH_MSG(
        _inferenceEngine.CheckBindings(bindingsDescription),
        "Mismatch in bindings on the inference engine"
    );
    _bufferBindings = std::make_unique<nva2x::BufferBindings>(bindingsDescription);

    const auto maxBatchSize = _inferenceEngine.GetMaxBatchSize(bindingsDescription);
    A2E_CHECK_ERROR_WITH_MSG(maxBatchSize > 0, "Unable to get maximum batch size", nva2x::ErrorCode::eInvalidValue);
    A2E_CHECK_ERROR_WITH_MSG(
        batchSize <= static_cast<std::size_t>(maxBatchSize),
        "Number of tracks / batch size cannot be greater than the network maximum batch size",
        nva2x::ErrorCode::eInvalidValue
        );
    if (batchSize == 0) {
        // Use the maximum batch size.
        batchSize = static_cast<std::size_t>(maxBatchSize);
    }
    _batchSize = batchSize;

    const auto postProcessParams = GetPostProcessParameters(classifierParams);

    A2E_CHECK_RESULT(
        EmotionExecutorCoreBase::Init(
            nbTracks, cudaStream, postProcessParams, classifierParams.inferencesToSkip + 1, batchSize
            )
        );

    A2E_CHECK_ERROR_WITH_MSG(
        _networkInfo.emotionLength == classifierParams.postProcessData.inferenceEmotionLength,
        "Emotion length must be equal to the post-process data inference emotion length",
        nva2x::ErrorCode::eMismatch
        );
    A2E_CHECK_RESULT_WITH_MSG(
        _inferenceAudioBuffer.Allocate(_networkInfo.bufferLength * batchSize),
        "Unable to initialize inference audio buffer"
        );

    A2E_CHECK_RESULT_WITH_MSG(BindBuffers(batchSize), "Unable to bind buffers");

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutorCore::GetProgressParameters(
    nva2x::WindowProgressParameters& outProgressParams,
    const NetworkInfo& networkInfo,
    std::size_t inferencesToSkip,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator
    ) {
    A2E_CHECK_ERROR_WITH_MSG(
        networkInfo.bufferLength > 0,
        "Buffer length must be greater than 0",
        nva2x::ErrorCode::eInvalidValue
    );
    A2E_CHECK_ERROR_WITH_MSG(
        networkInfo.bufferSamplerate > 0,
        "Buffer samplerate must be greater than 0",
        nva2x::ErrorCode::eInvalidValue
    );
    A2E_CHECK_ERROR_WITH_MSG(
        frameRateNumerator > 0,
        "Frame rate numerator must be greater than 0",
        nva2x::ErrorCode::eInvalidValue
    );
    A2E_CHECK_ERROR_WITH_MSG(
        frameRateDenominator > 0,
        "Frame rate denominator must be greater than 0",
        nva2x::ErrorCode::eInvalidValue
    );

    nva2x::WindowProgressParameters progressParams;
    progressParams.windowSize = networkInfo.bufferLength;
    progressParams.targetOffset = static_cast<nva2x::WindowProgressParameters::timestamp_t>(
        networkInfo.bufferLength / 2
        );
    progressParams.startOffset = -progressParams.targetOffset;
    progressParams.strideNum = networkInfo.bufferSamplerate * frameRateDenominator;
    progressParams.strideNum *= inferencesToSkip + 1;
    progressParams.strideDenom = frameRateNumerator;

    A2E_CHECK_ERROR_WITH_MSG(
        progressParams.strideNum <= progressParams.windowSize * progressParams.strideDenom,
        "Stride (including skipped inferences) cannot be greater than the window size",
        nva2x::ErrorCode::eInvalidValue
    );

    outProgressParams = progressParams;
    return nva2x::ErrorCode::eSuccess;
}

IPostProcessModel::EmotionExecutorCreationParameters EmotionExecutorCore::GetPostProcessParameters(
    const IClassifierModel::EmotionExecutorCreationParameters& classifierParams
    ) {
    nva2e::IPostProcessModel::EmotionExecutorCreationParameters postProcessParams;
    postProcessParams.samplingRate = classifierParams.networkInfo.bufferSamplerate;
    postProcessParams.inputStrength = classifierParams.inputStrength;
    postProcessParams.frameRateNumerator = classifierParams.frameRateNumerator;
    postProcessParams.frameRateDenominator = classifierParams.frameRateDenominator;
    postProcessParams.postProcessData = classifierParams.postProcessData;
    postProcessParams.postProcessParams = classifierParams.postProcessParams;
    postProcessParams.sharedPreferredEmotionAccumulators = classifierParams.sharedPreferredEmotionAccumulators;
    return postProcessParams;
}

} // namespace nva2e::IClassiferModel
