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
#include "audio2emotion/internal/executor_core.h"
#include "audio2emotion/internal/macros.h"
#include "audio2emotion/internal/postprocess.h"
#include "audio2x/error.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"

#include <cassert>
#include <numeric>

namespace nva2e {

nva2x::WindowProgress EmotionExecutorCoreBase::GetSingleFrameProgress(const nva2x::WindowProgress& progress) const {
    const auto nbFramesPerExecution = _nbFramesPerExecution;
    assert(nbFramesPerExecution > 0);
    return nva2x::GetFrameProgress(progress, nbFramesPerExecution);
}

std::error_code EmotionExecutorCoreBase::Reset(std::size_t trackIndex) {
    A2E_CHECK_RESULT_WITH_MSG(_postProcessor.Reset(trackIndex), "Unable to reset post-processor");
    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutorCoreBase::ReadPreferredEmotion(
    const nva2x::IEmotionAccumulator* emotionAccumulator, std::size_t batchIndex, std::size_t target
    ) {
    if (emotionAccumulator && _postProcessor.GetParameters(batchIndex)->enablePreferredEmotion) {
        const auto preferredEmotion = _postProcessor.GetPreferredEmotion(batchIndex);
        A2E_CHECK_RESULT_WITH_MSG(
            emotionAccumulator->Read(preferredEmotion, target, _cudaStream),
            "Unable to read preferred emotion from emotion accumulator"
        );
    }

    return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionExecutorCoreBase::RunPostProcess() {
    // Post-process might not have the same size as inference.
    const auto outputEmotionSize = _postProcessor.GetOutputEmotionsSize();
    auto& postProcessOutput = _outputEmotions;
    nva2x::TensorBatchInfo postProcessOutputInfo;
    postProcessOutputInfo.offset = 0;
    postProcessOutputInfo.size = outputEmotionSize;
    postProcessOutputInfo.stride = outputEmotionSize;

    const auto nbTracks = _outputEmotions.Size() / outputEmotionSize;

    const auto inputEmotionSize = _postProcessor.GetInputEmotionsSize();
    const auto postProcessInput = _inferenceOutputBuffer.View(0, inputEmotionSize * nbTracks);
    nva2x::TensorBatchInfo postProcessInputInfo;
    postProcessInputInfo.offset = 0;
    postProcessInputInfo.size = inputEmotionSize;
    postProcessInputInfo.stride = inputEmotionSize;

    A2E_CHECK_RESULT_WITH_MSG(
        _postProcessor.PostProcess(
            postProcessInput, postProcessInputInfo, postProcessOutput, postProcessOutputInfo
            ),
        "Unable to run post-processing"
        );
    return nva2x::ErrorCode::eSuccess;
}

nva2x::DeviceTensorFloatView EmotionExecutorCoreBase::GetInferenceOutputBuffer(std::size_t start, std::size_t count) {
    const auto stride = _postProcessor.GetInputEmotionsSize();
    return _inferenceOutputBuffer.View(start * stride, count * stride);
}

nva2x::DeviceTensorFloatConstView EmotionExecutorCoreBase::GetInferenceOutputBuffer(std::size_t start, std::size_t count) const {
    const auto stride = _postProcessor.GetInputEmotionsSize();
    return _inferenceOutputBuffer.View(start * stride, count * stride);
}

nva2x::DeviceTensorFloatView EmotionExecutorCoreBase::GetOutputEmotions(std::size_t start, std::size_t count) {
    const auto stride = _postProcessor.GetOutputEmotionsSize();
    return _outputEmotions.View(start * stride, count * stride);
}

nva2x::DeviceTensorFloatConstView EmotionExecutorCoreBase::GetOutputEmotions(std::size_t start, std::size_t count) const {
    const auto stride = _postProcessor.GetOutputEmotionsSize();
    return _outputEmotions.View(start * stride, count * stride);
}

std::error_code EmotionExecutorCoreBase::Init(
    std::size_t nbTracks, cudaStream_t cudaStream,
    const nva2e::IPostProcessModel::EmotionExecutorCreationParameters& params,
    std::size_t nbFramesPerExecution,
    std::size_t batchSize
    ) {
    A2E_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
    A2E_CHECK_ERROR_WITH_MSG(batchSize > 0, "Batch size must be greater than 0", nva2x::ErrorCode::eInvalidValue);

    A2E_CHECK_ERROR_WITH_MSG(params.frameRateNumerator > 0, "Frame rate numerator must be greater than 0", nva2x::ErrorCode::eInvalidValue);
    A2E_CHECK_ERROR_WITH_MSG(params.frameRateDenominator > 0, "Frame rate denominator must be greater than 0", nva2x::ErrorCode::eInvalidValue);

    _cudaStream = cudaStream;
    _inputStrength = params.inputStrength;

    _frameRateNumerator = params.frameRateNumerator;
    _frameRateDenominator = params.frameRateDenominator;
    const auto divisor = std::gcd(_frameRateNumerator, _frameRateDenominator);
    _frameRateNumerator /= divisor;
    _frameRateDenominator /= divisor;

    _samplingRate = params.samplingRate;
    _nbFramesPerExecution = nbFramesPerExecution;
    _batchSize = batchSize;

    A2E_CHECK_RESULT_WITH_MSG(
        _inferenceOutputBuffer.Allocate(params.postProcessData.inferenceEmotionLength * batchSize),
        "Unable to initialize inference output buffer"
    );
    A2E_CHECK_RESULT_WITH_MSG(
        _outputEmotions.Allocate(params.postProcessData.outputEmotionLength * nbTracks),
        "Unable to initialize output emotions buffer"
    );

    auto postProcessParams = params.postProcessParams;
    postProcessParams.fixedDt = static_cast<float>(_frameRateDenominator) / _frameRateNumerator;

    std::vector<float> defaultPreferredEmotion;
    if (postProcessParams.preferredEmotion.Size() == 0) {
        // Not specifying a default preferred emotion is supported if preferred emotion accumulators are provided
        // for every track.
        A2E_CHECK_ERROR_WITH_MSG(
            params.sharedPreferredEmotionAccumulators,
            "No preferred emotion specified and no preferred emotion accumulators provided",
            nva2x::ErrorCode::eInvalidValue
        );
        for (std::size_t i = 0; i < nbTracks; ++i) {
            A2E_CHECK_ERROR_WITH_MSG(
                params.sharedPreferredEmotionAccumulators[i],
                "No preferred emotion specified and not all preferred emotion accumulators provided",
                nva2x::ErrorCode::eInvalidValue
            );
        }

        // All accumulators are there, provide a default preferred emotion.
        defaultPreferredEmotion.resize(params.postProcessData.outputEmotionLength, 0.0f);
        postProcessParams.preferredEmotion = nva2x::ToConstView(defaultPreferredEmotion);
    }

    A2E_CHECK_RESULT_WITH_MSG(_postProcessor.SetCudaStream(_cudaStream), "Unable to set CUDA stream");
    A2E_CHECK_RESULT_WITH_MSG(
        _postProcessor.Init(params.postProcessData, postProcessParams, nbTracks),
        "Unable to initialize post-processor"
    );

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2e
