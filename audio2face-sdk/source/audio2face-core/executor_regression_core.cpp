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
#include "audio2face/internal/executor_regression_core.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"

namespace nva2f::IRegressionModel {

std::error_code GeometryExecutorCore::Reset(std::size_t trackIndex) {
    A2F_CHECK_RESULT(BaseReset(trackIndex));

    if (_skinPcaAnimator) {
        A2F_CHECK_RESULT_WITH_MSG(_skinPcaAnimator->Reset(trackIndex), "Unable to reset skin PCA animator");
    }
    if (_tonguePcaAnimator) {
        A2F_CHECK_RESULT_WITH_MSG(_tonguePcaAnimator->Reset(trackIndex), "Unable to reset tongue PCA animator");
    }

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorCore::ReadAudioBuffer(
    const nva2x::IAudioAccumulator& audioAccumulator, std::size_t batchIndex, std::size_t start
    ) {
    const auto audioBuffer = GetInferenceInputBuffers().GetInput(batchIndex);
    A2F_CHECK_RESULT_WITH_MSG(
        audioAccumulator.Read(audioBuffer, start, _inputStrength, _cudaStream),
        "Unable to read audio buffer"
        );

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorCore::BindBuffers(std::size_t batchSize) {
    A2F_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetInputBinding(kEmotionTensorIndex, _inferenceInputBuffers.GetEmotionTensor(batchSize)),
        "Unable to bind emotion buffer"
        );
    A2F_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetInputBinding(kInputTensorIndex, _inferenceInputBuffers.GetInputTensor(batchSize)),
        "Unable to bind input buffer"
        );
    A2F_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetOutputBinding(kResultTensorIndex, _inferenceOutputBuffers.GetResultTensor(batchSize)),
        "Unable to bind result buffer"
        );
    A2F_CHECK_RESULT_WITH_MSG(
        _inferenceEngine.BindBuffers(*_bufferBindings, batchSize),
        "Unable to bind buffers"
        );

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorCore::ReadExplicitEmotion(
    const nva2x::IEmotionAccumulator* emotionAccumulator, std::size_t batchIndex, std::size_t target
    ) {
    if (emotionAccumulator) {
        const auto emotion = _inferenceInputBuffers.GetExplicitEmotions(batchIndex);
        A2F_CHECK_RESULT_WITH_MSG(
            emotionAccumulator->Read(emotion, target, _cudaStream),
            "Unable to read explicit emotion from emotion accumulator"
        );
    }

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorCore::Init(
    std::size_t nbTracks, cudaStream_t stream,
    const nva2f::IRegressionModel::GeometryExecutorCreationParameters& regressionParams,
    std::size_t batchSize
    ) {
    A2F_CHECK_RESULT(
        BaseInit(
            nbTracks,
            stream,
            regressionParams.inputStrength,
            regressionParams.frameRateNumerator,
            regressionParams.frameRateDenominator,
            regressionParams.networkData,
            regressionParams.networkDataSize,
            GetBindingsDescription(),
            regressionParams.networkInfo.bufferSamplerate,
            regressionParams.networkInfo.resultSkinSize,
            regressionParams.networkInfo.resultTongueSize,
            regressionParams.initializationSkinParams,
            regressionParams.initializationTongueParams,
            regressionParams.initializationTeethParams,
            regressionParams.initializationEyesParams,
            batchSize
            )
        );

    _networkInfo = regressionParams.networkInfo;

    A2F_CHECK_ERROR_WITH_MSG(regressionParams.emotionDatabase, "Emotion database cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_ERROR_WITH_MSG(regressionParams.sourceShot, "Source shot cannot be null", nva2x::ErrorCode::eNullPointer);

    A2F_CHECK_RESULT_WITH_MSG(_inferenceInputBuffers.Init(_networkInfo, _batchSize), "Unable to initialize inference input buffers");
    A2F_CHECK_RESULT_WITH_MSG(_inferenceOutputBuffers.Init(_networkInfo, _batchSize), "Unable to initialize inference output buffers");
    // When there is only one track, the results buffers are scaled to the batch size.
    const auto resultBufferCount = _nbTracks > 1 ? _nbTracks : _batchSize;
    A2F_CHECK_RESULT_WITH_MSG(_resultBuffers.Init(_networkInfo, resultBufferCount), "Unable to initialize result buffers");

    A2F_CHECK_RESULT_WITH_MSG(BindBuffers(_batchSize), "Unable to bind buffers");

    if (regressionParams.initializationSkinParams) {
        A2F_CHECK_ERROR_WITH_MSG(
            regressionParams.initializationSkinParams->pcaData.shapeSize == _networkInfo.resultSkinSize,
            "PCA shape size does not match network info skin dimension",
            nva2x::ErrorCode::eMismatch
            );
        A2F_CHECK_ERROR_WITH_MSG(
            regressionParams.initializationSkinParams->pcaData.shapesMatrix.Size() == _networkInfo.resultSkinSize * _networkInfo.numShapesSkin,
            "PCA shapes matrix size does not match network info skin dimension",
            nva2x::ErrorCode::eMismatch
            );

        _skinPcaAnimator = std::make_unique<MultiTrackAnimatorPcaReconstruction>();
        A2F_CHECK_RESULT_WITH_MSG(
            _skinPcaAnimator->SetCudaStream(_cudaStream), "Unable to set CUDA stream on skin PCA animator"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            _skinPcaAnimator->Init(_batchSize),  "Unable to initialize skin PCA animator"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            _skinPcaAnimator->SetAnimatorData(regressionParams.initializationSkinParams->pcaData),
            "Unable to set data on skin PCA animator"
            );
    }

    if (regressionParams.initializationTongueParams) {
        A2F_CHECK_ERROR_WITH_MSG(
            regressionParams.initializationTongueParams->pcaData.shapeSize == _networkInfo.resultTongueSize,
            "PCA shape size does not match network info tongue dimension",
            nva2x::ErrorCode::eMismatch
            );
        A2F_CHECK_ERROR_WITH_MSG(
            regressionParams.initializationTongueParams->pcaData.shapesMatrix.Size() == _networkInfo.resultTongueSize * _networkInfo.numShapesTongue,
            "PCA shapes matrix size does not match network info tongue dimension",
            nva2x::ErrorCode::eMismatch
            );

        _tonguePcaAnimator = std::make_unique<MultiTrackAnimatorPcaReconstruction>();
        A2F_CHECK_RESULT_WITH_MSG(
            _tonguePcaAnimator->SetCudaStream(_cudaStream), "Unable to set CUDA stream on tongue PCA animator"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            _tonguePcaAnimator->Init(_batchSize),  "Unable to initialize tongue PCA animator"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            _tonguePcaAnimator->SetAnimatorData(regressionParams.initializationTongueParams->pcaData),
            "Unable to set data on tongue PCA animator"
            );
    }

    for (std::size_t batchIndex = 0; batchIndex < _batchSize; ++batchIndex) {
        A2F_CHECK_RESULT_WITH_MSG(
            regressionParams.emotionDatabase->GetEmotion(
                regressionParams.sourceShot,
                regressionParams.sourceFrame,
                _inferenceInputBuffers.GetImplicitEmotions(batchIndex)
                ),
            "Unable to get implicit emotion"
            );
    }

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorCore::GetProgressParameters(
    nva2x::WindowProgressParameters& outProgressParams,
    const NetworkInfo& networkInfo,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator
    ) {
    nva2x::WindowProgressParameters progressParams;
    progressParams.windowSize = networkInfo.bufferLength;
    progressParams.targetOffset = static_cast<timestamp_t>(networkInfo.bufferOffset);
    progressParams.startOffset = -progressParams.targetOffset;
    progressParams.strideNum = networkInfo.bufferSamplerate * frameRateDenominator;
    progressParams.strideDenom = frameRateNumerator;

    outProgressParams = progressParams;
    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f::IRegressionModel
