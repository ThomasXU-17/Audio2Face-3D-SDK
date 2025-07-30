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
#include "audio2face/internal/executor_diffusion_core.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"

namespace nva2f::IDiffusionModel {

std::error_code GeometryExecutorCore::Reset(std::size_t trackIndex) {
    A2F_CHECK_RESULT(BaseReset(trackIndex));

    A2F_CHECK_RESULT_WITH_MSG(
        _inferenceStateBuffers.Reset(_cudaStream, trackIndex),
        "Unable to reset inference state buffers"
        );

    if (_noiseGenerator) {
        A2F_CHECK_RESULT_WITH_MSG(
            _noiseGenerator->Reset(trackIndex, 0),
            "Unable to reset noise generator"
            );
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

std::error_code GeometryExecutorCore::BindBuffers() {
    A2F_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetInputBinding(kEmotionTensorIndex, _inferenceInputBuffers.GetEmotionTensor()),
        "Unable to bind emotion buffer"
        );
    A2F_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetInputBinding(kIdentityTensorIndex, _inferenceInputBuffers.GetIdentityTensor()),
        "Unable to bind identity buffer"
        );
    A2F_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetInputBinding(kInputLatentsTensorIndex, _inferenceStateBuffers.GetInputGRUStateTensor()),
        "Unable to bind GRU input buffer"
        );
    A2F_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetInputBinding(kNoiseTensorIndex, _inferenceInputBuffers.GetNoiseTensor()),
        "Unable to bind noise buffer"
        );
    A2F_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetInputBinding(kWindowTensorIndex, _inferenceInputBuffers.GetInputTensor()),
        "Unable to bind input buffer"
        );
    A2F_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetOutputBinding(kOutputLatentsTensorIndex, _inferenceStateBuffers.GetOutputGRUStateTensor()),
        "Unable to bind GRU output buffer"
        );
    A2F_CHECK_RESULT_WITH_MSG(
        _bufferBindings->SetOutputBinding(kPredictionTensorIndex, _inferenceOutputBuffers.GetResultTensor()),
        "Unable to bind result buffer"
        );
    A2F_CHECK_RESULT_WITH_MSG(
        _inferenceEngine.BindBuffers(*_bufferBindings, _nbTracks),
        "Unable to bind buffers"
        );

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorCore::ReadEmotion(
    const nva2x::IEmotionAccumulator* emotionAccumulator, std::size_t batchIndex, std::size_t target, std::size_t frameIndex
    ) {
    if (emotionAccumulator) {
        const auto emotion = _inferenceInputBuffers.GetEmotions(frameIndex, batchIndex);
        A2F_CHECK_RESULT_WITH_MSG(
            emotionAccumulator->Read(emotion, target, _cudaStream),
            "Unable to read explicit emotion from emotion accumulator"
        );
    }

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorCore::Init(
    std::size_t nbTracks, cudaStream_t stream,
    const nva2f::IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams,
    std::size_t batchSize
    ) {
    std::size_t frameRateNumerator{0};
    std::size_t frameRateDenominator{0};
    A2F_CHECK_RESULT_WITH_MSG(
        GetFrameRate(frameRateNumerator, frameRateDenominator, diffusionParams.networkInfo),
        "Unable to get frame rate"
        );

    A2F_CHECK_RESULT(
        BaseInit(
            nbTracks,
            stream,
            diffusionParams.inputStrength,
            frameRateNumerator,
            frameRateDenominator,
            diffusionParams.networkData,
            diffusionParams.networkDataSize,
            GetBindingsDescription(),
            diffusionParams.networkInfo.bufferSamplerate,
            diffusionParams.networkInfo.skinDim,
            diffusionParams.networkInfo.tongueDim,
            diffusionParams.initializationSkinParams,
            diffusionParams.initializationTongueParams,
            diffusionParams.initializationTeethParams,
            diffusionParams.initializationEyesParams,
            batchSize
            )
        );

    _networkInfo = diffusionParams.networkInfo;

    A2F_CHECK_RESULT_WITH_MSG(_inferenceInputBuffers.Init(_networkInfo, _batchSize), "Unable to initialize inference input buffers");
    A2F_CHECK_RESULT_WITH_MSG(_inferenceStateBuffers.Init(_networkInfo, _batchSize), "Unable to initialize inference state buffers");
    A2F_CHECK_RESULT_WITH_MSG(_inferenceOutputBuffers.Init(_networkInfo, _batchSize), "Unable to initialize inference output buffers");
    A2F_CHECK_RESULT_WITH_MSG(_resultBuffers.Init(_networkInfo, _nbTracks), "Unable to initialize result buffers");

    // Set the tensors.
    A2F_CHECK_RESULT_WITH_MSG(
        nva2x::FillOnDevice(_inferenceStateBuffers.GetInputGRUStateTensor(), 0.0f, _cudaStream),
        "Unable to fill input GRU state with zeros"
        );

    A2F_CHECK_ERROR_WITH_MSG(
        diffusionParams.identityIndex < diffusionParams.networkInfo.identityLength,
        "Identity index out of bounds",
        nva2x::ErrorCode::eOutOfBounds
        );
    A2F_CHECK_RESULT_WITH_MSG(
        nva2x::FillOnDevice(_inferenceInputBuffers.GetIdentity(0), 0.0f, _cudaStream),
        "Unable to fill identity with zeros"
        );
    A2F_CHECK_RESULT_WITH_MSG(
        nva2x::FillOnDevice(_inferenceInputBuffers.GetIdentity(0).View(diffusionParams.identityIndex, 1), 1.0f, _cudaStream),
        "Unable to set one-hot identity"
        );
    for (std::size_t i = 1; i < _nbTracks; ++i) {
        A2F_CHECK_RESULT_WITH_MSG(
            nva2x::CopyDeviceToDevice(_inferenceInputBuffers.GetIdentity(i), _inferenceInputBuffers.GetIdentity(0), _cudaStream),
            "Unable to copy identity"
            );
    }

    auto noiseGenerator = std::make_unique<NoiseGenerator>();
    const auto noiseSize = _inferenceInputBuffers.GetNoise(0).Size();
    A2F_CHECK_RESULT_WITH_MSG(noiseGenerator->SetCudaStream(_cudaStream), "Unable to set cuda stream");
    if (diffusionParams.constantNoise) {
        // Fill with constant noise, only once.
        A2F_CHECK_RESULT_WITH_MSG(noiseGenerator->Init(1, noiseSize), "Unable to initialize noise generator");
        A2F_CHECK_RESULT_WITH_MSG(noiseGenerator->Generate(0, _inferenceInputBuffers.GetNoise(0)), "Unable to generate noise");
        for (std::size_t i = 1; i < _nbTracks; ++i) {
            A2F_CHECK_RESULT_WITH_MSG(
                nva2x::CopyDeviceToDevice(_inferenceInputBuffers.GetNoise(i), _inferenceInputBuffers.GetNoise(0), _cudaStream),
                "Unable to copy noise"
                );
        }
        noiseGenerator.reset();
    } else {
        // Noise will be generated at each inference.
        A2F_CHECK_RESULT_WITH_MSG(noiseGenerator->Init(_nbTracks, noiseSize), "Unable to initialize noise generator");
    }
    _noiseGenerator = std::move(noiseGenerator);

    A2F_CHECK_ERROR_WITH_MSG(
        diffusionParams.networkInfo.skinDim % 3 == 0,
        "Skin dimension must be a multiple of 3",
        nva2x::ErrorCode::eInvalidValue
        );
    A2F_CHECK_ERROR_WITH_MSG(
        diffusionParams.networkInfo.tongueDim % 3 == 0,
        "Tongue dimension must be a multiple of 3",
        nva2x::ErrorCode::eInvalidValue
        );

    A2F_CHECK_RESULT_WITH_MSG(BindBuffers(), "Unable to bind buffers");

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorCore::GetProgressParameters(
    nva2x::WindowProgressParameters& outProgressParams,
    const NetworkInfo& networkInfo
    ) {
    const auto nbFramesPerInference = networkInfo.numFramesLeftTruncate + networkInfo.numFramesRightTruncate +
        networkInfo.numFramesCenter;

    nva2x::WindowProgressParameters progressParams;
    progressParams.windowSize = networkInfo.bufferLength;
    progressParams.targetOffset = (networkInfo.bufferLength * networkInfo.numFramesLeftTruncate) /
        nbFramesPerInference;
    progressParams.startOffset = -static_cast<timestamp_t>(networkInfo.paddingLeft);
    progressParams.strideNum = networkInfo.bufferLength * networkInfo.numFramesCenter;
    progressParams.strideDenom = nbFramesPerInference;

    outProgressParams = progressParams;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorCore::GetFrameRate(
    std::size_t& outFrameRateNumerator,
    std::size_t& outFrameRateDenominator,
    const NetworkInfo& networkInfo
    ) {
    const auto nbFramesPerInference = networkInfo.numFramesLeftTruncate + networkInfo.numFramesRightTruncate +
        networkInfo.numFramesCenter;

    // Frame rate is number of frames / time = number of frames / (buffer length / sampling rate)
    const std::size_t frameRateNumerator = nbFramesPerInference * networkInfo.bufferSamplerate;
    const std::size_t frameRateDenominator = networkInfo.bufferLength;

    outFrameRateNumerator = frameRateNumerator;
    outFrameRateDenominator = frameRateDenominator;
    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f::IRegressionModel
