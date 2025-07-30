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

#include "audio2face/internal/executor_core.h"
#include "audio2face/executor_diffusion.h"
#include "audio2face/internal/model_diffusion.h"
#include "audio2face/internal/noise.h"
#include "audio2x/internal/audio_accumulator.h"

namespace nva2f {

namespace IDiffusionModel {

class GeometryExecutorCore : public GeometryExecutorCoreBase {
public:
    inline const NetworkInfo& GetNetworkInfo() const { return _networkInfo; }

    inline InferenceInputBuffers& GetInferenceInputBuffers() { return _inferenceInputBuffers; }
    inline const InferenceInputBuffers& GetInferenceInputBuffers() const { return _inferenceInputBuffers; }
    inline InferenceStateBuffers& GetInferenceStateBuffers() { return _inferenceStateBuffers; }
    inline const InferenceStateBuffers& GetInferenceStateBuffers() const { return _inferenceStateBuffers; }
    inline InferenceOutputBuffers& GetInferenceOutputBuffers() { return _inferenceOutputBuffers; }
    inline const InferenceOutputBuffers& GetInferenceOutputBuffers() const { return _inferenceOutputBuffers; }
    inline ResultBuffers& GetResultBuffers() { return _resultBuffers; }
    inline const ResultBuffers& GetResultBuffers() const { return _resultBuffers; }
    inline NoiseGenerator* GetNoiseGenerator() { return _noiseGenerator.get(); }
    inline const NoiseGenerator* GetNoiseGenerator() const { return _noiseGenerator.get(); }

    std::error_code Reset(std::size_t trackIndex);

    std::error_code ReadAudioBuffer(
        const nva2x::IAudioAccumulator& audioAccumulator, std::size_t batchIndex, std::size_t start
        );
    std::error_code BindBuffers();
    std::error_code ReadEmotion(
        const nva2x::IEmotionAccumulator* emotionAccumulator, std::size_t batchIndex, std::size_t target, size_t frameIndex
        );

    std::error_code Init(
        std::size_t nbTracks, cudaStream_t stream,
        const nva2f::IDiffusionModel::GeometryExecutorCreationParameters& params,
        std::size_t batchSize
        );

    static std::error_code GetProgressParameters(
        nva2x::WindowProgressParameters& outProgressParams,
        const NetworkInfo& networkInfo
        );
    static std::error_code GetFrameRate(
        std::size_t& outFrameRateNumerator,
        std::size_t& outFrameRateDenominator,
        const NetworkInfo& networkInfo
        );
    using GeometryExecutorCoreBase::GetFrameRate;

private:
    NetworkInfo _networkInfo{};
    InferenceInputBuffers _inferenceInputBuffers;
    InferenceStateBuffers _inferenceStateBuffers;
    InferenceOutputBuffers _inferenceOutputBuffers;
    ResultBuffers _resultBuffers;
    std::unique_ptr<NoiseGenerator> _noiseGenerator;
};

} // namespace IDiffusionModel

} // namespace nva2f
