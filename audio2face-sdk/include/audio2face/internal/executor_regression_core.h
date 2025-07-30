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
#include "audio2face/executor_regression.h"
#include "audio2face/internal/model_regression.h"
#include "audio2x/internal/audio_accumulator.h"

namespace nva2f {

namespace IRegressionModel {

class GeometryExecutorCore : public GeometryExecutorCoreBase {
public:
    inline const NetworkInfo& GetNetworkInfo() const { return _networkInfo; }

    inline InferenceInputBuffers& GetInferenceInputBuffers() { return _inferenceInputBuffers; }
    inline const InferenceInputBuffers& GetInferenceInputBuffers() const { return _inferenceInputBuffers; }
    inline InferenceOutputBuffers& GetInferenceOutputBuffers() { return _inferenceOutputBuffers; }
    inline const InferenceOutputBuffers& GetInferenceOutputBuffers() const { return _inferenceOutputBuffers; }
    inline ResultBuffers& GetResultBuffers() { return _resultBuffers; }
    inline const ResultBuffers& GetResultBuffers() const { return _resultBuffers; }

    inline MultiTrackAnimatorPcaReconstruction* GetSkinPcaAnimator() { return _skinPcaAnimator.get(); }
    inline const MultiTrackAnimatorPcaReconstruction* GetSkinPcaAnimator() const { return _skinPcaAnimator.get(); }
    inline MultiTrackAnimatorPcaReconstruction* GetTonguePcaAnimator() { return _tonguePcaAnimator.get(); }
    inline const MultiTrackAnimatorPcaReconstruction* GetTonguePcaAnimator() const { return _tonguePcaAnimator.get(); }

    std::error_code Reset(std::size_t trackIndex);

    std::error_code ReadAudioBuffer(
        const nva2x::IAudioAccumulator& audioAccumulator, std::size_t batchIndex, std::size_t start
        );
    std::error_code BindBuffers(std::size_t batchSize);
    std::error_code ReadExplicitEmotion(
        const nva2x::IEmotionAccumulator* emotionAccumulator, std::size_t batchIndex, std::size_t target
        );

    std::error_code Init(
        std::size_t nbTracks, cudaStream_t stream,
        const nva2f::IRegressionModel::GeometryExecutorCreationParameters& params,
        std::size_t batchSize
        );

    static std::error_code GetProgressParameters(
        nva2x::WindowProgressParameters& outProgressParams,
        const NetworkInfo& networkInfo,
        std::size_t frameRateNumerator,
        std::size_t frameRateDenominator
        );

private:
    NetworkInfo _networkInfo{};
    InferenceInputBuffers _inferenceInputBuffers;
    InferenceOutputBuffers _inferenceOutputBuffers;
    ResultBuffers _resultBuffers;

    std::unique_ptr<MultiTrackAnimatorPcaReconstruction> _skinPcaAnimator;
    std::unique_ptr<MultiTrackAnimatorPcaReconstruction> _tonguePcaAnimator;
};

} // namespace IRegressionModel

} // namespace nva2f
