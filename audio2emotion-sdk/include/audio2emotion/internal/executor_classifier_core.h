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

#include "audio2emotion/executor_classifier.h"
#include "audio2emotion/internal/executor_core.h"
#include "audio2x/internal/inference_engine.h"

namespace nva2e {

namespace IClassifierModel {

class EmotionExecutorCore : public EmotionExecutorCoreBase {
public:
    std::error_code ReadAudioBuffer(
        const nva2x::IAudioAccumulator& audioAccumulator, std::size_t batchIndex, std::size_t start
        );
    std::error_code BindBuffers(std::size_t batchSize);
    std::error_code RunInference();

    nva2x::DeviceTensorFloatView GetInferenceAudioBuffer(std::size_t start, std::size_t count);
    nva2x::DeviceTensorFloatConstView GetInferenceAudioBuffer(std::size_t start, std::size_t count) const;

    std::error_code Init(
        std::size_t nbTracks, cudaStream_t cudaStream,
        const nva2e::IClassifierModel::EmotionExecutorCreationParameters& classifierParams,
        std::size_t batchSize
        );

    inline const NetworkInfo& GetNetworkInfo() const { return _networkInfo; }
    inline std::size_t GetInferencesToSkip() const { return _inferencesToSkip; }
    inline void SetInferencesToSkip(std::size_t inferencesToSkip) { _inferencesToSkip = inferencesToSkip; _nbFramesPerExecution = inferencesToSkip + 1; }

    static std::error_code GetProgressParameters(
        nva2x::WindowProgressParameters& outProgressParams,
        const NetworkInfo& networkInfo,
        std::size_t inferencesToSkip,
        std::size_t frameRateNumerator,
        std::size_t frameRateDenominator
        );

    static IPostProcessModel::EmotionExecutorCreationParameters GetPostProcessParameters(
        const IClassifierModel::EmotionExecutorCreationParameters& classifierParams
        );

private:
    NetworkInfo _networkInfo{};
    std::size_t _inferencesToSkip{0};
    nva2x::InferenceEngine _inferenceEngine;
    std::unique_ptr<nva2x::BufferBindings> _bufferBindings;
    nva2x::DeviceTensorFloat _inferenceAudioBuffer;
};

} // namespace IClassifierModel

} // namespace nva2e
