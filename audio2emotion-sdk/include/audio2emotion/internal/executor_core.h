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

#include "audio2emotion/executor_postprocess.h"
#include "audio2emotion/executor.h"
#include "audio2emotion/internal/multitrack_postprocess.h"
#include "audio2x/internal/audio_accumulator.h"

namespace nva2e {

class EmotionExecutorCoreBase {
public:
    using timestamp_t = IEmotionExecutor::timestamp_t;

    nva2x::WindowProgress GetSingleFrameProgress(const nva2x::WindowProgress& progress) const;

    std::error_code Reset(std::size_t trackIndex);

    std::error_code ReadPreferredEmotion(
        const nva2x::IEmotionAccumulator* emotionAccumulator, std::size_t batchIndex, std::size_t target
        );
    std::error_code RunPostProcess();

    nva2x::DeviceTensorFloatView GetInferenceOutputBuffer(std::size_t start, std::size_t count);
    nva2x::DeviceTensorFloatConstView GetInferenceOutputBuffer(std::size_t start, std::size_t count) const;
    nva2x::DeviceTensorFloatView GetOutputEmotions(std::size_t start, std::size_t count);
    nva2x::DeviceTensorFloatConstView GetOutputEmotions(std::size_t start, std::size_t count) const;

    std::error_code Init(
        std::size_t nbTracks, cudaStream_t cudaStream,
        const nva2e::IPostProcessModel::EmotionExecutorCreationParameters& postProcessParams,
        std::size_t nbFramesPerExecution,
        std::size_t batchSize
        );

    inline cudaStream_t GetCudaStream() const { return _cudaStream; }
    inline float GetInputStrength() const { return _inputStrength; }
    inline void SetInputStrength(float inputStrength) { _inputStrength = inputStrength; }
    inline void GetFrameRate(std::size_t& numerator, std::size_t& denominator) const {
        numerator = _frameRateNumerator;
        denominator = _frameRateDenominator;
    }
    inline std::size_t GetSamplingRate() const { return _samplingRate; }
    inline std::size_t GetNbFramesPerExecution() const { return _nbFramesPerExecution; }
    inline std::size_t GetBatchSize() const { return _batchSize; }
    inline MultiTrackPostProcessor& GetPostProcessor() { return _postProcessor; }
    inline const MultiTrackPostProcessor& GetPostProcessor() const { return _postProcessor; }

protected:
    cudaStream_t _cudaStream{};
    float _inputStrength{1.0f};
    std::size_t _frameRateNumerator{0};
    std::size_t _frameRateDenominator{0};
    std::size_t _samplingRate{0};
    std::size_t _nbFramesPerExecution{0};
    nva2x::DeviceTensorFloat _inferenceOutputBuffer;
    nva2x::DeviceTensorFloat _outputEmotions;
    std::size_t _batchSize{0};

    MultiTrackPostProcessor _postProcessor;
};

} // namespace nva2e
