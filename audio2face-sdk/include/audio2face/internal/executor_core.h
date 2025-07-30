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

#include "audio2face/executor.h"
#include "audio2face/animator.h"
#include "audio2face/internal/animator.h"
#include "audio2face/internal/multitrack_animator.h"
#include "audio2x/internal/inference_engine.h"

namespace nva2f {

class GeometryExecutorCoreBase {
public:
    using timestamp_t = IGeometryExecutor::timestamp_t;
    using ExecutionOption = IGeometryExecutor::ExecutionOption;

    inline cudaStream_t GetCudaStream() const { return _cudaStream; }
    inline float GetInputStrength() const { return _inputStrength; }
    inline void SetInputStrength(float inputStrength) { _inputStrength = inputStrength; }
    inline void GetFrameRate(std::size_t& numerator, std::size_t& denominator) const {
        numerator = _frameRateNumerator;
        denominator = _frameRateDenominator;
    }
    inline std::size_t GetNbTracks() const { return _nbTracks; }
    inline std::size_t GetBatchSize() const { return _batchSize; }

    inline MultiTrackAnimatorSkin* GetSkinAnimator() { return _skinAnimator.get(); }
    inline const MultiTrackAnimatorSkin* GetSkinAnimator() const { return _skinAnimator.get(); }
    inline MultiTrackAnimatorTongue* GetTongueAnimator() { return _tongueAnimator.get(); }
    inline const MultiTrackAnimatorTongue* GetTongueAnimator() const { return _tongueAnimator.get(); }
    inline MultiTrackAnimatorTeeth* GetTeethAnimator() { return _teethAnimator.get(); }
    inline const MultiTrackAnimatorTeeth* GetTeethAnimator() const { return _teethAnimator.get(); }
    inline MultiTrackAnimatorEyes* GetEyesAnimator() { return _eyesAnimator.get(); }
    inline const MultiTrackAnimatorEyes* GetEyesAnimator() const { return _eyesAnimator.get(); }

    inline std::size_t GetSamplingRate() const { return _samplingRate; }
    inline std::size_t GetSkinGeometrySize() const { return _skinGeometrySize; }
    inline std::size_t GetTongueGeometrySize() const { return _tongueGeometrySize; }
    inline std::size_t GetJawTransformSize() const { return 16; }
    inline std::size_t GetEyesRotationSize() const { return 6; }

    std::error_code RunInference();

    std::error_code SetExecutionOption(ExecutionOption executionOption);
    ExecutionOption GetExecutionOption() const;

protected:
    inline nva2x::BufferBindings& GetBufferBindings() { return *_bufferBindings; }
    inline nva2x::InferenceEngine& GetInferenceEngine() { return _inferenceEngine; }

    std::error_code BaseReset(std::size_t trackIndex);

    std::error_code BaseInit(
        std::size_t nbTracks,
        cudaStream_t cudaStream,
        float inputStrength,
        std::size_t frameRateNumerator,
        std::size_t frameRateDenominator,
        const void* networkData,
        std::size_t networkDataSize,
        const nva2x::BufferBindingsDescription& bindingsDescription,
        std::size_t samplingRate,
        std::size_t skinGeometrySize,
        std::size_t tongueGeometrySize,
        const IAnimatorSkin::InitData* skinParams,
        const IAnimatorTongue::InitData* tongueParams,
        const IAnimatorTeeth::InitData* teethParams,
        const IAnimatorEyes::InitData* eyesParams,
        std::size_t batchSize
        );

    cudaStream_t _cudaStream{};
    float _inputStrength{1.0f};
    std::size_t _frameRateNumerator{0};
    std::size_t _frameRateDenominator{0};
    nva2x::InferenceEngine _inferenceEngine;
    std::unique_ptr<nva2x::BufferBindings> _bufferBindings;
    std::size_t _nbTracks{0};
    std::size_t _batchSize{0};

    std::size_t _samplingRate{0};
    std::size_t _skinGeometrySize{0};
    std::size_t _tongueGeometrySize{0};

    ExecutionOption _executionOption{ExecutionOption::None};

    std::unique_ptr<MultiTrackAnimatorSkin> _skinAnimator;
    std::unique_ptr<MultiTrackAnimatorTongue> _tongueAnimator;
    std::unique_ptr<MultiTrackAnimatorTeeth> _teethAnimator;
    std::unique_ptr<MultiTrackAnimatorEyes> _eyesAnimator;
};

} // namespace nva2f
