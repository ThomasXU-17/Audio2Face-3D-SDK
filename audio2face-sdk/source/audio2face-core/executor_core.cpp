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
#include "audio2face/internal/executor_core.h"
#include "audio2face/internal/executor.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"

#include <numeric>

namespace nva2f {

// For IGeometryExecutor::ExecutionOption operators.
using namespace ::nva2f::internal;


std::error_code GeometryExecutorCoreBase::RunInference() {
    A2F_CHECK_RESULT_WITH_MSG(_inferenceEngine.Run(_cudaStream), "Unable to run inference engine");
    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorCoreBase::SetExecutionOption(ExecutionOption executionOption) {
    A2F_CHECK_ERROR_WITH_MSG(
        !IsAnySet(executionOption, ExecutionOption::Skin) || _skinAnimator,
        "Skin cannot be enabled without being initialized",
        nva2x::ErrorCode::eNotInitialized
        );
    A2F_CHECK_ERROR_WITH_MSG(
        !IsAnySet(executionOption, ExecutionOption::Tongue) || _tongueAnimator,
        "Tongue cannot be enabled without being initialized",
        nva2x::ErrorCode::eNotInitialized
        );
    A2F_CHECK_ERROR_WITH_MSG(
        !IsAnySet(executionOption, ExecutionOption::Jaw) || _teethAnimator,
        "Jaw cannot be enabled without being initialized",
        nva2x::ErrorCode::eNotInitialized
        );
    A2F_CHECK_ERROR_WITH_MSG(
        !IsAnySet(executionOption, ExecutionOption::Eyes) || _eyesAnimator,
        "Eyes cannot be enabled without being initialized",
        nva2x::ErrorCode::eNotInitialized
        );

    _executionOption = executionOption;
    return nva2x::ErrorCode::eSuccess;
}

IGeometryExecutor::ExecutionOption GeometryExecutorCoreBase::GetExecutionOption() const {
    return _executionOption;
}

std::error_code GeometryExecutorCoreBase::BaseReset(std::size_t trackIndex) {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _nbTracks, "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);

    if (_skinAnimator) {
        A2F_CHECK_RESULT_WITH_MSG(_skinAnimator->Reset(trackIndex), "Unable to reset skin animator");
    }
    if (_tongueAnimator) {
        A2F_CHECK_RESULT_WITH_MSG(_tongueAnimator->Reset(trackIndex), "Unable to reset tongue animator");
    }
    if (_teethAnimator) {
        A2F_CHECK_RESULT_WITH_MSG(_teethAnimator->Reset(trackIndex), "Unable to reset teeth animator");
    }
    if (_eyesAnimator) {
        A2F_CHECK_RESULT_WITH_MSG(_eyesAnimator->Reset(trackIndex), "Unable to reset eyes animator");
    }

    return nva2x::ErrorCode::eSuccess;
}

std::error_code GeometryExecutorCoreBase::BaseInit(
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
    ) {
    if (batchSize != nbTracks) {
        A2F_CHECK_ERROR_WITH_MSG(nbTracks == 1, "Number of tracks must be 1", nva2x::ErrorCode::eInvalidValue);
    }
    else {
        A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
    }

    A2F_CHECK_ERROR_WITH_MSG(frameRateNumerator > 0, "Frame rate numerator must be greater than 0", nva2x::ErrorCode::eInvalidValue);
    A2F_CHECK_ERROR_WITH_MSG(frameRateDenominator > 0, "Frame rate denominator must be greater than 0", nva2x::ErrorCode::eInvalidValue);

    _cudaStream = cudaStream;

    _inputStrength = inputStrength;

    _frameRateNumerator = frameRateNumerator;
    _frameRateDenominator = frameRateDenominator;
    const auto divisor = std::gcd(_frameRateNumerator, _frameRateDenominator);
    _frameRateNumerator /= divisor;
    _frameRateDenominator /= divisor;

    A2F_CHECK_ERROR_WITH_MSG(networkData, "Network data cannot be null", nva2x::ErrorCode::eNullPointer);
    A2F_CHECK_ERROR_WITH_MSG(networkDataSize, "Network data size cannot be zero", nva2x::ErrorCode::eInvalidValue);
    A2F_CHECK_RESULT_WITH_MSG(
        _inferenceEngine.Init(networkData, networkDataSize),
        "Unable to initialize inference engine"
    );
    A2F_CHECK_RESULT_WITH_MSG(
        _inferenceEngine.CheckBindings(bindingsDescription),
            "Mismatch in bindings on the inference engine"
    );
    _bufferBindings = std::make_unique<nva2x::BufferBindings>(bindingsDescription);

    const auto maxBatchSize = _inferenceEngine.GetMaxBatchSize(bindingsDescription);
    A2F_CHECK_ERROR_WITH_MSG(maxBatchSize > 0, "Unable to get maximum batch size", nva2x::ErrorCode::eInvalidValue);
    A2F_CHECK_ERROR_WITH_MSG(
        nbTracks <= static_cast<std::size_t>(maxBatchSize),
        "Number of tracks cannot be greater than the network maximum batch size",
        nva2x::ErrorCode::eInvalidValue
        );
    if (batchSize == 0) {
        // Use the maximum batch size.
        batchSize = static_cast<std::size_t>(maxBatchSize);
    }
    _nbTracks = nbTracks;
    _batchSize = batchSize;

    _samplingRate = samplingRate;
    _skinGeometrySize = skinGeometrySize;
    _tongueGeometrySize = tongueGeometrySize;

    // Initialize the animator data based on the provided parameters.
    const float dt = static_cast<float>(_frameRateDenominator) / _frameRateNumerator;

    _executionOption = ExecutionOption::None;
    if (skinParams) {
        A2F_CHECK_ERROR_WITH_MSG(
            skinParams->data.neutralPose.Size() == skinGeometrySize,
            "Neutral pose size does not match network info skin dimension",
            nva2x::ErrorCode::eMismatch
            );
        A2F_CHECK_ERROR_WITH_MSG(
            skinParams->data.lipOpenPoseDelta.Size() == skinGeometrySize,
            "Lip open pose delta size does not match network info skin dimension",
            nva2x::ErrorCode::eMismatch
            );
        A2F_CHECK_ERROR_WITH_MSG(
            skinParams->data.eyeClosePoseDelta.Size() == skinGeometrySize,
            "Eye close pose delta size does not match network info skin dimension",
            nva2x::ErrorCode::eMismatch
            );

        _skinAnimator = std::make_unique<MultiTrackAnimatorSkin>();
        A2F_CHECK_RESULT_WITH_MSG(
            _skinAnimator->SetCudaStream(_cudaStream), "Unable to set CUDA stream on skin animator"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            _skinAnimator->Init(skinParams->params, _nbTracks),
            "Unable to initialize skin animator"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            _skinAnimator->SetAnimatorData(skinParams->data, dt),
            "Unable to set data on skin animator"
            );

        _executionOption |= ExecutionOption::Skin;
    }

    if (tongueParams) {
        A2F_CHECK_ERROR_WITH_MSG(
            tongueParams->data.neutralPose.Size() == tongueGeometrySize,
            "Neutral pose size does not match network info tongue dimension",
            nva2x::ErrorCode::eMismatch
            );

        _tongueAnimator = std::make_unique<MultiTrackAnimatorTongue>();
        A2F_CHECK_RESULT_WITH_MSG(
            _tongueAnimator->SetCudaStream(_cudaStream), "Unable to set CUDA stream on tongue animator"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            _tongueAnimator->Init(tongueParams->params, _nbTracks),
            "Unable to initialize tongue animator"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            _tongueAnimator->SetAnimatorData(tongueParams->data),
            "Unable to set data on tongue animator"
            );

        _executionOption |= ExecutionOption::Tongue;
    }

    if (teethParams) {
        _teethAnimator = std::make_unique<MultiTrackAnimatorTeeth>();
        A2F_CHECK_RESULT_WITH_MSG(
            _teethAnimator->SetCudaStream(_cudaStream), "Unable to set CUDA stream on teeth animator"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            _teethAnimator->Init(teethParams->params, _nbTracks),
            "Unable to initialize teeth animator"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            _teethAnimator->SetAnimatorData(teethParams->data),
            "Unable to set data on teeth animator"
            );

        _executionOption |= ExecutionOption::Jaw;
    }

    if (eyesParams) {
        _eyesAnimator = std::make_unique<MultiTrackAnimatorEyes>();
        A2F_CHECK_RESULT_WITH_MSG(
            _eyesAnimator->SetCudaStream(_cudaStream), "Unable to set CUDA stream on eyes animator"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            _eyesAnimator->Init(eyesParams->params, _nbTracks),
            "Unable to initialize eyes animator"
            );
        A2F_CHECK_RESULT_WITH_MSG(
            _eyesAnimator->SetAnimatorData(eyesParams->data, dt),
            "Unable to set data on eyes animator"
            );

        _executionOption |= ExecutionOption::Eyes;
    }

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f
