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
#include "audio2emotion/internal/postprocess.h"
#include "audio2emotion/internal/logger.h"
#include "audio2emotion/internal/macros.h"
#include "audio2x/error.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <numeric>

namespace {

    void inplace_softmax(nva2x::HostTensorFloatView values) {
        if (values.Size() == 0) {
            return;
        }

        const auto maxElemIt = std::max_element(nva2x::begin(values), nva2x::end(values));
        const auto maxElem = *maxElemIt;

        float expSum = 0.0f;
        for (auto& value : values) {
            const float v = std::exp(value - maxElem);
            value = v;
            expSum += v;
        }

        for (auto& value : values) {
            value /= expSum;
        }
    }

    void ApplyEmotionContrast(nva2x::HostTensorFloatView a2eEmotion, float emotionContrast) {
        for (auto& emotion : a2eEmotion) {
            emotion *= emotionContrast;
        }
        inplace_softmax(a2eEmotion);
    }

    void ApplyNullifyUnmappedEmotion(nva2x::HostTensorFloatView a2eEmotion,
        const int* emotionCorrespondence, std::size_t emotionCorrespondenceSize
        ) {
        for (std::size_t i = 0; i < emotionCorrespondenceSize; ++i) {
            if (emotionCorrespondence[i] == -1) {
                a2eEmotion.Data()[i] = 0.0f;
            }
        }
    }

    void ApplyMaxEmotions(nva2x::HostTensorFloatView a2eEmotion, std::size_t maxEmotions, std::vector<std::size_t>& indices) {
        assert(a2eEmotion.Size() == indices.size());
        if (maxEmotions >= a2eEmotion.Size()) {
            return;
        }

        // Initialize indices from 0 to (a2eEmotion.Size() - 1)
        std::iota(indices.begin(), indices.end(), 0);

        // Identify the n-th smallest weights.
        const auto it_limit = indices.end() - maxEmotions;
        std::nth_element(indices.begin(), it_limit, indices.end(), [&a2eEmotion](auto a, auto b) {
            return a2eEmotion.Data()[a] < a2eEmotion.Data()[b];
        });

        // Setting the probabilities at zero for the smallest indices
        for (auto it = indices.begin(); it != it_limit; ++it) {
            a2eEmotion.Data()[*it] = 0.0f;
        }
    }

    void ApplyA2FEmotionIndexConversion(
        nva2x::HostTensorFloatView a2fEmotion, nva2x::HostTensorFloatConstView a2eEmotion,
        const int* emotionCorrespondence, std::size_t emotionCorrespondenceSize
        ) {
        assert(a2eEmotion.Size() == emotionCorrespondenceSize);

        for (std::size_t i = 0; i < emotionCorrespondenceSize; ++i) {
            const int j = emotionCorrespondence[i];
            assert(j >= -1);
            if (j == -1) {
                continue;
            }
            assert(j < static_cast<int>(a2fEmotion.Size()));
            a2fEmotion.Data()[j] = a2eEmotion.Data()[i];
        }
    }

    void ApplyBlending(
        nva2x::HostTensorFloatView a2fEmotion, float blendCoef, nva2x::HostTensorFloatConstView blendSource
        ) {
        assert(a2fEmotion.Size() == blendSource.Size());
        std::transform(
            nva2x::begin(a2fEmotion),
            nva2x::end(a2fEmotion),
            nva2x::begin(blendSource),
            nva2x::begin(a2fEmotion),
            [blendCoef](float emotion, float blendSource) {
                return (1.0f - blendCoef) * emotion + blendCoef * blendSource;
            }
        );
    }

    void ApplyTransitionSmoothing(
        nva2x::HostTensorFloatView a2fEmotion, float dt, float liveTransitionTime, nva2x::HostTensorFloatConstView prevBlendEmotion
        ) {
        assert(a2fEmotion.Size() == prevBlendEmotion.Size());
        // ensure the transition time is at least 1e-3 seconds.
        const float transitionTime = std::max(liveTransitionTime, 1e-3f);
        // ensure w is at most 1.
        const float w = std::min(dt / transitionTime, 1.0f);

        ApplyBlending(a2fEmotion, 1.0f - w, prevBlendEmotion);
    }

    void ApplyEmotionStrength(nva2x::HostTensorFloatView a2fEmotion, float emotionStrength) {
        for (auto& emotion : a2fEmotion) {
            emotion *= emotionStrength;
        }
    }
} // Anonymous namespace


namespace nva2e {

IPostProcessor::~IPostProcessor() = default;

std::error_code PostProcessor::Init(const PostProcessData& data, const PostProcessParams& params) {
    A2E_CHECK_ERROR_WITH_MSG(
        data.emotionCorrespondenceSize == data.inferenceEmotionLength,
        "Wrong emotion correspondence size",
        nva2x::ErrorCode::eInvalidValue
        );
    A2E_CHECK_ERROR_WITH_MSG(
        data.emotionCorrespondence != nullptr,
        "emotion correspondence cannot be null",
        nva2x::ErrorCode::eNullPointer
        );
    for (std::size_t i = 0; i < data.emotionCorrespondenceSize; ++i) {
        const auto correspondence = data.emotionCorrespondence[i];
        A2E_CHECK_ERROR_WITH_MSG(
            (-1 <= correspondence) && (correspondence < static_cast<int>(data.outputEmotionLength)),
            "Wrong emotion correspondence value",
            nva2x::ErrorCode::eInvalidValue
            );
    }

    _data = data;

    // Need to take copies.
    _dataBuffers.emotionCorrespondence.resize(_data.emotionCorrespondenceSize);
    std::copy(
        _data.emotionCorrespondence,
        _data.emotionCorrespondence + _data.emotionCorrespondenceSize,
        _dataBuffers.emotionCorrespondence.begin()
    );
    _data.emotionCorrespondence = _dataBuffers.emotionCorrespondence.data();

    _workingBuffers.indices.resize(_data.inferenceEmotionLength);

    A2E_CHECK_RESULT_WITH_MSG(SetParameters(params), "Unable to set post-processor parameters");
    A2E_CHECK_RESULT_WITH_MSG(Reset(), "Unable to reset post-processor");

    return nva2x::ErrorCode::eSuccess;
}

const PostProcessData& PostProcessor::GetData() const {
    return _data;
}

std::error_code PostProcessor::SetParameters(const PostProcessParams& params) {
    A2E_CHECK_ERROR_WITH_MSG(
        params.beginningEmotion.Size() == _data.outputEmotionLength,
        "Wrong beginning emotion size",
        nva2x::ErrorCode::eInvalidValue
        );
    A2E_CHECK_ERROR_WITH_MSG(
        params.preferredEmotion.Size() == _data.outputEmotionLength,
        "Wrong preferred emotion size",
        nva2x::ErrorCode::eInvalidValue
        );

    _params = params;

    // Need to take copies.
    A2E_CHECK_RESULT_WITH_MSG(
        _paramsBuffers.beginningEmotion.Init(_params.beginningEmotion),
        "Unable to copy beginning emotion"
    );
    _params.beginningEmotion = _paramsBuffers.beginningEmotion;

    A2E_CHECK_RESULT_WITH_MSG(
        _paramsBuffers.preferredEmotion.Init(_params.preferredEmotion),
        "Unable to copy preferred emotion"
    );
    _params.preferredEmotion = _paramsBuffers.preferredEmotion;

    return nva2x::ErrorCode::eSuccess;
}

const PostProcessParams& PostProcessor::GetParameters() const {
    return _params;
}

std::error_code PostProcessor::Reset() {
    _stateBuffers.firstFrame = true;
    A2E_CHECK_RESULT(_stateBuffers.prevEmotion.Allocate(_data.outputEmotionLength));
    A2E_CHECK_RESULT(nva2x::FillOnHost(_stateBuffers.prevEmotion, 0.0f));
    A2E_CHECK_RESULT(_stateBuffers.prevBlendedEmotion.Allocate(_data.outputEmotionLength));
    A2E_CHECK_RESULT(nva2x::FillOnHost(_stateBuffers.prevBlendedEmotion, 0.0f));

    return nva2x::ErrorCode::eSuccess;
}

std::error_code PostProcessor::PostProcess(
    nva2x::HostTensorFloatView outputEmotions, nva2x::HostTensorFloatConstView inputEmotions
    ) {
    A2E_CHECK_ERROR_WITH_MSG(inputEmotions.Size() == _data.inferenceEmotionLength, "Mismatch size for input emotions", nva2x::ErrorCode::eMismatch);
    A2E_CHECK_ERROR_WITH_MSG(outputEmotions.Size() == _data.outputEmotionLength, "Mismatch size for output emotions", nva2x::ErrorCode::eMismatch);

    _workingBuffers.a2eEmotions.Init(inputEmotions);
    if (_stateBuffers.firstFrame) {
        _workingBuffers.a2fEmotions.Init(_params.beginningEmotion);
    }
    assert(_workingBuffers.a2eEmotions.Size() == _data.inferenceEmotionLength);
    assert(_workingBuffers.a2fEmotions.Size() == _data.outputEmotionLength);

    ApplyEmotionContrast(_workingBuffers.a2eEmotions, _params.emotionContrast);
    ApplyNullifyUnmappedEmotion(_workingBuffers.a2eEmotions, _data.emotionCorrespondence, _data.emotionCorrespondenceSize);
    ApplyMaxEmotions(_workingBuffers.a2eEmotions, _params.maxEmotions, _workingBuffers.indices);
    ApplyA2FEmotionIndexConversion(
        _workingBuffers.a2fEmotions,
        _workingBuffers.a2eEmotions,
        _data.emotionCorrespondence,
        _data.emotionCorrespondenceSize
    );

    // Blend with either the beginning emotion or the previous one.
    nva2x::HostTensorFloatConstView blendSource = _stateBuffers.firstFrame ? _paramsBuffers.beginningEmotion : _stateBuffers.prevEmotion;
    ApplyBlending(_workingBuffers.a2fEmotions, _params.liveBlendCoef, blendSource);
    A2E_CHECK_RESULT_WITH_MSG(
        nva2x::CopyHostToHost(_stateBuffers.prevEmotion, _workingBuffers.a2fEmotions),
        "Unable to save previous emotion"
    );

    // Blend with preferred emotion.
    if (_params.enablePreferredEmotion) {
        ApplyBlending(_workingBuffers.a2fEmotions, _params.preferredEmotionStrength, _paramsBuffers.preferredEmotion);
    }

    // Apply transition.
    if (!_stateBuffers.firstFrame) {
        ApplyTransitionSmoothing(
            _workingBuffers.a2fEmotions, _params.fixedDt, _params.liveTransitionTime, _stateBuffers.prevBlendedEmotion
            );
    }
    A2E_CHECK_RESULT_WITH_MSG(
        nva2x::CopyHostToHost(_stateBuffers.prevBlendedEmotion, _workingBuffers.a2fEmotions),
        "Unable to save previous blended emotion"
    );

    // Apply strength.
    ApplyEmotionStrength(_workingBuffers.a2fEmotions, _params.emotionStrength);

    A2E_CHECK_RESULT_WITH_MSG(
        nva2x::CopyHostToHost(outputEmotions, _workingBuffers.a2fEmotions),
        "Unable to copy output emotion"
    );

    _stateBuffers.firstFrame = false;

    return nva2x::ErrorCode::eSuccess;
}

void PostProcessor::Destroy() {
    delete this;
}

IPostProcessor *CreatePostProcessor_INTERNAL() {
    LOG_DEBUG("CreatePostProcessor_INTERNAL()");
    return new PostProcessor();
}

bool AreEqual_INTERNAL(const PostProcessParams& a, const PostProcessParams& b) {
    return (
        a.emotionContrast == b.emotionContrast &&
        a.maxEmotions == b.maxEmotions &&
        a.beginningEmotion.Size() == b.beginningEmotion.Size() &&
        std::equal(
            a.beginningEmotion.Data(),
            a.beginningEmotion.Data() + a.beginningEmotion.Size(),
            b.beginningEmotion.Data()
        ) &&
        a.preferredEmotion.Size() == b.preferredEmotion.Size() &&
        std::equal(
            a.preferredEmotion.Data(),
            a.preferredEmotion.Data() + a.preferredEmotion.Size(),
            b.preferredEmotion.Data()
        ) &&
        a.liveBlendCoef == b.liveBlendCoef &&
        a.enablePreferredEmotion == b.enablePreferredEmotion &&
        a.preferredEmotionStrength == b.preferredEmotionStrength &&
        a.liveTransitionTime == b.liveTransitionTime &&
        a.fixedDt == b.fixedDt &&
        a.emotionStrength == b.emotionStrength
    );
}

} // namespace nva2e
