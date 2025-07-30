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

#include "audio2emotion/postprocess.h"
#include "audio2x/internal/tensor.h"

namespace nva2e {

class PostProcessor : public IPostProcessor {
public:
    std::error_code Init(const PostProcessData& data, const PostProcessParams& params) override;

    const PostProcessData& GetData() const override;

    std::error_code SetParameters(const PostProcessParams& params) override;
    const PostProcessParams& GetParameters() const override;

    std::error_code Reset() override;
    std::error_code PostProcess(nva2x::HostTensorFloatView outputEmotions,
                                nva2x::HostTensorFloatConstView inputEmotions) override;

    void Destroy() override;

private:
    struct WorkingBuffers {
        nva2x::HostTensorFloat a2eEmotions;
        nva2x::HostTensorFloat a2fEmotions;
        std::vector<std::size_t> indices;
    };

    struct DataBuffers {
        std::vector<int> emotionCorrespondence;
    };

    struct ParamsBuffers {
        // Used as the prevEmotion for the first frame
        nva2x::HostTensorFloat beginningEmotion;
        // Preferred emotion for a particular stream (10 dimensions)
        nva2x::HostTensorFloat preferredEmotion;
    };

    struct StateBuffers {
        // Indicate if the stream is being run for the first time.
        bool firstFrame{true};
        // Post-processed emotion vector from the previous timestamp (10 dimensions)
        nva2x::HostTensorFloat prevEmotion;
        // The emotion vector after blending with preferred emotion in the previous timestamp (10 dimensions)
        nva2x::HostTensorFloat prevBlendedEmotion;
        std::size_t inferencesSkippedCount{0};
    };

    WorkingBuffers _workingBuffers;
    PostProcessData _data;
    PostProcessParams _params;
    DataBuffers _dataBuffers;
    ParamsBuffers _paramsBuffers;
    StateBuffers _stateBuffers;
};

IPostProcessor* CreatePostProcessor_INTERNAL();

bool AreEqual_INTERNAL(const PostProcessParams& a, const PostProcessParams& b);

} // namespace nva2e
