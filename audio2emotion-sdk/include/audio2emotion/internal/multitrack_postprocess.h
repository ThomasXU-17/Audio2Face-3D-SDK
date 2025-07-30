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

#include "audio2emotion/multitrack_postprocess.h"
#include "audio2emotion/internal/postprocess.h"

#include <vector>

namespace nva2e {

class MultiTrackPostProcessor : public IMultiTrackPostProcessor {
public:
  MultiTrackPostProcessor(bool useFastPath = true);

  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(const HostData& data, const Params& params, std::size_t nbTracks) override; // GPU Async

  std::error_code SetParameters(std::size_t trackIndex, const Params& params) override;
  const Params* GetParameters(std::size_t trackIndex) const override;

  std::error_code Reset(std::size_t trackIndex) override;
  std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) override;
  std::error_code PostProcess(
    nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
    nva2x::DeviceTensorFloatView outputEmotions, const nva2x::TensorBatchInfo& outputEmotionsInfo
    ) override; // GPU Async

  std::size_t GetInputEmotionsSize() const override;
  std::size_t GetOutputEmotionsSize() const override;

  nva2x::DeviceTensorFloatView GetPreferredEmotion(std::size_t trackIndex) override;
  nva2x::DeviceTensorFloatConstView GetPreferredEmotion(std::size_t trackIndex) const override;

  void Destroy() override;

private:
  std::error_code SetParametersInternal(std::size_t trackIndex, const Params& params, std::vector<float>& paramsHost);

  cudaStream_t _cudaStream{nullptr};
  std::size_t _inferenceEmotionLength{0};
  std::size_t _outputEmotionLength{0};
  std::vector<nva2e::PostProcessParams> _params;
  nva2x::HostTensorFloat _beginningEmotionHost;
  nva2x::HostTensorFloat _preferredEmotionHost;

  // This is just a more compact bitset for the active tracks.
  // We would rather use an off-the-shelf bitset, but std ones does not support a dynamic number of bits.
  using bits_type = std::uint64_t;
  static constexpr const std::size_t nb_bits = 8 * sizeof(bits_type);
  std::vector<bits_type> _activeTracks;
  nva2x::DeviceTensorUInt64 _activeTracksDevice;

  nva2x::DeviceTensorInt64 _a2eEmotionCorrespondence;
  nva2x::DeviceTensorInt64 _a2fEmotionCorrespondence;
  nva2x::DeviceTensorFloat _postProcessParams;
  std::size_t _postProcessParamsStride{0};
  nva2x::DeviceTensorFloat _preferredEmotion;
  nva2x::DeviceTensorFloat _stateAndWorkBuffers;
  std::size_t _stateAndWorkBuffersStride{0};

  bool _initialized{false};
  bool _useFastPath{true};
};


IMultiTrackPostProcessor* CreateMultiTrackPostProcessor_INTERNAL();

} // namespace nva2e
