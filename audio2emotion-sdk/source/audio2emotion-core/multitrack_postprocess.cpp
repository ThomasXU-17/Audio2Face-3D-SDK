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
#include "audio2emotion/internal/multitrack_postprocess.h"
#include "audio2emotion/internal/multitrack_postprocess_cuda.h"
#include "audio2emotion/internal/logger.h"
#include "audio2emotion/internal/macros.h"
#include "audio2x/error.h"

#include <cassert>
#include <limits>

#include <cuda_runtime_api.h>

namespace nva2e {

IMultiTrackPostProcessor::~IMultiTrackPostProcessor() = default;

MultiTrackPostProcessor::MultiTrackPostProcessor(bool useFastPath)
  : _useFastPath(useFastPath) {
}

std::error_code MultiTrackPostProcessor::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackPostProcessor::Init(const HostData& data, const Params& params, std::size_t nbTracks) {
  _initialized = false;

  if (_useFastPath) {
    int current_device;
    A2E_CUDA_CHECK_ERROR(cudaGetDevice(&current_device), nva2x::ErrorCode::eCudaDeviceGetError);
    cudaDeviceProp deviceProp;
    A2E_CUDA_CHECK_ERROR(cudaGetDeviceProperties(&deviceProp, current_device), nva2x::ErrorCode::eCudaDeviceGetError);
    if (deviceProp.warpSize != nva2e::cuda::kExpectedWarpSize) {
      // We assert because we want to adapt the code if warp size ever changes.
      assert(!"Warp size is not 32, using slow path");
      _useFastPath = false;
    }
  }

  A2E_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);

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

  A2E_CHECK_ERROR_WITH_MSG(
    data.outputEmotionLength == params.beginningEmotion.Size(),
    "Wrong beginning emotion length",
    nva2x::ErrorCode::eMismatch
    );
  A2E_CHECK_ERROR_WITH_MSG(
    data.outputEmotionLength == params.preferredEmotion.Size(),
    "Wrong preferred emotion length",
    nva2x::ErrorCode::eMismatch
  );

  // Take the CPU copies.
  _inferenceEmotionLength = data.inferenceEmotionLength;
  _outputEmotionLength = data.outputEmotionLength;

  _params.clear();
  _params.resize(nbTracks, params);

  A2E_CHECK_RESULT_WITH_MSG(
    _beginningEmotionHost.Allocate(_outputEmotionLength * nbTracks),
    "Unable to allocate beginning emotion"
    );
  A2E_CHECK_RESULT_WITH_MSG(
    _preferredEmotionHost.Allocate(_outputEmotionLength * nbTracks),
    "Unable to allocate preferred emotion"
    );

  _postProcessParamsStride = 8 + data.outputEmotionLength;
  A2E_CHECK_RESULT_WITH_MSG(
    _postProcessParams.Allocate(nbTracks * _postProcessParamsStride),
    "Unable to allocate post-process params"
    );
  A2E_CHECK_RESULT_WITH_MSG(
    _preferredEmotion.Allocate(_outputEmotionLength * nbTracks),
    "Unable to allocate preferred emotion"
    );
  std::vector<float> paramsHost;
  for (std::size_t i = 0; i < nbTracks; ++i) {
    A2E_CHECK_RESULT_WITH_MSG(SetParametersInternal(i, _params[i], paramsHost), "Unable to set post-process parameters");
  }

  const auto nbBitMasks = (nbTracks + nb_bits - 1) / nb_bits;
  _activeTracks.clear();
  _activeTracks.resize(nbBitMasks, ~bits_type(0));

  A2E_CHECK_RESULT_WITH_MSG(_activeTracksDevice.Allocate(nbBitMasks), "Unable to allocate active tracks");

  // Upload data to GPU.
  std::vector<std::int64_t> a2eEmotionCorrespondenceHost(
    data.emotionCorrespondence, data.emotionCorrespondence + data.emotionCorrespondenceSize
    );
  A2E_CHECK_RESULT_WITH_MSG(
    _a2eEmotionCorrespondence.Init(nva2x::ToConstView(a2eEmotionCorrespondenceHost), _cudaStream),
    "Unable to initialize a2e emotion correspondence"
    );
  std::vector<std::int64_t> a2fEmotionCorrespondenceHost(data.outputEmotionLength, -1);
  for (std::size_t i = 0; i < data.emotionCorrespondenceSize; ++i) {
    const auto j = data.emotionCorrespondence[i];
    if (j == -1) {
      continue;
    }
    A2E_CHECK_ERROR_WITH_MSG(j < data.outputEmotionLength, "Emotion correspondence is out of bounds", nva2x::ErrorCode::eOutOfBounds);
    a2fEmotionCorrespondenceHost[j] = i;
  }
  A2E_CHECK_RESULT_WITH_MSG(
    _a2fEmotionCorrespondence.Init(nva2x::ToConstView(a2fEmotionCorrespondenceHost), _cudaStream),
    "Unable to initialize a2f emotion correspondence"
    );

  // State and work buffers:
  // - firstFrame
  // - a2e work buffer
  // - a2f work buffer and previous emotion
  // - a2f prevEmotion
  // - a2f prevBlendedEmotion
  _stateAndWorkBuffersStride = 1 + data.inferenceEmotionLength + data.outputEmotionLength * 3;
  A2E_CHECK_RESULT_WITH_MSG(
    _stateAndWorkBuffers.Allocate(_stateAndWorkBuffersStride * nbTracks),
    "Unable to allocate state and work buffers"
    );
  A2E_CHECK_RESULT_WITH_MSG(
    nva2x::FillOnDevice(_stateAndWorkBuffers, 1.0f, _cudaStream),
    "Unable to initialize state and work buffers"
    );

  A2E_CUDA_CHECK_ERROR(cudaStreamSynchronize(_cudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError);

  _initialized = true;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackPostProcessor::SetParameters(std::size_t trackIndex, const Params& params) {
  A2E_CHECK_ERROR_WITH_MSG(_initialized, "Post-processor is not initialized", nva2x::ErrorCode::eNotInitialized);

  A2E_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nva2x::ErrorCode::eOutOfBounds);

  std::vector<float> paramsHost;
  return SetParametersInternal(trackIndex, params, paramsHost);
}

const MultiTrackPostProcessor::Params* MultiTrackPostProcessor::GetParameters(std::size_t trackIndex) const {
  A2E_CHECK_ERROR_WITH_MSG(_initialized, "Post-processor is not initialized", nullptr);

  A2E_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nullptr);
  return &_params[trackIndex];
}

std::error_code MultiTrackPostProcessor::Reset(std::size_t trackIndex) {
  A2E_CHECK_ERROR_WITH_MSG(_initialized, "Post-processor is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2E_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nva2x::ErrorCode::eOutOfBounds);
  A2E_CHECK_RESULT_WITH_MSG(
    nva2x::FillOnDevice(
      _stateAndWorkBuffers.View(trackIndex * _stateAndWorkBuffersStride, _stateAndWorkBuffersStride),
      1.0f,
      _cudaStream
      ),
    "Unable to reset state and work buffers"
    );
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackPostProcessor::SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) {
  A2E_CHECK_ERROR_WITH_MSG(_initialized, "Post-processor is not initialized", nva2x::ErrorCode::eNotInitialized);
  if (activeTracks == 0) {
    // Set all tracks to active.
    std::fill(_activeTracks.begin(), _activeTracks.end(), ~bits_type(0));
  }
  else {
    A2E_CHECK_ERROR_WITH_MSG(activeTracksSize == _activeTracks.size(), "Mismatch in active tracks size", nva2x::ErrorCode::eMismatch);
    std::copy(activeTracks, activeTracks + activeTracksSize, _activeTracks.begin());
  }

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackPostProcessor::PostProcess(
  nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
  nva2x::DeviceTensorFloatView outputEmotions, const nva2x::TensorBatchInfo& outputEmotionsInfo
  ) {
  A2E_CHECK_ERROR_WITH_MSG(_initialized, "Post-processor is not initialized", nva2x::ErrorCode::eNotInitialized);

  const auto nbTracks = _params.size();
  A2E_CHECK_ERROR_WITH_MSG(
    inputEmotionsInfo.stride * nbTracks == inputEmotions.Size(),
    "Input emotions size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2E_CHECK_RESULT_WITH_MSG(nva2x::ValidateTensorBatchInfo(inputEmotions, inputEmotionsInfo), "Input emotions is invalid");
  A2E_CHECK_ERROR_WITH_MSG(
    outputEmotionsInfo.stride * nbTracks == outputEmotions.Size(),
    "Output emotions size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2E_CHECK_RESULT_WITH_MSG(nva2x::ValidateTensorBatchInfo(outputEmotions, outputEmotionsInfo), "Output emotions is invalid");

  A2E_CHECK_ERROR_WITH_MSG(_inferenceEmotionLength == inputEmotionsInfo.size, "Input emotions size does not match", nva2x::ErrorCode::eMismatch);
  A2E_CHECK_ERROR_WITH_MSG(_outputEmotionLength == outputEmotionsInfo.size, "Output emotions size does not match", nva2x::ErrorCode::eMismatch);

  auto checkAllSet = [](const std::vector<bits_type>& bits, std::size_t nbTracks) {
    const auto wholeCount = nbTracks / (8 * sizeof(bits_type));
    for (std::size_t i = 0; i < wholeCount; ++i) {
      if (bits[i] != ~bits_type(0)) {
        return false;
      }
    }

    const auto partialCount = nbTracks % (8 * sizeof(bits_type));
    if (partialCount > 0) {
      const auto mask = (bits_type(1) << partialCount) - 1;
      if ((bits[wholeCount] & mask) != mask) {
        return false;
      }
    }

    return true;
  };
  const bool allTracksActive = checkAllSet(_activeTracks, nbTracks);

  const std::uint64_t* activeTracks = nullptr;
  if (!allTracksActive) {
    A2E_CHECK_RESULT_WITH_MSG(
      cuda::PostProcess_Set(_activeTracksDevice.Data(), _activeTracks.data(), _activeTracks.size(), _cudaStream),
      "Unable to set active tracks"
      );
    activeTracks = _activeTracksDevice.Data();
  }

  A2E_CHECK_RESULT_WITH_MSG(
    cuda::PostProcess(
      outputEmotions.Data(), outputEmotionsInfo.offset, outputEmotionsInfo.stride, outputEmotionsInfo.size,
      inputEmotions.Data(), inputEmotionsInfo.offset, inputEmotionsInfo.stride, inputEmotionsInfo.size,
      _a2eEmotionCorrespondence.Data(), _a2fEmotionCorrespondence.Data(),
      _postProcessParams.Data(), _postProcessParamsStride,
      _preferredEmotion.Data(),
      _stateAndWorkBuffers.Data(), _stateAndWorkBuffersStride,
      activeTracks,
      nbTracks,
      _cudaStream,
      _useFastPath
      ),
    "Unable to post-process"
    );

  return nva2x::ErrorCode::eSuccess;
}

std::size_t MultiTrackPostProcessor::GetInputEmotionsSize() const {
  A2E_CHECK_ERROR_WITH_MSG(_initialized, "Post-processor is not initialized", 0);
  return _inferenceEmotionLength;
}

std::size_t MultiTrackPostProcessor::GetOutputEmotionsSize() const {
  A2E_CHECK_ERROR_WITH_MSG(_initialized, "Post-processor is not initialized", 0);
  return _outputEmotionLength;
}

nva2x::DeviceTensorFloatView MultiTrackPostProcessor::GetPreferredEmotion(std::size_t trackIndex) {
  A2E_CHECK_ERROR_WITH_MSG(_initialized, "Post-processor is not initialized", {});
  A2E_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", {});
  return _preferredEmotion.View(trackIndex * _outputEmotionLength, _outputEmotionLength);
}

nva2x::DeviceTensorFloatConstView MultiTrackPostProcessor::GetPreferredEmotion(std::size_t trackIndex) const {
  A2E_CHECK_ERROR_WITH_MSG(_initialized, "Post-processor is not initialized", {});
  A2E_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", {});
  return _preferredEmotion.View(trackIndex * _outputEmotionLength, _outputEmotionLength);
}

void MultiTrackPostProcessor::Destroy() {
  delete this;
}

std::error_code MultiTrackPostProcessor::SetParametersInternal(std::size_t trackIndex, const Params& params, std::vector<float>& paramsHost) {
  // There is no actual proxy validation here, but keep the name.
  auto& proxy = _params[trackIndex];
  proxy.maxEmotions = params.maxEmotions;
  proxy.emotionContrast = params.emotionContrast;
  proxy.liveBlendCoef = params.liveBlendCoef;
  proxy.enablePreferredEmotion = params.enablePreferredEmotion;
  proxy.preferredEmotionStrength = params.preferredEmotionStrength;
  proxy.liveTransitionTime = params.liveTransitionTime;
  proxy.fixedDt = params.fixedDt;
  proxy.emotionStrength = params.emotionStrength;

  const auto beginningEmotion = _beginningEmotionHost.View(trackIndex * _outputEmotionLength, _outputEmotionLength);
  A2E_CHECK_RESULT_WITH_MSG(
    nva2x::CopyHostToHost(beginningEmotion, params.beginningEmotion, _cudaStream),
    "Unable to copy beginning emotion"
    );
  proxy.beginningEmotion = beginningEmotion;

  const auto preferredEmotion = _preferredEmotionHost.View(trackIndex * _outputEmotionLength, _outputEmotionLength);
  A2E_CHECK_RESULT_WITH_MSG(
    nva2x::CopyHostToHost(preferredEmotion, params.preferredEmotion, _cudaStream),
    "Unable to copy preferred emotion"
    );
  proxy.preferredEmotion = preferredEmotion;

  // Copy to device.
  paramsHost.resize(8 + _outputEmotionLength);
  // We store the max emotion as a float, but it's an int.
  static_assert(
    sizeof(float) == sizeof(std::uint32_t),
    "float and uint32_t must have the same size since we reinterpret the float as an uint32_t"
    );
  assert(params.maxEmotions <= std::numeric_limits<std::uint32_t>::max());
  std::uint32_t maxEmotions = static_cast<std::uint32_t>(params.maxEmotions);
  paramsHost[0] = *reinterpret_cast<const float*>(&maxEmotions);
  paramsHost[1] = params.emotionContrast;
  paramsHost[2] = params.liveBlendCoef;
  paramsHost[3] = params.enablePreferredEmotion ? 1.0f : 0.0f;
  paramsHost[4] = params.preferredEmotionStrength;
  paramsHost[5] = params.liveTransitionTime;
  paramsHost[6] = params.fixedDt;
  paramsHost[7] = params.emotionStrength;
  std::copy(
    nva2x::begin(params.beginningEmotion),
    nva2x::end(params.beginningEmotion),
    paramsHost.begin() + 8
    );

  A2E_CHECK_RESULT_WITH_MSG(
    nva2x::CopyHostToDevice(
      _postProcessParams.View(trackIndex * _postProcessParamsStride, paramsHost.size()),
      nva2x::ToConstView(paramsHost),
      _cudaStream
      ),
    "Unable to copy post-process params to device"
    );
  A2E_CHECK_RESULT_WITH_MSG(
    nva2x::CopyHostToDevice(
      _preferredEmotion.View(trackIndex * _outputEmotionLength, _outputEmotionLength),
      params.preferredEmotion,
      _cudaStream
      ),
    "Unable to copy preferred emotion to device"
    );
  A2E_CUDA_CHECK_ERROR(cudaStreamSynchronize(_cudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError);

  return nva2x::ErrorCode::eSuccess;
}


IMultiTrackPostProcessor *CreateMultiTrackPostProcessor_INTERNAL() {
  LOG_DEBUG("CreateMultiTrackPostProcessor_INTERNAL()");
  return new MultiTrackPostProcessor();
}

} // namespace nva2e
