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
#include "audio2face/internal/multitrack_animator.h"
#include "audio2face/internal/multitrack_animator_cuda.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2x/error.h"

#include <cassert>


namespace {

  template <typename bits_type>
  inline bool checkAllSet(const std::vector<bits_type>& bits, std::size_t nbTracks) {
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

}
namespace nva2f {

//////////////////////////////////////////////

IMultiTrackAnimatorPcaReconstruction::~IMultiTrackAnimatorPcaReconstruction() = default;

std::error_code MultiTrackAnimatorPcaReconstruction::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  if (_cublasHandle.Data() != nullptr) {
    A2F_CHECK_RESULT_WITH_MSG(_cublasHandle.SetCudaStream(_cudaStream), "Unable to set CUDA stream on cublas handle");
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstruction::Init(std::size_t nbTracks) {
  _initialized = false;
  _animatorDataIsSet = false;

  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  _nbTracks = nbTracks;

  A2F_CHECK_RESULT_WITH_MSG(_cublasHandle.Init(), "Unable to initialize cublas handle");
  if (_cublasHandle.Data() != nullptr) {
    A2F_CHECK_RESULT_WITH_MSG(_cublasHandle.SetCudaStream(_cudaStream), "Unable to set CUDA stream on cublas handle");
  }

  _initialized = true;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstruction::SetAnimatorData(const HostData& data) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "PCA reconstruction is not initialized", nva2x::ErrorCode::eNotInitialized);
  _animatorDataIsSet = false;

  A2F_CHECK_ERROR_WITH_MSG(data.shapesMatrix.Data() != nullptr, "Shapes matrix must not be null", nva2x::ErrorCode::eNullPointer);
  A2F_CHECK_ERROR_WITH_MSG(data.shapesMatrix.Size() > 0, "Shapes matrix must not be empty", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(data.shapesMatrix.Size() % data.shapeSize == 0, "Shapes matrix size is not a multiple of the shape size", nva2x::ErrorCode::eMismatch);

  A2F_CHECK_RESULT_WITH_MSG(_data.Init(data, _cudaStream), "Unable to initialize animator data");

  _animatorDataIsSet = true;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstruction::Reset(std::size_t trackIndex) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "PCA reconstruction is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "PCA reconstruction data is not set", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _nbTracks, "Track index is out of bounds", nva2x::ErrorCode::eOutOfBounds);
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstruction::SetActiveTracks(
  const std::uint64_t* activeTracks, std::size_t activeTracksSize
  ) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "PCA reconstruction is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "PCA reconstruction data is not set", nva2x::ErrorCode::eNotInitialized);
  // PCA reconstruction does not hold state, so we can ignore the active tracks.
  // The overhead of handling "sparsely" active tracks is likely to be higher
  // than the benefits of running the animation on a subset of tracks.
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorPcaReconstruction::Animate(
  nva2x::DeviceTensorFloatConstView inputPcaCoefs, const nva2x::TensorBatchInfo& inputPcaCoefsInfo,
  nva2x::DeviceTensorFloatView outputShapes, const nva2x::TensorBatchInfo& outputShapesInfo,
  std::size_t batchSize
  ) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "PCA reconstruction is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "PCA reconstruction data is not set", nva2x::ErrorCode::eNotInitialized);

  if (batchSize == 0) {
    batchSize = _nbTracks;
  }
  A2F_CHECK_ERROR_WITH_MSG(batchSize <= _nbTracks, "Batch size is greater than the number of tracks", nva2x::ErrorCode::eOutOfBounds);

  A2F_CHECK_ERROR_WITH_MSG(
    inputPcaCoefsInfo.stride * batchSize == inputPcaCoefs.Size(),
    "Input PCA coefs size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2F_CHECK_RESULT_WITH_MSG(nva2x::ValidateTensorBatchInfo(inputPcaCoefs, inputPcaCoefsInfo), "Input PCA coefs is invalid");
  A2F_CHECK_ERROR_WITH_MSG(
    outputShapesInfo.stride * batchSize == outputShapes.Size(),
    "Output shapes size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2F_CHECK_RESULT_WITH_MSG(nva2x::ValidateTensorBatchInfo(outputShapes, outputShapesInfo), "Output shapes is invalid");

  const float alpha = 1.0f;
  const float beta = 0.0f;

  const auto data = _data.GetDeviceView();
  A2F_CHECK_ERROR_WITH_MSG(data.shapeSize == outputShapesInfo.size, "Shape size does not match", nva2x::ErrorCode::eMismatch);
  A2F_CHECK_ERROR_WITH_MSG(data.shapesMatrix.Size() / data.shapeSize == inputPcaCoefsInfo.size, "Input PCA coefs size does not match", nva2x::ErrorCode::eMismatch);
  cublasStatus_t status = cublasSgemm(
    _cublasHandle.Data(), CUBLAS_OP_N, CUBLAS_OP_N,
    static_cast<int>(data.shapeSize), static_cast<int>(batchSize), static_cast<int>(inputPcaCoefsInfo.size),
    &alpha, data.shapesMatrix.Data(), static_cast<int>(data.shapeSize),
    inputPcaCoefs.Data() + inputPcaCoefsInfo.offset, static_cast<int>(inputPcaCoefsInfo.stride),
    &beta, outputShapes.Data() + outputShapesInfo.offset, static_cast<int>(outputShapesInfo.stride)
    );
  A2F_CHECK_ERROR_WITH_MSG(status == CUBLAS_STATUS_SUCCESS, "Unable to run matrix multiplication", ErrorCode::eCublasExecutionError);

  return nva2x::ErrorCode::eSuccess;
}

void MultiTrackAnimatorPcaReconstruction::Destroy() {
  delete this;
}

//////////////////////////////////////////////

IMultiTrackAnimatorSkin::~IMultiTrackAnimatorSkin() = default;

std::error_code MultiTrackAnimatorSkin::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkin::Init(const Params& params, std::size_t nbTracks) {
  _initialized = false;
  _animatorDataIsSet = false;

  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  _params.clear();
  _params.resize(nbTracks, params);

  const auto nbBitMasks = (nbTracks + nb_bits - 1) / nb_bits;
  _activeTracks.clear();
  _activeTracks.resize(nbBitMasks, ~bits_type(0));
  _initializedTracks.clear();
  _initializedTracks.resize(nbBitMasks, bits_type(0));

  A2F_CHECK_RESULT_WITH_MSG(_activeTracksDevice.Allocate(nbBitMasks), "Unable to allocate active tracks");
  A2F_CHECK_RESULT_WITH_MSG(_initializedTracksDevice.Allocate(nbBitMasks), "Unable to allocate initialized tracks");

  _initialized = true;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkin::SetAnimatorData(const HostData& data, float dt) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Skin animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  _animatorDataIsSet = false;

  // Store the views from/to where the data will be read/written.
  A2F_CHECK_ERROR_WITH_MSG(data.neutralPose.Data() != nullptr, "Neutral pose must not be null", nva2x::ErrorCode::eNullPointer);
  A2F_CHECK_ERROR_WITH_MSG(data.neutralPose.Size() > 0, "Neutral pose must not be empty", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(data.neutralPose.Size() % 3 == 0, "Neutral pose size must be a multiple of 3", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(data.lipOpenPoseDelta.Data() != nullptr, "Lip open pose delta must not be null", nva2x::ErrorCode::eNullPointer);
  A2F_CHECK_ERROR_WITH_MSG(data.lipOpenPoseDelta.Size() == data.neutralPose.Size(), "Lip open pose delta size does not match neutral pose size", nva2x::ErrorCode::eMismatch);
  A2F_CHECK_ERROR_WITH_MSG(data.eyeClosePoseDelta.Data() != nullptr, "Eye close pose delta must not be null", nva2x::ErrorCode::eNullPointer);
  A2F_CHECK_ERROR_WITH_MSG(data.eyeClosePoseDelta.Size() == data.neutralPose.Size(), "Eye close pose delta size does not match neutral pose size", nva2x::ErrorCode::eMismatch);

  // Initialize the animator data.
  // 3 arrays are packed in a single array for memory efficiency.
  const auto poseSize = data.neutralPose.Size();
  const auto numVertices = poseSize / 3;
  const auto nbTracks = _params.size();

  _animatorDataStride = 3;
  std::vector<float> animatorDataHost(poseSize * _animatorDataStride);
  for (std::size_t i = 0; i < poseSize; ++i) {
    animatorDataHost[i * _animatorDataStride + 0] = data.eyeClosePoseDelta.Data()[i];
    animatorDataHost[i * _animatorDataStride + 1] = data.lipOpenPoseDelta.Data()[i];
    animatorDataHost[i * _animatorDataStride + 2] = data.neutralPose.Data()[i];
  }
  A2F_CHECK_RESULT_WITH_MSG(_animatorData.Init(nva2x::ToConstView(animatorDataHost), _cudaStream), "Unable to initialize animator data");

  // Initialize the face mask lower.
  // It is kept separate tensor since there is 1 value per vertex instead of per vertex component.
  // We could explore packing it in the animator data, but this is not done for now.
  A2F_CHECK_RESULT_WITH_MSG(
    _faceMaskLower.Allocate(numVertices * nbTracks), "Unable to allocate face mask lower"
    );

  // Initialize the interp data.
  // It is kept separate tensor since data will be written to that tensor as well as read.
  _interpDataStride = kInterpDegree * 2;
  A2F_CHECK_RESULT_WITH_MSG(_interpData.Allocate(poseSize * nbTracks * _interpDataStride), "Unable to allocate interp data");

  // The skin parameters are packed and stored per track.
  _dt = dt;

  _skinParamsStride = 9;
  A2F_CHECK_RESULT_WITH_MSG(_skinParams.Allocate(nbTracks * _skinParamsStride), "Unable to allocate skin params");
  std::vector<float> paramsHost;
  for (std::size_t i = 0; i < nbTracks; ++i) {
    A2F_CHECK_RESULT_WITH_MSG(SetParametersInternal(i, _params[i], paramsHost, true), "Unable to set skin parameters");
  }

  _animatorDataIsSet = true;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkin::SetParameters(std::size_t trackIndex, const Params& params) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Skin animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Skin animator data is not set", nva2x::ErrorCode::eNotInitialized);

  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nva2x::ErrorCode::eOutOfBounds);

  const bool needToUpdateMask = (
    _params[trackIndex].faceMaskLevel != params.faceMaskLevel ||
    _params[trackIndex].faceMaskSoftness != params.faceMaskSoftness
  );

  std::vector<float> paramsHost;
  return SetParametersInternal(trackIndex, params, paramsHost, needToUpdateMask);
}

const MultiTrackAnimatorSkin::Params* MultiTrackAnimatorSkin::GetParameters(std::size_t trackIndex) const {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Skin animator is not initialized", nullptr);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Skin animator data is not set", nullptr);

  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nullptr);
  return &_params[trackIndex];
}

std::error_code MultiTrackAnimatorSkin::Reset(std::size_t trackIndex) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Skin animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Skin animator data is not set", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nva2x::ErrorCode::eOutOfBounds);
  _initializedTracks[trackIndex / nb_bits] &= ~(bits_type(1) << (trackIndex % nb_bits));
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkin::SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Skin animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Skin animator data is not set", nva2x::ErrorCode::eNotInitialized);
  if (activeTracks == 0) {
    // Set all tracks to active.
    std::fill(_activeTracks.begin(), _activeTracks.end(), ~bits_type(0));
  }
  else {
    A2F_CHECK_ERROR_WITH_MSG(activeTracksSize == _activeTracks.size(), "Mismatch in active tracks size", nva2x::ErrorCode::eMismatch);
    std::copy(activeTracks, activeTracks + activeTracksSize, _activeTracks.begin());
  }

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorSkin::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Skin animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Skin animator data is not set", nva2x::ErrorCode::eNotInitialized);

  const auto nbTracks = _params.size();
  A2F_CHECK_ERROR_WITH_MSG(
    inputDeltasInfo.stride * nbTracks == inputDeltas.Size(),
    "Input deltas size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2F_CHECK_RESULT_WITH_MSG(nva2x::ValidateTensorBatchInfo(inputDeltas, inputDeltasInfo), "Input deltas is invalid");
  A2F_CHECK_ERROR_WITH_MSG(
    outputVerticesInfo.stride * nbTracks == outputVertices.Size(),
    "Output vertices size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2F_CHECK_RESULT_WITH_MSG(nva2x::ValidateTensorBatchInfo(outputVertices, outputVerticesInfo), "Output vertices is invalid");

  const auto poseSize = _animatorData.Size() / _animatorDataStride;
  A2F_CHECK_ERROR_WITH_MSG(poseSize == outputVerticesInfo.size, "Output vertices size does not match", nva2x::ErrorCode::eMismatch);
  A2F_CHECK_ERROR_WITH_MSG(poseSize == inputDeltasInfo.size, "Input deltas size does not match", nva2x::ErrorCode::eMismatch);

  assert(_activeTracks.size() == _initializedTracks.size());
  const bool allTracksActive = checkAllSet(_activeTracks, nbTracks);
  const bool allTracksInitialized = checkAllSet(_initializedTracks, nbTracks);

  if (allTracksActive && allTracksInitialized) {
    A2F_CHECK_RESULT_WITH_MSG(
      cuda::AnimateSkin_Everything(
        outputVertices.Data(), outputVerticesInfo.offset, outputVerticesInfo.stride,
        inputDeltas.Data(), inputDeltasInfo.offset, inputDeltasInfo.stride,
        _animatorData.Data(), _animatorDataStride,
        _faceMaskLower.Data(),
        _interpData.Data(), _interpDataStride,
        _skinParams.Data(), _skinParamsStride,
        poseSize,
        nbTracks,
        _cudaStream
        ),
      "Unable to animate with everything active and initialized"
      );
  }
  else {
    A2F_CHECK_RESULT_WITH_MSG(
      cuda::Tracks_Set(_activeTracksDevice.Data(), _activeTracks.data(), _activeTracks.size(), _cudaStream),
      "Unable to set active tracks"
      );
    A2F_CHECK_RESULT_WITH_MSG(
      cuda::Tracks_Set(_initializedTracksDevice.Data(), _initializedTracks.data(), _initializedTracks.size(), _cudaStream),
      "Unable to set initialized tracks"
      );
    A2F_CHECK_RESULT_WITH_MSG(
      cuda::AnimateSkin_Control(
        outputVertices.Data(), outputVerticesInfo.offset, outputVerticesInfo.stride,
        inputDeltas.Data(), inputDeltasInfo.offset, inputDeltasInfo.stride,
        _animatorData.Data(), _animatorDataStride,
        _faceMaskLower.Data(),
        _interpData.Data(), _interpDataStride,
        _skinParams.Data(), _skinParamsStride,
        _activeTracksDevice.Data(),
        _initializedTracksDevice.Data(),
        poseSize,
        nbTracks,
        _cudaStream
        ),
      "Unable to animate with some tracks active and initialized"
      );

    for (std::size_t i = 0; i < _activeTracks.size(); ++i) {
      _initializedTracks[i] |= _activeTracks[i];
    }
  }

  return nva2x::ErrorCode::eSuccess;
}

void MultiTrackAnimatorSkin::Destroy() {
  delete this;
}

std::error_code MultiTrackAnimatorSkin::SetParametersInternal(
  std::size_t trackIndex, const Params& params, std::vector<float>& paramsHost, bool updateMask
  ) {
  // Set with validation.
  ValidatorProxy<Params> proxy(_params[trackIndex]);
  A2F_CHECK_RESULT_WITH_MSG(proxy.lowerFaceSmoothing(params.lowerFaceSmoothing), "Unable to set lower face smoothing");
  A2F_CHECK_RESULT_WITH_MSG(proxy.upperFaceSmoothing(params.upperFaceSmoothing), "Unable to set upper face smoothing");
  A2F_CHECK_RESULT_WITH_MSG(proxy.lowerFaceStrength(params.lowerFaceStrength), "Unable to set lower face strength");
  A2F_CHECK_RESULT_WITH_MSG(proxy.upperFaceStrength(params.upperFaceStrength), "Unable to set upper face strength");
  A2F_CHECK_RESULT_WITH_MSG(proxy.faceMaskLevel(params.faceMaskLevel), "Unable to set face mask level");
  A2F_CHECK_RESULT_WITH_MSG(proxy.faceMaskSoftness(params.faceMaskSoftness), "Unable to set face mask softness");
  A2F_CHECK_RESULT_WITH_MSG(proxy.skinStrength(params.skinStrength), "Unable to set skin strength");
  A2F_CHECK_RESULT_WITH_MSG(proxy.blinkStrength(params.blinkStrength), "Unable to set blink strength");
  A2F_CHECK_RESULT_WITH_MSG(proxy.eyelidOpenOffset(params.eyelidOpenOffset), "Unable to set eyelid open offset");
  A2F_CHECK_RESULT_WITH_MSG(proxy.lipOpenOffset(params.lipOpenOffset), "Unable to set lip open offset");
  A2F_CHECK_RESULT_WITH_MSG(proxy.blinkOffset(params.blinkOffset), "Unable to set blink offset");

  // Copy to device.
  paramsHost.resize(9);
  paramsHost[0] = params.skinStrength;
  paramsHost[1] = params.eyelidOpenOffset;
  paramsHost[2] = params.blinkOffset;
  paramsHost[3] = params.blinkStrength;
  paramsHost[4] = params.lipOpenOffset;
  paramsHost[5] = params.lowerFaceSmoothing > 0.0f ? 1.0f - std::pow(0.5f, _dt / params.lowerFaceSmoothing) : -1.0f;
  paramsHost[6] = params.upperFaceSmoothing > 0.0f ? 1.0f - std::pow(0.5f, _dt / params.upperFaceSmoothing) : -1.0f;
  paramsHost[7] = params.lowerFaceStrength;
  paramsHost[8] = params.upperFaceStrength;

  A2F_CHECK_RESULT_WITH_MSG(
    nva2x::CopyHostToDevice(
      _skinParams.View(trackIndex * _skinParamsStride, paramsHost.size()), nva2x::ToConstView(paramsHost),_cudaStream),
    "Unable to copy skin params to device"
    );
  A2F_CUDA_CHECK_ERROR(cudaStreamSynchronize(_cudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError);

  if (updateMask) {
    const auto nbTracks = _params.size();
    assert(_faceMaskLower.Size() % nbTracks == 0);
    const auto numVertices = _faceMaskLower.Size() / nbTracks;
    const auto poseSize = numVertices * 3;

    A2F_CHECK_RESULT_WITH_MSG(
      cuda::CalculateFaceMaskLowerFromPackedNeutralPose(
        _faceMaskLower.Data() + trackIndex * numVertices,
        _animatorData.Data(),
        _animatorDataStride,
        poseSize,
        params.faceMaskLevel,
        params.faceMaskSoftness,
        _cudaStream
        ),
      "Unable to calculate face mask lower"
      );
  }

  return nva2x::ErrorCode::eSuccess;
}

//////////////////////////////////////////////

IMultiTrackAnimatorTongue::~IMultiTrackAnimatorTongue() = default;

std::error_code MultiTrackAnimatorTongue::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongue::Init(const Params& params, std::size_t nbTracks) {
  _initialized = false;
  _animatorDataIsSet = false;

  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  _params.clear();
  _params.resize(nbTracks, params);

  _initialized = true;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongue::SetAnimatorData(const HostData& data) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Tongue animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  _animatorDataIsSet = false;

  A2F_CHECK_ERROR_WITH_MSG(data.neutralPose.Data() != nullptr, "Neutral pose must not be null", nva2x::ErrorCode::eNullPointer);
  A2F_CHECK_ERROR_WITH_MSG(data.neutralPose.Size() > 0, "Neutral pose must not be empty", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(data.neutralPose.Size() % 3 == 0, "Neutral pose size must be a multiple of 3", nva2x::ErrorCode::eInvalidValue);

  // Initialize the animator data.
  A2F_CHECK_RESULT_WITH_MSG(_data.Init(data, _cudaStream), "Unable to initialize tongue animator data");

  const auto nbTracks = _params.size();

  _tongueParamsStride = 3;
  A2F_CHECK_RESULT_WITH_MSG(_tongueParams.Allocate(nbTracks * _tongueParamsStride), "Unable to allocate tongue params");
  std::vector<float> paramsHost;
  for (std::size_t i = 0; i < nbTracks; ++i) {
    A2F_CHECK_RESULT_WITH_MSG(SetParametersInternal(i, _params[i], paramsHost), "Unable to set tongue parameters");
  }

  _animatorDataIsSet = true;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongue::SetParameters(std::size_t trackIndex, const Params& params) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Tongue animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Tongue animator data is not set", nva2x::ErrorCode::eNotInitialized);

  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nva2x::ErrorCode::eOutOfBounds);

  std::vector<float> paramsHost;
  return SetParametersInternal(trackIndex, params, paramsHost);
}

const MultiTrackAnimatorTongue::Params* MultiTrackAnimatorTongue::GetParameters(std::size_t trackIndex) const {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Tongue animator is not initialized", nullptr);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Tongue animator data is not set", nullptr);

  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nullptr);
  return &_params[trackIndex];
}

std::error_code MultiTrackAnimatorTongue::Reset(std::size_t trackIndex) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Tongue animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Tongue animator data is not set", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nva2x::ErrorCode::eOutOfBounds);
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongue::SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Tongue animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Tongue animator data is not set", nva2x::ErrorCode::eNotInitialized);
  // Tongue animator does not hold state, so we can ignore the active tracks.
  // The overhead of handling "sparsely" active tracks is likely to be higher
  // than the benefits of running the animation on a subset of tracks.
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTongue::Animate(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
  ) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Tongue animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Tongue animator data is not set", nva2x::ErrorCode::eNotInitialized);

  const auto nbTracks = _params.size();
  A2F_CHECK_ERROR_WITH_MSG(
    inputDeltasInfo.stride * nbTracks == inputDeltas.Size(),
    "Input deltas size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2F_CHECK_RESULT_WITH_MSG(nva2x::ValidateTensorBatchInfo(inputDeltas, inputDeltasInfo), "Input deltas is invalid");
  A2F_CHECK_ERROR_WITH_MSG(
    outputVerticesInfo.stride * nbTracks == outputVertices.Size(),
    "Output vertices size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2F_CHECK_RESULT_WITH_MSG(nva2x::ValidateTensorBatchInfo(outputVertices, outputVerticesInfo), "Output vertices is invalid");

  const auto poseSize = _data.GetDeviceView().neutralPose.Size();

  A2F_CHECK_RESULT_WITH_MSG(
    cuda::AnimateTongue(
      outputVertices.Data(),
      outputVerticesInfo.offset,
      outputVerticesInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _data.GetDeviceView().neutralPose.Data(),
      _tongueParams.Data(),
      _tongueParamsStride,
      poseSize,
      nbTracks,
      _cudaStream
      ),
    "Unable to animate animator"
    );

  return nva2x::ErrorCode::eSuccess;
}

void MultiTrackAnimatorTongue::Destroy() {
  delete this;
}

std::error_code MultiTrackAnimatorTongue::SetParametersInternal(
  std::size_t trackIndex, const Params& params, std::vector<float>& paramsHost
  ) {
  // Set with validation.
  ValidatorProxy<Params> proxy(_params[trackIndex]);
  A2F_CHECK_RESULT_WITH_MSG(proxy.tongueStrength(params.tongueStrength), "Unable to set tongue strength");
  A2F_CHECK_RESULT_WITH_MSG(proxy.tongueHeightOffset(params.tongueHeightOffset), "Unable to set tongue height offset");
  A2F_CHECK_RESULT_WITH_MSG(proxy.tongueDepthOffset(params.tongueDepthOffset), "Unable to set tongue depth offset");

  // Copy to device.
  paramsHost.resize(3);
  paramsHost[0] = params.tongueStrength;
  paramsHost[1] = params.tongueHeightOffset;
  paramsHost[2] = params.tongueDepthOffset;

  A2F_CHECK_RESULT_WITH_MSG(
    nva2x::CopyHostToDevice(
      _tongueParams.View(trackIndex * _tongueParamsStride, paramsHost.size()), nva2x::ToConstView(paramsHost),_cudaStream),
    "Unable to copy tongue params to device"
    );
  A2F_CUDA_CHECK_ERROR(cudaStreamSynchronize(_cudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError);

  return nva2x::ErrorCode::eSuccess;
}

//////////////////////////////////////////////

IMultiTrackAnimatorTeeth::~IMultiTrackAnimatorTeeth() = default;

std::error_code MultiTrackAnimatorTeeth::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeeth::Init(const Params& params, std::size_t nbTracks) {
  _initialized = false;
  _animatorDataIsSet = false;

  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  _params.clear();
  _params.resize(nbTracks, params);

  _initialized = true;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeeth::SetAnimatorData(const HostData& data) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Teeth animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  _animatorDataIsSet = false;

  A2F_CHECK_ERROR_WITH_MSG(data.neutralJaw.Data() != nullptr, "Neutral jaw must not be null", nva2x::ErrorCode::eNullPointer);
  A2F_CHECK_ERROR_WITH_MSG(data.neutralJaw.Size() > 0, "Neutral jaw must not be empty", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(data.neutralJaw.Size() % 3 == 0, "Neutral jaw size must be a multiple of 3", nva2x::ErrorCode::eInvalidValue);
  _nbPoints = data.neutralJaw.Size() / 3;

  // Initialize the animator data.
  // Add the neutral jaw mean so it's available.
  std::vector<float> neutralJawHost(data.neutralJaw.Data(), data.neutralJaw.Data() + data.neutralJaw.Size());
  float mean[3] = {0.0f, 0.0f, 0.0f};
  for (std::size_t i = 0; i < _nbPoints; ++i) {
    mean[0] += neutralJawHost[i * 3 + 0];
    mean[1] += neutralJawHost[i * 3 + 1];
    mean[2] += neutralJawHost[i * 3 + 2];
  }
  for (std::size_t i = 0; i < 3; ++i) {
    neutralJawHost.emplace_back(mean[i] / _nbPoints);
  }
  A2F_CHECK_RESULT_WITH_MSG(
    _data.Init({nva2x::ToConstView(neutralJawHost)}, _cudaStream),
    "Unable to initialize neutral jaw"
    );

  const auto nbTracks = _params.size();

  _teethParamsStride = 4;
  A2F_CHECK_RESULT_WITH_MSG(_teethParams.Allocate(nbTracks * _teethParamsStride), "Unable to allocate teeth params");
  std::vector<float> paramsHost;
  for (std::size_t i = 0; i < nbTracks; ++i) {
    A2F_CHECK_RESULT_WITH_MSG(SetParametersInternal(i, _params[i], paramsHost), "Unable to set teeth parameters");
  }

  _animatorDataIsSet = true;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeeth::SetParameters(std::size_t trackIndex, const Params& params) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Teeth animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Teeth animator data is not set", nva2x::ErrorCode::eNotInitialized);

  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nva2x::ErrorCode::eOutOfBounds);

  std::vector<float> paramsHost;
  return SetParametersInternal(trackIndex, params, paramsHost);
}

const MultiTrackAnimatorTeeth::Params* MultiTrackAnimatorTeeth::GetParameters(std::size_t trackIndex) const {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Teeth animator is not initialized", nullptr);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Teeth animator data is not set", nullptr);

  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nullptr);
  return &_params[trackIndex];
}

std::error_code MultiTrackAnimatorTeeth::Reset(std::size_t trackIndex) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Teeth animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Teeth animator data is not set", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nva2x::ErrorCode::eOutOfBounds);
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeeth::SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Teeth animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Teeth animator data is not set", nva2x::ErrorCode::eNotInitialized);
  // Teeth animator does not hold state, so we can ignore the active tracks.
  // The overhead of handling "sparsely" active tracks is likely to be higher
  // than the benefits of running the animation on a subset of tracks.
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeeth::ComputeJawTransform(
  nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
  nva2x::DeviceTensorFloatView outputTransforms, const nva2x::TensorBatchInfo& outputTransformsInfo
  ) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Teeth animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Teeth animator data is not set", nva2x::ErrorCode::eNotInitialized);

  const auto nbTracks = _params.size();
  A2F_CHECK_ERROR_WITH_MSG(
    inputDeltasInfo.stride * nbTracks == inputDeltas.Size(),
    "Input deltas size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2F_CHECK_RESULT_WITH_MSG(nva2x::ValidateTensorBatchInfo(inputDeltas, inputDeltasInfo), "Input deltas is invalid");
  A2F_CHECK_ERROR_WITH_MSG(
    outputTransformsInfo.stride * nbTracks == outputTransforms.Size(),
    "Output transforms size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2F_CHECK_RESULT_WITH_MSG(nva2x::ValidateTensorBatchInfo(outputTransforms, outputTransformsInfo), "Output transforms is invalid");

  A2F_CHECK_RESULT_WITH_MSG(
    cuda::ComputeJawTransform(
      outputTransforms.Data(),
      outputTransformsInfo.offset,
      outputTransformsInfo.stride,
      inputDeltas.Data(),
      inputDeltasInfo.offset,
      inputDeltasInfo.stride,
      _data.GetDeviceView().neutralJaw.Data(),
      _teethParams.Data(),
      _teethParamsStride,
      _nbPoints,
      nbTracks,
      _cudaStream
      ),
    "Unable to compute jaw transform"
    );

  return nva2x::ErrorCode::eSuccess;
}

void MultiTrackAnimatorTeeth::Destroy() {
  delete this;
}

std::error_code MultiTrackAnimatorTeeth::SetParametersInternal(
  std::size_t trackIndex, const Params& params, std::vector<float>& paramsHost
  ) {
  // Set with validation.
  ValidatorProxy<Params> proxy(_params[trackIndex]);
  A2F_CHECK_RESULT_WITH_MSG(proxy.lowerTeethStrength(params.lowerTeethStrength), "Unable to set lower teeth strength");
  A2F_CHECK_RESULT_WITH_MSG(proxy.lowerTeethHeightOffset(params.lowerTeethHeightOffset), "Unable to set lower teeth height offset");
  A2F_CHECK_RESULT_WITH_MSG(proxy.lowerTeethDepthOffset(params.lowerTeethDepthOffset), "Unable to set lower teeth depth offset");

  // Copy to device.
  paramsHost.resize(4);
  paramsHost[0] = params.lowerTeethStrength;
  paramsHost[1] = 0.0f;
  paramsHost[2] = params.lowerTeethHeightOffset;
  paramsHost[3] = params.lowerTeethDepthOffset;

  A2F_CHECK_RESULT_WITH_MSG(
    nva2x::CopyHostToDevice(
      _teethParams.View(trackIndex * _teethParamsStride, paramsHost.size()), nva2x::ToConstView(paramsHost),_cudaStream),
    "Unable to copy teeth params to device"
    );
  A2F_CUDA_CHECK_ERROR(cudaStreamSynchronize(_cudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeeth::Data::Init(const HostData& data, cudaStream_t cudaStream) {
  CHECK_RESULT_WITH_MSG(_neutralJaw.Init(data.neutralJaw, cudaStream),
      "Unable to initialize neutral jaw");
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorTeeth::Data::Deallocate() {
  CHECK_RESULT_WITH_MSG(_neutralJaw.Deallocate(), "Unable to deallocate neutral jaw");
  return nva2x::ErrorCode::eSuccess;
}

MultiTrackAnimatorTeeth::DeviceData MultiTrackAnimatorTeeth::Data::GetDeviceView() const {
 return {_neutralJaw};
}

//////////////////////////////////////////////

IMultiTrackAnimatorEyes::~IMultiTrackAnimatorEyes() = default;

std::error_code MultiTrackAnimatorEyes::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorEyes::Init(const Params& params, std::size_t nbTracks) {
  _initialized = false;
  _animatorDataIsSet = false;

  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  _params.clear();
  _params.resize(nbTracks, params);

  const auto nbBitMasks = (nbTracks + nb_bits - 1) / nb_bits;
  _activeTracks.clear();
  _activeTracks.resize(nbBitMasks, ~bits_type(0));

  A2F_CHECK_RESULT_WITH_MSG(_activeTracksDevice.Allocate(nbBitMasks), "Unable to allocate active tracks");

  _initialized = true;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorEyes::SetAnimatorData(const HostData& data, float dt) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Eyes animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  _animatorDataIsSet = false;

  // Store the views from/to where the data will be read/written.
  A2F_CHECK_ERROR_WITH_MSG(data.saccadeRot.Data() != nullptr, "Saccade rotation matrix must not be null", nva2x::ErrorCode::eNullPointer);
  A2F_CHECK_ERROR_WITH_MSG(data.saccadeRot.Size() > 0, "Saccade rotation matrix must not be empty", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(data.saccadeRot.Size() % 2 == 0, "Saccade rotation matrix must be an even size", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_RESULT_WITH_MSG(_saccadeRot.Init(data.saccadeRot), "Unable to initialize saccade rot");

  const auto nbTracks = _params.size();

  // Initialize the live time data.
  A2F_CHECK_RESULT_WITH_MSG(_liveTime.Allocate(nbTracks), "Unable to allocate live time");
  A2F_CHECK_RESULT_WITH_MSG(nva2x::FillOnDevice(_liveTime, 0.0f, _cudaStream), "Unable to initialize live time");

  _dt = dt;

  // The eyes parameters are packed and stored per track.
  _eyesParamsStride = 8;
  A2F_CHECK_RESULT_WITH_MSG(_eyesParams.Allocate(nbTracks * _eyesParamsStride), "Unable to allocate eyes params");
  std::vector<float> paramsHost;
  for (std::size_t i = 0; i < nbTracks; ++i) {
    A2F_CHECK_RESULT_WITH_MSG(SetParametersInternal(i, _params[i], paramsHost), "Unable to set eyes parameters");
  }

  _animatorDataIsSet = true;

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorEyes::SetParameters(std::size_t trackIndex, const Params& params) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Eyes animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Eyes animator data is not set", nva2x::ErrorCode::eNotInitialized);

  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nva2x::ErrorCode::eOutOfBounds);

  std::vector<float> paramsHost;
  return SetParametersInternal(trackIndex, params, paramsHost);
}

const MultiTrackAnimatorEyes::Params* MultiTrackAnimatorEyes::GetParameters(std::size_t trackIndex) const {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Eyes animator is not initialized", nullptr);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Eyes animator data is not set", nullptr);

  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nullptr);
  return &_params[trackIndex];
}

std::error_code MultiTrackAnimatorEyes::Reset(std::size_t trackIndex) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Eyes animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Eyes animator data is not set", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nva2x::ErrorCode::eOutOfBounds);

  A2F_CHECK_RESULT_WITH_MSG(
    nva2x::FillOnDevice(_liveTime.View(trackIndex, 1), 0.0f, _cudaStream), "Unable to reset live time"
    );

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorEyes::SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Eyes animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Eyes animator data is not set", nva2x::ErrorCode::eNotInitialized);
  if (activeTracks == 0) {
    // Set all tracks to active.
    std::fill(_activeTracks.begin(), _activeTracks.end(), ~bits_type(0));
  }
  else {
    A2F_CHECK_ERROR_WITH_MSG(activeTracksSize == _activeTracks.size(), "Mismatch in active tracks size", nva2x::ErrorCode::eMismatch);
    std::copy(activeTracks, activeTracks + activeTracksSize, _activeTracks.begin());
  }

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorEyes::SetLiveTime(std::size_t trackIndex, float liveTime) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Eyes animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Eyes animator data is not set", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _params.size(), "Track index is out of bounds", nva2x::ErrorCode::eOutOfBounds);

  // This function simply sets the live time, but it doesn't wrap it.
  // The next call to ComputeEyesRotation will wrap it.
  A2F_CHECK_RESULT_WITH_MSG(
    nva2x::FillOnDevice(_liveTime.View(trackIndex, 1), liveTime * 30.0f, _cudaStream), "Unable to reset live time"
    );

  return nva2x::ErrorCode::eSuccess;
}

std::error_code MultiTrackAnimatorEyes::ComputeEyesRotation(
  nva2x::DeviceTensorFloatConstView inputEyesRotationResult, const nva2x::TensorBatchInfo& inputEyesRotationResultInfo,
  nva2x::DeviceTensorFloatView outputEyesRotation, const nva2x::TensorBatchInfo& outputEyesRotationInfo
  ) {
  A2F_CHECK_ERROR_WITH_MSG(_initialized, "Eyes animator is not initialized", nva2x::ErrorCode::eNotInitialized);
  A2F_CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "Eyes animator data is not set", nva2x::ErrorCode::eNotInitialized);

  const auto nbTracks = _eyesParams.Size() / _eyesParamsStride;
  A2F_CHECK_ERROR_WITH_MSG(
    inputEyesRotationResultInfo.stride * nbTracks == inputEyesRotationResult.Size(),
    "Input eyes rotation result size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2F_CHECK_RESULT_WITH_MSG(nva2x::ValidateTensorBatchInfo(inputEyesRotationResult, inputEyesRotationResultInfo), "Input eyes rotation result is invalid");
  A2F_CHECK_ERROR_WITH_MSG(
    outputEyesRotationInfo.stride * nbTracks == outputEyesRotation.Size(),
    "Output eyes rotation size does not match the batch size and the stride",
    nva2x::ErrorCode::eMismatch
    );
  A2F_CHECK_RESULT_WITH_MSG(nva2x::ValidateTensorBatchInfo(outputEyesRotation, outputEyesRotationInfo), "Output eyes rotation is invalid");

  A2F_CHECK_ERROR_WITH_MSG(4 == inputEyesRotationResultInfo.size, "Input eyes rotation result size must be equal to 4", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(6 == outputEyesRotationInfo.size, "Output eyes rotation size must be equal to 6", nva2x::ErrorCode::eInvalidValue);

  const bool allTracksActive = checkAllSet(_activeTracks, nbTracks);

  if (allTracksActive) {
    A2F_CHECK_RESULT_WITH_MSG(
      cuda::ComputeEyesRotation_Everything(
        outputEyesRotation.Data(), outputEyesRotationInfo.offset, outputEyesRotationInfo.stride,
        inputEyesRotationResult.Data(), inputEyesRotationResultInfo.offset, inputEyesRotationResultInfo.stride,
        _eyesParams.Data(), _eyesParamsStride,
        _saccadeRot.Data(), _saccadeRot.Size(),
        _dt,
        _liveTime.Data(),
        nbTracks,
        _cudaStream
        ),
        "Unable to compute eyes rotation with everything active"
      );
  }
  else {
    A2F_CHECK_RESULT_WITH_MSG(
      cuda::Tracks_Set(_activeTracksDevice.Data(), _activeTracks.data(), _activeTracks.size(), _cudaStream),
      "Unable to set active tracks"
      );
    A2F_CHECK_RESULT_WITH_MSG(
      cuda::ComputeEyesRotation_Control(
        outputEyesRotation.Data(), outputEyesRotationInfo.offset, outputEyesRotationInfo.stride,
        inputEyesRotationResult.Data(), inputEyesRotationResultInfo.offset, inputEyesRotationResultInfo.stride,
        _eyesParams.Data(), _eyesParamsStride,
        _saccadeRot.Data(), _saccadeRot.Size(),
        _dt,
        _liveTime.Data(),
        _activeTracksDevice.Data(),
        nbTracks,
        _cudaStream
        ),
        "Unable to compute eyes rotation with some tracks active"
      );
  }

  return nva2x::ErrorCode::eSuccess;
}

void MultiTrackAnimatorEyes::Destroy() {
  delete this;
}

std::error_code MultiTrackAnimatorEyes::SetParametersInternal(
  std::size_t trackIndex, const Params& params, std::vector<float>& paramsHost
  ) {
  // Set with validation.
  ValidatorProxy<Params> proxy(_params[trackIndex]);
  A2F_CHECK_RESULT_WITH_MSG(proxy.eyeballsStrength(params.eyeballsStrength), "Unable to set eyeballs strength");
  A2F_CHECK_RESULT_WITH_MSG(proxy.saccadeStrength(params.saccadeStrength), "Unable to set saccade strength");
  A2F_CHECK_RESULT_WITH_MSG(proxy.rightEyeballRotationOffsetX(params.rightEyeballRotationOffsetX), "Unable to set right eyeball rotation offset X");
  A2F_CHECK_RESULT_WITH_MSG(proxy.rightEyeballRotationOffsetY(params.rightEyeballRotationOffsetY), "Unable to set right eyeball rotation offset Y");
  A2F_CHECK_RESULT_WITH_MSG(proxy.leftEyeballRotationOffsetX(params.leftEyeballRotationOffsetX), "Unable to set left eyeball rotation offset X");
  A2F_CHECK_RESULT_WITH_MSG(proxy.leftEyeballRotationOffsetY(params.leftEyeballRotationOffsetY), "Unable to set left eyeball rotation offset Y");
  A2F_CHECK_RESULT_WITH_MSG(proxy.saccadeSeed(params.saccadeSeed), "Unable to set saccade seed");

  // Copy to device.
  paramsHost.resize(7);
  paramsHost[0] = params.eyeballsStrength;
  paramsHost[1] = params.saccadeStrength;
  paramsHost[2] = params.rightEyeballRotationOffsetX;
  paramsHost[3] = params.rightEyeballRotationOffsetY;
  paramsHost[4] = params.leftEyeballRotationOffsetX;
  paramsHost[5] = params.leftEyeballRotationOffsetY;
  paramsHost[6] = params.saccadeSeed;

  A2F_CHECK_RESULT_WITH_MSG(
    nva2x::CopyHostToDevice(
      _eyesParams.View(trackIndex * _eyesParamsStride, paramsHost.size()), nva2x::ToConstView(paramsHost),_cudaStream),
    "Unable to copy eyes params to device"
    );
  A2F_CUDA_CHECK_ERROR(cudaStreamSynchronize(_cudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError);

  return nva2x::ErrorCode::eSuccess;
}

//////////////////////////////////////////////

IMultiTrackAnimatorPcaReconstruction *CreateMultiTrackAnimatorPcaReconstruction_INTERNAL() {
  LOG_DEBUG("CreateMultiTrackAnimatorPcaReconstruction_INTERNAL()");
  return new MultiTrackAnimatorPcaReconstruction();
}

IMultiTrackAnimatorSkin *CreateMultiTrackAnimatorSkin_INTERNAL() {
  LOG_DEBUG("CreateMultiTrackAnimatorSkin_INTERNAL()");
  return new MultiTrackAnimatorSkin();
}

IMultiTrackAnimatorTongue *CreateMultiTrackAnimatorTongue_INTERNAL() {
  LOG_DEBUG("CreateMultiTrackAnimatorTongue_INTERNAL()");
  return new MultiTrackAnimatorTongue();
}

IMultiTrackAnimatorTeeth *CreateMultiTrackAnimatorTeeth_INTERNAL() {
  LOG_DEBUG("CreateMultiTrackAnimatorTeeth_INTERNAL()");
  return new MultiTrackAnimatorTeeth();
}

IMultiTrackAnimatorEyes *CreateMultiTrackAnimatorEyes_INTERNAL() {
  LOG_DEBUG("CreateMultiTrackAnimatorEyes_INTERNAL()");
  return new MultiTrackAnimatorEyes();
}

} // namespace nva2f
