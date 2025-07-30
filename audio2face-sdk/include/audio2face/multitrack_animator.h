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

#include "audio2face/animator.h"

namespace nva2f {


// Interface for multi-track PCA reconstruction animator.
// Handles PCA reconstruction across multiple audio tracks simultaneously.
class IMultiTrackAnimatorPcaReconstruction {
public:
  // Type alias for the host data.
  using HostData = IAnimatorPcaReconstruction::HostData;

  // Set the CUDA stream for GPU operations.
  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;

  // Initialize the animator with the specified number of tracks.
  virtual std::error_code Init(std::size_t nbTracks) = 0; // GPU Async

  // Set the animator data to use for reconstruction.
  virtual std::error_code SetAnimatorData(const HostData& data) = 0; // GPU Async

  // Reset the animator state for a specific track.
  virtual std::error_code Reset(std::size_t trackIndex) = 0;

  // Set which tracks are currently active for reconstruction.
  // This is used to make sure the right data is used from input batches and
  // no inactive track state gets updated.  It can also provide opportunities for optimizations.
  // If activeTracks is nullptr, all tracks are active.
  // For PCA reconstruction, active tracks are not used since there is no state,
  // everything is computed all the time.
  virtual std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) = 0;

  // Animate using PCA coefficients and output vertex deltas data.
  virtual std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputPcaCoefs, const nva2x::TensorBatchInfo& inputPcaCoefsInfo,
    nva2x::DeviceTensorFloatView outputShapes, const nva2x::TensorBatchInfo& outputShapesInfo,
    std::size_t batchSize = 0
    ) = 0; // GPU Async

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IMultiTrackAnimatorPcaReconstruction();
};


// Interface for multi-track skin animator.
// Handles skin animation across multiple audio tracks simultaneously.
class IMultiTrackAnimatorSkin {
public:
  // Type alias for the host data.
  using HostData = IAnimatorSkin::HostData;
  // Type alias for the parameters.
  using Params = AnimatorSkinParams;

  // Set the CUDA stream for GPU operations.
  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;

  // Initialize the animator with parameters and number of tracks.
  virtual std::error_code Init(const Params& params, std::size_t nbTracks) = 0; // GPU Async

  // Set the animator data and time step to use for animation.
  virtual std::error_code SetAnimatorData(const HostData& data, float dt) = 0; // GPU Async

  // Set animator parameters for a specific track.
  virtual std::error_code SetParameters(std::size_t trackIndex, const Params& params) = 0;

  // Get the current parameters for a specific track.
  virtual const Params* GetParameters(std::size_t trackIndex) const = 0;

  // Reset the animator state for a specific track.
  virtual std::error_code Reset(std::size_t trackIndex) = 0;

  // Set which tracks are currently active for animation.
  // This is used to make sure the right data is used from input batches and
  // no inactive track state gets updated.  It can also provide opportunities for optimizations.
  // If activeTracks is nullptr, all tracks are active.
  virtual std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) = 0;

  // Animate skin using delta inputs and output vertex positions data.
  virtual std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) = 0; // GPU Async

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IMultiTrackAnimatorSkin();
};


// Interface for multi-track tongue animator.
// Handles tongue animation across multiple audio tracks simultaneously.
class IMultiTrackAnimatorTongue {
public:
  // Type alias for the host data.
  using HostData = IAnimatorTongue::HostData;
  // Type alias for the parameters.
  using Params = AnimatorTongueParams;

  // Set the CUDA stream for GPU operations.
  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;

  // Initialize the animator with parameters and number of tracks.
  virtual std::error_code Init(const Params& params, std::size_t nbTracks) = 0; // GPU Async

  // Set the animator data to use for animation.
  virtual std::error_code SetAnimatorData(const HostData& data) = 0; // GPU Async

  // Set animator parameters for a specific track.
  virtual std::error_code SetParameters(std::size_t trackIndex, const Params& params) = 0;

  // Get the current parameters for a specific track.
  virtual const Params* GetParameters(std::size_t trackIndex) const = 0;

  // Reset the animator state for a specific track.
  virtual std::error_code Reset(std::size_t trackIndex) = 0;

  // Set which tracks are currently active for animation.
  // This is used to make sure the right data is used from input batches and
  // no inactive track state gets updated.  It can also provide opportunities for optimizations.
  // If activeTracks is nullptr, all tracks are active.
  // For tongue animation, active tracks are not used since there is no state,
  // everything is computed all the time.
  virtual std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) = 0;

  // Animate tongue using delta inputs and output vertex positions data.
  virtual std::error_code Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputVertices, const nva2x::TensorBatchInfo& outputVerticesInfo
    ) = 0; // GPU Async

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IMultiTrackAnimatorTongue();
};


// Interface for multi-track teeth animator.
// Handles teeth animation across multiple audio tracks simultaneously.
class IMultiTrackAnimatorTeeth {
public:
  // Type alias for the host data.
  using HostData = IAnimatorTeeth::HostData;
  // Type alias for the parameters.
  using Params = AnimatorTeethParams;

  // Set the CUDA stream for GPU operations.
  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;

  // Initialize the animator with parameters and number of tracks.
  virtual std::error_code Init(const Params& params, std::size_t nbTracks) = 0; // GPU Async

  // Set the animator data to use for animation.
  virtual std::error_code SetAnimatorData(const HostData& data) = 0; // GPU Async

  // Set animator parameters for a specific track.
  virtual std::error_code SetParameters(std::size_t trackIndex, const Params& params) = 0;

  // Get the current parameters for a specific track.
  virtual const Params* GetParameters(std::size_t trackIndex) const = 0;

  // Reset the animator state for a specific track.
  virtual std::error_code Reset(std::size_t trackIndex) = 0;

  // Set which tracks are currently active for animation.
  // This is used to make sure the right data is used from input batches and
  // no inactive track state gets updated.  It can also provide opportunities for optimizations.
  // If activeTracks is nullptr, all tracks are active.
  // For teeth animation, active tracks are not used since there is no state,
  // everything is computed all the time.
  virtual std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) = 0;

  // Compute jaw transforms using delta inputs.
  virtual std::error_code ComputeJawTransform(
    nva2x::DeviceTensorFloatConstView inputDeltas, const nva2x::TensorBatchInfo& inputDeltasInfo,
    nva2x::DeviceTensorFloatView outputTransforms, const nva2x::TensorBatchInfo& outputTransformsInfo
    ) = 0; // GPU Async

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IMultiTrackAnimatorTeeth();
};


// Interface for multi-track eyes animator.
// Handles eye rotation animation across multiple audio tracks simultaneously.
class IMultiTrackAnimatorEyes {
public:
  // Type alias for the host data.
  using HostData = IAnimatorEyes::HostData;
  // Type alias for the parameters.
  using Params = AnimatorEyesParams;

  // Set the CUDA stream for GPU operations.
  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;

  // Initialize the animator with parameters and number of tracks.
  virtual std::error_code Init(const Params& params, std::size_t nbTracks) = 0; // GPU Async

  // Set the animator data and time step to use for animation.
  virtual std::error_code SetAnimatorData(const HostData& data, float dt) = 0; // GPU Async

  // Set animator parameters for a specific track.
  virtual std::error_code SetParameters(std::size_t trackIndex, const Params& params) = 0;

  // Get the current parameters for a specific track.
  virtual const Params* GetParameters(std::size_t trackIndex) const = 0;

  // Reset the animator state for a specific track.
  virtual std::error_code Reset(std::size_t trackIndex) = 0;

  // Set which tracks are currently active for animation.
  // This is used to make sure the right data is used from input batches and
  // no inactive track state gets updated.  It can also provide opportunities for optimizations.
  // If activeTracks is nullptr, all tracks are active.
  virtual std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) = 0;

  // Set the live time for a specific track.
  virtual std::error_code SetLiveTime(std::size_t trackIndex, float liveTime) = 0;

  // Compute eye rotations using input rotation results.
  virtual std::error_code ComputeEyesRotation(
    nva2x::DeviceTensorFloatConstView inputEyesRotationResult, const nva2x::TensorBatchInfo& inputEyesRotationResultInfo,
    nva2x::DeviceTensorFloatView outputEyesRotation, const nva2x::TensorBatchInfo& outputEyesRotationInfo
    ) = 0; // GPU Async

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IMultiTrackAnimatorEyes();
};


} // namespace nva2f
