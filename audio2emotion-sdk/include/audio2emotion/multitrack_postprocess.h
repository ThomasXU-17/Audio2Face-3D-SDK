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

namespace nva2e {


// Interface for multi-track emotion post-processing operations.
// Handles post-processing of emotion data across multiple audio tracks simultaneously.
// It improves the IPostProcessor by running on the GPU (avoiding CPU-GPU synchronization)
// and allowing to process multiple tracks in parallel.
class IMultiTrackPostProcessor {
public:
  // Type alias for host-side post-processing data.
  using HostData = PostProcessData;
  // Type alias for post-processing parameters.
  using Params = PostProcessParams;

  // Set the CUDA stream for GPU operations.
  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;

  // Initialize the post-processor with data, parameters, and number of tracks.
  virtual std::error_code Init(const HostData& data, const Params& params, std::size_t nbTracks) = 0; // GPU Async

  // Set post-processing parameters for a specific track.
  virtual std::error_code SetParameters(std::size_t trackIndex, const Params& params) = 0;

  // Get the current parameters for a specific track.
  virtual const Params* GetParameters(std::size_t trackIndex) const = 0;

  // Reset the post-processor state for a specific track.
  virtual std::error_code Reset(std::size_t trackIndex) = 0;

  // Set which tracks are currently active for processing.
  // This is used to make sure the right data is used from input batches and
  // no inactive track state gets updated by post-processing.
  virtual std::error_code SetActiveTracks(const std::uint64_t* activeTracks, std::size_t activeTracksSize) = 0;

  // Perform post-processing on emotion data for all active tracks.
  virtual std::error_code PostProcess(
    nva2x::DeviceTensorFloatConstView inputEmotions, const nva2x::TensorBatchInfo& inputEmotionsInfo,
    nva2x::DeviceTensorFloatView outputEmotions, const nva2x::TensorBatchInfo& outputEmotionsInfo
    ) = 0; // GPU Async

  // Get the number of emotions used as post-processing input.
  virtual std::size_t GetInputEmotionsSize() const = 0;

  // Get the number of emotions used as post-processing output.
  virtual std::size_t GetOutputEmotionsSize() const = 0;

  // Get a writable view of the preferred emotion tensor for a specific track.
  virtual nva2x::DeviceTensorFloatView GetPreferredEmotion(std::size_t trackIndex) = 0;

  // Get a read-only view of the preferred emotion tensor for a specific track.
  virtual nva2x::DeviceTensorFloatConstView GetPreferredEmotion(std::size_t trackIndex) const = 0;

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IMultiTrackPostProcessor();
};


} // namespace nva2e
