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

#include "audio2x/tensor.h"

namespace nva2f {

// Interface for generating noise data on GPU.
// It generates normally distributed random numbers with mean 0 and standard deviation 1.
class INoiseGenerator {
public:
  // Set the CUDA stream for GPU noise generation operations.
  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;

  // Initialize the noise generator with the specified number of tracks and generation size.
  virtual std::error_code Init(std::size_t nbTracks, std::size_t sizeToGenerate) = 0;

  // Generate noise data for the specified track and store it in the provided tensor.
  virtual std::error_code Generate(std::size_t trackIndex, nva2x::DeviceTensorFloatView tensor) = 0;

  // Reset the noise generation state for the specified track and generation index.
  // After this call, the generator will be in the same state as if a count of generateIndex calls
  // to Generate() had been made.
  virtual std::error_code Reset(std::size_t trackIndex, std::size_t generateIndex) = 0;

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~INoiseGenerator();
};

} // namespace nva2f
