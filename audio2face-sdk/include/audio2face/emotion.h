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

// Interface for managing emotion database operations.
class IEmotionDatabase {
public:
  // Set the CUDA stream for GPU operations.
  virtual std::error_code SetCudaStream(cudaStream_t cudaStream) = 0;

  // Initialize the emotion database from a file.
  virtual std::error_code InitFromFile(const char *emotionDBPath) = 0;

  // Get the length of emotion data in each frame.
  virtual std::size_t GetEmotionLength() const = 0;

  // Get emotion data for a specific shot and frame into GPU memory.
  virtual std::error_code GetEmotion(const char *emotionShot, unsigned int emotionFrame,
                          nva2x::DeviceTensorFloatView emotion) const = 0; // GPU Async

  // Get the name of an emotion shot by index.
  virtual const char* GetEmotionShotName(std::size_t index) const = 0;

  // Get the total number of emotion shots in the database.
  virtual std::size_t GetNbEmotionShots() const = 0;

  // Get the size of a specific emotion shot in frames.
  virtual uint32_t GetEmotionShotSize(const char* emotionShotName) const = 0;

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IEmotionDatabase();
};

} // namespace nva2f
