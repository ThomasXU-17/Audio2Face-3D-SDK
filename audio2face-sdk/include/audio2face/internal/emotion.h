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

#include "audio2face/emotion.h"
#include "audio2x/internal/tensor.h"

#include <map>
#include <string>

namespace nva2f {

class EmotionDatabase : public IEmotionDatabase {
public:
  struct EmotionShotInfo {
    uint32_t start;
    uint32_t size;
  };

  EmotionDatabase();
  ~EmotionDatabase();

  EmotionDatabase(EmotionDatabase&& other) = default;
  EmotionDatabase& operator=(EmotionDatabase&& other) = default;

  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code InitFromFile(const char *emotionDBPath) override;
  std::size_t GetEmotionLength() const override;
  std::error_code GetEmotion(const char *emotionShot, unsigned int emotionFrame,
                  nva2x::DeviceTensorFloatView emotion) const override; // GPU Async
  const char* GetEmotionShotName(std::size_t index) const override;
  std::size_t GetNbEmotionShots() const override;
  uint32_t GetEmotionShotSize(const char* emotionShotName) const override;
  void Destroy() override;

private:
  std::error_code InitFromNpz(const char *emotionDBPath);

  cudaStream_t _cudaStream;
  std::map<std::string, EmotionShotInfo> _shotList;
  nva2x::DeviceTensorFloat _emotionData;
  size_t _emotionLength;
  bool _initialized;
};

IEmotionDatabase *CreateEmotionDatabase_INTERNAL();

} // namespace nva2f
