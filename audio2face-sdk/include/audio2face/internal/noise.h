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

#include "audio2face/noise.h"

#include <vector>

#include <curand.h>

namespace nva2f {

// This class is a simpler wrapper around curandGenerator_t.
class CurandGeneratorHandle {
public:
  CurandGeneratorHandle();
  CurandGeneratorHandle(CurandGeneratorHandle&&);
  ~CurandGeneratorHandle();
  CurandGeneratorHandle& operator=(CurandGeneratorHandle&&);

  CurandGeneratorHandle(const CurandGeneratorHandle&) = delete;
  CurandGeneratorHandle& operator=(const CurandGeneratorHandle&) = delete;

  std::error_code Init();
  std::error_code Deallocate();

  std::error_code SetCudaStream(cudaStream_t cudaStream);

  curandGenerator_t Data() const;

private:
  curandGenerator_t _curandGenerator{nullptr};
};

class NoiseGenerator : public INoiseGenerator {
public:
  std::error_code SetCudaStream(cudaStream_t cudaStream) override;
  std::error_code Init(std::size_t nbTracks, std::size_t sizeToGenerate) override;

  std::error_code Generate(std::size_t trackIndex, nva2x::DeviceTensorFloatView tensor) override;
  std::error_code Reset(std::size_t trackIndex, std::size_t generateIndex) override;

  void Destroy() override;

private:
  cudaStream_t _cudaStream{nullptr};
  std::size_t _sizeToGenerate{0};
  std::vector<CurandGeneratorHandle> _handles;
};

INoiseGenerator* CreateNoiseGenerator_INTERNAL(
  std::size_t nbTracks, std::size_t sizeToGenerate
  );

} // namespace nva2f
