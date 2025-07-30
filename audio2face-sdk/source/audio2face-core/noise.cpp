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

#include "audio2face/internal/noise.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2face/error.h"
#include "audio2x/error.h"

#include <memory>

namespace {

  // We are using Philox which is supposed to be newer, faster then the default Xorwow.
  // Also, resetting the default Xorwow generator to replay the same sequence is not working with large enough sizes.
  constexpr const auto kRngToUse = CURAND_RNG_PSEUDO_PHILOX4_32_10;

}

namespace nva2f {

CurandGeneratorHandle::CurandGeneratorHandle() = default;

CurandGeneratorHandle::CurandGeneratorHandle(CurandGeneratorHandle&& other)
    : _curandGenerator(other._curandGenerator) {
  other._curandGenerator = nullptr;
}

CurandGeneratorHandle::~CurandGeneratorHandle() {
  Deallocate();
}

CurandGeneratorHandle& CurandGeneratorHandle::operator=(CurandGeneratorHandle&& other) {
  std::swap(_curandGenerator, other._curandGenerator);
  return *this;
}

std::error_code CurandGeneratorHandle::Init() {
  A2F_CHECK_RESULT(Deallocate());
  A2F_CHECK_ERROR_WITH_MSG(
    CURAND_STATUS_SUCCESS == curandCreateGenerator(&_curandGenerator, kRngToUse),
    "Unable to create curand generator",
    ErrorCode::eCurandCreateError
    );
  return nva2x::ErrorCode::eSuccess;
}

std::error_code CurandGeneratorHandle::Deallocate() {
  if (_curandGenerator != nullptr) {
    A2F_CHECK_ERROR_WITH_MSG(
      CURAND_STATUS_SUCCESS == curandDestroyGenerator(_curandGenerator),
      "Unable to destroy curand generator",
      ErrorCode::eCurandDestroyError
      );
    _curandGenerator = nullptr;
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code CurandGeneratorHandle::SetCudaStream(cudaStream_t cudaStream) {
  A2F_CHECK_ERROR_WITH_MSG(
    _curandGenerator != nullptr,
    "CurandGeneratorHandle is not initialized",
    nva2x::ErrorCode::eNotInitialized
    );
  A2F_CHECK_ERROR_WITH_MSG(
    CURAND_STATUS_SUCCESS == curandSetStream(_curandGenerator, cudaStream),
    "Unable to set cuda stream",
    ErrorCode::eCurandSetStreamError
    );
  return nva2x::ErrorCode::eSuccess;
}

curandGenerator_t CurandGeneratorHandle::Data() const {
  return _curandGenerator;
}


INoiseGenerator::~INoiseGenerator() = default;

std::error_code NoiseGenerator::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  for (auto& generator : _handles) {
    A2F_CHECK_RESULT(generator.SetCudaStream(cudaStream));
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code NoiseGenerator::Init(std::size_t nbTracks, std::size_t sizeToGenerate) {
  A2F_CHECK_ERROR_WITH_MSG(nbTracks > 0, "Number of tracks must be greater than 0", nva2x::ErrorCode::eInvalidValue);
  A2F_CHECK_ERROR_WITH_MSG(sizeToGenerate % 2 == 0, "Size to generate must be even", nva2x::ErrorCode::eInvalidValue);
  _sizeToGenerate = sizeToGenerate;
  _handles.resize(nbTracks);
  for (auto& generator : _handles) {
    A2F_CHECK_RESULT(generator.Init());
    A2F_CHECK_RESULT(generator.SetCudaStream(_cudaStream));
    // Generate seeds right away so it doesn't happen at the first call to Generate().
    A2F_CHECK_ERROR_WITH_MSG(
      CURAND_STATUS_SUCCESS == curandGenerateSeeds(generator.Data()),
      "Unable to generate seeds",
      ErrorCode::eCurandGenerateSeedsError
      );
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code NoiseGenerator::Generate(std::size_t trackIndex, nva2x::DeviceTensorFloatView tensor) {
  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _handles.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
  A2F_CHECK_ERROR_WITH_MSG(tensor.Size() == _sizeToGenerate, "Tensor size does not match size to generate", nva2x::ErrorCode::eMismatch);
  A2F_CHECK_ERROR_WITH_MSG(
    CURAND_STATUS_SUCCESS == curandGenerateNormal(_handles[trackIndex].Data(), tensor.Data(), tensor.Size(), 0.0f, 1.0f),
    "Unable to generate random numbers",
    ErrorCode::eCurandGenerateError
    );
  return nva2x::ErrorCode::eSuccess;
}

std::error_code NoiseGenerator::Reset(std::size_t trackIndex, std::size_t generateIndex) {
  A2F_CHECK_ERROR_WITH_MSG(trackIndex < _handles.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
  A2F_CHECK_ERROR_WITH_MSG(
    CURAND_STATUS_SUCCESS == curandSetGeneratorOffset(_handles[trackIndex].Data(), generateIndex * _sizeToGenerate),
    "Unable to reset curand generator",
    ErrorCode::eCurandSetOffsetError
    );
  return nva2x::ErrorCode::eSuccess;
}

void NoiseGenerator::Destroy() {
  delete this;
}

} // namespace nva2f

nva2f::INoiseGenerator* nva2f::CreateNoiseGenerator_INTERNAL(
    std::size_t nbTracks, std::size_t sizeToGenerate
    ) {
  LOG_DEBUG("CreateNoiseGenerator()");
  auto noiseGenerator = std::make_unique<nva2f::NoiseGenerator>();
  if (noiseGenerator->Init(nbTracks, sizeToGenerate)) {
    LOG_ERROR("Unable to create noise generator");
    return nullptr;
  }
  return noiseGenerator.release();
}
