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
#include "audio2x/internal/cuda_stream.h"
#include "audio2x/internal/audio2x.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/error.h"

#include <cuda_runtime_api.h>


namespace nva2x {

ICudaStream::~ICudaStream() = default;

CudaStream::CudaStream() : _cudaStream(nullptr) {
  A2X_LOG_DEBUG("CudaStream::CudaStream()");
}

CudaStream::CudaStream(CudaStream&& other)
: _cudaStream(other._cudaStream) {
  A2X_LOG_DEBUG("CudaStream::CudaStream(CudaStream&&)");
  other._cudaStream = nullptr;
}

CudaStream::~CudaStream() {
  A2X_LOG_DEBUG("CudaStream::~CudaStream()");
  Deallocate();
}

std::error_code CudaStream::Synchronize() const {
  A2X_CUDA_CHECK_ERROR(cudaStreamSynchronize(_cudaStream), ErrorCode::eCudaStreamSynchronizeError);
  return ErrorCode::eSuccess;
}

cudaStream_t CudaStream::Data() const {
  return _cudaStream;
}

void CudaStream::Destroy() {
  A2X_LOG_DEBUG("CudaStream::Destroy()");
  delete this;
}

std::error_code CudaStream::Init() {
  A2X_CHECK_RESULT_WITH_MSG(Deallocate(), "Unable to destroy CUDA stream before init");
  A2X_LOG_DEBUG("Creating CUDA stream");
  A2X_CUDA_CHECK_ERROR(cudaStreamCreate(&_cudaStream), ErrorCode::eCudaStreamCreateError);
  return ErrorCode::eSuccess;
}

std::error_code CudaStream::Deallocate() {
  if (_cudaStream != nullptr) {
    A2X_LOG_DEBUG("Deleting CUDA stream");
    A2X_CUDA_CHECK_ERROR(cudaStreamDestroy(_cudaStream), ErrorCode::eCudaStreamDestroyError);
    _cudaStream = nullptr;
  }
  return ErrorCode::eSuccess;
}


} // namespace nva2x


nva2x::ICudaStream* nva2x::internal::CreateCudaStream() {
  A2X_LOG_DEBUG("CreateCudaStream()");
  CudaStream cudaStream;
  if (cudaStream.Init()) {
    A2X_LOG_ERROR("Unable to create CUDA stream");
    return nullptr;
  }
  return new CudaStream(std::move(cudaStream));
}

nva2x::ICudaStream* nva2x::internal::CreateDefaultCudaStream() {
  A2X_LOG_DEBUG("CreateDefaultCudaStream()");
  return new CudaStream;
}
