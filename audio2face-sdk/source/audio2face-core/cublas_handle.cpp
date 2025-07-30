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

#include "audio2face/internal/cublas_handle.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2face/error.h"
#include "audio2x/error.h"

namespace nva2f {

CublasHandle::CublasHandle() = default;

CublasHandle::CublasHandle(CublasHandle&& other)
    : _cublasHandle(other._cublasHandle) {
  other._cublasHandle = nullptr;
}

CublasHandle::~CublasHandle() {
  Deallocate();
}

CublasHandle& CublasHandle::operator=(CublasHandle&& other) {
  std::swap(_cublasHandle, other._cublasHandle);
  return *this;
}

std::error_code CublasHandle::Init() {
  A2F_CHECK_RESULT(Deallocate());
  A2F_CHECK_ERROR_WITH_MSG(
    CUBLAS_STATUS_SUCCESS == cublasCreate(&_cublasHandle),
    "Unable to create cublas handle",
    ErrorCode::eCublasCreateError
    );
  return nva2x::ErrorCode::eSuccess;
}

std::error_code CublasHandle::Deallocate() {
  if (_cublasHandle != nullptr) {
    A2F_CHECK_ERROR_WITH_MSG(
      CUBLAS_STATUS_SUCCESS == cublasDestroy(_cublasHandle),
      "Unable to destroy cublas handle",
      ErrorCode::eCublasDestroyError
      );
    _cublasHandle = nullptr;
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code CublasHandle::SetCudaStream(cudaStream_t cudaStream) {
  A2F_CHECK_ERROR_WITH_MSG(
    _cublasHandle != nullptr,
    "CublasHandle is not initialized",
    nva2x::ErrorCode::eNotInitialized
    );
  A2F_CHECK_ERROR_WITH_MSG(
    CUBLAS_STATUS_SUCCESS == cublasSetStream(_cublasHandle, cudaStream),
    "Unable to set cuda stream",
    ErrorCode::eCublasSetStreamError
    );
  return nva2x::ErrorCode::eSuccess;
}

cublasHandle_t CublasHandle::Data() const {
  return _cublasHandle;
}

} // namespace nva2f
