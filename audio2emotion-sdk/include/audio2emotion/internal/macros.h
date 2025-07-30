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

#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"

// A2E SDK specific macros.
#define A2E_CHECK_RESULT(errorCode)                                                \
    A2X_BASE_CHECK_RESULT("A2E SDK", errorCode)

#define A2E_CHECK_ERROR_WITH_MSG(expression, message, errorCode)                   \
    A2X_BASE_CHECK_ERROR_WITH_MSG("A2E SDK", expression, message, errorCode)

#define A2E_CHECK_RESULT_WITH_MSG(func, message)                                   \
    A2X_BASE_CHECK_RESULT_WITH_MSG("A2E SDK", func, message)

#define A2E_CUDA_CHECK_ERROR(expression, errorCode)                                \
    A2X_BASE_CUDA_CHECK_ERROR("A2E SDK", expression, errorCode)


#define CHECK_TRUE(expression, message)                                        \
  {                                                                            \
    if (!(expression)) {                                                       \
      LOG_ERROR(message);                                                      \
      return false;                                                            \
    }                                                                          \
  }

#define CHECK_ERROR(expression, errorCode)                                     \
  {                                                                            \
    if (!(expression)) {                                                       \
      std::error_code ec = (errorCode);                                        \
      LOG_ERROR(ec.message());                                                 \
      return (errorCode);                                                      \
    }                                                                          \
  }

#define CHECK_ERROR_WITH_MSG(expression, message, errorCode)                   \
  {                                                                            \
    if (!(expression)) {                                                       \
      std::error_code ec = (errorCode);                                        \
      LOG_ERROR(message);                                                      \
      return (errorCode);                                                      \
    }                                                                          \
  }

#define CHECK_NO_ERROR(expression)                                             \
  {                                                                            \
    std::error_code ec = (expression);                                         \
    if (ec) {                                                                  \
      return ec;                                                               \
    }                                                                          \
  }

#define CHECK_NOT_NULL(expression, message)                                    \
  {                                                                            \
    if ((expression) == nullptr) {                                             \
      LOG_ERROR(message);                                                      \
      return false;                                                            \
    }                                                                          \
  }

#define CHECK_CUDA_ERROR(expression, errorCode)                                \
  {                                                                            \
    cudaError_t status = (expression);                                         \
    if (status != cudaSuccess) {                                               \
      LOG_ERROR("CUDA error: " << cudaGetErrorString(status));                 \
      return (errorCode);                                                      \
    }                                                                          \
  }

#define CUDA_CHECK(expression)                                                 \
  {                                                                            \
    cudaError_t status = (expression);                                         \
    if (status != cudaSuccess) {                                               \
      LOG_ERROR("CUDA error: " << cudaGetErrorString(status));                 \
      return false;                                                            \
    }                                                                          \
  }

#define CUDA_ERROR_CHECK()                                                     \
  {                                                                            \
    cudaError_t status = cudaGetLastError();                                   \
    if (status != cudaSuccess) {                                               \
      LOG_ERROR("CUDA kernel error: " << cudaGetErrorString(status));          \
      return false;                                                            \
    }                                                                          \
  }
