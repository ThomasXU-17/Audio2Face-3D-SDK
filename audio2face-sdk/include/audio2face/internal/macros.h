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

// A2F SDK specific macros.
#define A2F_CHECK_RESULT(errorCode)                                                \
    A2X_BASE_CHECK_RESULT("A2F SDK", errorCode)

#define A2F_CHECK_ERROR_WITH_MSG(expression, message, errorCode)                   \
    A2X_BASE_CHECK_ERROR_WITH_MSG("A2F SDK", expression, message, errorCode)

#define A2F_CHECK_RESULT_WITH_MSG(func, message)                                   \
    A2X_BASE_CHECK_RESULT_WITH_MSG("A2F SDK", func, message)

#define A2F_CUDA_CHECK_ERROR(expression, errorCode)                                \
    A2X_BASE_CUDA_CHECK_ERROR("A2F SDK", expression, errorCode)


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
        LOG_ERROR(errorCode.message());                                        \
        return errorCode;                                                      \
    }                                                                          \
  }

#define CHECK_ERROR_WITH_MSG(expression, message, errorCode)                   \
  {                                                                            \
    if (!(expression)) {                                                       \
        LOG_ERROR(message);                                                    \
        return errorCode;                                                      \
    }                                                                          \
  }

#define CHECK_NO_ERROR(expression)                                             \
  {                                                                            \
    std::error_code error = (expression);                                      \
    if (error) {                                                               \
        return error;                                                          \
    }                                                                          \
  }

#define CHECK_RESULT(func)                                                     \
  {                                                                            \
    std::error_code error = func;                                              \
    if (error) {                                                               \
      LOG_ERROR(error.message());                                              \
      return error;                                                            \
    }                                                                          \
  }

#define CHECK_RESULT_WITH_MSG(func, message)                                   \
  {                                                                            \
    std::error_code error = func;                                              \
    if (error) {                                                               \
      LOG_ERROR(message);                                                      \
      return error;                                                            \
    }                                                                          \
  }

#define CHECK_NOT_NULL(expression, message)                                    \
  {                                                                            \
    if ((expression) == nullptr) {                                             \
      LOG_ERROR(message);                                                      \
      return false;                                                            \
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

#define CUDA_CHECK_ERROR(expression, errorCode)                                \
  {                                                                            \
    cudaError_t status = (expression);                                         \
    if (status != cudaSuccess) {                                               \
      LOG_ERROR("CUDA error: " << cudaGetErrorString(status));                 \
      return errorCode;                                                        \
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

#define CUBLAS_CHECK(expression)                                               \
  {                                                                            \
    cublasStatus_t status = (expression);                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      LOG_ERROR("CUBLAS error[" << status << "] file:" << __FILE__             \
                                << " line:" << __LINE__);                      \
      return false;                                                            \
    }                                                                          \
  }

#define CUBLAS_CHECK_ERROR(expression, errorCode)                              \
  {                                                                            \
    cublasStatus_t status = (expression);                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      LOG_ERROR("CUBLAS error[" << status << "] file:" << __FILE__             \
                                << " line:" << __LINE__);                      \
      return errorCode;                                                        \
    }                                                                          \
  }
  
