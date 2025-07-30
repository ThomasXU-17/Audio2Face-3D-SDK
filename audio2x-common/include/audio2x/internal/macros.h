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

// Generic macros that can be used to define SDK-specific ones.
#define A2X_BASE_CHECK_RESULT(category, expression)                                \
  {                                                                                \
    const std::error_code errorCode = (expression);                                \
    if ((errorCode)) {                                                             \
        A2X_BASE_LOG_ERROR(category, errorCode.message());                         \
        return errorCode;                                                          \
    }                                                                              \
  }

#define A2X_BASE_CHECK_ERROR_WITH_MSG(category, expression, message, errorCode)    \
  {                                                                                \
    if (!(expression)) {                                                           \
        A2X_BASE_LOG_ERROR(category, message);                                     \
        return errorCode;                                                          \
    }                                                                              \
  }

#define A2X_BASE_CHECK_RESULT_WITH_MSG(category, expression, message)              \
  {                                                                                \
    const std::error_code errorCode = (expression);                                \
    if (errorCode) {                                                               \
      A2X_BASE_LOG_ERROR(category, message);                                       \
      return errorCode;                                                            \
    }                                                                              \
  }

#define A2X_BASE_CUDA_CHECK_ERROR(category, expression, errorCode)                 \
  {                                                                                \
    cudaError_t status = (expression);                                             \
    if (status != cudaSuccess) {                                                   \
      A2X_BASE_LOG_ERROR(category, "CUDA error: " << cudaGetErrorString(status));  \
      return errorCode;                                                            \
    }                                                                              \
  }

// A2X SDK specific macros.
#define A2X_CHECK_RESULT(expression)                                               \
    A2X_BASE_CHECK_RESULT("A2X SDK", expression)

#define A2X_CHECK_ERROR_WITH_MSG(expression, message, errorCode)                   \
    A2X_BASE_CHECK_ERROR_WITH_MSG("A2X SDK", expression, message, errorCode)

#define A2X_CHECK_RESULT_WITH_MSG(expression, message)                             \
    A2X_BASE_CHECK_RESULT_WITH_MSG("A2X SDK", expression, message)

#define A2X_CUDA_CHECK_ERROR(expression, errorCode)                                \
    A2X_BASE_CUDA_CHECK_ERROR("A2X SDK", expression, errorCode)

#define IDIVUP(a, b) ((a) % (b) == 0 ? (a) / (b) : (a) / (b) + 1)
