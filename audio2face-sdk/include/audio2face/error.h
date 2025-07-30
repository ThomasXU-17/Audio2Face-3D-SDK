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

#ifdef _WIN32
#ifdef AUDIO2FACE_SDK_DLL_EXPORTS
#define AUDIO2FACE_ERROR_DLL_API __declspec(dllexport)
#else
//#define AUDIO2FACE_ERROR_DLL_API __declspec(dllimport)
#define AUDIO2FACE_ERROR_DLL_API
#endif
#else
#define AUDIO2FACE_ERROR_DLL_API
#endif

#include <system_error>

namespace nva2f {

enum class ErrorCode {
  eNotANumber = 1,
  eOutOfRange,
  // CUDA
  eCudaEventCreateError = 20,
  eCudaEventDestroyError,
  eCudaEventRecordError,
  eCudaStreamWaitEventError,
  // Animator
  eDataNotSet = 60,
  // Audio
  eSampleRateTooLow = 100,
  // cuBlas
  eCublasCreateError = 200,
  eCublasDestroyError,
  eCublasSetStreamError,
  eCublasExecutionError,
  eCublasMatMatFailed,
  // cuRand
  eCurandCreateError = 300,
  eCurandDestroyError,
  eCurandSetStreamError,
  eCurandGenerateSeedsError,
  eCurandGenerateError,
  eCurandSetOffsetError,
  // Executor
  eEmotionNotAvailable = 400,
  // misc.
  eEndOfEnum
};

AUDIO2FACE_ERROR_DLL_API std::error_code make_error_code(ErrorCode code);
AUDIO2FACE_ERROR_DLL_API ErrorCode get_error_code(std::error_code error);

}//namespace nva2f

namespace std {
  template <>
  struct is_error_code_enum<nva2f::ErrorCode> : true_type {};
}
