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
#include "audio2x/internal/logger_trt.h"

#include "audio2x/internal/logger.h"

namespace nva2x {

void TRTLogger::log(nvinfer1::ILogger::Severity severity,
                    const char *msg) noexcept {
  switch (severity) {
  case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
  case nvinfer1::ILogger::Severity::kERROR:
    A2X_BASE_LOG_ERROR("TensorRT", msg);
    break;
  case nvinfer1::ILogger::Severity::kWARNING:
  case nvinfer1::ILogger::Severity::kINFO:
    A2X_BASE_LOG_INFO("TensorRT", msg);
    break;
  case nvinfer1::ILogger::Severity::kVERBOSE:
    A2X_BASE_LOG_DEBUG("TensorRT", msg);
    break;
  default:
    break;
  }
}

TRTLogger trtLogger;

} // namespace nva2x
