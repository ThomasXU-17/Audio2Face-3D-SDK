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
#include "audio2x/error.h"

namespace {

  struct Audio2XErrorCategory : public std::error_category {
    const char* name() const noexcept override {
      return "Audio2X";
    }

    std::string message(int ev) const override{
      switch (static_cast<nva2x::ErrorCode>(ev))
      {
      case nva2x::ErrorCode::eSuccess:
        return "No error";
      case nva2x::ErrorCode::eMismatch:
        return "Mismatch between values";
      case nva2x::ErrorCode::eOutOfBounds:
        return "Out of bounds access";
      case nva2x::ErrorCode::eNullPointer:
        return "Null pointer";
      case nva2x::ErrorCode::eInvalidValue:
        return "Invalid value";
      case nva2x::ErrorCode::eUnsupported:
        return "Unsupported operation";
      case nva2x::ErrorCode::eCudaDeviceGetError:
        return "Error getting CUDA device";
      case nva2x::ErrorCode::eCudaDeviceSetError:
        return "Error setting CUDA device";
      case nva2x::ErrorCode::eCudaMemoryAllocationError:
        return "Error allocating CUDA memory";
      case nva2x::ErrorCode::eCudaMemoryFreeError:
        return "Error freeing CUDA memory";
      case nva2x::ErrorCode::eCudaMemcpyDeviceToDeviceError:
        return "Error copying CUDA memory from device to device";
      case nva2x::ErrorCode::eCudaMemcpyHostToDeviceError:
        return "Error copying CUDA memory from host to device";
      case nva2x::ErrorCode::eCudaMemcpyDeviceToHostError:
        return "Error copying CUDA memory from device to host";
      case nva2x::ErrorCode::eCudaMemcpyHostToHostError:
        return "Error copying CUDA memory from host to host";
      case nva2x::ErrorCode::eCudaStreamCreateError:
        return "Error creating CUDA stream";
      case nva2x::ErrorCode::eCudaStreamDestroyError:
        return "Error destroying CUDA stream";
      case nva2x::ErrorCode::eCudaStreamSynchronizeError:
        return "Error synchronizing CUDA stream";
      case nva2x::ErrorCode::eCudaThrustError:
        return "Error running Thrust call";
      case nva2x::ErrorCode::eInitNvInferPluginsFailed:
        return "Error initializing NVInfer plugins";
      case nva2x::ErrorCode::eCreateInferRuntimeFailed:
        return "Error creating NVInfer runtime";
      case nva2x::ErrorCode::eDeserializeCudaEngineFailed:
        return "Error deserializing CUDA engine";
      case nva2x::ErrorCode::eCreateExecutionContextFailed:
        return "Error creating NVInfer execution context";
      case nva2x::ErrorCode::eEnqueueFailed:
        return "Error enqueuing NVInfer operation";
      case nva2x::ErrorCode::eNotAllInputDimensionsSpecified:
        return "Not all input dimensions specified";
      case nva2x::ErrorCode::eMismatchEngineIOBindings:
        return "Mismatch between engine and input/output bindings";
      case nva2x::ErrorCode::eSetTensorAddress:
        return "Error setting tensor address";
      case nva2x::ErrorCode::eSetInputShape:
        return "Error setting input shape";
      case nva2x::ErrorCode::eNotInitialized:
        return "Not initialized";
      case nva2x::ErrorCode::eInterrupted:
        return "Interrupted";
      case nva2x::ErrorCode::eOpenFileFailed:
        return "Error opening file";
      case nva2x::ErrorCode::eReadFileFailed:
        return "Error reading file";
      case nva2x::ErrorCode::eExecutionAlreadyStarted:
        return "Execution already started";
      case nva2x::ErrorCode::eNoTracksToExecute:
        return "No tracks to execute";
      case nva2x::ErrorCode::eEndOfEnum:
        return "End of enum";
      default:
        return "Unrecognized error";
      }
    }
  };

  const Audio2XErrorCategory category;

}

AUDIO2X_SDK_EXPORT std::error_code nva2x::make_error_code(ErrorCode code) {
  return {static_cast<int>(code), category};
}
