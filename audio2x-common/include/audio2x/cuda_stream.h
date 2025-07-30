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

#include "audio2x/export.h"
#include "audio2x/cuda_fwd.h"

#include <system_error>

namespace nva2x {

// This class creates, manages, and synchronizes CUDA streams.
class ICudaStream {
public:
  // Synchronize the CUDA stream.
  virtual std::error_code Synchronize() const = 0;

  // Get the underlying CUDA stream handle
  virtual cudaStream_t Data() const = 0;

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~ICudaStream();
};

// Create a new CUDA stream instance
AUDIO2X_SDK_EXPORT ICudaStream* CreateCudaStream();

// Create a CUDA stream instance using the default CUDA stream
AUDIO2X_SDK_EXPORT ICudaStream* CreateDefaultCudaStream();

} // namespace nva2x
