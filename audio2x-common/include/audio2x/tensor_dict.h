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

#include <system_error>

namespace nva2x {

class IHostTensorFloat;

// This class contains multiple named float tensors in host memory.
// It supports reading tensor data from binary files or NPZ (NumPy compressed) files.
// Each tensor is identified by a string name and contains float data of arbitrary size.
class IHostTensorDict {
public:
  // Read tensor data from a memory buffer containing a custom binary format.
  // The buffer should contain: number of tensors (uint32), followed by pairs of
  // tensor name length (uint32), tensor name (char array), tensor data length (uint32),
  // and tensor data (float array) for each tensor.
  virtual std::error_code ReadFromBuffer(const void* data, size_t size) = 0;

  // Read tensor data from a file. Supports both binary format (.bin) and NPZ format (.npz).
  // For NPZ files, uses the cnpy library to read NumPy compressed arrays.
  // For binary files, uses the same format as ReadFromBuffer.
  virtual std::error_code ReadFromFile(const char* filePath) = 0;

  // Retrieve a tensor by its name and return a const pointer to the tensor,
  // or nullptr if the tensor with the specified name doesn't exist.
  // The returned pointer remains valid until the IHostTensorDict is destroyed.
  virtual const IHostTensorFloat* At(const char* tensorName) const = 0;

  // Return the number of tensors stored in this dictionary.
  virtual size_t Size() const = 0;

  // Delete this object.
  virtual void Destroy() = 0;

protected:
  virtual ~IHostTensorDict();
};

// Create a new host tensor dictionary instance.
AUDIO2X_SDK_EXPORT IHostTensorDict* CreateHostTensorDict();

} // namespace nva2x
