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

#include "audio2x/tensor_float.h"
#include "audio2x/tensor_int64.h"
#include "audio2x/tensor_uint64.h"
#include "audio2x/tensor_bool.h"
#include "audio2x/tensor_void.h"

namespace nva2x {

// Information describing how multiple slices are batched together in a tensor.
struct TensorBatchInfo {
  // Offset from the start of the tensor to the first slice.
  std::size_t offset{0};
  // Size of a batch slice.
  std::size_t size{0};
  // Stride between consecutive slices.
  std::size_t stride{0};
};

// Validates that the given tensor batch info is compatible with the provided tensor.
std::error_code ValidateTensorBatchInfo(DeviceTensorFloatConstView tensor, const TensorBatchInfo& info);

} // namespace nva2x
