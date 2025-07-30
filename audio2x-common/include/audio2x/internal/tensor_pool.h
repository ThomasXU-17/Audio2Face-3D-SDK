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

#include "audio2x/internal/tensor.h"
#include "audio2x/internal/tensor_pool.h"

#include <memory>
#include <vector>

namespace nva2x {


// This class is not thread-safe.  Any synchronization must be done externally.
class DeviceTensorPool {
public:
  std::error_code Allocate(std::size_t tensorSize, std::size_t tensorCount);
  std::error_code Deallocate();

  std::unique_ptr<DeviceTensorFloat> Obtain();
  std::error_code Return(std::unique_ptr<DeviceTensorFloat> tensor);

  inline std::size_t TensorSize() const {return _tensorSize; }

private:
  std::size_t _tensorSize{0};
  std::vector<std::unique_ptr<DeviceTensorFloat>> _pool;
};


} // namespace nva2x
