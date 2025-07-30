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

#include <cstdint>
#include <cstddef>
#include <system_error>

// Refer to tensor_float.h for documentation.
//
// This file contains the same implementation as tensor_float.h, but for void data.
namespace nva2x {

class DeviceTensorVoidConstView;

class AUDIO2X_SDK_EXPORT DeviceTensorVoidView
{
public:
  DeviceTensorVoidView(): _data(nullptr), _size(0) {}

  void* Data() const { return _data; }
  std::size_t Size() const { return _size; }

  operator DeviceTensorVoidConstView() const;

  class Accessor;

private:
  DeviceTensorVoidView(void* data, std::size_t size) : _data(data), _size(size) {}

  void* _data;
  std::size_t _size;
};
static_assert(sizeof(DeviceTensorVoidView) == 16);
static_assert(alignof(DeviceTensorVoidView) == 8);

class AUDIO2X_SDK_EXPORT DeviceTensorVoidConstView
{
public:
  DeviceTensorVoidConstView(): _data(nullptr), _size(0) {}

  const void* Data() const { return _data; }
  std::size_t Size() const { return _size; }

  class Accessor;

private:
  DeviceTensorVoidConstView(const void* data, std::size_t size) : _data(data), _size(size) {}

  const void* _data;
  std::size_t _size;
};
static_assert(sizeof(DeviceTensorVoidConstView) == 16);
static_assert(alignof(DeviceTensorVoidConstView) == 8);

} // namespace nva2x
