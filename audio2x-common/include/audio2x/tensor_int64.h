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
#include "audio2x/tensor_void.h"

#include <cstdint>
#include <cstddef>
#include <system_error>

// Refer to tensor_float.h for documentation.
//
// This file contains the same implementation as tensor_float.h, but for int64_t data.
namespace nva2x {

class AUDIO2X_SDK_EXPORT DeviceTensorInt64View
{
public:
  DeviceTensorInt64View();

  DeviceTensorInt64View View(std::size_t viewOffset, std::size_t viewSize) const;

  int64_t* Data() const;
  std::size_t Size() const;

  operator DeviceTensorVoidView();
  operator DeviceTensorVoidConstView() const;

  class Accessor;

private:
  DeviceTensorInt64View(int64_t* data, std::size_t size);

  int64_t* _data;
  std::size_t _size;
};
static_assert(sizeof(DeviceTensorInt64View) == 16);
static_assert(alignof(DeviceTensorInt64View) == 8);

class AUDIO2X_SDK_EXPORT DeviceTensorInt64ConstView
{
public:
  DeviceTensorInt64ConstView();
  DeviceTensorInt64ConstView(const DeviceTensorInt64View&);

  DeviceTensorInt64ConstView View(std::size_t viewOffset, std::size_t viewSize) const;

  const int64_t* Data() const;
  std::size_t Size() const;

  operator DeviceTensorVoidConstView() const;

  class Accessor;

private:
  DeviceTensorInt64ConstView(const int64_t* data, std::size_t size);

  const int64_t* _data;
  std::size_t _size;
};
static_assert(sizeof(DeviceTensorInt64ConstView) == 16);
static_assert(alignof(DeviceTensorInt64ConstView) == 8);


class AUDIO2X_SDK_EXPORT HostTensorInt64View
{
public:
  HostTensorInt64View();
  HostTensorInt64View(int64_t* data, std::size_t size);

  HostTensorInt64View View(std::size_t viewOffset, std::size_t viewSize) const;

  int64_t* Data() const;
  std::size_t Size() const;

private:
  int64_t* _data;
  std::size_t _size;
};
static_assert(sizeof(HostTensorInt64View) == 16);
static_assert(alignof(HostTensorInt64View) == 8);

inline int64_t* begin(HostTensorInt64View view) { return view.Data(); }
inline int64_t* end(HostTensorInt64View view) { return view.Data() + view.Size(); }

class AUDIO2X_SDK_EXPORT HostTensorInt64ConstView
{
public:
  HostTensorInt64ConstView();
  HostTensorInt64ConstView(const int64_t* data, std::size_t size);
  HostTensorInt64ConstView(const HostTensorInt64View&);

  HostTensorInt64ConstView View(std::size_t viewOffset, std::size_t viewSize) const;

  const int64_t* Data() const;
  std::size_t Size() const;

private:
  const int64_t* _data;
  std::size_t _size;
};
static_assert(sizeof(HostTensorInt64ConstView) == 16);
static_assert(alignof(HostTensorInt64ConstView) == 8);

inline const int64_t* begin(HostTensorInt64ConstView view) { return view.Data(); }
inline const int64_t* end(HostTensorInt64ConstView view) { return view.Data() + view.Size(); }


class IDeviceTensorInt64 {
public:
  virtual operator DeviceTensorInt64View() = 0;
  virtual operator DeviceTensorInt64ConstView() const = 0;
  virtual DeviceTensorInt64View View(std::size_t viewOffset, std::size_t viewSize) = 0;
  virtual DeviceTensorInt64ConstView View(std::size_t viewOffset, std::size_t viewSize) const = 0;
  virtual int64_t* Data() = 0;
  virtual const int64_t* Data() const = 0;
  virtual std::size_t Size() const = 0;
  virtual void Destroy() = 0;

  operator DeviceTensorVoidView() { return operator DeviceTensorInt64View(); }
  operator DeviceTensorVoidConstView() const { return operator DeviceTensorInt64ConstView(); }

protected:
  virtual ~IDeviceTensorInt64() = default;
};

class IHostTensorInt64 {
public:
  virtual operator HostTensorInt64View() = 0;
  virtual operator HostTensorInt64ConstView() const = 0;
  virtual HostTensorInt64View View(std::size_t viewOffset, std::size_t viewSize) = 0;
  virtual HostTensorInt64ConstView View(std::size_t viewOffset, std::size_t viewSize) const = 0;
  virtual int64_t* Data() = 0;
  virtual const int64_t* Data() const = 0;
  virtual std::size_t Size() const = 0;
  virtual void Destroy() = 0;

protected:
  virtual ~IHostTensorInt64() = default;
};


AUDIO2X_SDK_EXPORT IDeviceTensorInt64* CreateDeviceTensorInt64(std::size_t size);
AUDIO2X_SDK_EXPORT IDeviceTensorInt64* CreateDeviceTensorInt64(HostTensorInt64ConstView source, cudaStream_t cudaStream);
AUDIO2X_SDK_EXPORT IHostTensorInt64* CreateHostTensorInt64(std::size_t size);
AUDIO2X_SDK_EXPORT IHostTensorInt64* CreateHostPinnedTensorInt64(std::size_t size);

AUDIO2X_SDK_EXPORT DeviceTensorInt64View GetDeviceTensorInt64View(int64_t* data, std::size_t size);
AUDIO2X_SDK_EXPORT DeviceTensorInt64ConstView GetDeviceTensorInt64ConstView(const int64_t* data, std::size_t size);


AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToDevice(
  DeviceTensorInt64View destination, DeviceTensorInt64ConstView source, cudaStream_t cudaStream
  );
AUDIO2X_SDK_EXPORT std::error_code CopyHostToDevice(
  DeviceTensorInt64View destination, HostTensorInt64ConstView source, cudaStream_t cudaStream
  );
AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToHost(
  HostTensorInt64View destination, DeviceTensorInt64ConstView source, cudaStream_t cudaStream
  );
AUDIO2X_SDK_EXPORT std::error_code CopyHostToHost(
  HostTensorInt64View destination, HostTensorInt64ConstView source, cudaStream_t cudaStream
  );

AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToDevice(
  DeviceTensorInt64View destination, DeviceTensorInt64ConstView source
  );
AUDIO2X_SDK_EXPORT std::error_code CopyHostToDevice(
  DeviceTensorInt64View destination, HostTensorInt64ConstView source
  );
AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToHost(
  HostTensorInt64View destination, DeviceTensorInt64ConstView source
  );
AUDIO2X_SDK_EXPORT std::error_code CopyHostToHost(
  HostTensorInt64View destination, HostTensorInt64ConstView source
  );

AUDIO2X_SDK_EXPORT std::error_code FillOnDevice(
  DeviceTensorInt64View destination, int64_t value, cudaStream_t cudaStream
  );
AUDIO2X_SDK_EXPORT std::error_code FillOnHost(
  HostTensorInt64View destination, int64_t value
  );

} // namespace nva2x
