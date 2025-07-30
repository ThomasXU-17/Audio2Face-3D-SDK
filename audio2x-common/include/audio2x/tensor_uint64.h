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
// This file contains the same implementation as tensor_float.h, but for uint64_t data.
namespace nva2x {

class AUDIO2X_SDK_EXPORT DeviceTensorUInt64View
{
public:
  DeviceTensorUInt64View();

  DeviceTensorUInt64View View(std::size_t viewOffset, std::size_t viewSize) const;

  uint64_t* Data() const;
  std::size_t Size() const;

  operator DeviceTensorVoidView();
  operator DeviceTensorVoidConstView() const;

  class Accessor;

private:
  DeviceTensorUInt64View(uint64_t* data, std::size_t size);

  uint64_t* _data;
  std::size_t _size;
};
static_assert(sizeof(DeviceTensorUInt64View) == 16);
static_assert(alignof(DeviceTensorUInt64View) == 8);

class AUDIO2X_SDK_EXPORT DeviceTensorUInt64ConstView
{
public:
  DeviceTensorUInt64ConstView();
  DeviceTensorUInt64ConstView(const DeviceTensorUInt64View&);

  DeviceTensorUInt64ConstView View(std::size_t viewOffset, std::size_t viewSize) const;

  const uint64_t* Data() const;
  std::size_t Size() const;

  operator DeviceTensorVoidConstView() const;

  class Accessor;

private:
  DeviceTensorUInt64ConstView(const uint64_t* data, std::size_t size);

  const uint64_t* _data;
  std::size_t _size;
};
static_assert(sizeof(DeviceTensorUInt64ConstView) == 16);
static_assert(alignof(DeviceTensorUInt64ConstView) == 8);


class AUDIO2X_SDK_EXPORT HostTensorUInt64View
{
public:
  HostTensorUInt64View();
  HostTensorUInt64View(uint64_t* data, std::size_t size);

  HostTensorUInt64View View(std::size_t viewOffset, std::size_t viewSize) const;

  uint64_t* Data() const;
  std::size_t Size() const;

private:
  uint64_t* _data;
  std::size_t _size;
};
static_assert(sizeof(HostTensorUInt64View) == 16);
static_assert(alignof(HostTensorUInt64View) == 8);

inline uint64_t* begin(HostTensorUInt64View view) { return view.Data(); }
inline uint64_t* end(HostTensorUInt64View view) { return view.Data() + view.Size(); }

class AUDIO2X_SDK_EXPORT HostTensorUInt64ConstView
{
public:
  HostTensorUInt64ConstView();
  HostTensorUInt64ConstView(const uint64_t* data, std::size_t size);
  HostTensorUInt64ConstView(const HostTensorUInt64View&);

  HostTensorUInt64ConstView View(std::size_t viewOffset, std::size_t viewSize) const;

  const uint64_t* Data() const;
  std::size_t Size() const;

private:
  const uint64_t* _data;
  std::size_t _size;
};
static_assert(sizeof(HostTensorUInt64ConstView) == 16);
static_assert(alignof(HostTensorUInt64ConstView) == 8);

inline const uint64_t* begin(HostTensorUInt64ConstView view) { return view.Data(); }
inline const uint64_t* end(HostTensorUInt64ConstView view) { return view.Data() + view.Size(); }


class IDeviceTensorUInt64 {
public:
  virtual operator DeviceTensorUInt64View() = 0;
  virtual operator DeviceTensorUInt64ConstView() const = 0;
  virtual DeviceTensorUInt64View View(std::size_t viewOffset, std::size_t viewSize) = 0;
  virtual DeviceTensorUInt64ConstView View(std::size_t viewOffset, std::size_t viewSize) const = 0;
  virtual uint64_t* Data() = 0;
  virtual const uint64_t* Data() const = 0;
  virtual std::size_t Size() const = 0;
  virtual void Destroy() = 0;

  operator DeviceTensorVoidView() { return operator DeviceTensorUInt64View(); }
  operator DeviceTensorVoidConstView() const { return operator DeviceTensorUInt64ConstView(); }

protected:
  virtual ~IDeviceTensorUInt64() = default;
};

class IHostTensorUInt64 {
public:
  virtual operator HostTensorUInt64View() = 0;
  virtual operator HostTensorUInt64ConstView() const = 0;
  virtual HostTensorUInt64View View(std::size_t viewOffset, std::size_t viewSize) = 0;
  virtual HostTensorUInt64ConstView View(std::size_t viewOffset, std::size_t viewSize) const = 0;
  virtual uint64_t* Data() = 0;
  virtual const uint64_t* Data() const = 0;
  virtual std::size_t Size() const = 0;
  virtual void Destroy() = 0;

protected:
  virtual ~IHostTensorUInt64() = default;
};


AUDIO2X_SDK_EXPORT IDeviceTensorUInt64* CreateDeviceTensorUInt64(std::size_t size);
AUDIO2X_SDK_EXPORT IDeviceTensorUInt64* CreateDeviceTensorUInt64(HostTensorUInt64ConstView source, cudaStream_t cudaStream);
AUDIO2X_SDK_EXPORT IHostTensorUInt64* CreateHostTensorUInt64(std::size_t size);
AUDIO2X_SDK_EXPORT IHostTensorUInt64* CreateHostPinnedTensorUInt64(std::size_t size);

AUDIO2X_SDK_EXPORT DeviceTensorUInt64View GetDeviceTensorUInt64View(uint64_t* data, std::size_t size);
AUDIO2X_SDK_EXPORT DeviceTensorUInt64ConstView GetDeviceTensorUInt64ConstView(const uint64_t* data, std::size_t size);


AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToDevice(
  DeviceTensorUInt64View destination, DeviceTensorUInt64ConstView source, cudaStream_t cudaStream
  );
AUDIO2X_SDK_EXPORT std::error_code CopyHostToDevice(
  DeviceTensorUInt64View destination, HostTensorUInt64ConstView source, cudaStream_t cudaStream
  );
AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToHost(
  HostTensorUInt64View destination, DeviceTensorUInt64ConstView source, cudaStream_t cudaStream
  );
AUDIO2X_SDK_EXPORT std::error_code CopyHostToHost(
  HostTensorUInt64View destination, HostTensorUInt64ConstView source, cudaStream_t cudaStream
  );

AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToDevice(
  DeviceTensorUInt64View destination, DeviceTensorUInt64ConstView source
  );
AUDIO2X_SDK_EXPORT std::error_code CopyHostToDevice(
  DeviceTensorUInt64View destination, HostTensorUInt64ConstView source
  );
AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToHost(
  HostTensorUInt64View destination, DeviceTensorUInt64ConstView source
  );
AUDIO2X_SDK_EXPORT std::error_code CopyHostToHost(
  HostTensorUInt64View destination, HostTensorUInt64ConstView source
  );

AUDIO2X_SDK_EXPORT std::error_code FillOnDevice(
  DeviceTensorUInt64View destination, uint64_t value, cudaStream_t cudaStream
  );
AUDIO2X_SDK_EXPORT std::error_code FillOnHost(
  HostTensorUInt64View destination, uint64_t value
  );

} // namespace nva2x
