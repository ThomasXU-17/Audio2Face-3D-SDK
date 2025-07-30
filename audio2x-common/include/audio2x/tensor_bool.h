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
// This file contains the same implementation as tensor_float.h, but for bool data.
namespace nva2x {

class AUDIO2X_SDK_EXPORT DeviceTensorBoolView
{
public:
  DeviceTensorBoolView();

  DeviceTensorBoolView View(std::size_t viewOffset, std::size_t viewSize) const;

  bool* Data() const;
  std::size_t Size() const;

  operator DeviceTensorVoidView();
  operator DeviceTensorVoidConstView() const;

  class Accessor;

private:
  DeviceTensorBoolView(bool* data, std::size_t size);

  bool* _data;
  std::size_t _size;
};
static_assert(sizeof(DeviceTensorBoolView) == 16);
static_assert(alignof(DeviceTensorBoolView) == 8);

class AUDIO2X_SDK_EXPORT DeviceTensorBoolConstView
{
public:
  DeviceTensorBoolConstView();
  DeviceTensorBoolConstView(const DeviceTensorBoolView&);

  DeviceTensorBoolConstView View(std::size_t viewOffset, std::size_t viewSize) const;

  const bool* Data() const;
  std::size_t Size() const;

  operator DeviceTensorVoidConstView() const;

  class Accessor;

private:
  DeviceTensorBoolConstView(const bool* data, std::size_t size);

  const bool* _data;
  std::size_t _size;
};
static_assert(sizeof(DeviceTensorBoolConstView) == 16);
static_assert(alignof(DeviceTensorBoolConstView) == 8);


class AUDIO2X_SDK_EXPORT HostTensorBoolView
{
public:
  HostTensorBoolView();
  HostTensorBoolView(bool* data, std::size_t size);

  HostTensorBoolView View(std::size_t viewOffset, std::size_t viewSize) const;

  bool* Data() const;
  std::size_t Size() const;

private:
  bool* _data;
  std::size_t _size;
};
static_assert(sizeof(HostTensorBoolView) == 16);
static_assert(alignof(HostTensorBoolView) == 8);

inline bool* begin(HostTensorBoolView view) { return view.Data(); }
inline bool* end(HostTensorBoolView view) { return view.Data() + view.Size(); }

class AUDIO2X_SDK_EXPORT HostTensorBoolConstView
{
public:
  HostTensorBoolConstView();
  HostTensorBoolConstView(const bool* data, std::size_t size);
  HostTensorBoolConstView(const HostTensorBoolView&);

  HostTensorBoolConstView View(std::size_t viewOffset, std::size_t viewSize) const;

  const bool* Data() const;
  std::size_t Size() const;

private:
  const bool* _data;
  std::size_t _size;
};
static_assert(sizeof(HostTensorBoolConstView) == 16);
static_assert(alignof(HostTensorBoolConstView) == 8);

inline const bool* begin(HostTensorBoolConstView view) { return view.Data(); }
inline const bool* end(HostTensorBoolConstView view) { return view.Data() + view.Size(); }


class IDeviceTensorBool {
public:
  virtual operator DeviceTensorBoolView() = 0;
  virtual operator DeviceTensorBoolConstView() const = 0;
  virtual DeviceTensorBoolView View(std::size_t viewOffset, std::size_t viewSize) = 0;
  virtual DeviceTensorBoolConstView View(std::size_t viewOffset, std::size_t viewSize) const = 0;
  virtual bool* Data() = 0;
  virtual const bool* Data() const = 0;
  virtual std::size_t Size() const = 0;
  virtual void Destroy() = 0;

  operator DeviceTensorVoidView() { return operator DeviceTensorBoolView(); }
  operator DeviceTensorVoidConstView() const { return operator DeviceTensorBoolConstView(); }

protected:
  virtual ~IDeviceTensorBool() = default;
};

class IHostTensorBool {
public:
  virtual operator HostTensorBoolView() = 0;
  virtual operator HostTensorBoolConstView() const = 0;
  virtual HostTensorBoolView View(std::size_t viewOffset, std::size_t viewSize) = 0;
  virtual HostTensorBoolConstView View(std::size_t viewOffset, std::size_t viewSize) const = 0;
  virtual bool* Data() = 0;
  virtual const bool* Data() const = 0;
  virtual std::size_t Size() const = 0;
  virtual void Destroy() = 0;

protected:
  virtual ~IHostTensorBool() = default;
};


AUDIO2X_SDK_EXPORT IDeviceTensorBool* CreateDeviceTensorBool(std::size_t size);
AUDIO2X_SDK_EXPORT IDeviceTensorBool* CreateDeviceTensorBool(HostTensorBoolConstView source, cudaStream_t cudaStream);
AUDIO2X_SDK_EXPORT IHostTensorBool* CreateHostTensorBool(std::size_t size);
AUDIO2X_SDK_EXPORT IHostTensorBool* CreateHostPinnedTensorBool(std::size_t size);

AUDIO2X_SDK_EXPORT DeviceTensorBoolView GetDeviceTensorBoolView(bool* data, std::size_t size);
AUDIO2X_SDK_EXPORT DeviceTensorBoolConstView GetDeviceTensorBoolConstView(const bool* data, std::size_t size);


AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToDevice(
  DeviceTensorBoolView destination, DeviceTensorBoolConstView source, cudaStream_t cudaStream
  );
AUDIO2X_SDK_EXPORT std::error_code CopyHostToDevice(
  DeviceTensorBoolView destination, HostTensorBoolConstView source, cudaStream_t cudaStream
  );
AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToHost(
  HostTensorBoolView destination, DeviceTensorBoolConstView source, cudaStream_t cudaStream
  );
AUDIO2X_SDK_EXPORT std::error_code CopyHostToHost(
  HostTensorBoolView destination, HostTensorBoolConstView source, cudaStream_t cudaStream
  );

AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToDevice(
  DeviceTensorBoolView destination, DeviceTensorBoolConstView source
  );
AUDIO2X_SDK_EXPORT std::error_code CopyHostToDevice(
  DeviceTensorBoolView destination, HostTensorBoolConstView source
  );
AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToHost(
  HostTensorBoolView destination, DeviceTensorBoolConstView source
  );
AUDIO2X_SDK_EXPORT std::error_code CopyHostToHost(
  HostTensorBoolView destination, HostTensorBoolConstView source
  );

AUDIO2X_SDK_EXPORT std::error_code FillOnDevice(
  DeviceTensorBoolView destination, bool value, cudaStream_t cudaStream
  );
AUDIO2X_SDK_EXPORT std::error_code FillOnHost(
  HostTensorBoolView destination, bool value
  );

} // namespace nva2x
