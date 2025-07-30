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

namespace nva2x {

// Tensor views are light-weight non-owning representation of memory buffers,
// basically the equivalent of std::string_view or std::span.
// They are meant to be used to pass parameters to functions, etc., with light copy
// and no ownership transfer.
// Device tensor views should only be obtained from associated device tensor objects.

// Non-const view of a device tensor for float data.
class AUDIO2X_SDK_EXPORT DeviceTensorFloatView
{
public:
  // Default constructor creating an empty view.
  DeviceTensorFloatView();

  // Create a sub-view of the current view.
  DeviceTensorFloatView View(std::size_t viewOffset, std::size_t viewSize) const;

  // Return a pointer to the underlying float data in device memory.
  float* Data() const;

  // Return the number of float elements in the view.
  std::size_t Size() const;

  // Convert to DeviceTensorVoidView for type-erased operations.
  operator DeviceTensorVoidView();

  // Convert to DeviceTensorVoidConstView for const type-erased operations.
  operator DeviceTensorVoidConstView() const;

  // Accessor class for internal implementation.
  class Accessor;

private:
  // Private constructor for internal use.
  DeviceTensorFloatView(float* data, std::size_t size);

  float* _data;
  std::size_t _size;
};
static_assert(sizeof(DeviceTensorFloatView) == 16);
static_assert(alignof(DeviceTensorFloatView) == 8);

// Const view of a device tensor for float data.
class AUDIO2X_SDK_EXPORT DeviceTensorFloatConstView
{
public:
  // Default constructor creating an empty view.
  DeviceTensorFloatConstView();

  // Constructs a const view from a non-const view.
  DeviceTensorFloatConstView(const DeviceTensorFloatView&);

  // Create a sub-view of the current view.
  DeviceTensorFloatConstView View(std::size_t viewOffset, std::size_t viewSize) const;

  // Return a pointer to the underlying const float data in device memory.
  const float* Data() const;

  // Return the number of float elements in the view.
  std::size_t Size() const;

  // Convert to DeviceTensorVoidConstView for const type-erased operations.
  operator DeviceTensorVoidConstView() const;

  // Accessor class for internal implementation.
  class Accessor;

private:
  // Private constructor for internal use.
  DeviceTensorFloatConstView(const float* data, std::size_t size);

  const float* _data;
  std::size_t _size;
};
static_assert(sizeof(DeviceTensorFloatConstView) == 16);
static_assert(alignof(DeviceTensorFloatConstView) == 8);


// Non-const view of a host tensor for float data.
class AUDIO2X_SDK_EXPORT HostTensorFloatView
{
public:
  // Default constructor creating an empty view.
  HostTensorFloatView();

  // Construct a view from existing host memory.
  HostTensorFloatView(float* data, std::size_t size);

  // Create a sub-view of the current view.
  HostTensorFloatView View(std::size_t viewOffset, std::size_t viewSize) const;

  // Return a pointer to the underlying float data in host memory.
  float* Data() const;

  // Return the number of float elements in the view.
  std::size_t Size() const;

private:
  float* _data;
  std::size_t _size;
};
static_assert(sizeof(HostTensorFloatView) == 16);
static_assert(alignof(HostTensorFloatView) == 8);

// Return an iterator to the beginning of the view.
inline float* begin(HostTensorFloatView view) { return view.Data(); }

// Return an iterator to the end of the view.
inline float* end(HostTensorFloatView view) { return view.Data() + view.Size(); }

// Const view of a host tensor for float data.
class AUDIO2X_SDK_EXPORT HostTensorFloatConstView
{
public:
  // Default constructor creating an empty view.
  HostTensorFloatConstView();

  // Construct a const view from existing host memory.
  HostTensorFloatConstView(const float* data, std::size_t size);

  // Construct a const view from a non-const view.
  HostTensorFloatConstView(const HostTensorFloatView&);

  // Create a sub-view of the current view.
  HostTensorFloatConstView View(std::size_t viewOffset, std::size_t viewSize) const;

  // Return a pointer to the underlying const float data in host memory.
  const float* Data() const;

  // Return the number of float elements in the view.
  std::size_t Size() const;

private:
  const float* _data;
  std::size_t _size;
};
static_assert(sizeof(HostTensorFloatConstView) == 16);
static_assert(alignof(HostTensorFloatConstView) == 8);

// Return an iterator to the beginning of the view.
inline const float* begin(HostTensorFloatConstView view) { return view.Data(); }

// Return an iterator to the end of the view.
inline const float* end(HostTensorFloatConstView view) { return view.Data() + view.Size(); }


// This class provides an abstract interface for managing float data in GPU memory
// with proper ownership semantics. Implementations handle memory allocation, deallocation,
// and provide views for efficient access.
class IDeviceTensorFloat {
public:
  // Convert to DeviceTensorFloatView for non-const access.
  virtual operator DeviceTensorFloatView() = 0;

  // Convert to DeviceTensorFloatConstView for const access.
  virtual operator DeviceTensorFloatConstView() const = 0;

  // Create a sub-view of the device tensor.
  virtual DeviceTensorFloatView View(std::size_t viewOffset, std::size_t viewSize) = 0;

  // Create a const sub-view of the device tensor.
  virtual DeviceTensorFloatConstView View(std::size_t viewOffset, std::size_t viewSize) const = 0;

  // Return a pointer to the underlying float data in device memory.
  virtual float* Data() = 0;

  // Return a pointer to the underlying const float data in device memory.
  virtual const float* Data() const = 0;

  // Return the number of float elements in the tensor.
  virtual std::size_t Size() const = 0;

  // Destroy the tensor and free associated memory.
  virtual void Destroy() = 0;

  // Convert to DeviceTensorVoidView for type-erased operations
  operator DeviceTensorVoidView() { return operator DeviceTensorFloatView(); }

  // Convert to DeviceTensorVoidConstView for const type-erased operations
  operator DeviceTensorVoidConstView() const { return operator DeviceTensorFloatConstView(); }

protected:
  virtual ~IDeviceTensorFloat() = default;
};

// This class provides an abstract interface for managing float data in host memory
// with proper ownership semantics. Implementations handle memory allocation, deallocation,
// and provide views for efficient access.
class IHostTensorFloat {
public:
  // Convert to HostTensorFloatView for non-const access.
  virtual operator HostTensorFloatView() = 0;

  // Convert to HostTensorFloatConstView for const access.
  virtual operator HostTensorFloatConstView() const = 0;

  // Create a sub-view of the host tensor.
  virtual HostTensorFloatView View(std::size_t viewOffset, std::size_t viewSize) = 0;

  // Create a const sub-view of the host tensor.
  virtual HostTensorFloatConstView View(std::size_t viewOffset, std::size_t viewSize) const = 0;

  // Return a pointer to the underlying float data in host memory.
  virtual float* Data() = 0;

  // Return a pointer to the underlying const float data in host memory.
  virtual const float* Data() const = 0;

  // Return the number of float elements in the tensor.
  virtual std::size_t Size() const = 0;

  // Destroy the tensor and free associated memory.
  virtual void Destroy() = 0;

protected:
  virtual ~IHostTensorFloat() = default;
};


//
// Creation functions.
//

// Create a device tensor for floats with specified size
AUDIO2X_SDK_EXPORT IDeviceTensorFloat* CreateDeviceTensorFloat(std::size_t size);

// Create a device tensor for floats by copying data from host memory
AUDIO2X_SDK_EXPORT IDeviceTensorFloat* CreateDeviceTensorFloat(HostTensorFloatConstView source, cudaStream_t cudaStream);

// Create a host tensor for floats with specified size
AUDIO2X_SDK_EXPORT IHostTensorFloat* CreateHostTensorFloat(std::size_t size);

// Create a host pinned tensor for floats with specified size
AUDIO2X_SDK_EXPORT IHostTensorFloat* CreateHostPinnedTensorFloat(std::size_t size);

// These functions must be used with extreme care, they should only be used when the
// pointer is GPU device. They should not be necessary when using IDeviceTensorFloat
// classes directly.

// Create a device tensor float view from existing device memory
AUDIO2X_SDK_EXPORT DeviceTensorFloatView GetDeviceTensorFloatView(float* data, std::size_t size);

// Create a device tensor float const view from existing device memory
AUDIO2X_SDK_EXPORT DeviceTensorFloatConstView GetDeviceTensorFloatConstView(const float* data, std::size_t size);

//
// Copy and fill functions.
//
// They could be overloaded based on the type of the views, but they are named
// explicitly after the type of copy for extra clarity.
//

// Asynchronous copies.

// Asynchronously copy float data from device to device
AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToDevice(
  DeviceTensorFloatView destination, DeviceTensorFloatConstView source, cudaStream_t cudaStream
  );

// Asynchronously copy float data from host to device
AUDIO2X_SDK_EXPORT std::error_code CopyHostToDevice(
  DeviceTensorFloatView destination, HostTensorFloatConstView source, cudaStream_t cudaStream
  );

// Asynchronously copy float data from device to host
AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToHost(
  HostTensorFloatView destination, DeviceTensorFloatConstView source, cudaStream_t cudaStream
  );

// Asynchronously copy float data from host to host
AUDIO2X_SDK_EXPORT std::error_code CopyHostToHost(
  HostTensorFloatView destination, HostTensorFloatConstView source, cudaStream_t cudaStream
  );

// Synchronous copies.

// Synchronously copy float data from device to device
AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToDevice(
  DeviceTensorFloatView destination, DeviceTensorFloatConstView source
  );

// Synchronously copy float data from host to device
AUDIO2X_SDK_EXPORT std::error_code CopyHostToDevice(
  DeviceTensorFloatView destination, HostTensorFloatConstView source
  );

// Synchronously copy float data from device to host
AUDIO2X_SDK_EXPORT std::error_code CopyDeviceToHost(
  HostTensorFloatView destination, DeviceTensorFloatConstView source
  );

// Synchronously copy float data from host to host
AUDIO2X_SDK_EXPORT std::error_code CopyHostToHost(
  HostTensorFloatView destination, HostTensorFloatConstView source
  );

// Fill functions.

// Asynchronously fill device memory with a constant float value
AUDIO2X_SDK_EXPORT std::error_code FillOnDevice(
  DeviceTensorFloatView destination, float value, cudaStream_t cudaStream
  );

// Synchronously fill host memory with a constant float value
AUDIO2X_SDK_EXPORT std::error_code FillOnHost(
  HostTensorFloatView destination, float value
  );

} // namespace nva2x
