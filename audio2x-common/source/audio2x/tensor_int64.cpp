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

#include "audio2x/tensor_int64.h"
#include "audio2x/internal/tensor.h"
#include <cassert>

namespace nva2x::internal {

//
// Create functions
//
IDeviceTensorInt64* CreateDeviceTensorInt64(HostTensorInt64ConstView source, cudaStream_t cudaStream) {
  return TensorUtility<int64_t>::CreateDeviceTensor(source, cudaStream);
}
IDeviceTensorInt64* CreateDeviceTensorInt64(std::size_t size) {
  return TensorUtility<int64_t>::CreateTensor<Location::Device>(size);
}
IHostTensorInt64* CreateHostTensorInt64(std::size_t size) {
  return TensorUtility<int64_t>::CreateTensor<Location::Host>(size);
}
IHostTensorInt64* CreateHostPinnedTensorInt64(std::size_t size) {
  return TensorUtility<int64_t>::CreateTensor<Location::HostPinned>(size);
}

// These functions must be used with extreme care, they should only be used when the
// pointer is GPU device. They should not be necessary when using IDeviceTensorInt64
// classes directly.
DeviceTensorInt64View GetDeviceTensorInt64View(int64_t* data, std::size_t size) {
  return TensorUtility<int64_t>::GetDeviceTensorView(data, size);
}
DeviceTensorInt64ConstView GetDeviceTensorInt64ConstView(const int64_t* data, std::size_t size) {
  return TensorUtility<int64_t>::GetDeviceTensorConstView(data, size);
}

//
// Asynchronous copies.
//
std::error_code CopyDeviceToDevice(
  DeviceTensorInt64View destination, DeviceTensorInt64ConstView source, cudaStream_t cudaStream) {
  return TensorUtility<int64_t>::CopyDeviceToDevice(destination, source, cudaStream);
}
std::error_code CopyHostToDevice(
  DeviceTensorInt64View destination, HostTensorInt64ConstView source, cudaStream_t cudaStream) {
  return TensorUtility<int64_t>::CopyHostToDevice(destination, source, cudaStream);
}
std::error_code CopyDeviceToHost(
  HostTensorInt64View destination, DeviceTensorInt64ConstView source, cudaStream_t cudaStream) {
  return TensorUtility<int64_t>::CopyDeviceToHost(destination, source, cudaStream);
}
std::error_code CopyHostToHost(
  HostTensorInt64View destination, HostTensorInt64ConstView source, cudaStream_t cudaStream) {
  return TensorUtility<int64_t>::CopyHostToHost(destination, source, cudaStream);
}

//
// Synchronous copies.
//
std::error_code CopyDeviceToDevice(
  DeviceTensorInt64View destination, DeviceTensorInt64ConstView source) {
  return TensorUtility<int64_t>::CopyDeviceToDevice(destination, source);
}
std::error_code CopyHostToDevice(
  DeviceTensorInt64View destination, HostTensorInt64ConstView source) {
  return TensorUtility<int64_t>::CopyHostToDevice(destination, source);
}
std::error_code CopyDeviceToHost(
  HostTensorInt64View destination, DeviceTensorInt64ConstView source) {
  return TensorUtility<int64_t>::CopyDeviceToHost(destination, source);
}
std::error_code CopyHostToHost(
  HostTensorInt64View destination, HostTensorInt64ConstView source) {
  return TensorUtility<int64_t>::CopyHostToHost(destination, source);
}

//
// Fill functions.
//
std::error_code FillOnDevice(DeviceTensorInt64View destination, int64_t value, cudaStream_t cudaStream) {
  return TensorUtility<int64_t>::FillOnDevice(destination, value, cudaStream);
}
std::error_code FillOnHost(HostTensorInt64View destination, int64_t value) {
  return TensorUtility<int64_t>::FillOnHost(destination, value);
}

} // namespace nva2x::internal
