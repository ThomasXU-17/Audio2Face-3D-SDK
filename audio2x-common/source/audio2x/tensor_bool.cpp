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

#include "audio2x/tensor_bool.h"
#include "audio2x/internal/tensor.h"
#include <cassert>

namespace nva2x::internal {

//
// Create functions
//
IDeviceTensorBool* CreateDeviceTensorBool(HostTensorBoolConstView source, cudaStream_t cudaStream) {
  return TensorUtility<bool>::CreateDeviceTensor(source, cudaStream);
}
IDeviceTensorBool* CreateDeviceTensorBool(std::size_t size) {
  return TensorUtility<bool>::CreateTensor<Location::Device>(size);
}
IHostTensorBool* CreateHostTensorBool(std::size_t size) {
  return TensorUtility<bool>::CreateTensor<Location::Host>(size);
}
IHostTensorBool* CreateHostPinnedTensorBool(std::size_t size) {
  return TensorUtility<bool>::CreateTensor<Location::HostPinned>(size);
}

// These functions must be used with extreme care, they should only be used when the
// pointer is GPU device. They should not be necessary when using IDeviceTensorBool
// classes directly.
DeviceTensorBoolView GetDeviceTensorBoolView(bool* data, std::size_t size) {
  return TensorUtility<bool>::GetDeviceTensorView(data, size);
}
DeviceTensorBoolConstView GetDeviceTensorBoolConstView(const bool* data, std::size_t size) {
  return TensorUtility<bool>::GetDeviceTensorConstView(data, size);
}

//
// Asynchronous copies.
//
std::error_code CopyDeviceToDevice(
  DeviceTensorBoolView destination, DeviceTensorBoolConstView source, cudaStream_t cudaStream) {
  return TensorUtility<bool>::CopyDeviceToDevice(destination, source, cudaStream);
}
std::error_code CopyHostToDevice(
  DeviceTensorBoolView destination, HostTensorBoolConstView source, cudaStream_t cudaStream) {
  return TensorUtility<bool>::CopyHostToDevice(destination, source, cudaStream);
}
std::error_code CopyDeviceToHost(
  HostTensorBoolView destination, DeviceTensorBoolConstView source, cudaStream_t cudaStream) {
  return TensorUtility<bool>::CopyDeviceToHost(destination, source, cudaStream);
}
std::error_code CopyHostToHost(
  HostTensorBoolView destination, HostTensorBoolConstView source, cudaStream_t cudaStream) {
  return TensorUtility<bool>::CopyHostToHost(destination, source, cudaStream);
}

//
// Synchronous copies.
//
std::error_code CopyDeviceToDevice(
  DeviceTensorBoolView destination, DeviceTensorBoolConstView source) {
  return TensorUtility<bool>::CopyDeviceToDevice(destination, source);
}
std::error_code CopyHostToDevice(
  DeviceTensorBoolView destination, HostTensorBoolConstView source) {
  return TensorUtility<bool>::CopyHostToDevice(destination, source);
}
std::error_code CopyDeviceToHost(
  HostTensorBoolView destination, DeviceTensorBoolConstView source) {
  return TensorUtility<bool>::CopyDeviceToHost(destination, source);
}
std::error_code CopyHostToHost(
  HostTensorBoolView destination, HostTensorBoolConstView source) {
  return TensorUtility<bool>::CopyHostToHost(destination, source);
}

//
// Fill functions.
//
std::error_code FillOnDevice(DeviceTensorBoolView destination, bool value, cudaStream_t cudaStream) {
  return TensorUtility<bool>::FillOnDevice(destination, value, cudaStream);
}
std::error_code FillOnHost(HostTensorBoolView destination, bool value) {
  return TensorUtility<bool>::FillOnHost(destination, value);
}

} // namespace nva2x::internal
