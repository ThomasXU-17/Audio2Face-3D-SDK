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

#include "audio2x/tensor_float.h"
#include "audio2x/internal/tensor.h"
#include <cassert>

namespace nva2x::internal {

//
// Create functions
//
IDeviceTensorFloat* CreateDeviceTensorFloat(HostTensorFloatConstView source, cudaStream_t cudaStream) {
  return TensorUtility<float>::CreateDeviceTensor(source, cudaStream);
}
IDeviceTensorFloat* CreateDeviceTensorFloat(std::size_t size) {
  return TensorUtility<float>::CreateTensor<Location::Device>(size);
}
IHostTensorFloat* CreateHostTensorFloat(std::size_t size) {
  return TensorUtility<float>::CreateTensor<Location::Host>(size);
}
IHostTensorFloat* CreateHostPinnedTensorFloat(std::size_t size) {
  return TensorUtility<float>::CreateTensor<Location::HostPinned>(size);
}

// These functions must be used with extreme care, they should only be used when the
// pointer is GPU device. They should not be necessary when using IDeviceTensorFloat
// classes directly.
DeviceTensorFloatView GetDeviceTensorFloatView(float* data, std::size_t size) {
  return TensorUtility<float>::GetDeviceTensorView(data, size);
}
DeviceTensorFloatConstView GetDeviceTensorFloatConstView(const float* data, std::size_t size) {
  return TensorUtility<float>::GetDeviceTensorConstView(data, size);
}

//
// Asynchronous copies.
//
std::error_code CopyDeviceToDevice(
  DeviceTensorFloatView destination, DeviceTensorFloatConstView source, cudaStream_t cudaStream) {
  return TensorUtility<float>::CopyDeviceToDevice(destination, source, cudaStream);
}
std::error_code CopyHostToDevice(
  DeviceTensorFloatView destination, HostTensorFloatConstView source, cudaStream_t cudaStream) {
  return TensorUtility<float>::CopyHostToDevice(destination, source, cudaStream);
}
std::error_code CopyDeviceToHost(
  HostTensorFloatView destination, DeviceTensorFloatConstView source, cudaStream_t cudaStream) {
  return TensorUtility<float>::CopyDeviceToHost(destination, source, cudaStream);
}
std::error_code CopyHostToHost(
  HostTensorFloatView destination, HostTensorFloatConstView source, cudaStream_t cudaStream) {
  return TensorUtility<float>::CopyHostToHost(destination, source, cudaStream);
}

//
// Synchronous copies.
//
std::error_code CopyDeviceToDevice(
  DeviceTensorFloatView destination, DeviceTensorFloatConstView source) {
  return TensorUtility<float>::CopyDeviceToDevice(destination, source);
}
std::error_code CopyHostToDevice(
  DeviceTensorFloatView destination, HostTensorFloatConstView source) {
  return TensorUtility<float>::CopyHostToDevice(destination, source);
}
std::error_code CopyDeviceToHost(
  HostTensorFloatView destination, DeviceTensorFloatConstView source) {
  return TensorUtility<float>::CopyDeviceToHost(destination, source);
}
std::error_code CopyHostToHost(
  HostTensorFloatView destination, HostTensorFloatConstView source) {
  return TensorUtility<float>::CopyHostToHost(destination, source);
}

//
// Fill functions.
//
std::error_code FillOnDevice(DeviceTensorFloatView destination, float value, cudaStream_t cudaStream) {
  return TensorUtility<float>::FillOnDevice(destination, value, cudaStream);
}
std::error_code FillOnHost(HostTensorFloatView destination, float value) {
  return TensorUtility<float>::FillOnHost(destination, value);
}

} // namespace nva2x::internal
