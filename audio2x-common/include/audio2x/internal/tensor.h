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

#include "audio2x/tensor.h"
#include "audio2x/internal/logger.h"
#include <array>
#include <vector>

namespace nva2x {

//
// Tensor location enumeration
//
enum class Location {
  Host = 0,
  HostPinned = 1,
  Device = 2
};

//
// Forward declaration
//
template <typename Scalar, Location L>
class Tensor;

using DeviceTensorFloat = Tensor<float, Location::Device>;
using HostTensorFloat = Tensor<float, Location::Host>;
using HostPinnedTensorFloat = Tensor<float, Location::HostPinned>;

using DeviceTensorBool = Tensor<bool, Location::Device>;
using HostTensorBool = Tensor<bool, Location::Host>;
using HostPinnedTensorBool = Tensor<bool, Location::HostPinned>;

using DeviceTensorInt64 = Tensor<int64_t, Location::Device>;
using HostTensorInt64 = Tensor<int64_t, Location::Host>;
using HostPinnedTensorInt64 = Tensor<int64_t, Location::HostPinned>;

using DeviceTensorUInt64 = Tensor<uint64_t, Location::Device>;
using HostTensorUInt64 = Tensor<uint64_t, Location::Host>;
using HostPinnedTensorUInt64 = Tensor<uint64_t, Location::HostPinned>;

//
// Helper class to get the correct interface and view types for a given scalar typee.
//
template <typename Scalar>
struct TensorHelper;

template <>
struct TensorHelper<float> {
    using Scalar = float;

    using DeviceInterface = IDeviceTensorFloat;
    using DeviceView = DeviceTensorFloatView;
    using DeviceConstView = DeviceTensorFloatConstView;

    using HostInterface = IHostTensorFloat;
    using HostView = HostTensorFloatView;
    using HostConstView = HostTensorFloatConstView;

    using DeviceTensor = DeviceTensorFloat;
    using HostTensor = HostTensorFloat;
    using HostPinnedTensor = HostPinnedTensorFloat;
};

template <>
struct TensorHelper<bool> {
    using Scalar = bool;

    using DeviceInterface = nva2x::IDeviceTensorBool;
    using DeviceView = nva2x::DeviceTensorBoolView;
    using DeviceConstView = nva2x::DeviceTensorBoolConstView;

    using HostInterface = nva2x::IHostTensorBool;
    using HostView = nva2x::HostTensorBoolView;
    using HostConstView = nva2x::HostTensorBoolConstView;

    using DeviceTensor = DeviceTensorBool;
    using HostTensor = HostTensorBool;
    using HostPinnedTensor = HostPinnedTensorBool;
};

template <>
struct TensorHelper<int64_t> {
    using Scalar = int64_t;
    using DeviceInterface = nva2x::IDeviceTensorInt64;
    using DeviceView = nva2x::DeviceTensorInt64View;
    using DeviceConstView = nva2x::DeviceTensorInt64ConstView;

    using HostInterface = nva2x::IHostTensorInt64;
    using HostView = nva2x::HostTensorInt64View;
    using HostConstView = nva2x::HostTensorInt64ConstView;

    using DeviceTensor = DeviceTensorInt64;
    using HostTensor = HostTensorInt64;
    using HostPinnedTensor = HostPinnedTensorInt64;
};

template <>
struct TensorHelper<uint64_t> {
    using Scalar = uint64_t;
    using DeviceInterface = nva2x::IDeviceTensorUInt64;
    using DeviceView = nva2x::DeviceTensorUInt64View;
    using DeviceConstView = nva2x::DeviceTensorUInt64ConstView;

    using HostInterface = nva2x::IHostTensorUInt64;
    using HostView = nva2x::HostTensorUInt64View;
    using HostConstView = nva2x::HostTensorUInt64ConstView;

    using DeviceTensor = DeviceTensorUInt64;
    using HostTensor = HostTensorUInt64;
    using HostPinnedTensor = HostPinnedTensorUInt64;
};

//
// Helper class to get the correct interface and view types for a given location.
//
template <typename Scalar, Location L>
struct TensorLocationHelper;

template <typename Scalar>
struct TensorLocationHelper<Scalar, Location::Host> {
  using Interface = typename TensorHelper<Scalar>::HostInterface;
  using TensorView = typename TensorHelper<Scalar>::HostView;
  using TensorConstView = typename TensorHelper<Scalar>::HostConstView;
  using HostConstView = typename TensorHelper<Scalar>::HostConstView;
};

template <typename Scalar>
struct TensorLocationHelper<Scalar, Location::HostPinned> {
  using Interface = typename TensorHelper<Scalar>::HostInterface;
  using TensorView = typename TensorHelper<Scalar>::HostView;
  using TensorConstView = typename TensorHelper<Scalar>::HostConstView;
  using HostConstView = typename TensorHelper<Scalar>::HostConstView;
};

template <typename Scalar>
struct TensorLocationHelper<Scalar, Location::Device> {
  using Interface = typename TensorHelper<Scalar>::DeviceInterface;
  using TensorView = typename TensorHelper<Scalar>::DeviceView;
  using TensorConstView = typename TensorHelper<Scalar>::DeviceConstView;
  using HostConstView = typename TensorHelper<Scalar>::HostConstView;
};

//
// Template class for Tensor
//
template <typename Scalar, Location L>
class Tensor : public TensorLocationHelper<Scalar, L>::Interface {
public:
  // Deduce view types based on Scalar and Location
  using TensorView = typename TensorLocationHelper<Scalar, L>::TensorView;
  using TensorConstView = typename TensorLocationHelper<Scalar, L>::TensorConstView;
  using HostConstView = typename TensorLocationHelper<Scalar, L>::HostConstView;

  Tensor();
  ~Tensor();

  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  operator TensorView() override;
  operator TensorConstView() const override;
  TensorView View(std::size_t viewOffset, std::size_t viewSize) override;
  TensorConstView View(std::size_t viewOffset, std::size_t viewSize) const override;

  Scalar* Data() override;
  const Scalar* Data() const override;
  std::size_t Size() const override;
  void Destroy() override;

  std::error_code Allocate(std::size_t size);
  std::error_code Init(HostConstView source, cudaStream_t cudaStream);
  std::error_code Init(HostConstView source);
  std::error_code Deallocate();

private:
  Scalar* _data;
  std::size_t _size;
};

//
// extern instantiated template classes
//
extern template class Tensor<float, Location::Device>;
extern template class Tensor<float, Location::Host>;
extern template class Tensor<float, Location::HostPinned>;

extern template class Tensor<bool, Location::Device>;
extern template class Tensor<bool, Location::Host>;
extern template class Tensor<bool, Location::HostPinned>;

extern template class Tensor<int64_t, Location::Device>;
extern template class Tensor<int64_t, Location::Host>;
extern template class Tensor<int64_t, Location::HostPinned>;

extern template class Tensor<uint64_t, Location::Device>;
extern template class Tensor<uint64_t, Location::Host>;
extern template class Tensor<uint64_t, Location::HostPinned>;


// Helpers pass vectors and arrays as views.
// With C++20, we could just have one set taking std::span
template<typename Scalar>
inline typename TensorHelper<Scalar>::HostConstView ToConstView(const std::vector<Scalar>& vec) {
  // Note: this function does not work for std::vector<bool>, because it is a specialization
  // which does not support the data() method.
  return {vec.data(), vec.size()};
}

template<typename Scalar>
inline typename TensorHelper<Scalar>::HostView ToView(std::vector<Scalar>& vec) {
  // Note: this function does not work for std::vector<bool>, because it is a specialization
  // which does not support the data() method.
  return {vec.data(), vec.size()};
}

template <typename Scalar, std::size_t T>
inline typename TensorHelper<Scalar>::HostConstView ToConstView(const std::array<Scalar, T>& arr) {
  return {arr.data(), arr.size()};
}

template<typename Scalar, std::size_t T>
inline typename TensorHelper<Scalar>::HostView ToView(std::array<Scalar, T>& arr) {
  return {arr.data(), arr.size()};
}

//
// Utility class for Tensor
//
template<typename Scalar>
struct TensorUtility {

  using DeviceInterface = typename TensorHelper<Scalar>::DeviceInterface;
  using HostInterface   = typename TensorHelper<Scalar>::HostInterface;
  using DeviceView      = typename TensorHelper<Scalar>::DeviceView;
  using DeviceConstView = typename TensorHelper<Scalar>::DeviceConstView;
  using HostView        = typename TensorHelper<Scalar>::HostView;
  using HostConstView   = typename TensorHelper<Scalar>::HostConstView;

// Create functions
template<Location L>
static Tensor<Scalar, L>* CreateTensor(std::size_t size);

static DeviceInterface* CreateDeviceTensor(HostConstView source, cudaStream_t cudaStream);

// These functions must be used with extreme care, they should only be used when the
// pointer is GPU device. They should not be necessary when using IDeviceTensor{Scalar}
// classes directly.
static DeviceView GetDeviceTensorView(Scalar* data, std::size_t size);
static DeviceConstView GetDeviceTensorConstView(const Scalar* data, std::size_t size);

// Asynchronous copies.
static std::error_code CopyDeviceToDevice(DeviceView destination, DeviceConstView source, cudaStream_t cudaStream);
static std::error_code CopyHostToDevice(DeviceView destination, HostConstView source, cudaStream_t cudaStream);
static std::error_code CopyDeviceToHost(HostView destination, DeviceConstView source, cudaStream_t cudaStream);
static std::error_code CopyHostToHost(HostView destination, HostConstView source, cudaStream_t cudaStream);

// Synchronous copies.
static std::error_code CopyDeviceToDevice(DeviceView destination, DeviceConstView source);
static std::error_code CopyHostToDevice(DeviceView destination, HostConstView source);
static std::error_code CopyDeviceToHost(HostView destination, DeviceConstView source);
static std::error_code CopyHostToHost(HostView destination, HostConstView source);

// Fill functions.
static std::error_code FillOnDevice(DeviceView destination, Scalar value, cudaStream_t cudaStream);
static std::error_code FillOnHost(HostView destination, Scalar value);

}; // struct TensorUtility

//
// extern instantiated template classes
//
extern template struct TensorUtility<float>;
extern template struct TensorUtility<bool>;
extern template struct TensorUtility<int64_t>;
extern template struct TensorUtility<uint64_t>;

} // namespace nva2x
