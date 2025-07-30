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
#include "audio2x/internal/tensor.h"
#include "audio2x/internal/audio2x.h"
#include "audio2x/internal/tensor_cuda.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/error.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cstring>

namespace nva2x {

DeviceTensorVoidView::operator DeviceTensorVoidConstView() const {
    return DeviceTensorVoidConstView::Accessor::Build(_data, _size);
}

// Accessor classes to allow only specific functions to create views.
#define DECLARE_TENSOR_VIEW_ACCESSOR(SCALAR, VIEW)  \
class VIEW::Accessor {  \
public: \
    static inline VIEW Build(SCALAR* data, std::size_t size) { return {data, size}; } \
}

DECLARE_TENSOR_VIEW_ACCESSOR(float, DeviceTensorFloatView);
DECLARE_TENSOR_VIEW_ACCESSOR(const float, DeviceTensorFloatConstView);

DECLARE_TENSOR_VIEW_ACCESSOR(bool, DeviceTensorBoolView);
DECLARE_TENSOR_VIEW_ACCESSOR(const bool, DeviceTensorBoolConstView);

DECLARE_TENSOR_VIEW_ACCESSOR(int64_t, DeviceTensorInt64View);
DECLARE_TENSOR_VIEW_ACCESSOR(const int64_t, DeviceTensorInt64ConstView);

DECLARE_TENSOR_VIEW_ACCESSOR(uint64_t, DeviceTensorUInt64View);
DECLARE_TENSOR_VIEW_ACCESSOR(const uint64_t, DeviceTensorUInt64ConstView);

//
// Helper to allocate memory for a tensor.
//
template <typename Scalar, Location L>
struct AllocateHelper;

// Specialization for device.
template <typename Scalar>
struct AllocateHelper<Scalar, Location::Device> {

    using DeviceView = typename TensorHelper<Scalar>::DeviceView;
    using DeviceConstView = typename TensorHelper<Scalar>::DeviceConstView;
    using HostView = typename TensorHelper<Scalar>::HostView;
    using HostConstView = typename TensorHelper<Scalar>::HostConstView;

    static DeviceView Build(Scalar* data, std::size_t size) {
        return DeviceView::Accessor::Build(data, size);
    }

    static DeviceConstView Build(const Scalar* data, std::size_t size) {
        return DeviceConstView::Accessor::Build(data, size);
    }

    static std::error_code Allocate(Scalar** ptr, std::size_t size) {
        A2X_LOG_DEBUG("Allocating device tensor of size " << size);
        A2X_CUDA_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(ptr), size * sizeof(Scalar)),
            ErrorCode::eCudaMemoryAllocationError);
        return nva2x::ErrorCode::eSuccess;
    }

    static std::error_code Copy(DeviceView destination, HostConstView source, cudaStream_t cudaStream) {
        return CopyHostToDevice(destination, source, cudaStream);
    }

    static std::error_code Copy(DeviceView destination, HostConstView source) {
        return CopyHostToDevice(destination, source);
    }

    static std::error_code Deallocate(Scalar* ptr) {
        A2X_LOG_DEBUG("Deleting device tensor");
        A2X_CUDA_CHECK_ERROR(cudaFree(ptr), ErrorCode::eCudaMemoryFreeError);
        return nva2x::ErrorCode::eSuccess;
    }
};

// Specialization for host, not pinned.
template <typename Scalar>
struct AllocateHelper<Scalar, Location::Host> {

    using DeviceView = typename TensorHelper<Scalar>::DeviceView;
    using DeviceConstView = typename TensorHelper<Scalar>::DeviceConstView;
    using HostView = typename TensorHelper<Scalar>::HostView;
    using HostConstView = typename TensorHelper<Scalar>::HostConstView;

    static HostView Build(Scalar* data, std::size_t size) {
        return {data, size};
    }

    static HostConstView Build(const Scalar* data, std::size_t size) {
        return {data, size};
    }

    static std::error_code Allocate(Scalar** ptr, std::size_t size) {
        A2X_LOG_DEBUG("Allocating host tensor of size " << size);
        *ptr = new Scalar[size];
        if (*ptr == nullptr) {
            return nva2x::ErrorCode::eCudaMemoryAllocationError;
        }
        std::memset(*ptr, 0, size * sizeof(Scalar));
        return nva2x::ErrorCode::eSuccess;
    }

    static std::error_code Copy(HostView destination, HostConstView source, cudaStream_t cudaStream) {
        return CopyHostToHost(destination, source, cudaStream);
    }

    static std::error_code Copy(HostView destination, HostConstView source) {
        return CopyHostToHost(destination, source);
    }

    static std::error_code Deallocate(Scalar* ptr) {
        A2X_LOG_DEBUG("Deleting host tensor");
        delete [] ptr;
        return nva2x::ErrorCode::eSuccess;
    }
};

// Specialization for host, pinned.
template <typename Scalar>
struct AllocateHelper<Scalar, Location::HostPinned> {

    using DeviceView = typename TensorHelper<Scalar>::DeviceView;
    using DeviceConstView = typename TensorHelper<Scalar>::DeviceConstView;
    using HostView = typename TensorHelper<Scalar>::HostView;
    using HostConstView = typename TensorHelper<Scalar>::HostConstView;

    static HostView Build(Scalar* data, std::size_t size) {
        return {data, size};
    }

    static HostConstView Build(const Scalar* data, std::size_t size) {
        return {data, size};
    }

    static std::error_code Allocate(Scalar** ptr, std::size_t size) {
        A2X_LOG_DEBUG("Allocating host pinned tensor of size " << size);
        A2X_CUDA_CHECK_ERROR(cudaMallocHost(reinterpret_cast<void**>(ptr), size * sizeof(Scalar)),
            ErrorCode::eCudaMemoryAllocationError);
        return nva2x::ErrorCode::eSuccess;
    }

    static std::error_code Copy(HostView destination, HostConstView source, cudaStream_t cudaStream) {
        return CopyHostToHost(destination, source, cudaStream);
    }

    static std::error_code Copy(HostView destination, HostConstView source) {
        return CopyHostToHost(destination, source);
    }

    static std::error_code Deallocate(Scalar* ptr) {
        A2X_LOG_DEBUG("Deleting host pinned tensor");
        A2X_CUDA_CHECK_ERROR(cudaFreeHost(ptr), ErrorCode::eCudaMemoryFreeError);
        return nva2x::ErrorCode::eSuccess;
    }
};

//
// Tensor implementation.
//
template <typename Scalar, Location L>
Tensor<Scalar, L>::Tensor() : _data(nullptr), _size(0) {
    A2X_LOG_DEBUG("Tensor::Tensor()");
}

template <typename Scalar, Location L>
Tensor<Scalar, L>::~Tensor() {
    A2X_LOG_DEBUG("Tensor::~Tensor()");
    Deallocate();
}

template <typename Scalar, Location L>
Tensor<Scalar, L>::Tensor(Tensor&& other) noexcept : _data(other._data), _size(other._size) {
    A2X_LOG_DEBUG("Tensor::Tensor(Tensor&&)");
    other._data = nullptr;
    other._size = 0;
}

template <typename Scalar, Location L>
typename Tensor<Scalar, L>::Tensor& Tensor<Scalar, L>::operator=(Tensor&& other) noexcept {
    std::swap(_data, other._data);
    std::swap(_size, other._size);
    return *this;
}

template <typename Scalar, Location L>
Tensor<Scalar, L>::operator TensorView() {
    return AllocateHelper<Scalar, L>::Build(_data, _size);
}

template <typename Scalar, Location L>
Tensor<Scalar, L>::operator TensorConstView() const {
    return AllocateHelper<Scalar, L>::Build(_data, _size);
}

template <typename Scalar, Location L>
typename Tensor<Scalar, L>::TensorView Tensor<Scalar, L>::View(std::size_t viewOffset, std::size_t viewSize) {
    assert(viewOffset + viewSize <= _size);
    return AllocateHelper<Scalar, L>::Build(Data() + viewOffset, viewSize);
}

template <typename Scalar, Location L>
typename Tensor<Scalar, L>::TensorConstView Tensor<Scalar, L>::View(std::size_t viewOffset, std::size_t viewSize) const {
    assert(viewOffset + viewSize <= _size);
    return AllocateHelper<Scalar, L>::Build(Data() + viewOffset, viewSize);
}

template <typename Scalar, Location L>
Scalar* Tensor<Scalar, L>::Data() {
    return _data;
}

template <typename Scalar, Location L>
const Scalar* Tensor<Scalar, L>::Data() const {
    return _data;
}

template <typename Scalar, Location L>
std::size_t Tensor<Scalar, L>::Size() const {
    return _size;
}

template <typename Scalar, Location L>
void Tensor<Scalar, L>::Destroy() {
    A2X_LOG_DEBUG("Tensor::Destroy()");
    delete this;
}

template <typename Scalar, Location L>
std::error_code Tensor<Scalar, L>::Allocate(std::size_t size) {
    if (size == _size) {
        A2X_LOG_DEBUG("Reusing allocated tensor of size " << size);
        return ErrorCode::eSuccess;
    }
    A2X_CHECK_RESULT_WITH_MSG(Deallocate(), "Unable to deallocate tensor before allocation");
    auto e = AllocateHelper<Scalar, L>::Allocate(&_data, size);
    _size = size;
    return e;
}

template <typename Scalar, Location L>
std::error_code Tensor<Scalar, L>::Init(HostConstView source, cudaStream_t cudaStream) {
    A2X_CHECK_RESULT(Allocate(source.Size()));
    return AllocateHelper<Scalar, L>::Copy(*this, source, cudaStream);
}

template <typename Scalar, Location L>
std::error_code Tensor<Scalar, L>::Init(HostConstView source) {
    A2X_CHECK_RESULT(Allocate(source.Size()));
    return AllocateHelper<Scalar, L>::Copy(*this, source);
}

template <typename Scalar, Location L>
std::error_code Tensor<Scalar, L>::Deallocate() {
    if (_data == nullptr) {
        return ErrorCode::eSuccess;
    }
    auto e = AllocateHelper<Scalar, L>::Deallocate(_data);
    _data = nullptr;
    _size = 0;
    return e;
}

//
// instantiate tensor template classes
//
template class Tensor<float, Location::Device>;
template class Tensor<float, Location::Host>;
template class Tensor<float, Location::HostPinned>;

template class Tensor<bool, Location::Device>;
template class Tensor<bool, Location::Host>;
template class Tensor<bool, Location::HostPinned>;

template class Tensor<int64_t, Location::Device>;
template class Tensor<int64_t, Location::Host>;
template class Tensor<int64_t, Location::HostPinned>;

template class Tensor<uint64_t, Location::Device>;
template class Tensor<uint64_t, Location::Host>;
template class Tensor<uint64_t, Location::HostPinned>;

//
// Utility class for Tensor
//
template <typename Scalar>
template <Location L>
Tensor<Scalar, L>* TensorUtility<Scalar>::CreateTensor(std::size_t size) {
    Tensor<Scalar, L> tensor;
    if (tensor.Allocate(size)) {
        A2X_LOG_ERROR("Unable to allocate tensor; size=" << size);
        return nullptr;
    }
    return new Tensor<Scalar, L>(std::move(tensor));
}

template <typename Scalar>
typename TensorUtility<Scalar>::DeviceInterface*
TensorUtility<Scalar>::CreateDeviceTensor(HostConstView source, cudaStream_t cudaStream) {
    Tensor<Scalar, Location::Device> tensor;
    if (tensor.Init(source, cudaStream)) {
        A2X_LOG_ERROR("Unable to initialize device tensor; size=" << source.Size());
        return nullptr;
    }
    return new Tensor<Scalar, Location::Device>(std::move(tensor));
}

// These functions must be used with extreme care, they should only be used when the
// pointer is GPU device. They should not be necessary when using IDeviceTensor{Scalar}
// classes directly.
template <typename Scalar>
typename TensorUtility<Scalar>::DeviceView
TensorUtility<Scalar>::GetDeviceTensorView(Scalar* data, std::size_t size) {
    return DeviceView::Accessor::Build(data, size);
}

template <typename Scalar>
typename TensorUtility<Scalar>::DeviceConstView
TensorUtility<Scalar>::GetDeviceTensorConstView(const Scalar* data, std::size_t size) {
    return DeviceConstView::Accessor::Build(data, size);
}

//
// Asynchronous copies.
//
template <typename Scalar>
std::error_code TensorUtility<Scalar>::CopyDeviceToDevice(DeviceView destination, DeviceConstView source, cudaStream_t cudaStream) {
    A2X_CHECK_ERROR_WITH_MSG(destination.Size() == source.Size(), "CopyDeviceToDevice: size mismatch: dst:"
        << destination.Size() << " vs. source:" << source.Size(), ErrorCode::eMismatch);
    A2X_CUDA_CHECK_ERROR(cudaMemcpyAsync(destination.Data(), source.Data(), source.Size() * sizeof(Scalar),
        cudaMemcpyDeviceToDevice, cudaStream), ErrorCode::eCudaMemcpyDeviceToDeviceError);
    return ErrorCode::eSuccess;
}

template <typename Scalar>
std::error_code TensorUtility<Scalar>::CopyHostToDevice(DeviceView destination, HostConstView source, cudaStream_t cudaStream)  {
    A2X_CHECK_ERROR_WITH_MSG(destination.Size() == source.Size(), "CopyHostToDevice: size mismatch: dst:"
        << destination.Size() << " vs. source:" << source.Size(), ErrorCode::eMismatch);
    A2X_CUDA_CHECK_ERROR(cudaMemcpyAsync(destination.Data(), source.Data(), source.Size() * sizeof(Scalar),
        cudaMemcpyHostToDevice, cudaStream), ErrorCode::eCudaMemcpyHostToDeviceError);
    return ErrorCode::eSuccess;
}

template <typename Scalar>
std::error_code TensorUtility<Scalar>::CopyDeviceToHost(HostView destination, DeviceConstView source, cudaStream_t cudaStream) {
    A2X_CHECK_ERROR_WITH_MSG(destination.Size() == source.Size(), "CopyDeviceToHost: size mismatch: dst:"
        << destination.Size() << " vs. source:" << source.Size(), ErrorCode::eMismatch);
    A2X_CUDA_CHECK_ERROR(cudaMemcpyAsync(destination.Data(), source.Data(), source.Size() * sizeof(Scalar),
        cudaMemcpyDeviceToHost, cudaStream), ErrorCode::eCudaMemcpyDeviceToHostError);
    return ErrorCode::eSuccess;
}

template <typename Scalar>
std::error_code TensorUtility<Scalar>::CopyHostToHost(HostView destination, HostConstView source, cudaStream_t cudaStream) {
    A2X_CHECK_ERROR_WITH_MSG(destination.Size() == source.Size(), "CopyHostToHost: size mismatch: dst:"
        << destination.Size() << " vs. source:" << source.Size(), ErrorCode::eMismatch);
    A2X_CUDA_CHECK_ERROR(cudaMemcpyAsync(destination.Data(), source.Data(), source.Size() * sizeof(Scalar),
        cudaMemcpyHostToHost, cudaStream), ErrorCode::eCudaMemcpyHostToHostError);
    return ErrorCode::eSuccess;
}

//
// Synchronous copies.
//
template <typename Scalar>
std::error_code TensorUtility<Scalar>::CopyDeviceToDevice(DeviceView destination, DeviceConstView source) {
    A2X_CHECK_ERROR_WITH_MSG(destination.Size() == source.Size(), "CopyDeviceToDevice: size mismatch: dst:"
        << destination.Size() << " vs. source:" << source.Size(), ErrorCode::eMismatch);
    A2X_CUDA_CHECK_ERROR(cudaMemcpy(destination.Data(), source.Data(), source.Size() * sizeof(Scalar),
        cudaMemcpyDeviceToDevice), ErrorCode::eCudaMemcpyDeviceToDeviceError);
    return ErrorCode::eSuccess;
}

template <typename Scalar>
std::error_code TensorUtility<Scalar>::CopyHostToDevice(DeviceView destination, HostConstView source)  {
    A2X_CHECK_ERROR_WITH_MSG(destination.Size() == source.Size(), "CopyHostToDevice: size mismatch: dst:"
        << destination.Size() << " vs. source:" << source.Size(), ErrorCode::eMismatch);
    A2X_CUDA_CHECK_ERROR(cudaMemcpy(destination.Data(), source.Data(), source.Size() * sizeof(Scalar),
        cudaMemcpyHostToDevice), ErrorCode::eCudaMemcpyHostToDeviceError);
    return ErrorCode::eSuccess;
}

template <typename Scalar>
std::error_code TensorUtility<Scalar>::CopyDeviceToHost(HostView destination, DeviceConstView source) {
    A2X_CHECK_ERROR_WITH_MSG(destination.Size() == source.Size(), "CopyDeviceToHost: size mismatch: dst:"
        << destination.Size() << " vs. source:" << source.Size(), ErrorCode::eMismatch);
    A2X_CUDA_CHECK_ERROR(cudaMemcpy(destination.Data(), source.Data(), source.Size() * sizeof(Scalar),
        cudaMemcpyDeviceToHost), ErrorCode::eCudaMemcpyDeviceToHostError);
    return ErrorCode::eSuccess;
}

template <typename Scalar>
std::error_code TensorUtility<Scalar>::CopyHostToHost(HostView destination, HostConstView source) {
    A2X_CHECK_ERROR_WITH_MSG(destination.Size() == source.Size(), "CopyHostToHost: size mismatch: dst:"
        << destination.Size() << " vs. source:" << source.Size(), ErrorCode::eMismatch);
    A2X_CUDA_CHECK_ERROR(cudaMemcpy(destination.Data(), source.Data(), source.Size() * sizeof(Scalar),
        cudaMemcpyHostToHost), ErrorCode::eCudaMemcpyHostToHostError);
    return ErrorCode::eSuccess;
}

//
// Fill functions.
//
template <typename Scalar>
std::error_code TensorUtility<Scalar>::FillOnDevice(DeviceView destination, Scalar value, cudaStream_t cudaStream) {
    return cuda::FillOnDevice(destination.Data(), destination.Size(), value, cudaStream);
}

template <typename Scalar>
std::error_code TensorUtility<Scalar>::FillOnHost(HostView destination, Scalar value) {
    std::fill(destination.Data(), destination.Data() + destination.Size(), value);
    return ErrorCode::eSuccess;
}

//
// instantiate tensor utility template classes and functions.
//
template struct TensorUtility<float>;
template Tensor<float, Location::Device>* TensorUtility<float>::CreateTensor<Location::Device>(std::size_t size);
template Tensor<float, Location::Host>* TensorUtility<float>::CreateTensor<Location::Host>(std::size_t size);
template Tensor<float, Location::HostPinned>* TensorUtility<float>::CreateTensor<Location::HostPinned>(std::size_t size);

template struct TensorUtility<bool>;
template Tensor<bool, Location::Device>* TensorUtility<bool>::CreateTensor<Location::Device>(std::size_t size);
template Tensor<bool, Location::Host>* TensorUtility<bool>::CreateTensor<Location::Host>(std::size_t size);
template Tensor<bool, Location::HostPinned>* TensorUtility<bool>::CreateTensor<Location::HostPinned>(std::size_t size);

template struct TensorUtility<int64_t>;
template Tensor<int64_t, Location::Device>* TensorUtility<int64_t>::CreateTensor<Location::Device>(std::size_t size);
template Tensor<int64_t, Location::Host>* TensorUtility<int64_t>::CreateTensor<Location::Host>(std::size_t size);
template Tensor<int64_t, Location::HostPinned>* TensorUtility<int64_t>::CreateTensor<Location::HostPinned>(std::size_t size);

template struct TensorUtility<uint64_t>;
template Tensor<uint64_t, Location::Device>* TensorUtility<uint64_t>::CreateTensor<Location::Device>(std::size_t size);
template Tensor<uint64_t, Location::Host>* TensorUtility<uint64_t>::CreateTensor<Location::Host>(std::size_t size);
template Tensor<uint64_t, Location::HostPinned>* TensorUtility<uint64_t>::CreateTensor<Location::HostPinned>(std::size_t size);


std::error_code ValidateTensorBatchInfo(DeviceTensorFloatConstView tensor, const TensorBatchInfo& info) {
  A2X_CHECK_ERROR_WITH_MSG(tensor.Data() != nullptr, "Tensor is not set", ErrorCode::eNullPointer);
  A2X_CHECK_ERROR_WITH_MSG(
    tensor.Size() % info.stride == 0,
    "Tensor size is not a multiple of the stride",
    ErrorCode::eMismatch
    );
  A2X_CHECK_ERROR_WITH_MSG(
    info.offset + info.size <= info.stride,
    "Batched tensor info is invalid",
    ErrorCode::eInvalidValue
    );
  return ErrorCode::eSuccess;
}

} // namespace nva2x
