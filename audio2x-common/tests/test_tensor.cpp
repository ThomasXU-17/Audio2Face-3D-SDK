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
#include "audio2x/internal/cuda_stream.h"
#include "audio2x/error.h"
#include "utils.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <array>

namespace {
  constexpr std::size_t kSize = 1234;
}

template <typename T>
class Tensor : public ::testing::Test {
};

// Define the tensor types to test
using TensorTypes = ::testing::Types<float, bool, int64_t, uint64_t>;
TYPED_TEST_SUITE(Tensor, TensorTypes);

TYPED_TEST(Tensor, Copy) {
  using Helper = nva2x::TensorHelper<TypeParam>;
  using Scalar = typename Helper::Scalar;
  using DeviceView = typename Helper::DeviceView;
  using DeviceConstView = typename Helper::DeviceConstView;
  using HostView = typename Helper::HostView;
  using HostConstView = typename Helper::HostConstView;
  using DeviceTensor = typename Helper::DeviceTensor;
  using HostPinnedTensor = typename Helper::HostPinnedTensor;

  const std::array<Scalar, kSize> source = []() {
    std::array<Scalar, kSize> source;
    for (size_t t = 0; t < source.size(); ++t) {
      source[t] = static_cast<Scalar>(rand());
    }
    return source;
  }();

  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  for (bool async : {false, true}) {
    std::function<std::error_code(DeviceView, DeviceConstView)> copyDeviceToDevice;
    std::function<std::error_code(DeviceView, HostConstView)> copyHostToDevice;
    std::function<std::error_code(HostView, DeviceConstView)> copyDeviceToHost;
    std::function<std::error_code(HostView, HostConstView)> copyHostToHost;
    std::function<std::error_code()> synchronize;

    if (async) {
      copyDeviceToDevice = [cudaStream=cudaStream.Data()](auto destination, auto source) {
        return nva2x::CopyDeviceToDevice(destination, source, cudaStream);
      };
      copyHostToDevice = [cudaStream=cudaStream.Data()](auto destination, auto source) {
        return nva2x::CopyHostToDevice(destination, source, cudaStream);
      };
      copyDeviceToHost = [cudaStream=cudaStream.Data()](auto destination, auto source) {
        return nva2x::CopyDeviceToHost(destination, source, cudaStream);
      };
      copyHostToHost = [cudaStream=cudaStream.Data()](auto destination, auto source) {
        return nva2x::CopyHostToHost(destination, source, cudaStream);
      };
      synchronize = [&cudaStream]() {
        return cudaStream.Synchronize();
      };
    }
    else {
      copyDeviceToDevice = [](auto destination, auto source) {
        return nva2x::CopyDeviceToDevice(destination, source);
      };
      copyHostToDevice = [](auto destination, auto source) {
        return nva2x::CopyHostToDevice(destination, source);
      };
      copyDeviceToHost = [](auto destination, auto source) {
        return nva2x::CopyDeviceToHost(destination, source);
      };
      copyHostToHost = [](auto destination, auto source) {
        return nva2x::CopyHostToHost(destination, source);
      };
      synchronize = [&cudaStream]() {
        return nva2x::ErrorCode::eSuccess;
      };
    }

    // Test copy to device and back.
    {
      std::array<Scalar, source.size()> destination{0};
      ASSERT_NE(source, destination);

      DeviceTensor tensor;
      ASSERT_EQ(0U, tensor.Size());
      ASSERT_TRUE(!tensor.Allocate(source.size()));
      ASSERT_EQ(source.size(), tensor.Size());

      ASSERT_TRUE(!copyHostToDevice(tensor, {source.data(), source.size()}));
      ASSERT_TRUE(!copyDeviceToHost({destination.data(), destination.size()}, tensor));
      ASSERT_TRUE(!synchronize());
      ASSERT_TRUE(!tensor.Deallocate());

      ASSERT_EQ(source, destination);
    }

    // Test copy to device and back using pinned tensors.
    {
      std::array<Scalar, source.size()> destination{0};
      ASSERT_NE(source, destination);

      HostPinnedTensor pinnedSource;
      ASSERT_EQ(0U, pinnedSource.Size());
      ASSERT_TRUE(!pinnedSource.Allocate(source.size()));
      ASSERT_EQ(source.size(), pinnedSource.Size());

      HostPinnedTensor pinnedDestination;
      ASSERT_EQ(0U, pinnedDestination.Size());
      ASSERT_TRUE(!pinnedDestination.Allocate(source.size()));
      ASSERT_EQ(source.size(), pinnedDestination.Size());

      // Copy data multiple times.
      static constexpr std::size_t kNbCopies = 1000;
      std::vector<DeviceTensor> tensors(kNbCopies + 1);
      for (auto& tensor : tensors) {
        ASSERT_EQ(0U, tensor.Size());
        ASSERT_TRUE(!tensor.Allocate(source.size()));
        ASSERT_EQ(source.size(), tensor.Size());
      }

      ASSERT_TRUE(!copyHostToHost(pinnedSource, {source.data(), source.size()}));
      ASSERT_TRUE(!copyHostToDevice(tensors.front(), pinnedSource));
      for (std::size_t i = 1; i < tensors.size(); ++i) {
        ASSERT_TRUE(!copyDeviceToDevice(tensors[i], tensors[i-1]));
      }
      ASSERT_TRUE(!copyDeviceToHost(pinnedDestination, tensors.back()));
      ASSERT_TRUE(!copyHostToHost({destination.data(), destination.size()}, pinnedDestination));
      ASSERT_TRUE(!synchronize());
      for (auto& tensor : tensors) {
        ASSERT_TRUE(!tensor.Deallocate());
      }

      // Make sure we can also read the pinned buffers from the host.
      ASSERT_TRUE(std::equal(source.begin(), source.end(), pinnedSource.Data()));
      ASSERT_TRUE(std::equal(source.begin(), source.end(), pinnedDestination.Data()));

      ASSERT_TRUE(!pinnedSource.Deallocate());
      ASSERT_TRUE(!pinnedDestination.Deallocate());

      ASSERT_EQ(source, destination);
    }
  }
}

TYPED_TEST(Tensor, Fill) {
  using Helper = nva2x::TensorHelper<TypeParam>;
  using Scalar = typename Helper::Scalar;

  const std::vector<Scalar> valuesToTry = []() {
    std::vector<Scalar> values;
    if constexpr (std::is_same_v<Scalar, bool>) {
      values = {false, true};
    }
    else if constexpr (std::is_integral_v<Scalar> && !std::is_signed_v<Scalar>) {
      for (Scalar value = 0; value <= 10; ++value) {
        values.push_back(value);
      }
    }
    else {
      for (Scalar value = -10; value <= 10; ++value) {
        values.push_back(value);
      }
    }
    return values;
  }();

  {
    // Simple on host.
    constexpr auto all_equal = [](const auto& data, auto value) {
      return std::all_of(data.begin(), data.end(), [value](Scalar element) { return value == element; });
    };
    std::array<Scalar, kSize> data;
    bool first = true;
    for (const auto value : valuesToTry) {
      ASSERT_TRUE(first || !all_equal(data, value));
      ASSERT_TRUE(!nva2x::FillOnHost({data.data(), data.size()}, value));
      ASSERT_TRUE(all_equal(data, value));
      first = false;
    }
  }

  {
    // Simple on device.
    constexpr auto all_equal = [](const auto& deviceData, auto value, cudaStream_t cudaStream) {
      std::array<Scalar, kSize> data;
      nva2x::CopyDeviceToHost({data.data(), data.size()}, deviceData, cudaStream);
      return std::all_of(data.begin(), data.end(), [value](Scalar element) { return value == element; });
    };
    typename Helper::DeviceTensor data;
    ASSERT_TRUE(!data.Allocate(kSize));
    bool first = true;
    for (const auto value : valuesToTry) {
      ASSERT_TRUE(first || !all_equal(data, value, nullptr));
      ASSERT_TRUE(!nva2x::FillOnDevice(data, value, nullptr));
      ASSERT_TRUE(all_equal(data, value, nullptr));
      first = false;
    }

    // Do it again with a stream.
    nva2x::CudaStream cudaStream;
    ASSERT_TRUE(!cudaStream.Init());
    for (const auto value : valuesToTry) {
      ASSERT_TRUE(!all_equal(data, value, cudaStream.Data()));
      ASSERT_TRUE(!nva2x::FillOnDevice(data, value, cudaStream.Data()));
      ASSERT_TRUE(all_equal(data, value, cudaStream.Data()));
    }
  }
}

TYPED_TEST(Tensor, View) {
  using Helper = nva2x::TensorHelper<TypeParam>;
  using Scalar = typename Helper::Scalar;
  // Simple hard-coded one on host.
  if constexpr (std::is_same_v<Scalar, bool>)
  {
    std::array<Scalar, 5> data{0,1,0,0,1};
    typename Helper::HostView view{data.data(), data.size()};

    std::array<Scalar, 2> replacement{1, 1};
    ASSERT_TRUE(!nva2x::CopyHostToHost(view.View(2, 2), {replacement.data(), replacement.size()}, nullptr));
    ASSERT_THAT(data, ::testing::ElementsAre(Scalar(0), Scalar(1), Scalar(1), Scalar(1), Scalar(1)));
  }
  else
  {
    std::array<Scalar, 5> data{1, 2, 3, 4, 5};
    typename Helper::HostView view{data.data(), data.size()};

    std::array<Scalar, 2> replacement{13, 14};
    ASSERT_TRUE(!nva2x::CopyHostToHost(view.View(2, 2), {replacement.data(), replacement.size()}, nullptr));
    ASSERT_THAT(data, ::testing::ElementsAre(Scalar(1), Scalar(2), Scalar(13), Scalar(14), Scalar(5)));
  }

  // Test on device.
  {
    typename Helper::DeviceTensor tensor;
    ASSERT_TRUE(!tensor.Allocate(kSize));
    ASSERT_TRUE(!nva2x::FillOnDevice(tensor, 0, nullptr));

    for (std::size_t i = 1; i < tensor.Size(); i += 2) {
      if constexpr (std::is_same_v<Scalar, bool>) {
        const Scalar value = true;
        ASSERT_TRUE(!nva2x::CopyHostToDevice(tensor.View(i, 1), {&value, 1}, nullptr));
      }
      else {
        const Scalar value = static_cast<Scalar>(i);
        ASSERT_TRUE(!nva2x::CopyHostToDevice(tensor.View(i, 1), {&value, 1}, nullptr));
      }
    }

    for (std::size_t i = 0; i < tensor.Size(); ++i) {
      Scalar value = 1;
      ASSERT_TRUE(!nva2x::CopyDeviceToHost({&value, 1}, tensor.View(i, 1), nullptr));
      if constexpr (std::is_same_v<Scalar, bool>) {
        const Scalar expectedValue = i % 2 ? true : 0;
        ASSERT_EQ(expectedValue, value);
      }
      else {
        const Scalar expectedValue = i % 2 ? static_cast<Scalar>(i) : 0;
        ASSERT_EQ(expectedValue, value);
      }
    }
  }
}

TYPED_TEST(Tensor, Allocate) {
  constexpr auto run_test = [](auto&& tensor) {
    static constexpr std::array kSizes{0, 111, 333, 222, -444, 333, 555, 444, 444, 0};
    for (const auto size : kSizes) {
      const bool deallocate = size < 0;
      const std::size_t size_to_alloc = static_cast<std::size_t>(size < 0 ? -size : size);
      if (deallocate) {
        ASSERT_TRUE(!tensor.Deallocate());
      }
      ASSERT_TRUE(!tensor.Allocate(size_to_alloc));
      ASSERT_EQ(size_to_alloc, tensor.Size());
      if (size_to_alloc > 0) {
        ASSERT_NE(nullptr, tensor.Data());
      }
    }
  };
  using Helper = nva2x::TensorHelper<TypeParam>;
  run_test(typename Helper::DeviceTensor());
  run_test(typename Helper::HostTensor());
  run_test(typename Helper::HostPinnedTensor());
}

TYPED_TEST(Tensor, DeviceView) {
  using Helper = nva2x::TensorHelper<TypeParam>;
  using Scalar = typename Helper::Scalar;
  std::array<Scalar, kSize> data;
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<Scalar>(i);
  }

  typename Helper::DeviceTensor sourceDevice;
  typename Helper::HostTensor middleHost;
  typename Helper::DeviceTensor destinationDevice;

  ASSERT_TRUE(!sourceDevice.Init(nva2x::ToConstView(data)));
  ASSERT_TRUE(!middleHost.Allocate(sourceDevice.Size()));
  ASSERT_TRUE(!destinationDevice.Init(nva2x::ToConstView(data)));

  // Usually, the data would not come from a DeviceTensorFloat directly like that.
  const auto sourceView = nva2x::TensorUtility<Scalar>::GetDeviceTensorConstView(sourceDevice.Data(), sourceDevice.Size());
  ASSERT_EQ(sourceDevice.Data(), sourceView.Data());
  ASSERT_EQ(sourceDevice.Size(), sourceView.Size());

  ASSERT_TRUE(!nva2x::CopyDeviceToHost(middleHost, sourceView));
  for (std::size_t i = 0; i < middleHost.Size(); ++i) {
    if constexpr (std::is_same_v<Scalar, bool>) {
      middleHost.Data()[i] = true;
    } else {
      middleHost.Data()[i] += Scalar(123);
    }
  }

  const auto destinationView = nva2x::TensorUtility<Scalar>::GetDeviceTensorView(destinationDevice.Data(), destinationDevice.Size());
  ASSERT_EQ(destinationDevice.Data(), destinationView.Data());
  ASSERT_EQ(destinationDevice.Size(), destinationView.Size());
  std::array<Scalar, kSize> results;
  ASSERT_TRUE(!nva2x::CopyDeviceToHost(nva2x::ToView(results), destinationView));
  for (std::size_t i = 0; i < results.size(); ++i) {
    ASSERT_EQ(static_cast<Scalar>(i), results[i]);
  }

  ASSERT_TRUE(!nva2x::CopyHostToDevice(destinationView, middleHost));

  ASSERT_TRUE(!nva2x::CopyDeviceToHost(nva2x::ToView(results), destinationView));
  for (std::size_t i = 0; i < results.size(); ++i) {
    if constexpr (std::is_same_v<Scalar, bool>) {
      ASSERT_EQ(true, results[i]);
    }
    else {
      ASSERT_EQ(static_cast<Scalar>(i + 123) , results[i]);
    }
  }
}
