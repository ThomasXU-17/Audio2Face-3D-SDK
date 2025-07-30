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
#include "audio2x/internal/animated_tensor.h"
#include "audio2x/internal/cuda_stream.h"
#include "utils.h"

#include <gtest/gtest.h>

#include <algorithm>

TEST(AnimatedDeviceTensor, Simple) {
  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::AnimatedDeviceTensor animatedTensor;

  // Do it multiple times with a reset in between.
  for (std::size_t k = 0; k < 3; ++k) {
    ASSERT_TRUE(!animatedTensor.Allocate(10, 30, 30));
    ASSERT_TRUE(!animatedTensor.Allocate(10, 30, 30));
    ASSERT_TRUE(!animatedTensor.Allocate(10, 30, 15));
    ASSERT_TRUE(!animatedTensor.Allocate(10, 30, 30));
    ASSERT_TRUE(!animatedTensor.Allocate(10, 30, 45));
    ASSERT_TRUE(!animatedTensor.Allocate(10, 30, 30));

    ASSERT_TRUE(!animatedTensor.Deallocate());
    ASSERT_TRUE(!animatedTensor.Allocate(10, 30, 30));

    ASSERT_TRUE(!animatedTensor.Allocate(10, 10, 30));
    ASSERT_TRUE(!animatedTensor.Allocate(10, 30, 15));
    ASSERT_TRUE(!animatedTensor.Allocate(10, 2, 1000));
    ASSERT_TRUE(!animatedTensor.Allocate(10, 30, 30));

    ASSERT_TRUE(animatedTensor.Allocate(0, 30, 30));
    ASSERT_TRUE(animatedTensor.Allocate(30, 0, 30));
    ASSERT_TRUE(!animatedTensor.Allocate(30, 30, 0));
    ASSERT_TRUE(!animatedTensor.Allocate(10, 30, 30));

    nva2x::DeviceTensorFloat key;
    ASSERT_TRUE(!key.Allocate(10));
    for (std::size_t i = 0; i < 1000; ++i) {
      ASSERT_TRUE(!nva2x::FillOnDevice(key, (i + 1) * 10.0f, cudaStream.Data()));
      ASSERT_TRUE(!animatedTensor.AddKey(i, key, cudaStream.Data()));
      EXPECT_EQ(i + 1, animatedTensor.GetKeyCount()) << "index is " << i;
    }

    static constexpr std::size_t keyIndexToRead = 123;
    nva2x::DeviceTensorFloat resultDevice;
    ASSERT_TRUE(!resultDevice.Allocate(key.Size()));
    ASSERT_TRUE(!animatedTensor.GetKeyValue(resultDevice, keyIndexToRead, cudaStream.Data()));

    std::vector<float> resultsHost(resultDevice.Size(), 0.0f);
    ASSERT_TRUE(!nva2x::CopyDeviceToHost(nva2x::ToView(resultsHost), resultDevice, cudaStream.Data()));
    for (std::size_t i = 0; i < resultsHost.size(); ++i) {
      const float expectedValue = (keyIndexToRead + 1) * 10.0f;
      ASSERT_EQ(expectedValue, resultsHost[i]) << "index is " << i;
    }

    ASSERT_TRUE(animatedTensor.AddKey(999, key, cudaStream.Data()));
    EXPECT_EQ(1000U, animatedTensor.GetKeyCount());
    ASSERT_TRUE(!animatedTensor.AddKey(1000, key, cudaStream.Data()));
    EXPECT_EQ(1001U, animatedTensor.GetKeyCount());
    ASSERT_TRUE(!key.Allocate(9));
    ASSERT_TRUE(animatedTensor.AddKey(1001, key, cudaStream.Data()));
    EXPECT_EQ(1001U, animatedTensor.GetKeyCount());

    ASSERT_TRUE(!animatedTensor.DropKeysBefore(300));
    ASSERT_EQ(701U, animatedTensor.GetKeyCount());
    ASSERT_TRUE(!animatedTensor.DropKeysBefore(150));
    ASSERT_EQ(701U, animatedTensor.GetKeyCount());

    ASSERT_EQ(10U, animatedTensor.GetKeySize());
    ASSERT_TRUE(!animatedTensor.Reset());
    ASSERT_EQ(10U, animatedTensor.GetKeySize());

    ASSERT_EQ(0U, animatedTensor.GetKeyCount());
    ASSERT_TRUE(animatedTensor.GetKeyValue(resultDevice, 0, cudaStream.Data()));

    ASSERT_EQ(10U, animatedTensor.GetKeySize());
    ASSERT_TRUE(!animatedTensor.Deallocate());
    ASSERT_EQ(0U, animatedTensor.GetKeySize());

    ASSERT_EQ(0U, animatedTensor.GetKeyCount());
    ASSERT_TRUE(animatedTensor.GetKeyValue(resultDevice, 0, cudaStream.Data()));
  }
}

namespace {

  void AddKey(
    nva2x::AnimatedDeviceTensor& animatedTensor,
    nva2x::HostTensorFloatConstView tensorHost,
    nva2x::AnimatedDeviceTensor::timestamp_t timestamp,
    nva2x::CudaStream& cudaStream
  ) {
    ASSERT_TRUE(!animatedTensor.AddKey(timestamp, tensorHost, cudaStream.Data()));
  }

  void AddKey(
    nva2x::AnimatedDeviceTensor& animatedTensor,
    float value,
    nva2x::AnimatedDeviceTensor::timestamp_t timestamp,
    nva2x::CudaStream& cudaStream
  ) {
    AddKey(animatedTensor, {&value, 1}, timestamp, cudaStream);
  }

  void ValidateKey(
    nva2x::AnimatedDeviceTensor& animatedTensor,
    nva2x::DeviceTensorFloatView tensorDevice,
    nva2x::HostTensorFloatView tensorHost,
    nva2x::HostTensorFloatConstView expectedTensorHost,
    nva2x::AnimatedDeviceTensor::timestamp_t timestamp,
    nva2x::CudaStream& cudaStream
  ) {
    ASSERT_TRUE(!animatedTensor.Sample(tensorDevice, timestamp, cudaStream.Data()));
    ASSERT_TRUE(!nva2x::CopyDeviceToHost(tensorHost, tensorDevice, cudaStream.Data()));
    ASSERT_TRUE(!cudaStream.Synchronize());

    ASSERT_EQ(expectedTensorHost.Size(), tensorHost.Size());
    for (std::size_t i = 0; i < tensorHost.Size(); ++i) {
      ASSERT_EQ(expectedTensorHost.Data()[i], tensorHost.Data()[i]) << "index is " << i;
    }
  }

}

TEST(AnimatedDeviceTensor, FindSurroundingKeys) {
  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::AnimatedDeviceTensor animatedTensor;

  // Test cases.
  using Key = nva2x::AnimatedDeviceTensor::timestamp_t;
  using Curve = std::vector<Key>;
  using Result = std::tuple<nva2x::AnimatedDeviceTensor::timestamp_t, std::size_t, std::size_t>;
  using TestCase = std::pair<Curve, std::vector<Result>>;
  std::vector<TestCase> testCases = {
    // Single key.
    {
      {3},
      {{3,0,0},{-3,0,0},{6,0,0}}
    },
    // 2 keys.
    {
      {2,6},
      {{1,0,0},{2,0,0},{3,0,1},{4,0,1},{5,0,1},{6,1,1},{7,1,1}}
    },
    // 3 keys.
    {
      {2,6,8},
      {{1,0,0},{2,0,0},{3,0,1},{4,0,1},{5,0,1},{6,1,1},{7,1,2},{8,2,2},{9,2,2}}
    },
    // More keys.
    {
      {2,6,8,11,12},
      {{1,0,0},{2,0,0},{3,0,1},{4,0,1},{5,0,1},{6,1,1},{7,1,2},{8,2,2},{9,2,3},{10,2,3},{11,3,3},{12,4,4},{13,4,4}}
    },
    // More keys with non-sequential access
    {
      {2,6,8,11,12},
      {{1,0,0},{2,0,0},{3,0,1},{4,0,1},{4,0,1},{5,0,1},{6,1,1},{3,0,1},{7,1,2},{8,2,2},{9,2,3},{10,2,3},{13,4,4},{11,3,3},{12,4,4},{13,4,4}}
    },
  };

  nva2x::DeviceTensorFloat tensorDevice;
  ASSERT_TRUE(!tensorDevice.Allocate(1));
  for (const auto& testCase : testCases) {
    ASSERT_TRUE(!animatedTensor.Reset());
    ASSERT_TRUE(!animatedTensor.Allocate(1, 10, 0));

    for (const auto& key : testCase.first) {
      const float value = static_cast<float>(key);
      AddKey(animatedTensor, value, key, cudaStream);
    }

    auto accessor = animatedTensor.GetAccessor();
    for (const auto& value : testCase.second) {
      {
        std::size_t indexBefore = std::numeric_limits<std::size_t>::max();
        std::size_t indexAfter = std::numeric_limits<std::size_t>::max();
        ASSERT_TRUE(!animatedTensor.FindSurroundingKeys(indexBefore, indexAfter, std::get<0>(value)));
        ASSERT_EQ(std::get<1>(value), indexBefore);
        ASSERT_EQ(std::get<2>(value), indexAfter);
      }

      {
        std::size_t indexBefore = std::numeric_limits<std::size_t>::max();
        std::size_t indexAfter = std::numeric_limits<std::size_t>::max();
        ASSERT_TRUE(!accessor.FindSurroundingKeys(indexBefore, indexAfter, std::get<0>(value)));
        ASSERT_EQ(std::get<1>(value), indexBefore);
        ASSERT_EQ(std::get<2>(value), indexAfter);
      }
    }
  }
}

TEST(AnimatedDeviceTensor, Sample) {
  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::AnimatedDeviceTensor animatedTensor;

  // Test cases.
  struct Key {
    nva2x::AnimatedDeviceTensor::timestamp_t timestamp;
    std::vector<float> key;
  };
  using Curve = std::vector<Key>;
  using TestCase = std::pair<Curve, Curve>;
  std::vector<TestCase> testCases = {
    // Single key.
    {
      {{3,{1.0f, 2.0f, 3.0f}}},
      {{3,{1.0f, 2.0f, 3.0f}},{-3,{1.0f, 2.0f, 3.0f}},{6,{1.0f, 2.0f, 3.0f}}}
    },
    // 2 keys.
    {
      {{2,{1.0f, 2.0f, 3.0f}},{6,{5.0f, 10.0f, 15.0f}}},
      {{1,{1.0f, 2.0f, 3.0f}},{2,{1.0f, 2.0f, 3.0f}},{3,{2.0f, 4.0f, 6.0f}},
        {4,{3.0f, 6.0f, 9.0f}},{5,{4.0f, 8.0f, 12.0f}},{6,{5.0f, 10.0f, 15.0f}}}
    },
  };

  nva2x::HostTensorFloat tensorHost;
  nva2x::DeviceTensorFloat tensorDevice;
  ASSERT_TRUE(!tensorHost.Allocate(3));
  ASSERT_TRUE(!tensorDevice.Allocate(3));
  for (const auto& testCase : testCases) {
    ASSERT_TRUE(!animatedTensor.Reset());
    ASSERT_TRUE(!animatedTensor.Allocate(3, 10, 0));

    for (const auto& key : testCase.first) {
      AddKey(animatedTensor, {key.key.data(), key.key.size()}, key.timestamp, cudaStream);
    }

    for (const auto& value : testCase.second) {
      ValidateKey(
        animatedTensor,
        tensorDevice,
        tensorHost,
        {value.key.data(), value.key.size()},
        value.timestamp,
        cudaStream
        );
    }
  }
}

TEST(AnimatedDeviceTensor, DropKeysBefore) {
  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::AnimatedDeviceTensor animatedTensor;

  static constexpr std::size_t keysPerBuffer = 10;
  ASSERT_TRUE(!animatedTensor.Allocate(1, keysPerBuffer, 0));

  nva2x::DeviceTensorFloat tensorDevice;
  ASSERT_TRUE(!tensorDevice.Allocate(1));

  static constexpr std::size_t nbKeys = 1000;
  for (std::size_t i = 0; i < nbKeys; ++i) {
    const float value = i + 1.0f;
    AddKey(animatedTensor, value, i, cudaStream);
  }

  for (std::size_t i = 0; i < nbKeys / keysPerBuffer; ++i) {
    const std::size_t expectedNbKeysBefore = nbKeys - i * keysPerBuffer;
    for (std::size_t j = 0; j < keysPerBuffer; ++j) {
      ASSERT_EQ(expectedNbKeysBefore, animatedTensor.GetKeyCount()) << "i " << i << ", j " << j;
       ASSERT_TRUE(!animatedTensor.DropKeysBefore(i * keysPerBuffer + j + 1));
    }
  }
  ASSERT_EQ(keysPerBuffer, animatedTensor.GetKeyCount());
  ASSERT_TRUE(!animatedTensor.DropKeysBefore(nbKeys * 100));
  ASSERT_EQ(keysPerBuffer, animatedTensor.GetKeyCount());
}

TEST(AnimatedDeviceTensor, FindSurroundingKeysRandom) {
  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::AnimatedDeviceTensor animatedTensor;

  static constexpr std::size_t keysPerBuffer = 10;
  ASSERT_TRUE(!animatedTensor.Allocate(1, keysPerBuffer, 0));

  nva2x::DeviceTensorFloat tensorDevice;
  ASSERT_TRUE(!tensorDevice.Allocate(1));

  // Add random cases.
  const int seed = static_cast<unsigned int>(time(NULL));
  std::cout << "Current srand seed: " << seed << std::endl;
  std::srand(seed); // make random inputs reproducible

  constexpr std::size_t kNbRuns = 1000;
  constexpr nva2x::AnimatedDeviceTensor::timestamp_t kMaxTimestamp = 1000;
  constexpr std::size_t kMaxNbKeysPerCurve = 100;
  constexpr std::size_t kNbSamples = 1000;
  constexpr std::size_t kNbSamplesSkip = 50;

  std::vector<nva2x::AnimatedDeviceTensor::timestamp_t> timestamps;
  timestamps.reserve(kMaxNbKeysPerCurve);
  for (std::size_t run = 0; run < kNbRuns; ++run) {
    ASSERT_TRUE(!animatedTensor.Reset());
    ASSERT_TRUE(!animatedTensor.Allocate(1, 10, 0));

    // Generate curves.
    {
      timestamps.clear();
      const auto nbKeys = 1 + GetRandomInteger(kMaxNbKeysPerCurve);
      for (std::int64_t i = 0; i < nbKeys; ++i) {
        timestamps.emplace_back(GetRandomInteger(kMaxTimestamp));
      }
      std::sort(timestamps.begin(), timestamps.end());
      timestamps.erase(std::unique(timestamps.begin(), timestamps.end()), timestamps.end());
    }

    for (const auto timestamp : timestamps) {
      const float value = static_cast<float>(timestamp);
      AddKey(animatedTensor, value, timestamp, cudaStream);
    }

    // Generate time to sample.
    {
      timestamps.clear();
      for (std::size_t i = 0; i < kNbSamples; ++i) {
        timestamps.emplace_back(GetRandomInteger(kMaxTimestamp));
      }
    }

    // Run it once with random accesses and once with sorted accesses.
    for (const auto doSort : {false, true}) {
      if (doSort) {
        std::sort(timestamps.begin(), timestamps.end());
      }

      // Check the keys.
      {
        auto accessor = animatedTensor.GetAccessor();
        for (const auto timestamp : timestamps) {
          // Read the value without cache.
          std::size_t indexBefore = std::numeric_limits<std::size_t>::max();
          std::size_t indexAfter = std::numeric_limits<std::size_t>::max();
          ASSERT_TRUE(!animatedTensor.FindSurroundingKeys(indexBefore, indexAfter, timestamp));

          // Check if the found indices make sense according to the timestamps.
          if (indexBefore != indexAfter) {
            auto timestampBefore = std::numeric_limits<nva2x::AnimatedDeviceTensor::timestamp_t>::min();
            auto timestampAfter = std::numeric_limits<nva2x::AnimatedDeviceTensor::timestamp_t>::min();
            ASSERT_TRUE(!animatedTensor.GetKeyTimestamp(timestampBefore, indexBefore));
            ASSERT_TRUE(!animatedTensor.GetKeyTimestamp(timestampAfter, indexAfter));
            ASSERT_LT(timestampBefore, timestamp);
            ASSERT_LT(timestamp, timestampAfter);
          }
          else {
            if (animatedTensor.GetKeyCount() == 1) {
              // A single key, all timestamp would have returned it.
              ASSERT_EQ(0U, indexBefore);
            }
            else {
              auto timestampSame = std::numeric_limits<nva2x::AnimatedDeviceTensor::timestamp_t>::min();
              ASSERT_TRUE(!animatedTensor.GetKeyTimestamp(timestampSame, indexBefore));
              if (indexBefore == 0) {
                ASSERT_LE(timestamp, timestampSame);
              }
              else if (indexBefore == animatedTensor.GetKeyCount() - 1) {
                ASSERT_LE(timestampSame, timestamp);
              }
              else {
                ASSERT_EQ(timestamp, timestampSame);
              }
            }
          }

          // Check if cached timestamps give the same results.
          {
            std::size_t indexBeforeCached = std::numeric_limits<std::size_t>::max();
            std::size_t indexAfterCached = std::numeric_limits<std::size_t>::max();
            ASSERT_TRUE(!accessor.FindSurroundingKeys(indexBeforeCached, indexAfterCached, timestamp));
            ASSERT_EQ(indexBefore, indexBeforeCached);
            ASSERT_EQ(indexAfter, indexAfterCached);
          }
        }
      }

      // Check the sampling function.
      {
        auto timestampFirst = std::numeric_limits<nva2x::AnimatedDeviceTensor::timestamp_t>::min();
        auto timestampLast = std::numeric_limits<nva2x::AnimatedDeviceTensor::timestamp_t>::min();
        ASSERT_TRUE(!animatedTensor.GetKeyTimestamp(timestampFirst, 0));
        ASSERT_TRUE(!animatedTensor.GetKeyTimestamp(timestampLast, animatedTensor.GetKeyCount() - 1));

        auto accessor = animatedTensor.GetAccessor();
        for (std::size_t i = 0; i < timestamps.size(); i += kNbSamplesSkip) {
          const auto timestamp = timestamps[i];

          ASSERT_TRUE(!nva2x::FillOnDevice(tensorDevice, -1.f, cudaStream.Data()));
          ASSERT_TRUE(!animatedTensor.Sample(tensorDevice, timestamp, cudaStream.Data()));
          float value = -1;
          ASSERT_TRUE(!nva2x::CopyDeviceToHost({&value, 1}, tensorDevice, cudaStream.Data()));
          ASSERT_TRUE(!cudaStream.Synchronize());

          if (timestamp < timestampFirst) {
            ASSERT_EQ(static_cast<float>(timestampFirst), value);
          }
          else if (timestampLast < timestamp) {
            ASSERT_EQ(static_cast<float>(timestampLast), value);
          }
          else {
            ASSERT_NEAR(static_cast<float>(timestamp), value, 1e-4f);
          }

          ASSERT_TRUE(!nva2x::FillOnDevice(tensorDevice, -1.f, cudaStream.Data()));
          ASSERT_TRUE(!accessor.Sample(tensorDevice, timestamp, cudaStream.Data()));
          float valueCached = -1;
          ASSERT_TRUE(!nva2x::CopyDeviceToHost({&valueCached, 1}, tensorDevice, cudaStream.Data()));
          ASSERT_TRUE(!cudaStream.Synchronize());

          ASSERT_EQ(value, valueCached);
        }
      }
    }
  }
}
