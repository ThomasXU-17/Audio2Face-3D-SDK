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
#include "audio2x/internal/audio_accumulator.h"
#include "audio2x/internal/cuda_stream.h"
#include "utils.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>

TEST(AudioAccumulator, Simple) {
  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::AudioAccumulator accumulator;

  // Do it multiple times with a reset in between.
  for (std::size_t k = 0; k < 3; ++k) {
    ASSERT_TRUE(!accumulator.Allocate(16000, 30));
    ASSERT_TRUE(!accumulator.Allocate(16000, 30));
    ASSERT_TRUE(!accumulator.Allocate(16000, 15));
    ASSERT_TRUE(!accumulator.Allocate(16000, 30));
    ASSERT_TRUE(!accumulator.Allocate(16000, 45));
    ASSERT_TRUE(!accumulator.Allocate(16000, 30));

    ASSERT_TRUE(!accumulator.Deallocate());
    ASSERT_TRUE(!accumulator.Allocate(16000, 30));

    ASSERT_TRUE(!accumulator.Allocate(1000, 30));
    ASSERT_TRUE(!accumulator.Allocate(16000, 15));
    ASSERT_TRUE(!accumulator.Allocate(2, 1000));
    ASSERT_TRUE(!accumulator.Allocate(16000, 30));

    const std::vector<float> samples(1000, 1.0f);
    for (std::size_t i = 0; i < 16; ++i) {
      ASSERT_TRUE(!accumulator.Accumulate(nva2x::ToConstView(samples), cudaStream.Data()));
      EXPECT_EQ((i+1)*1000, accumulator.NbAccumulatedSamples()) << "index is " << i;
    }

    nva2x::DeviceTensorFloat resultDevice;
    ASSERT_TRUE(!resultDevice.Allocate(16000));
    for (std::size_t i = 0; i < 16; ++i) {
      ASSERT_TRUE(
        !accumulator.Read(
          resultDevice.View(samples.size() * i, samples.size()), 0, static_cast<float>(i), cudaStream.Data()
        )
      );
    }

    std::vector<float> resultsHost(resultDevice.Size(), 0.0f);
    ASSERT_TRUE(!nva2x::CopyDeviceToHost(nva2x::ToView(resultsHost), resultDevice, cudaStream.Data()));
    for (std::size_t i = 0; i < resultsHost.size(); ++i) {
      const float expectedValue = static_cast<float>(i / 1000);
      ASSERT_EQ(expectedValue, resultsHost[i]) << "index is " << i;
    }

    ASSERT_TRUE(!accumulator.DropSamplesBefore(16000));
    ASSERT_EQ(16000U, accumulator.NbDroppedSamples());
    ASSERT_TRUE(accumulator.Read(resultDevice.View(0, 1), 15999, 1.0f, cudaStream.Data()));

    ASSERT_TRUE(!accumulator.Reset());
  }
}

namespace {

  void Write(
      nva2x::AudioAccumulator& accumulator,
      std::size_t beginValue,
      std::size_t valueCount,
      nva2x::CudaStream& cudaStream
  ) {
    const auto nextSample = accumulator.NbAccumulatedSamples();
    std::vector<float> buffer(valueCount, 0.0f);
    for (std::size_t i = 0; i < valueCount; ++i) {
      buffer[i] = static_cast<float>(i + beginValue);
    }
    ASSERT_TRUE(!accumulator.Accumulate(nva2x::ToConstView(buffer), cudaStream.Data()));
    ASSERT_TRUE(!cudaStream.Synchronize());
    ASSERT_EQ(accumulator.NbAccumulatedSamples(), nextSample + buffer.size());
  }

  void Validate(
    nva2x::DeviceTensorFloatConstView buffer,
    std::size_t prependZerosCount,
    std::size_t integerBegin,
    std::size_t integerEnd,
    std::size_t postpendZerosCount
    ) {
    const auto integerCount = integerEnd - integerBegin;
    ASSERT_EQ(prependZerosCount + integerCount + postpendZerosCount, buffer.Size());
    std::vector<float> hostBuffer(buffer.Size(), -1.0f);
    ASSERT_TRUE(!nva2x::CopyDeviceToHost(nva2x::ToView(hostBuffer), buffer));
    for (std::size_t ii = 0; ii < prependZerosCount; ++ii) {
      const auto i = ii;
      ASSERT_EQ(0.0f, hostBuffer[i]) << "Index " << i;
    }
    for (std::size_t ii = 0; ii < integerCount; ++ii) {
      const auto i = ii + prependZerosCount;
      ASSERT_EQ(static_cast<float>(ii + 1 + integerBegin), hostBuffer[i]) << "Index " << i;
    }
    for (std::size_t ii = 0; ii < postpendZerosCount; ++ii) {
      const auto i = ii + prependZerosCount + integerCount;
      ASSERT_EQ(0.0f, hostBuffer[i]) << "Index " << i;
    }
  }

}

TEST(AudioAccumulator, Write) {
  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::AudioAccumulator accumulator;

  // Helpers.
  auto write = [&cudaStream, &accumulator](
    std::size_t beginValue, std::size_t valueCount
  ) {
    Write(accumulator, beginValue, valueCount, cudaStream);
  };
  auto validate = [&accumulator, &cudaStream](
    std::size_t integerBegin, std::size_t integerEnd
  ) {
    ASSERT_EQ(integerEnd, accumulator.NbAccumulatedSamples());
    if (integerBegin == integerEnd) {
      // Nothing else to validate, everything has been dropped.
      return;
    }
    nva2x::DeviceTensorFloat buffer;
    ASSERT_TRUE(!buffer.Allocate(accumulator.NbAccumulatedSamples() - integerBegin));
    ASSERT_TRUE(!accumulator.Read(buffer, integerBegin, 1.0f, cudaStream.Data()));
    Validate(buffer, 0, integerBegin, integerEnd, 0);
  };

  // Test cases.
  constexpr std::int64_t kBufferSize = 100;
  std::vector<std::vector<std::int64_t>> baseTestCases = {
    // Start at the beginning.
    // Start at the beginning of the buffer, but end in the buffer.
    {kBufferSize / 2},
    // Start at the beginning of the buffer, but end one before the end of the buffer.
    {kBufferSize - 1},
    // Start at the beginning of the buffer, but end at end of the buffer.
    {kBufferSize},
    // Start at the beginning of the buffer, but end one after the end of the buffer.
    {kBufferSize + 1},
    // Start at the beginning of the buffer, but end one before the end of the second buffer.
    {2*kBufferSize - 1},
    // Start at the beginning of the buffer, but end at the end of the second buffer.
    {2*kBufferSize },
    // Start at the beginning of the buffer, but end one after the end of the second buffer.
    {2*kBufferSize + 1},
    // Start at the beginning of the buffer, but end one before the end of the tenth buffer.
    {10*kBufferSize - 1},
    // Start at the beginning of the buffer, but end at the end of the tenth buffer.
    {10*kBufferSize },
    // Start at the beginning of the buffer, but end one after the end of the tenth buffer.
    {10*kBufferSize + 1},
  };

  // Add specific cases.
  auto testCases = baseTestCases;
  for (const auto& baseTestCase : baseTestCases) {
    // Do the same general idea, but add a first write at a different offset.
    for (auto offsets : baseTestCases) {
      assert(offsets.size() == 1);
      const auto offset = offsets[0];
      if (offset >= baseTestCase.back()) {
        continue;
      }

      // Add the case.
      auto testCase = baseTestCase;
      testCase.back() -= offset;
      testCase.insert(testCase.begin(), offset);
      testCases.emplace_back(testCase);

      // Add a case with a pack between the two.
      testCase.insert(testCase.begin() + 1, -1);
      testCases.emplace_back(testCase);
    }
  }

  // Add random cases.
  const int seed = static_cast<unsigned int>(time(NULL));
  std::cout << "Current srand seed: " << seed << std::endl;
  std::srand(seed); // make random inputs reproducible

  constexpr std::size_t kRandomTestCasesCount = 1000;
  constexpr std::size_t kWritePerTest = 20;
  for (std::size_t i = 0; i < kRandomTestCasesCount; ++i) {
    std::vector<std::int64_t> testCase;
    for (std::size_t j = 0; j < kWritePerTest; ++j) {
      // Number between 0 and 5
      const auto value = GetRandomInteger(6);
      // Write -1 (drop samples) if 0, otherwise write some data.
      testCase.emplace_back(value ? value * kBufferSize / 5 : -1);
    }
    testCases.emplace_back(std::move(testCase));
  }

  // Run the tests.
  std::cout << "Running " << testCases.size() << " tests\n";
  for (const auto& testCase : testCases) {
    ASSERT_TRUE(!accumulator.Allocate(kBufferSize, 0));
    std::size_t written = 0;
    std::size_t dropped = 0;
    for (const auto sizeToWrite : testCase) {
      if (sizeToWrite > 0) {
        const auto size = static_cast<std::size_t>(sizeToWrite);
        write(written + 1, size);
        written += size;
      }
      else {
        const auto toDrop = static_cast<std::size_t>((written / kBufferSize) * kBufferSize);
        ASSERT_TRUE(!accumulator.DropSamplesBefore(toDrop));
        dropped = toDrop;
      }
    }
    validate(dropped, written);
  }
}

TEST(AudioAccumulator, Read) {
  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::AudioAccumulator accumulator;

  // Test cases.
  struct TestCase {
    std::size_t amountToWrite;
    std::size_t amountToDrop;
    std::size_t sizeToRead;
    std::int64_t startSampleToRead;
    std::size_t prependZerosCount;
    std::size_t integerBegin;
    std::size_t integerEnd;
    std::size_t postpendZerosCount;
  };

  // Test cases.
  constexpr std::int64_t kBufferSize = 10;
  std::vector<TestCase> testCases {
    //
    // Reads in a single buffer.
    //
    // Start before the buffer, end before the buffer.
    {kBufferSize, 0, 10, -20, 10, 0, 0, 0},
    // Start before the buffer, end one before the start the buffer.
    {kBufferSize, 0, 19, -20, 19, 0, 0, 0},
    // Start before the buffer, end on the start of the buffer.
    {kBufferSize, 0, 20, -20, 20, 0, 0, 0},
    // Start before the buffer, end one after start of the buffer.
    {kBufferSize, 0, 21, -20, 20, 0, 1, 0},
    // More cases.
    {kBufferSize, 0, 25, -20, 20, 0, 5, 0},
    {kBufferSize, 0, 29, -20, 20, 0, 9, 0},
    {kBufferSize, 0, 30, -20, 20, 0, 10, 0},
    {kBufferSize, 0, 31, -20, 20, 0, 10, 1},
    {kBufferSize, 0, 35, -20, 20, 0, 10, 5},
    {kBufferSize, 0, 55, -20, 20, 0, 10, 25},
    // One before the start of the buffer.
    {kBufferSize, 0, 1, -1, 1, 0, 0, 0},
    {kBufferSize, 0, 2, -1, 1, 0, 1, 0},
    {kBufferSize, 0, 6, -1, 1, 0, 5, 0},
    {kBufferSize, 0, 10, -1, 1, 0, 9, 0},
    {kBufferSize, 0, 11, -1, 1, 0, 10, 0},
    {kBufferSize, 0, 12, -1, 1, 0, 10, 1},
    {kBufferSize, 0, 16, -1, 1, 0, 10, 5},
    {kBufferSize, 0, 36, -1, 1, 0, 10, 25},
    // On the start of the buffer.
    {kBufferSize, 0, 1, 0, 0, 0, 1, 0},
    {kBufferSize, 0, 5, 0, 0, 0, 5, 0},
    {kBufferSize, 0, 9, 0, 0, 0, 9, 0},
    {kBufferSize, 0, 10, 0, 0, 0, 10, 0},
    {kBufferSize, 0, 11, 0, 0, 0, 10, 1},
    {kBufferSize, 0, 15, 0, 0, 0, 10, 5},
    {kBufferSize, 0, 35, 0, 0, 0, 10, 25},
    // One after the start of the buffer.
    {kBufferSize, 0, 4, 1, 0, 1, 5, 0},
    {kBufferSize, 0, 8, 1, 0, 1, 9, 0},
    {kBufferSize, 0, 9, 1, 0, 1, 10, 0},
    {kBufferSize, 0, 10, 1, 0, 1, 10, 1},
    {kBufferSize, 0, 14, 1, 0, 1, 10, 5},
    {kBufferSize, 0, 34, 1, 0, 1, 10, 25},
    // Middle of the buffer.
    {kBufferSize, 0, 4, 5, 0, 5, 9, 0},
    {kBufferSize, 0, 5, 5, 0, 5, 10, 0},
    {kBufferSize, 0, 6, 5, 0, 5, 10, 1},
    {kBufferSize, 0, 10, 5, 0, 5, 10, 5},
    {kBufferSize, 0, 30, 5, 0, 5, 10, 25},
    // One before the end of the buffer.
    {kBufferSize, 0, 1, 9, 0, 9, 10, 0},
    {kBufferSize, 0, 2, 9, 0, 9, 10, 1},
    {kBufferSize, 0, 6, 9, 0, 9, 10, 5},
    {kBufferSize, 0, 26, 9, 0, 9, 10, 25},
    // End of the buffer.
    {kBufferSize, 0, 1, 10, 0, 10, 10, 1},
    {kBufferSize, 0, 5, 10, 0, 10, 10, 5},
    {kBufferSize, 0, 25, 10, 0, 10, 10, 25},
    // After of the buffer.
    {kBufferSize, 0, 4, 11, 0, 10, 10, 4},
    {kBufferSize, 0, 24, 11, 0, 10, 10, 24},
    {kBufferSize, 0, 14, 21, 0, 10, 10, 14},

    //
    // Reads in multiple buffers.
    //
    {3*kBufferSize, 0, 5, 2, 0, 2, 7, 0},
    {3*kBufferSize, 0, 4, 5, 0, 5, 9, 0},
    {3*kBufferSize, 0, 5, 5, 0, 5, 10, 0},
    {3*kBufferSize, 0, 6, 5, 0, 5, 11, 0},
    {3*kBufferSize, 0, 15, 5, 0, 5, 20, 0},
    {3*kBufferSize, 0, 25, 5, 0, 5, 30, 0},
    {3*kBufferSize, 0, 35, 5, 0, 5, 30, 10},
  };

  // Add random cases.
  const int seed = static_cast<unsigned int>(time(NULL));
  std::cout << "Current srand seed: " << seed << std::endl;
  std::srand(seed); // make random inputs reproducible

  constexpr std::size_t kRandomTestCasesCount = 100;
  constexpr std::size_t kReadPerTest = 20;
  for (std::size_t i = 0; i < kRandomTestCasesCount; ++i) {
    const auto amountToWrite = GetRandomInteger(10 * kBufferSize) + 1;
    const auto amountToDrop = GetRandomInteger(amountToWrite / kBufferSize) * kBufferSize;

    for (std::size_t j = 0; j < kReadPerTest; ++j) {
      auto startSampleToRead = GetRandomInteger(amountToWrite) - kBufferSize;
      if (amountToDrop > 0 && startSampleToRead < 0) {
        startSampleToRead = -startSampleToRead;
      }
      startSampleToRead = std::max(startSampleToRead, amountToDrop);
      const auto sizeToRead = GetRandomInteger(2 * kBufferSize) + 1;
      const auto prependZerosCount = startSampleToRead < 0 ? -startSampleToRead : 0;
      const auto integerBegin = std::max<std::int64_t>(startSampleToRead, 0);
      const auto integerEnd = std::min(amountToWrite, integerBegin + sizeToRead);
      const auto postpendZerosCount = std::max<std::int64_t>(0, integerBegin + sizeToRead - amountToWrite);
      testCases.emplace_back(TestCase{
        static_cast<std::size_t>(amountToWrite),
        static_cast<std::size_t>(amountToDrop),
        static_cast<std::size_t>(sizeToRead),
        startSampleToRead,
        static_cast<std::size_t>(prependZerosCount),
        static_cast<std::size_t>(integerBegin),
        static_cast<std::size_t>(integerEnd),
        static_cast<std::size_t>(postpendZerosCount)
        });
    }
  }

  // Run the tests.
  std::cout << "Running " << testCases.size() << " tests\n";
  nva2x::DeviceTensorFloat buffer;
  for (const auto& testCase : testCases) {
    ASSERT_TRUE(!accumulator.Allocate(kBufferSize, 0));

    Write(accumulator, 1, testCase.amountToWrite, cudaStream);
    ASSERT_TRUE(!accumulator.DropSamplesBefore(testCase.amountToDrop));

    // Re-use the buffer to avoid many allocations.
    if (buffer.Size() < testCase.sizeToRead) {
      ASSERT_TRUE(!buffer.Allocate(testCase.sizeToRead));
    }
    const auto bufferView = buffer.View(0 , testCase.sizeToRead);
    ASSERT_TRUE(!nva2x::FillOnDevice(bufferView, -1.0f, cudaStream.Data()));

    // Try to read when not closed.
    const bool read_beyond_end =
     testCase.startSampleToRead + static_cast<std::int64_t>(testCase.sizeToRead)
     >
     static_cast<std::int64_t>(accumulator.NbAccumulatedSamples());
    if (read_beyond_end) {
      // Check that reading beyond the end returns an error.
      ASSERT_TRUE(accumulator.Read(bufferView, testCase.startSampleToRead, 1.0f, cudaStream.Data()));

      if (testCase.startSampleToRead < static_cast<std::int64_t>(accumulator.NbAccumulatedSamples())) {
        // Check that reading until the end does not return an error.
        const auto sizeToRead = std::min(testCase.sizeToRead, accumulator.NbAccumulatedSamples() - testCase.startSampleToRead);
        const auto insideView = bufferView.View(0, sizeToRead);
        ASSERT_TRUE(insideView.Size() == 0 || !accumulator.Read(insideView, testCase.startSampleToRead, 1.0f, cudaStream.Data()));
        Validate(
          insideView,
          testCase.prependZerosCount,
          testCase.integerBegin,
          testCase.integerEnd,
          0
          );
      }
    }
    else {
      ASSERT_TRUE(!accumulator.Read(bufferView, testCase.startSampleToRead, 1.0f, cudaStream.Data()));
      Validate(
        bufferView,
        testCase.prependZerosCount,
        testCase.integerBegin,
        testCase.integerEnd,
        testCase.postpendZerosCount
        );
    }

    // Try to read when closed.
    ASSERT_TRUE(!accumulator.Close());
    ASSERT_TRUE(!accumulator.Read(bufferView, testCase.startSampleToRead, 1.0f, cudaStream.Data()));
    Validate(
      bufferView,
      testCase.prependZerosCount,
      testCase.integerBegin,
      testCase.integerEnd,
      testCase.postpendZerosCount
      );
  }
}


TEST(WindowProgress, Simple) {
    nva2x::WindowProgress progress({3, 0, 1, 2, 1});

    std::size_t nbAvailableSamples = 10;
    bool isClosed = false;
    ASSERT_EQ(4U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(0, 1, 3));
    progress.IncrementReadWindowCount();
    ASSERT_EQ(3U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(2, 3, 5));
    progress.IncrementReadWindowCount();
    ASSERT_EQ(2U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(4, 5, 7));
    progress.IncrementReadWindowCount();
    ASSERT_EQ(1U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(6, 7, 9));
    progress.IncrementReadWindowCount();
    ASSERT_EQ(0U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(8, 9, 11));
    progress.IncrementReadWindowCount();
    ASSERT_EQ(0U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(10, 11, 13));

    progress.ResetReadWindowCount();
    isClosed = true;
    ASSERT_EQ(5U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(0, 1, 3));
    progress.IncrementReadWindowCount();
    ASSERT_EQ(4U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(2, 3, 5));
    progress.IncrementReadWindowCount();
    ASSERT_EQ(3U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(4, 5, 7));
    progress.IncrementReadWindowCount();
    ASSERT_EQ(2U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(6, 7, 9));
    progress.IncrementReadWindowCount();
    ASSERT_EQ(1U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(8, 9, 11));
    progress.IncrementReadWindowCount();
    ASSERT_EQ(0U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(10, 11, 13));
    progress.IncrementReadWindowCount();
    ASSERT_EQ(0U, progress.GetNbAvailableWindows(nbAvailableSamples, isClosed));
    ASSERT_THAT(progress.GetCurrentWindow(), ::testing::ElementsAre(12, 13, 15));
}

TEST(WindowProgress, SpecificCases) {
  struct TestCase {
    nva2x::WindowProgressParameters params;
    std::size_t nbAvailableSamples;
    bool isClosed;
    std::vector<std::array<std::int64_t, 3>> expectedWindows;
  };

  std::vector<TestCase> testCases{
    // Same cases as the simple case above.
    {{3,0,1,2,1}, 10, false, {{{0,1,3}}, {{2,3,5}}, {{4,5,7}}, {{6,7,9}}}},
    {{3,0,1,2,1}, 10, true, {{{0,1,3}}, {{2,3,5}}, {{4,5,7}}, {{6,7,9}}, {{8,9,11}}}},
    // Simple cases.
    {{3,0,1,2,1}, 11, false, {{{0,1,3}}, {{2,3,5}}, {{4,5,7}}, {{6,7,9}}, {{8,9,11}}}},
    {{3,0,1,2,1}, 12, false, {{{0,1,3}}, {{2,3,5}}, {{4,5,7}}, {{6,7,9}}, {{8,9,11}}}},
    {{3,0,1,2,1}, 13, false, {{{0,1,3}}, {{2,3,5}}, {{4,5,7}}, {{6,7,9}}, {{8,9,11}}, {{10,11,13}}}},
    {{3,0,1,2,1}, 14, false, {{{0,1,3}}, {{2,3,5}}, {{4,5,7}}, {{6,7,9}}, {{8,9,11}}, {{10,11,13}}}},
    {{3,0,1,2,1}, 15, false, {{{0,1,3}}, {{2,3,5}}, {{4,5,7}}, {{6,7,9}}, {{8,9,11}}, {{10,11,13}}, {{12,13,15}}}},
    {{3,0,1,2,1}, 11, true, {{{0,1,3}}, {{2,3,5}}, {{4,5,7}}, {{6,7,9}}, {{8,9,11}}}},
    {{3,0,1,2,1}, 12, true, {{{0,1,3}}, {{2,3,5}}, {{4,5,7}}, {{6,7,9}}, {{8,9,11}}, {{10,11,13}}}},
    {{3,0,1,2,1}, 13, true, {{{0,1,3}}, {{2,3,5}}, {{4,5,7}}, {{6,7,9}}, {{8,9,11}}, {{10,11,13}}}},
    {{3,0,1,2,1}, 14, true, {{{0,1,3}}, {{2,3,5}}, {{4,5,7}}, {{6,7,9}}, {{8,9,11}}, {{10,11,13}}, {{12,13,15}}}},
    {{3,0,1,2,1}, 15, true, {{{0,1,3}}, {{2,3,5}}, {{4,5,7}}, {{6,7,9}}, {{8,9,11}}, {{10,11,13}}, {{12,13,15}}}},
    // Start before the beginning, like diffusion.  Offset should be 16000*7/30 = 3733,
    // but we'll put 1 to keep things simple.
    {{16000,-16000,1,8000,1}, 16000, false, {{{-16000,-15999,0}}, {{-8000,-7999,8000}}, {{0,1,16000}}}},
    {{2,0,1,2,1}, 4, false, {{{0,1,2}}, {{2,3,4}}}},
    {{2,0,1,2,1}, 5, false, {{{0,1,2}}, {{2,3,4}}}},
    {{2,0,1,2,1}, 6, false, {{{0,1,2}}, {{2,3,4}}, {{4,5,6}}}},
    {{2,0,1,2,1}, 7, false, {{{0,1,2}}, {{2,3,4}}, {{4,5,6}}}},
    {{2,0,1,2,1}, 4, true, {{{0,1,2}}, {{2,3,4}}}},
    {{2,0,1,2,1}, 5, true, {{{0,1,2}}, {{2,3,4}}}},
    {{2,0,1,2,1}, 6, true, {{{0,1,2}}, {{2,3,4}}, {{4,5,6}}}},
    {{2,0,1,2,1}, 7, true, {{{0,1,2}}, {{2,3,4}}, {{4,5,6}}}},
    // Start aligned with the beginning, like regression.  For 30 FPS increments are 16000/30 = 533
    {{8320, -4160, 4160, 16000, 30}, 4160 + 3200, false,
     {{{-4160,0,4160}},
      {{-4160+533,533,4160+533}},
      {{-4160+533+533,533+533,4160+533+533}},
      {{-4160+533+533+534,533+533+534,4160+533+533+534}},
      {{-4160+533+533+534+533,533+533+534+533,4160+533+533+534+533}},
      {{-4160+533+533+534+533+533,533+533+534+533+533,4160+533+533+534+533+533}},
      {{-4160+3200,3200,4160+3200}},
      }},
    {{8,-4,4,10,3}, 10, false, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}}},
    {{8,-4,4,10,3}, 11, false, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}}},
    {{8,-4,4,10,3}, 12, false, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}}},
    {{8,-4,4,10,3}, 13, false, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}}},
    {{8,-4,4,10,3}, 14, false, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}, {{6,10,14}}}},
    {{8,-4,4,10,3}, 15, false, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}, {{6,10,14}}}},
    {{8,-4,4,10,3}, 16, false, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}, {{6,10,14}}}},
    {{8,-4,4,10,3}, 17, false, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}, {{6,10,14}}, {{9,13,17}}}},
    {{8,-4,4,10,3}, 18, false, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}, {{6,10,14}}, {{9,13,17}}}},
    {{8,-4,4,10,3}, 10, true, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}}},
    {{8,-4,4,10,3}, 11, true, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}, {{6,10,14}}}},
    {{8,-4,4,10,3}, 12, true, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}, {{6,10,14}}}},
    {{8,-4,4,10,3}, 13, true, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}, {{6,10,14}}}},
    {{8,-4,4,10,3}, 14, true, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}, {{6,10,14}}, {{9,13,17}}}},
    {{8,-4,4,10,3}, 15, true, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}, {{6,10,14}}, {{9,13,17}}}},
    {{8,-4,4,10,3}, 16, true, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}, {{6,10,14}}, {{9,13,17}}}},
    {{8,-4,4,10,3}, 17, true, {{{-4,0,4}}, {{-1,3,7}}, {{2,6,10}}, {{6,10,14}}, {{9,13,17}}, {{12,16,20}}}},
    // One sample at a time.
    {{1,-2,0,1,1}, 4, false, {{{-2,-2,-1}}, {{-1,-1,0}}, {{0,0,1}}, {{1,1,2}}, {{2,2,3}}, {{3,3,4}}}},
  };

  std::cout << "Running " << testCases.size() << " tests\n";
  for (const auto& testCase : testCases) {
    nva2x::WindowProgress progress(testCase.params);

    for (std::size_t i = 0; i < testCase.expectedWindows.size(); ++i) {
      ASSERT_EQ(
        testCase.expectedWindows.size() - i,
        progress.GetNbAvailableWindows(testCase.nbAvailableSamples, testCase.isClosed)
      ) << "Index " << i;
      ASSERT_EQ(progress.GetCurrentWindow(), testCase.expectedWindows[i]) << "Index " << i;
      progress.IncrementReadWindowCount();
    }
    ASSERT_EQ(
      0U,
      progress.GetNbAvailableWindows(testCase.nbAvailableSamples, testCase.isClosed)
    ) << "After #1";
    progress.IncrementReadWindowCount();
    ASSERT_EQ(
      0U,
      progress.GetNbAvailableWindows(testCase.nbAvailableSamples, testCase.isClosed)
    ) << "After #2";
  }
}

TEST(WindowProgress, RandomCases) {
  struct TestCase {
    nva2x::WindowProgressParameters params;
    std::size_t nbAvailableSamples;
  };

  // Add random cases.
  const int seed = static_cast<unsigned int>(time(NULL));
  std::cout << "Current srand seed: " << seed << std::endl;
  std::srand(seed); // make random inputs reproducible

  std::vector<TestCase> testCases;
  constexpr std::size_t kRandomTestCasesCount = 1000;
  for (std::size_t i = 0; i < kRandomTestCasesCount; ++i) {
    const auto windowSize = GetRandomInteger(1000) + 1;
    const auto startOffset = GetRandomInteger(100) - 50;
    const auto targetOffset = GetRandomInteger(windowSize);
    const auto strideNum = GetRandomInteger(50) + 1;
    const auto strideDenom = GetRandomInteger(50) + 1;
    const auto nbAvailableSamples = GetRandomInteger(2000) + 2000;
    testCases.emplace_back(TestCase{
      {static_cast<std::size_t>(windowSize),
       startOffset,
       targetOffset,
       static_cast<std::size_t>(strideNum),
       static_cast<std::size_t>(strideDenom)},
       static_cast<std::size_t>(nbAvailableSamples)
    });
  }

  // Run the tests.
  std::cout << "Running " << testCases.size() << " tests\n";
  for (const auto& testCase : testCases) {
    for (auto isClosed : {false, true}) {
      nva2x::WindowProgress progress(testCase.params);

      const auto nbWindows = progress.GetNbAvailableWindows(testCase.nbAvailableSamples, isClosed);
      ASSERT_LT(1U, nbWindows);

      if (isClosed) {
        const auto expectedNbWindows =
          ((testCase.nbAvailableSamples - testCase.params.startOffset - testCase.params.targetOffset)
           * testCase.params.strideDenom + testCase.params.strideNum - 1)
          / testCase.params.strideNum;
        ASSERT_EQ(expectedNbWindows, nbWindows);
      }

      for (std::size_t i = 0; i < nbWindows; ++i) {
        ASSERT_EQ(
          nbWindows - i,
          progress.GetNbAvailableWindows(testCase.nbAvailableSamples, isClosed)
        ) << "Index " << i;
        progress.IncrementReadWindowCount();
      }
      ASSERT_EQ(
        0U,
        progress.GetNbAvailableWindows(testCase.nbAvailableSamples, isClosed)
      ) << "After #1";
      progress.IncrementReadWindowCount();
      ASSERT_EQ(
        0U,
        progress.GetNbAvailableWindows(testCase.nbAvailableSamples, isClosed)
      ) << "After #2";
    }
  }
}


TEST(AudioAccumulator, Thread) {
  std::mutex log_mutex;
#define A2X_TEST_LOG(x) do { std::lock_guard lock(log_mutex); std::cout << x << std::endl; } while(false)

  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::AudioAccumulator accumulator;
  ASSERT_TRUE(!accumulator.Allocate(16000, 0));

  constexpr nva2x::WindowProgressParameters progressParameters{16000, 0, 0, 4000, 1};
  nva2x::WindowProgress progress(progressParameters);

  ASSERT_EQ(0U, progress.GetNbAvailableWindows(accumulator.NbAccumulatedSamples(), false));

  std::vector<float> audioData(160000);
  std::iota(audioData.begin(), audioData.end(), 0.0f);
  std::atomic<bool> done = false;

  std::thread producer([&]() {
    constexpr std::size_t nbRuns = 10;
    constexpr std::size_t nbChunksByRun = 10;
    const std::size_t chunkSize = audioData.size() / nbRuns / nbChunksByRun;
    for (std::size_t i = 0; i < nbRuns; ++i) {
      for (std::size_t j = 0; j < nbChunksByRun; ++j) {
        const auto startSample = (i * nbChunksByRun + j) * chunkSize;
        A2X_TEST_LOG("Producer : Adding chunk " << j << " in run " << i);
        ASSERT_TRUE(!accumulator.Accumulate(nva2x::HostTensorFloatConstView{audioData.data() + startSample, chunkSize}, cudaStream.Data()));
      }

      A2X_TEST_LOG("Producer : Sleeping");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ASSERT_TRUE(!accumulator.Close());
    done = true;
  });

  std::size_t chunkIndex = 0;
  std::thread consumer([&]() {
    constexpr std::size_t kBufferCheckIncrement =
#ifdef NDEBUG
        1
#else
        100
#endif
        ;

    std::vector<float> chunk(16000);
    nva2x::DeviceTensorFloat tensor;
    ASSERT_TRUE(!tensor.Allocate(16000));

    while (true) {
      // Must read done first.
      const bool isDone = done;
      const auto nbWindows = progress.GetNbAvailableWindows(accumulator.NbAccumulatedSamples(), accumulator.IsClosed());
      if (nbWindows > 0) {
        A2X_TEST_LOG("Consumer : Read chunk #" << chunkIndex);
        std::int64_t start, target, end;
        progress.GetCurrentWindow(start, target, end);
        ASSERT_TRUE(!accumulator.Read(tensor, start, 1.0f, cudaStream.Data()));
        ASSERT_TRUE(!nva2x::CopyDeviceToHost(nva2x::ToView(chunk), tensor, cudaStream.Data()));
        ASSERT_TRUE(!cudaStream.Synchronize());

        for (std::size_t j = 0; j < chunk.size(); j += kBufferCheckIncrement) {
          const auto index = (chunkIndex * progressParameters.strideNum) / progressParameters.strideDenom + j;
          const auto value = index < audioData.size() ? index : 0;
          ASSERT_EQ(value, chunk[j]);
        }
        ++chunkIndex;
        progress.IncrementReadWindowCount();
      }
      else if (isDone) {
        break;
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(0U, progress.GetNbAvailableWindows(accumulator.NbAccumulatedSamples(), accumulator.IsClosed()));
  ASSERT_EQ(40U, chunkIndex);
#undef A2X_TEST_LOG
}
