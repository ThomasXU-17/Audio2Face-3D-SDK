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
#include "audio2x/internal/emotion_accumulator.h"
#include "audio2x/internal/cuda_stream.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

// EmotionAccumulator is a thin wrapper over AnimatedDeviceTensor,
// so testing will focus on what EmotionAccumulator adds.

TEST(EmotionAccumulator, Simple) {
  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::DeviceTensorFloat tensor;
  ASSERT_TRUE(!tensor.Allocate(1));

  nva2x::EmotionAccumulator accumulator;

  auto add_key = [&](nva2x::EmotionAccumulator::timestamp_t timestamp) {
    ASSERT_TRUE(!nva2x::FillOnDevice(tensor, static_cast<float>(timestamp), cudaStream.Data()));
    ASSERT_TRUE(!accumulator.Accumulate(timestamp, tensor, cudaStream.Data()));
  };
  auto read_key = [&](nva2x::EmotionAccumulator::timestamp_t timestamp, bool shouldFail = false) {
    ASSERT_TRUE(!nva2x::FillOnDevice(tensor, -1.f, cudaStream.Data()));
    const auto errorCode = accumulator.Read(tensor, timestamp, cudaStream.Data());
    ASSERT_TRUE(shouldFail ^ !errorCode);
  };

  // Do it multiple times with a reset in between.
  for (std::size_t k = 0; k < 3; ++k) {
    ASSERT_TRUE(!accumulator.Allocate(1, 2, 0));

    // Should fail when empty.
    ASSERT_TRUE(accumulator.IsEmpty());
    read_key(1, true);

    // Should not be closable when empty.
    ASSERT_TRUE(accumulator.IsEmpty());
    ASSERT_TRUE(!accumulator.IsClosed());
    ASSERT_TRUE(accumulator.Close());
    ASSERT_TRUE(!accumulator.IsClosed());

    // Run #1
    ASSERT_TRUE(accumulator.IsEmpty());
    add_key(1);
    ASSERT_TRUE(!accumulator.IsEmpty());
    read_key(0);
    read_key(1);
    read_key(0);
    read_key(2, true);
    ASSERT_TRUE(!accumulator.Close());
    read_key(2);

    ASSERT_TRUE(!accumulator.IsEmpty());
    ASSERT_TRUE(!accumulator.Reset());
    ASSERT_TRUE(accumulator.IsEmpty());

    // Run #2
    add_key(1);
    add_key(4);
    read_key(0);
    read_key(1);
    read_key(2);
    read_key(3);
    read_key(4);
    read_key(5, true);

    add_key(5);
    read_key(5);

    add_key(7);
    add_key(9);
    read_key(6);
    read_key(7);
    read_key(8);
    read_key(9);
    read_key(10, true);

    // Drop keys.
    ASSERT_EQ(0U, accumulator.NbDroppedEmotions());
    ASSERT_TRUE(!accumulator.DropEmotionsBefore(-1));
    ASSERT_EQ(0U, accumulator.NbDroppedEmotions());
    ASSERT_TRUE(!accumulator.DropEmotionsBefore(0));
    ASSERT_EQ(0U, accumulator.NbDroppedEmotions());
    ASSERT_TRUE(!accumulator.DropEmotionsBefore(1));
    ASSERT_EQ(0U, accumulator.NbDroppedEmotions());
    ASSERT_TRUE(!accumulator.DropEmotionsBefore(3));
    ASSERT_EQ(0U, accumulator.NbDroppedEmotions());
    ASSERT_TRUE(!accumulator.DropEmotionsBefore(4));
    ASSERT_EQ(0U, accumulator.NbDroppedEmotions());
    ASSERT_TRUE(!accumulator.DropEmotionsBefore(5));
    ASSERT_EQ(2U, accumulator.NbDroppedEmotions());

    ASSERT_TRUE(!accumulator.Reset());

    // Run #3
    for (nva2x::EmotionAccumulator::timestamp_t i = -10; i < 10; ++i) {
      add_key(i);
      read_key(i);
      ASSERT_TRUE(!accumulator.DropEmotionsBefore(i));
      read_key(i-1, true);
    }
    ASSERT_TRUE(!accumulator.Close());
    read_key(10);
    read_key(11);
    read_key(12);
  }
}

TEST(EmotionAccumulator, Thread) {
  std::mutex log_mutex;
#define A2F_TEST_LOG(x) do { std::lock_guard lock(log_mutex); std::cout << x << std::endl; } while(false)

  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::DeviceTensorFloat tensorWrite;
  ASSERT_TRUE(!tensorWrite.Allocate(1));
  nva2x::DeviceTensorFloat tensorRead;
  ASSERT_TRUE(!tensorRead.Allocate(1));

  nva2x::EmotionAccumulator accumulator;
  ASSERT_TRUE(!accumulator.Allocate(1, 2, 0));

  auto add_key = [&](nva2x::EmotionAccumulator::timestamp_t timestamp) {
    ASSERT_TRUE(!nva2x::FillOnDevice(tensorWrite, static_cast<float>(timestamp), cudaStream.Data()));
    ASSERT_TRUE(!accumulator.Accumulate(timestamp, tensorWrite, cudaStream.Data()));
  };
  auto read_key = [&](nva2x::EmotionAccumulator::timestamp_t timestamp, bool shouldFail = false) {
    ASSERT_TRUE(!nva2x::FillOnDevice(tensorRead, -1.f, cudaStream.Data()));
    const auto errorCode = accumulator.Read(tensorRead, timestamp, cudaStream.Data());
    ASSERT_TRUE(shouldFail ^ !errorCode);
  };

  std::atomic<bool> done = false;

  std::thread producer([&]() {
    constexpr std::size_t nbRuns = 10;
    constexpr std::size_t nbKeysByRun = 10;
    for (std::size_t i = 0; i < nbRuns; ++i) {
      for (std::size_t j = 0; j < nbKeysByRun; ++j) {
        const auto timestamp = i * nbKeysByRun + j;
        A2F_TEST_LOG("Producer : Adding key " << j << " in run " << i << " : " << timestamp);
        add_key(timestamp);
      }

      A2F_TEST_LOG("Producer : Sleeping");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ASSERT_TRUE(!accumulator.Close());
    done = true;
  });

  nva2x::EmotionAccumulator::timestamp_t timestamp = -10;
  std::thread consumer([&]() {
    while (true) {
      // Must read done first.
      const bool isDone = done;
      const auto availableTimestamp = accumulator.LastAccumulatedTimestamp();
      if (timestamp <= availableTimestamp) {
        A2F_TEST_LOG("Consumer : Read key #" << timestamp);
        read_key(timestamp);
        ++timestamp;
      }
      else if (isDone) {
        // Read a couple more.
        for (std::size_t i = 0; i < 10; ++i) {
          read_key(timestamp);
          ++timestamp;
        }
        break;
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(110U, timestamp);

#undef A2F_TEST_LOG
}
