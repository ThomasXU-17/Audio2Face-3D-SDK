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
#include "audio2x/internal/bit_vector.h"

#include <gtest/gtest.h>

TEST(BitVector, SizeAndResize) {
  nva2x::bit_vector<std::uint32_t> bv;
  EXPECT_EQ(bv.size(), 0U);

  bv.resize(100);
  EXPECT_EQ(bv.size(), 100U);

  bv.resize(50);
  EXPECT_EQ(bv.size(), 50U);
}

TEST(BitVector, BitOperations) {
  nva2x::bit_vector<std::uint32_t> bv;
  bv.resize(100);

  // Test setting and getting individual bits
  bv.set(0, true);
  EXPECT_TRUE(bv.test(0));
  EXPECT_TRUE(bv[0]);

  bv.set(1, false);
  EXPECT_FALSE(bv.test(1));
  EXPECT_FALSE(bv[1]);

  // Test setting bits across block boundaries
  bv.set(31, true);
  bv.set(32, true);
  EXPECT_FALSE(bv.test(30));
  EXPECT_TRUE(bv.test(31));
  EXPECT_TRUE(bv.test(32));
  EXPECT_FALSE(bv.test(33));
}

TEST(BitVector, SetAllAndResetAll) {
  nva2x::bit_vector<std::uint32_t> bv;
  bv.resize(100);

  // Test set_all
  bv.set_all();
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_TRUE(bv.test(i));
  }

  // Test reset_all
  bv.reset_all();
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_FALSE(bv.test(i));
  }
}

TEST(BitVector, BlockOperations) {
  nva2x::bit_vector<std::uint32_t> bv;
  bv.resize(100);

  // Test block size calculation
  EXPECT_EQ(bv.block_size(), (100U + 31U) / 32U); // 32 bits per uint32_t

  // Test block data access
  const uint32_t* block_data = bv.block_data();
  EXPECT_NE(block_data, nullptr);

  // Test block operations
  bv.set(0, true);
  EXPECT_EQ(block_data[0] & 1U, 1U);

  bv.set(31, true);
  EXPECT_EQ(block_data[0] & (1U << 31), 1U << 31);
}

TEST(BitVector, EdgeCases) {
  nva2x::bit_vector<std::uint32_t> bv;

  // Test empty vector
  EXPECT_EQ(bv.size(), 0U);
  EXPECT_EQ(bv.block_size(), 0U);

  // Test resize to 0
  bv.resize(0);
  EXPECT_EQ(bv.size(), 0U);
  EXPECT_EQ(bv.block_size(), 0U);

  // Test resize to exact block boundary
  bv.resize(32);
  EXPECT_EQ(bv.size(), 32U);
  EXPECT_EQ(bv.block_size(), 1U);
}

TEST(BitVector, OutOfBounds) {
  nva2x::bit_vector<std::uint32_t> bv;
  bv.resize(10);

  // These should assert in debug builds
  EXPECT_DEBUG_DEATH(bv.test(10), "");
  EXPECT_DEBUG_DEATH(bv.set(10, true), "");
}
