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
#include "audio2x/internal/integer_array.h"

#include <gtest/gtest.h>

namespace {

  enum class TestEnum : std::uint32_t {
    kValue1,
    kValue2,
    kValue3,
  };

}

TEST(IntegerArray, CompileTime) {
  // Validation can be done at compile time.
  using IntegerArray = nva2x::IntegerArray<std::uint32_t, std::uint32_t, 2>;
  static_assert(IntegerArray::nb_bits_per_element == 2U);
  static_assert(sizeof(IntegerArray) == sizeof(std::uint32_t));
  static_assert(sizeof(IntegerArray) == sizeof(IntegerArray::storage_type));

  {
    constexpr IntegerArray arr;
    static_assert(arr.Size() == 0U);
    static_assert(arr.Capacity() == 14U);
  }

  {
    constexpr auto arr = IntegerArray();
    static_assert(arr.Size() == 0U);
  }

  {
    constexpr IntegerArray arr = {{1U, 2U, 3U}};
    static_assert(arr.Size() == 3U);
    static_assert(arr.Get(0) == 1U);
    static_assert(arr.Get(1) == 2U);
    static_assert(arr.Get(2) == 3U);
  }

  using EnumArray = nva2x::IntegerArray<TestEnum, std::uint32_t, 2>;
  static_assert(sizeof(EnumArray) == sizeof(std::uint32_t));
  static_assert(sizeof(EnumArray) == sizeof(EnumArray::storage_type));
  {
    constexpr EnumArray arr;
    static_assert(arr.Size() == 0U);
  }

  {
    constexpr EnumArray arr;
    static_assert(arr.Size() == 0U);
  }

  {
    constexpr EnumArray arr = {{TestEnum::kValue1, TestEnum::kValue2}};
    static_assert(arr.Size() == 2U);
    static_assert(arr.Get(0) == TestEnum::kValue1);
    static_assert(arr.Get(1) == TestEnum::kValue2);
  }

  {
    constexpr IntegerArray arr = {{1U, 2U, 3U, 2U, 1U, 0U, 1U, 2U, 3U, 2U, 1U, 0U}};
    static_assert(arr.Size() == 12U);
    static_assert(arr.Get(0) == 1U);
    static_assert(arr.Get(1) == 2U);
    static_assert(arr.Get(2) == 3U);
    static_assert(arr.Get(3) == 2U);
    static_assert(arr.Get(4) == 1U);
    static_assert(arr.Get(5) == 0U);
    static_assert(arr.Get(6) == 1U);
    static_assert(arr.Get(7) == 2U);
    static_assert(arr.Get(8) == 3U);
    static_assert(arr.Get(9) == 2U);
    static_assert(arr.Get(10) == 1U);
    static_assert(arr.Get(11) == 0U);
  }
}

TEST(IntegerArray, RunTime) {
  {
    using IntegerArray = nva2x::IntegerArray<std::uint32_t, std::uint32_t, 2>;
    static_assert(sizeof(IntegerArray) == sizeof(std::uint32_t));
    static_assert(sizeof(IntegerArray) == sizeof(IntegerArray::storage_type));

    IntegerArray arr;
    ASSERT_EQ(arr.Size(), 0U);
    ASSERT_EQ(arr.Capacity(), 14U);
    arr.Add(1U);
    ASSERT_EQ(arr.Size(), 1U);
    ASSERT_EQ(arr.Get(0), 1U);
    arr.Add(2U);
    ASSERT_EQ(arr.Size(), 2U);
    ASSERT_EQ(arr.Get(0), 1U);
    ASSERT_EQ(arr.Get(1), 2U);
    arr.RemoveLast();
    ASSERT_EQ(arr.Size(), 1U);
    ASSERT_EQ(arr.Get(0), 1U);
    arr.Clear();
    ASSERT_EQ(arr.Size(), 0U);
  }

  {
    using IntegerArray = nva2x::IntegerArray<std::uint8_t, std::uint32_t, 3>;
    static_assert(sizeof(IntegerArray) == sizeof(std::uint32_t));
    static_assert(sizeof(IntegerArray) == sizeof(IntegerArray::storage_type));

    IntegerArray arr;
    ASSERT_EQ(arr.Size(), 0U);
    ASSERT_EQ(arr.Capacity(), 9U);
    arr.Add(1U);
    ASSERT_EQ(arr.Size(), 1U);
    ASSERT_EQ(arr.Get(0), 1U);
    arr.Add(0b111U);
    ASSERT_EQ(arr.Size(), 2U);
    ASSERT_EQ(arr.Get(0), 1U);
    ASSERT_EQ(arr.Get(1), 0b111U);
    arr.Set(1, 3U);
    ASSERT_EQ(arr.Get(1), 3U);
    arr.RemoveLast();
    ASSERT_EQ(arr.Size(), 1U);
    ASSERT_EQ(arr.Get(0), 1U);
    arr.Clear();
    ASSERT_EQ(arr.Size(), 0U);
  }

  {
    using IntegerArray = nva2x::IntegerArray<std::uint8_t, std::uint64_t, 5>;
    static_assert(sizeof(IntegerArray) == sizeof(std::uint64_t));
    static_assert(sizeof(IntegerArray) == sizeof(IntegerArray::storage_type));

    IntegerArray arr;
    ASSERT_EQ(arr.Size(), 0U);
    ASSERT_EQ(arr.Capacity(), 12U);
    arr.Add(1U);
    arr.Add(2U);
    arr.Add(3U);
    arr.Add(2U);
    arr.Add(1U);
    arr.Add(0U);
    arr.Add(1U);
    arr.Add(2U);
    arr.Add(3U);
    arr.Add(2U);
    arr.Add(1U);
    arr.Add(0U);
    ASSERT_EQ(arr.Size(), 12U);
    ASSERT_EQ(arr.Get(0), 1U);
    ASSERT_EQ(arr.Get(1), 2U);
    ASSERT_EQ(arr.Get(2), 3U);
    ASSERT_EQ(arr.Get(3), 2U);
    ASSERT_EQ(arr.Get(4), 1U);
    ASSERT_EQ(arr.Get(5), 0U);
    ASSERT_EQ(arr.Get(6), 1U);
    ASSERT_EQ(arr.Get(7), 2U);
    ASSERT_EQ(arr.Get(8), 3U);
    ASSERT_EQ(arr.Get(9), 2U);
    ASSERT_EQ(arr.Get(10), 1U);
    ASSERT_EQ(arr.Get(11), 0U);
    arr.Clear();
    ASSERT_EQ(arr.Size(), 0U);
  }
}
