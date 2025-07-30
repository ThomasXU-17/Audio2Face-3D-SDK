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
#include "audio2x/internal/io.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

TEST(IO, DataReader) {
  const std::vector<char> bytes = {
    // Bytes.
    0x1,
    0x2,
    // Shorts.
    0x3, 0x4,
    0x5, 0x6,
    // Ints.
    0x7, 0x8, 0x9, 0xa,
    0xb, 0xc, 0xd, 0xe,
    // Not enough at the end.
    0xf, 0xf,
  };
  nva2x::DataReader reader(bytes.data(), bytes.size());

  // WARNING: assuming little-endian byte order
  std::int8_t data_byte;
  ASSERT_TRUE(!reader.Read(&data_byte, sizeof(data_byte)));
  ASSERT_EQ(data_byte, 0x01);
  ASSERT_TRUE(!reader.Read(&data_byte, sizeof(data_byte)));
  ASSERT_EQ(data_byte, 0x02);

  std::int16_t data_short;
  ASSERT_TRUE(!reader.Read(&data_short, sizeof(data_short)));
  ASSERT_EQ(data_short, 0x0403);
  ASSERT_TRUE(!reader.Read(&data_short, sizeof(data_short)));
  ASSERT_EQ(data_short, 0x0605);

  std::int32_t data_int;
  ASSERT_TRUE(!reader.Read(&data_int, sizeof(data_int)));
  ASSERT_EQ(data_int, 0x0a090807);
  ASSERT_TRUE(!reader.Read(&data_int, sizeof(data_int)));
  ASSERT_EQ(data_int, 0x0e0d0c0b);

  // Out of range
  ASSERT_FALSE(!reader.Read(&data_int, sizeof(data_int)));

  reader.Reset();
  ASSERT_TRUE(!reader.Read(&data_byte, sizeof(data_byte)));
  ASSERT_EQ(data_byte, 0x01);
}

TEST(IO, DataBytesReadFromFile) {
  static constexpr char filePath[] = TEST_DATA_DIR "_data/generated/audio2x-common/tests/data/test_data_io.bin";

  // Get size using std::filesystem.
  const auto fileSize = std::filesystem::file_size(filePath);

  // Read the file.
  nva2x::DataBytes dataBytes;
  ASSERT_TRUE(!dataBytes.ReadFromFile(filePath));
  ASSERT_EQ(dataBytes.Size(), fileSize);

  // Read a file that does not exist.
  ASSERT_FALSE(!dataBytes.ReadFromFile("FILE_DOES_NOT_EXIST"));
}
