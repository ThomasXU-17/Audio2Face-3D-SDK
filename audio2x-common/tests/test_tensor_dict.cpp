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
#include "audio2x/internal/tensor_dict.h"
#include "audio2x/internal/io.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>

namespace {

  void Validate(const nva2x::HostTensorDict& testData) {
    auto tensor1 = testData.At("tensor1");
    ASSERT_NE(tensor1, nullptr);
    auto tensor2 = testData.At("tensor2");
    ASSERT_NE(tensor2, nullptr);
    auto tensor3 = testData.At("tensor3");
    ASSERT_NE(tensor3, nullptr);
    auto shouldBeNull = testData.At("TENSOR_DOES_NOT_EXIST");
    ASSERT_EQ(shouldBeNull, nullptr);

    ASSERT_EQ(tensor1->Size(), 16u * 1234u);
    ASSERT_EQ(tensor2->Size(), 16u * 56u);
    ASSERT_EQ(tensor3->Size(), 16u * 789u);
  }

  void Compare(const nva2x::HostTensorDict& testData1, const nva2x::HostTensorDict& testData2) {
    static constexpr std::array tensorNames = {"tensor1", "tensor2", "tensor3"};
    for (const auto tensorName : tensorNames) {
      const auto tensor1 = testData1.At(tensorName);
      const auto tensor2 = testData2.At(tensorName);
      ASSERT_NE(tensor1, nullptr);
      ASSERT_NE(tensor2, nullptr);
      EXPECT_EQ(tensor1->Size(), tensor2->Size());
      EXPECT_EQ(0, std::memcmp(tensor1->Data(), tensor2->Data(), tensor1->Size() * sizeof(float)));
    }
  }

}

TEST(TensorDict, ReadFromBin) {
  static constexpr char filePath[] = TEST_DATA_DIR "_data/generated/audio2x-common/tests/data/test_data_io.bin";

  nva2x::DataBytes testDataBytes;
  ASSERT_TRUE(!testDataBytes.ReadFromFile(filePath));

  nva2x::HostTensorDict testData1;
  ASSERT_TRUE(!testData1.ReadFromBuffer(testDataBytes.Data(), testDataBytes.Size()));

  Validate(testData1);

  nva2x::HostTensorDict testData2;
  ASSERT_TRUE(!testData2.ReadFromFile(filePath));

  Validate(testData2);

  Compare(testData1, testData2);
}

TEST(TensorDict, ReadFromNpz) {
  static constexpr char filePathReference[] = TEST_DATA_DIR "_data/generated/audio2x-common/tests/data/test_data_io.bin";
  nva2x::HostTensorDict testDataReference;
  ASSERT_TRUE(!testDataReference.ReadFromFile(filePathReference));
  Validate(testDataReference);

  {
    static constexpr char filePath[] = TEST_DATA_DIR "_data/generated/audio2x-common/tests/data/test_data_io.npz";
    nva2x::HostTensorDict testData;
    ASSERT_TRUE(!testData.ReadFromFile(filePath));
    Validate(testData);
    Compare(testDataReference, testData);
  }

  {
    static constexpr char filePath[] = TEST_DATA_DIR "_data/generated/audio2x-common/tests/data/test_data_io_compressed.npz";
    nva2x::HostTensorDict testData;
    ASSERT_TRUE(!testData.ReadFromFile(filePath));
    Validate(testData);
    Compare(testDataReference, testData);
  }

}
