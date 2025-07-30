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
#include "audio2x/internal/tensor_pool.h"

#include <gtest/gtest.h>

TEST(DeviceTensorPool, Simple) {
  nva2x::DeviceTensorPool pool;

  ASSERT_TRUE(!pool.Allocate(16000, 30));
  ASSERT_TRUE(!pool.Allocate(16000, 30));
  ASSERT_TRUE(!pool.Allocate(16000, 15));
  ASSERT_TRUE(!pool.Allocate(16000, 30));
  ASSERT_TRUE(!pool.Allocate(16000, 45));
  ASSERT_TRUE(!pool.Allocate(16000, 30));

  ASSERT_TRUE(!pool.Deallocate());
  ASSERT_TRUE(!pool.Allocate(16000, 30));

  ASSERT_TRUE(!pool.Allocate(1000, 30));
  ASSERT_TRUE(!pool.Allocate(16000, 15));
  ASSERT_TRUE(!pool.Allocate(2, 1000));
  ASSERT_TRUE(!pool.Allocate(16000, 30));

  {
    // Obtain and return a tensor.
    auto tensor1 = pool.Obtain();
    ASSERT_TRUE(tensor1);
    ASSERT_NE(tensor1->Data(), nullptr);
    ASSERT_EQ(tensor1->Size(), 16000ULL);
    ASSERT_TRUE(!pool.Return(std::move(tensor1)));
  }

  {
    // Obtain and return tensors in reverse order.
    auto tensor1 = pool.Obtain();
    ASSERT_TRUE(tensor1);
    ASSERT_NE(tensor1->Data(), nullptr);
    ASSERT_EQ(tensor1->Size(), 16000ULL);
    auto tensor2 = pool.Obtain();
    ASSERT_TRUE(tensor2);
    ASSERT_NE(tensor2->Data(), nullptr);
    ASSERT_EQ(tensor1->Size(), 16000ULL);
    ASSERT_TRUE(!pool.Return(std::move(tensor1)));
    ASSERT_TRUE(!pool.Return(std::move(tensor2)));
  }

  {
    // Get more tensors than available.
    ASSERT_TRUE(!pool.Allocate(1000, 1));
    auto tensor1 = pool.Obtain();
    ASSERT_TRUE(tensor1);
    ASSERT_NE(tensor1->Data(), nullptr);
    ASSERT_EQ(tensor1->Size(), 1000ULL);
    auto tensor2 = pool.Obtain();
    ASSERT_TRUE(tensor2);
    ASSERT_NE(tensor2->Data(), nullptr);
    ASSERT_EQ(tensor2->Size(), 1000ULL);
  }

  {
    // Return tensor allocated somewhere else.
    auto tensor1 = std::make_unique<nva2x::DeviceTensorFloat>();
    ASSERT_TRUE(tensor1);
    ASSERT_TRUE(!tensor1->Allocate(1000));
    ASSERT_NE(tensor1->Data(), nullptr);
    ASSERT_EQ(tensor1->Size(), 1000ULL);
    ASSERT_TRUE(!pool.Return(std::move(tensor1)));

    // Wrong size.
    auto tensor2 = pool.Obtain();
    ASSERT_TRUE(tensor2);
    ASSERT_NE(tensor2->Data(), nullptr);
    ASSERT_EQ(tensor2->Size(), 1000ULL);

    ASSERT_TRUE(!tensor2->Allocate(1001));
    ASSERT_TRUE(pool.Return(std::move(tensor2)));

    // Null pointer.
    ASSERT_TRUE(pool.Return({}));
  }
}
