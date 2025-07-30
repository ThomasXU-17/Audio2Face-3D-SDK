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
#include "audio2x/internal/npz_utils.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

TEST(NpzUtils, ParseStringArray) {
    cnpy::npz_t npzInputData;
    try {
        npzInputData = nva2x::npz_load(TEST_DATA_DIR "_data/generated/audio2x-common/tests/data/test_string_arr.npz");
    } catch(const std::exception& e [[maybe_unused]]) {

    }
    ASSERT_TRUE(npzInputData.count("names"));
    cnpy::NpyArray namesNpyArr = npzInputData["names"];
    std::vector<std::string> names = nva2x::parse_string_array_from_npy_array(namesNpyArr);
    EXPECT_EQ(names[0], "apple");
    EXPECT_EQ(names[1], "banana");
}
