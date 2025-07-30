# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import os

import audio2x.data_utils as data_utils
import numpy as np
from audio2x.common import TESTS_DATA_ROOT


def gen_data():
    data = {
        "tensor1": np.random.randn(16, 1234).astype(np.float32),
        "tensor2": np.random.randn(16, 56).astype(np.float32),
        "tensor3": np.random.randn(16, 789).astype(np.float32),
    }

    os.makedirs(TESTS_DATA_ROOT, exist_ok=True)
    data_utils.export_to_bin(data, os.path.join(TESTS_DATA_ROOT, "test_data_io.bin"))
    np.savez(os.path.join(TESTS_DATA_ROOT, "test_data_io.npz"), **data)
    np.savez_compressed(os.path.join(TESTS_DATA_ROOT, "test_data_io_compressed.npz"), **data)

    string_arr_data = {"names": np.array(["apple", "banana"], dtype="S")}
    np.savez_compressed(os.path.join(TESTS_DATA_ROOT, "test_string_arr.npz"), **string_arr_data)


if __name__ == "__main__":
    gen_data()
