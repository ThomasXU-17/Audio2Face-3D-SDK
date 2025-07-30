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

import data_utils
import numpy as np
from common import TESTS_DATA_ROOT
from util.interpolator import Interpolator


def gen_data():
    RAW_LEN = 300
    smoothing = 0.2
    interp = Interpolator(smoothing=smoothing)
    dt_arr = np.array([0.0, 0.1, 0.3, 0.2, 0.65], dtype=np.float32)
    raw_arr = np.random.randn(len(dt_arr), RAW_LEN).astype(np.float32)

    for i in range(len(dt_arr)):
        smoothed = interp.update(raw_arr[i], dt_arr[i])

    data = {"raw_arr": raw_arr, "dt_arr": dt_arr, "smoothed": smoothed}

    os.makedirs(TESTS_DATA_ROOT, exist_ok=True)
    data_utils.export_to_bin(data, os.path.join(TESTS_DATA_ROOT, "test_data_interpolator.bin"))


if __name__ == "__main__":
    gen_data()
