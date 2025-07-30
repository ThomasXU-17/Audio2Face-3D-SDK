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
import copy

import numpy as np


class Interpolator:
    def __init__(self, smoothing=0.8, degree=2):
        self.smoothing = smoothing
        self.degree = degree
        self.values = []

    def get(self):
        return self.values[-1] if len(self.values) != 0 else np.zeros(())

    def update(self, input, time_delta):
        v = self.values
        if len(v) != 0 and v[0].shape != input.shape:
            del v[:]

        if len(v) == 0:
            v.append(copy.deepcopy(input))
        v[0][:] = input
        while len(v) < self.degree + 1:
            v.append(copy.deepcopy(v[-1]))

        alpha = 1.0 - 0.5 ** (time_delta / self.smoothing) if self.smoothing > 0 else 1.0
        for i in range(1, len(v)):
            v[i] += (v[i - 1] - v[i]) * alpha
        return self.get()
