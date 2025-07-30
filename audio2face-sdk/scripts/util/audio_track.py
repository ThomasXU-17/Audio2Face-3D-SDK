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
import math

import numpy as np
import scipy.signal


class AudioTrack:
    def __init__(self, data=None, samplerate=48000):
        self.data = data.astype(np.float32) if data is not None else np.zeros(0, dtype=np.float32)
        self.samplerate = samplerate
        self.norm_factor = 1.0
        assert self.data.ndim == 1
        assert self.samplerate > 0

    def sec_to_sample(self, sec):
        return int(round(sec * self.samplerate))

    def get_padded_buffer(self, ofs, length):
        if ofs >= 0 and ofs + length <= self.data.size:
            return self.data[ofs : ofs + length]
        res = np.zeros(length, dtype=self.data.dtype)
        begin = max(0, -ofs)
        end = min(length, self.data.size - ofs)
        if begin < end:
            res[begin:end] = self.data[ofs + begin : ofs + end]
        return res

    def get_resampled_padded_buffer(self, input_buffer_pos, resampled_ofs, resampled_len, new_samplerate):
        if self.samplerate == new_samplerate:
            ofs = input_buffer_pos - resampled_ofs
            return self.get_padded_buffer(ofs, resampled_len)
        resample_ratio = float(new_samplerate) / self.samplerate
        resample_up = max(int(round(min(resample_ratio, 1) * 1000)), 1)
        resample_down = max(int(round(resample_up / resample_ratio)), 1)
        input_buffer_len = int(math.ceil(float(resampled_len) * resample_down / resample_up))
        input_buffer_ofs = int(round(float(resampled_ofs) * resample_down / resample_up))
        ofs = input_buffer_pos - input_buffer_ofs
        buffer_track = AudioTrack(self.get_padded_buffer(ofs, input_buffer_len), self.samplerate)
        buffer_track.resample(new_samplerate)
        return buffer_track.get_padded_buffer(0, resampled_len)

    def resample(self, new_samplerate):
        if self.samplerate == new_samplerate:
            return
        resample_ratio = float(new_samplerate) / self.samplerate
        resample_up = max(int(round(min(resample_ratio, 1) * 1000)), 1)
        resample_down = max(int(round(resample_up / resample_ratio)), 1)
        self.data = scipy.signal.resample_poly(self.data.astype(np.float32), resample_up, resample_down).astype(
            np.float32
        )
        self.samplerate = new_samplerate
