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
from common import AUDIO_TRACKS_DATA_ROOT, TESTS_DATA_ROOT
from pydub import AudioSegment
from util.audio_track import AudioTrack


def gen_data():
    TRACK_SAMPLERATE = 16000
    NEW_SAMPLERATE = 16000
    BUFFER_LEN = 8320
    BUFFER_OFS = 4160
    INPUT_STRENGTH = 1.5

    TRACK_LEN = int(TRACK_SAMPLERATE * 10.0)
    audio_data = np.random.randn(TRACK_LEN).astype(np.float32)
    track = AudioTrack(audio_data, TRACK_SAMPLERATE)

    def get_buffer(timestamp):
        cur_sample = track.sec_to_sample(timestamp)
        buffer = track.get_resampled_padded_buffer(cur_sample, BUFFER_OFS, BUFFER_LEN, NEW_SAMPLERATE)
        buffer = buffer.copy() * INPUT_STRENGTH
        return buffer

    timestamps = np.array([0.1, 4.0, 9.8], dtype=np.float32)

    buffer1 = get_buffer(timestamps[0])
    buffer2 = get_buffer(timestamps[1])
    buffer3 = get_buffer(timestamps[2])

    data = {
        "timestamps": timestamps,
        "audio_data": audio_data,
        "buffer1": buffer1,
        "buffer2": buffer2,
        "buffer3": buffer3,
    }

    os.makedirs(TESTS_DATA_ROOT, exist_ok=True)
    data_utils.export_to_bin(data, os.path.join(TESTS_DATA_ROOT, "test_data_audio.bin"))

    # generate resampled data
    TARGET_SAMPLE_RATE = 16000
    SAMPLE_AUDIO_FILES = ["audio_4sec_16k_s16le.wav", "audio_4sec_16k_s16le.wav"]
    for audio_fpath in SAMPLE_AUDIO_FILES:
        audio_data = AudioSegment.from_wav(os.path.join(AUDIO_TRACKS_DATA_ROOT, audio_fpath))
        audio_data = audio_data.set_channels(1)
        audio_data = audio_data.set_sample_width(2)
        audio_data = audio_data.set_frame_rate(TARGET_SAMPLE_RATE)
        audio_data.export(os.path.join(TESTS_DATA_ROOT, audio_fpath), format="wav", bitrate="16k")


if __name__ == "__main__":
    gen_data()
