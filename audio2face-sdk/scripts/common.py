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

from audio2x.common import CUR_OS, ROOT_DIR, normalize_path

AUDIO_INPUT_TENSOR_NAME = "input"
EMOTION_INPUT_TENSOR_NAME = "emotion"

SAMPLE_NETWORKS_PATH = normalize_path(ROOT_DIR, "_data/audio2face-models/")
AUDIO_TRACKS_DATA_ROOT = normalize_path(ROOT_DIR, "sample-data")
SAMPLES_DATA_ROOT = normalize_path(ROOT_DIR, "_data/generated/audio2face-sdk/samples/data/")
GOLDEN_DATA_ROOT = normalize_path(ROOT_DIR, "_data/audio2face-golden-data/")
AUDIO2FACE_SDK_NETS_SAMPLES_DATA_ROOT = normalize_path(
    ROOT_DIR, "_data/generated/audio2face-sdk/samples/data/audio2face-sdk-nets/"
)
TESTS_DATA_ROOT = normalize_path(ROOT_DIR, "_data/generated/audio2face-sdk/tests/data/")

os.makedirs(SAMPLES_DATA_ROOT, exist_ok=True)
os.makedirs(TESTS_DATA_ROOT, exist_ok=True)
