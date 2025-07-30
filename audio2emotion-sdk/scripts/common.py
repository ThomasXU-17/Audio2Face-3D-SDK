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
from os.path import join as opj

from audio2x.common import ROOT_DIR, normalize_path

X_TENSOR_NAME = "input_values"
Z_TENSOR_NAME = "output"


def normalize_path(base, sub):
    return os.path.normpath(os.path.join(base, sub))


AUDIO_TRACKS_DATA_ROOT = normalize_path(ROOT_DIR, "sample-data")
AUDIO2EMOTION_NET_ROOT = normalize_path(ROOT_DIR, "_data/audio2emotion-models/")
BENCHMARK_DATA_ROOT = normalize_path(ROOT_DIR, "_data/generated/audio2emotion-sdk/benchmark/data/")
SAMPLES_MODEL_ROOT = normalize_path(ROOT_DIR, "_data/generated/audio2emotion-sdk/samples/model/")
TESTS_DATA_ROOT = normalize_path(ROOT_DIR, "_data/generated/audio2emotion-sdk/tests/data/")

NETWORK_VERSION = "audio2emotion-v2.2"
ONNX_MODEL_FOLDER_PATH = opj(AUDIO2EMOTION_NET_ROOT, NETWORK_VERSION)
ONNX_MODEL_PATH = opj(ONNX_MODEL_FOLDER_PATH, "network.onnx")
NETWORK_INFO_PATH = opj(ONNX_MODEL_FOLDER_PATH, "network_info.json")
CONFIG_PATH = opj(ONNX_MODEL_FOLDER_PATH, "model_config.json")
