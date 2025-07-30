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
import argparse
import os
import tempfile

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CUR_DIR)))
DEPS_DIR = os.path.join(ROOT_DIR, "_deps/target-deps")
A2X_SDK_TMP_DIR = os.path.join(tempfile.gettempdir(), "a2x_sdk_tmp")
os.makedirs(A2X_SDK_TMP_DIR, exist_ok=True)
# for speed up development, we can set this to true system-wide to avoid re-converting the model
# use with caution, as it may lead to incorrect results
USE_TRT_CACHE = os.environ.get("A2X_SDK_USE_TRT_CACHE", "").lower() == "true"


def normalize_path(base, sub):
    return os.path.normpath(os.path.join(base, sub))


if os.name == "nt":
    CUR_OS = "windows"
else:
    CUR_OS = "linux"

TRT_DEVICE = 0

parser = argparse.ArgumentParser()
parser.add_argument("--want_nvinfer_dispatch", nargs=1, type=str)
parser.add_argument("--want_ampere_plus", nargs=1, type=str)
args = parser.parse_args()
WANT_NVINFER_DISPATCH = args.want_nvinfer_dispatch[0].lower() == "true" if args.want_nvinfer_dispatch else False
WANT_AMPERE_PLUS = args.want_ampere_plus[0].lower() == "true" if args.want_ampere_plus else False
del parser, args

TESTS_DATA_ROOT = normalize_path(ROOT_DIR, "_data/generated/audio2x-common/tests/data/")
