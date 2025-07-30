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
import hashlib
import shutil
from os.path import join as opj

import numpy as np

from .common import A2X_SDK_TMP_DIR


def get_trt_cache_path(onnx_model_fpath, cmd):
    """
    Get the cache path for the TRT model with a hash of the onnx model, trtexec and the arguments
    """
    # create a hash of the onnx model, trtexec and the arguments
    digest = hashlib.sha256()
    digest.update(open(onnx_model_fpath, "rb").read())
    digest.update(open(shutil.which("trtexec"), "rb").read())
    digest.update(
        "".join(cmd[3:]).encode()
    )  # starting with 3 to skip the absolute path of trtexec, onnx and trt_model_fpath
    # warning: this assumes that cmd starts with the absolute path of trtexec, onnx and trt_model_fpath
    trt_cache_hash = digest.hexdigest()

    return opj(A2X_SDK_TMP_DIR, f"{trt_cache_hash}.trt")


def int2bytes(i):
    return np.array([i], dtype=np.dtype("<i4")).tobytes()  # "little-endian"


# FIXME add format versions
def export_to_bin(tensors, fname):
    data = b""
    data += int2bytes(len(tensors.keys()))
    for key, w in tensors.items():
        print("Adding tensor:", key, w.shape)
        data += int2bytes(len(key))
        data += key.encode()
        data += int2bytes(w.size)
        data += w.tobytes()
    with open(fname, "wb") as f:
        f.write(data)
