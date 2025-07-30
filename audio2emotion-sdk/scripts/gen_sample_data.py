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
import shutil
from os.path import join as opj

import audio2x.trt
from common import ONNX_MODEL_FOLDER_PATH, SAMPLES_MODEL_ROOT


def gen_data():
    # Create the model folder with the real model data
    os.makedirs(SAMPLES_MODEL_ROOT, exist_ok=True)
    for filename in ["model_config.json", "network_info.json", "trt_info.json", "model.json"]:
        shutil.copy(opj(ONNX_MODEL_FOLDER_PATH, filename), opj(SAMPLES_MODEL_ROOT, filename))
    audio2x.trt.convert_onnx_to_trt_from_folders(ONNX_MODEL_FOLDER_PATH, SAMPLES_MODEL_ROOT)


if __name__ == "__main__":
    gen_data()
