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

import audio2x.trt
import torch
import torch.nn as nn
import torch.nn.functional as F
from audio2x import data_utils
from common import TESTS_DATA_ROOT, X_TENSOR_NAME, Z_TENSOR_NAME


def gen_data():

    ###########  Create test PyTorch model  ###########

    BATCH_SIZE = 10
    X_LEN = 30000
    Z_LEN = 6

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(X_LEN, 42)
            self.fc2 = nn.Linear(42, Z_LEN)

        def forward(self, x):
            h1 = F.relu(self.fc1(x))
            z = self.fc2(h1)
            return z

    model = Model()
    model.eval()

    ###########  Generate test data  ###########

    x = torch.rand((BATCH_SIZE, X_LEN), dtype=torch.float32)
    with torch.no_grad():
        z = model(x)

    data = {"x": x.numpy(), "z": z.numpy()}

    os.makedirs(TESTS_DATA_ROOT, exist_ok=True)
    data_utils.export_to_bin(data, os.path.join(TESTS_DATA_ROOT, "test_data_inference.bin"))

    ###########  Export model to ONNX  ###########

    os.makedirs(TESTS_DATA_ROOT, exist_ok=True)
    onnx_model_fpath = os.path.join(TESTS_DATA_ROOT, "test_data_inference_model.onnx")
    dummy_input = (torch.zeros((BATCH_SIZE, X_LEN), dtype=torch.float32),)
    dynamic_axes = {X_TENSOR_NAME: {0: "batch", 1: "seqLen"}}
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_fpath,
        verbose=False,
        input_names=[X_TENSOR_NAME],
        output_names=[Z_TENSOR_NAME],
        dynamic_axes=dynamic_axes,
    )

    ###########  Convert model to TensorRT  ###########

    os.makedirs(TESTS_DATA_ROOT, exist_ok=True)
    trt_model_fpath = os.path.join(TESTS_DATA_ROOT, "test_data_inference_model.trt")
    TRT_DEVICE = 0  # ADJUST
    dynamic_shapes = [
        (X_TENSOR_NAME, {"min": (1, X_LEN), "max": (BATCH_SIZE, X_LEN), "opt": (BATCH_SIZE, X_LEN)}),
    ]
    audio2x.trt.convert_onnx_to_trt(onnx_model_fpath, trt_model_fpath, TRT_DEVICE, dynamic_shapes)


if __name__ == "__main__":
    gen_data()
