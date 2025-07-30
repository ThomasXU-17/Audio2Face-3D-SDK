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

import audio2x.data_utils
import audio2x.trt
import torch
import torch.nn as nn
import torch.nn.functional as F
from audio2x.common import TESTS_DATA_ROOT


def gen_data():
    ###########  Create test PyTorch model  ###########

    # FIXME support batch size > 1
    BATCH_SIZE = 1
    INPUT_LEN = 8320
    EMO_LEN = 26
    RESULT_LEN = 300

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(INPUT_LEN, 42)
            self.fc2 = nn.Linear(EMO_LEN, 42)
            self.fc3 = nn.Linear(42, RESULT_LEN)

        def forward(self, x, emotion):
            h1 = F.relu(self.fc1(x))
            h2 = F.relu(self.fc2(emotion))
            h3 = h1 + h2 * 2
            res = self.fc3(h3)
            return res

    model = Model()
    model.eval()

    ###########  Generate test data  ###########

    x = torch.rand((BATCH_SIZE, INPUT_LEN), dtype=torch.float32)
    emo = torch.rand((BATCH_SIZE, EMO_LEN), dtype=torch.float32)
    with torch.no_grad():
        res = model(x, emo)

    data = {"input": x.numpy(), "emotion": emo.numpy(), "result": res.numpy()}

    os.makedirs(TESTS_DATA_ROOT, exist_ok=True)
    audio2x.data_utils.export_to_bin(data, os.path.join(TESTS_DATA_ROOT, "test_data_inference.bin"))

    ###########  Export model to ONNX  ###########

    INPUT_TENSOR_NAME = "input"
    EMOTION_TENSOR_NAME = "emotion"
    RESULT_TENSOR_NAME = "result"

    os.makedirs(TESTS_DATA_ROOT, exist_ok=True)
    onnx_network_fpath = os.path.join(TESTS_DATA_ROOT, "test_data_inference_network.onnx")
    dummy_input = (
        torch.zeros((BATCH_SIZE, 1, INPUT_LEN), dtype=torch.float32),
        torch.zeros((BATCH_SIZE, 1, EMO_LEN), dtype=torch.float32),
    )
    torch.onnx.export(
        model,
        dummy_input,
        onnx_network_fpath,
        verbose=False,
        input_names=[INPUT_TENSOR_NAME, EMOTION_TENSOR_NAME],
        output_names=[RESULT_TENSOR_NAME],
        dynamic_axes={
            "input": {0: "batch_size"},  # dynamic batch_size for input
            "emotion": {0: "batch_size"},  # dynamic batch_size for emotion output
        },
    )

    ###########  Convert model to TensorRT  ###########

    os.makedirs(TESTS_DATA_ROOT, exist_ok=True)
    TRT_DEVICE = 0  # ADJUST

    trt_network_fpath = os.path.join(TESTS_DATA_ROOT, "test_data_inference_network.trt")
    audio2x.trt.convert_onnx_to_trt(onnx_network_fpath, trt_network_fpath, TRT_DEVICE)

    trt_network_fpath_batched = os.path.join(TESTS_DATA_ROOT, "test_data_inference_network_batched.trt")
    dynamic_shapes = [
        (
            INPUT_TENSOR_NAME,
            {"min": (1, 1, INPUT_LEN), "max": (128, 1, INPUT_LEN), "opt": (8, 1, INPUT_LEN)},
        ),
        (
            EMOTION_TENSOR_NAME,
            {"min": (1, 1, EMO_LEN), "max": (128, 1, EMO_LEN), "opt": (8, 1, EMO_LEN)},
        ),
    ]
    audio2x.trt.convert_onnx_to_trt(
        onnx_network_fpath, trt_network_fpath_batched, TRT_DEVICE, dynamic_shapes=dynamic_shapes
    )


if __name__ == "__main__":
    gen_data()
