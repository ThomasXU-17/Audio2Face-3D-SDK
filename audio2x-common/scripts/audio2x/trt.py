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
import json
import os
import shutil
import string
import subprocess
import tempfile
from collections import defaultdict
from os.path import join as opj

from .common import TRT_DEVICE, USE_TRT_CACHE, WANT_AMPERE_PLUS, WANT_NVINFER_DISPATCH
from .data_utils import get_trt_cache_path


def convert_onnx_to_trt(onnx_model_fpath, trt_model_fpath, device_id, dynamic_shapes=None, extra_args=None):
    cmd = [
        "trtexec",
        "--onnx=" + onnx_model_fpath,
        "--saveEngine=" + trt_model_fpath,
        "--device=" + str(device_id),
    ]

    if dynamic_shapes:
        if isinstance(dynamic_shapes[0], str):
            dynamic_shapes_arg = dynamic_shapes
        else:
            dynamic_shapes_arg = [
                "--minShapes="
                + ",".join([d[0] + ":" + "x".join([str(i) for i in d[1]["min"]]) for d in dynamic_shapes]),
                "--optShapes="
                + ",".join([d[0] + ":" + "x".join([str(i) for i in d[1]["opt"]]) for d in dynamic_shapes]),
                "--maxShapes="
                + ",".join([d[0] + ":" + "x".join([str(i) for i in d[1]["max"]]) for d in dynamic_shapes]),
            ]
        cmd += dynamic_shapes_arg
        # save dynamic_shapes in trt_info.json
        trt_info_fpath = opj(os.path.splitext(trt_model_fpath)[0] + "_trt_info.json")
        with open(trt_info_fpath, "w") as f:
            json.dump({"trt_build_param": {"batch": dynamic_shapes_arg}}, f, indent=4)
    if WANT_NVINFER_DISPATCH:
        cmd.append("--versionCompatible")
    if WANT_AMPERE_PLUS:
        cmd += ["--hardwareCompatibilityLevel=ampere+"]
    if extra_args:
        cmd += extra_args

    if USE_TRT_CACHE:
        trt_cache_path = get_trt_cache_path(onnx_model_fpath, cmd)
        # check if model is already converted
        if os.path.exists(trt_cache_path):
            # show cache path; user may delete if needed
            print(f"Using cached TRT model {trt_cache_path} for {onnx_model_fpath}.")
            shutil.copy(trt_cache_path, trt_model_fpath)
            return
        print(
            f"Cached TRT model not found for {onnx_model_fpath}. Converting ONNX to TRT model and caching it at {trt_cache_path}."
        )
        subprocess.call(cmd)
        shutil.copy(trt_model_fpath, trt_cache_path)
    else:
        subprocess.call(cmd)


def get_dynamic_shapes_from_trt_shape_params(shape_params):
    """
    Parse trt_info.json and convert into dynamic_shapes

    Example:
    shape_params:
    [
        "--minShapes=input_values:1x5000",
        "--maxShapes=input_values:1x60000",
        "--optShapes=input_values:1x30000"
    ]

    returned:
    dynamic_shapes = [
        ('input_values', {'min': ('1', '5000'), 'max': ('1', '60000'), 'opt': ('1', '30000')}),
    ]
    """
    dynamic_shapes_map = defaultdict(dict)
    for shape in shape_params:
        shape_type, shape_defs = shape.split("=")
        for shape_def in shape_defs.split(","):
            tensor_name, shape = shape_def.split(":")
            dims = tuple(shape.split("x"))
            dynamic_shapes_map[tensor_name][f"{shape_type[2:5]}"] = dims

    dynamic_shapes = []
    for k, v in dynamic_shapes_map.items():
        dynamic_shapes.append((k, v))
    return dynamic_shapes


def load_trt_build_param(trt_info_fpath):
    """
    Parse trt_info.json and convert into dynamic_shapes

    Example:
    json content:
    "trt_build_param": {
        "batch": [
            "--minShapes=input_values:1x5000",
            "--maxShapes=input_values:1x60000",
            "--optShapes=input_values:1x30000"
            ]
    }

    returned:
    [
        "--minShapes=input_values:1x5000",
        "--maxShapes=input_values:1x60000",
        "--optShapes=input_values:1x30000"
    ]
    """
    with open(trt_info_fpath, "r") as f:
        trt_info = json.load(f)

    build_param = trt_info["trt_build_param"]

    # "--memPoolSize=tacticSharedMem:0.046875" not supported in earlier versions.

    return build_param


def load_default_trt_command_param(trt_info_fpath, override_defaults: dict = None):
    trt_build_param = load_trt_build_param(trt_info_fpath)
    trt_command_param = [arg for _, args in trt_build_param.items() for arg in args]

    # Only try to read the defaults if there are format variables in the trt_build_param
    has_format_vars = lambda s: any(fname for _, fname, _, _ in string.Formatter().parse(s) if fname is not None)
    needs_default = any(has_format_vars(param) for param in trt_command_param)
    if needs_default:
        with open(trt_info_fpath, "r") as f:
            trt_info = json.load(f)

        defaults = trt_info["defaults"]
        if override_defaults:
            defaults.update(override_defaults)
        trt_command_param = [param.format(**defaults) for param in trt_command_param]

    return trt_command_param


def convert_onnx_to_trt_from_trt_info(onnx_network_fpath, trt_network_fpath, device_id, trt_info_fpath):
    trt_command_param = load_default_trt_command_param(trt_info_fpath)

    convert_onnx_to_trt(onnx_network_fpath, trt_network_fpath, device_id, extra_args=trt_command_param)


def convert_onnx_to_trt_from_folders(
    source_path, dest_path, trt_info_fname="trt_info.json", network_trt_fname="network.trt"
):
    source_onnx_path = opj(source_path, "network.onnx")
    dest_onnx_path = opj(dest_path, "network.onnx")
    shutil.copy(source_onnx_path, dest_onnx_path)

    # trtexec doesn't support unicode path. generating to a temp folder and then copy to destination
    temp_trt_network_fpath = opj(tempfile.mkdtemp(), network_trt_fname)
    trt_info_fpath = opj(dest_path, trt_info_fname)
    device_id = TRT_DEVICE
    convert_onnx_to_trt_from_trt_info(source_onnx_path, temp_trt_network_fpath, device_id, trt_info_fpath)
    # explicitly specify the destination file path to allow overwrite
    shutil.move(temp_trt_network_fpath, opj(dest_path, network_trt_fname))
