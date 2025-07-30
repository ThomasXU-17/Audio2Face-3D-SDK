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
from os.path import join as opj

from common import CONFIG_PATH, NETWORK_INFO_PATH


def gen_config_data(sample_network_path, trt_info_fpath, sample_data_root):
    out_config_data_path = opj(sample_data_root, "a2e_ms_config.json")
    import json

    config = gen_config_contents(trt_info_fpath)
    config["net_path"] = sample_network_path

    with open(out_config_data_path, "w") as f:
        json.dump(config, f, indent=4)


def gen_config_contents(trt_info_fpath):
    import json

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    with open(NETWORK_INFO_PATH, "r") as f:
        network_info = json.load(f)

    config["device_id"] = 0

    with open(trt_info_fpath, "r") as f:
        trt_info = json.load(f)

    config["audio_params"] = {
        "buffer_len": int(trt_info["defaults"]["MAX_BUFFER_LEN"]),
        "samplerate": network_info["audio_params"]["samplerate"],
    }

    return config
