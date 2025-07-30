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
import glob
import json
import os
import os.path
import shutil
from os.path import join as opj

import audio2x.trt
from common import SAMPLE_NETWORKS_PATH, SAMPLES_DATA_ROOT


def gen_copy_support_files(source_path, dest_path):
    # Copy all json and npz files from source to dest
    for ext in ["*.json", "*.npz"]:
        files = glob.glob(os.path.join(source_path, ext))
        for f in files:
            shutil.copy(f, dest_path)


def gen_network_fp16(source_path, dest_path):
    # assuming gen_data_base has been called and generate the files in dest_path

    MODEL_JSON_FP16 = "model_fp16.json"
    TRT_INFO_JSON_FP16 = "trt_info_fp16.json"
    NETWORK_FP16 = "network_fp16.trt"

    # patch trt_info.json with fp16 arg
    with open(opj(dest_path, "trt_info.json"), "r") as f:
        trt_info = json.load(f)
    trt_info["trt_build_param"]["fp16"] = [
        "--fp16",
    ]
    with open(opj(dest_path, TRT_INFO_JSON_FP16), "w") as f:
        json.dump(trt_info, f, indent=4)

    # convert fp16 model
    audio2x.trt.convert_onnx_to_trt_from_folders(
        source_path, dest_path, trt_info_fname=TRT_INFO_JSON_FP16, network_trt_fname=NETWORK_FP16
    )

    # patch model.json with fp16 network
    with open(opj(dest_path, "model.json"), "r") as f:
        model = json.load(f)
    model["networkPath"] = NETWORK_FP16
    with open(opj(dest_path, MODEL_JSON_FP16), "w") as f:
        json.dump(model, f, indent=4)


def gen_data_base(source_model, dest_path):
    os.makedirs(dest_path, exist_ok=True)

    source_path = opj(SAMPLE_NETWORKS_PATH, source_model)

    gen_copy_support_files(source_path, dest_path)
    audio2x.trt.convert_onnx_to_trt_from_folders(source_path, dest_path)

    # generate fp16 is pretty slow. please uncomment as needed
    # gen_network_fp16(source_path, dest_path)


def diffusion_gen_data(source_model, custom_folder_name=None):
    dest_path = opj(SAMPLES_DATA_ROOT, custom_folder_name if custom_folder_name else source_model)
    gen_data_base(source_model, dest_path)


def regression_gen_data(source_model, custom_folder_name=None):
    dest_path = opj(SAMPLES_DATA_ROOT, custom_folder_name if custom_folder_name else source_model)
    gen_data_base(source_model, dest_path)

    gen_config_data(source_model, dest_path)


def gen_config_data(source_model, sample_data_root):
    actor = os.path.basename(source_model)
    source_config_data_path = opj(SAMPLE_NETWORKS_PATH, source_model, "a2f_ms_config.json")
    out_config_data_path = opj(sample_data_root, "a2f_ms_config.json")

    # Load the source config, replace the paths and store updated file in sample_data_root
    def replace_paths():
        with open(source_config_data_path, "r") as f:
            data = json.load(f)

        data["inputDataFilePath"] = data["inputDataFilePath"].format(ROOT=sample_data_root)
        data["netPath"] = data["netPath"].format(ROOT=sample_data_root)
        data["emotionDatabaseFilePath"] = data["emotionDatabaseFilePath"].format(ROOT=sample_data_root)
        data["blendshape_params"]["bsDataPath"] = data["blendshape_params"]["bsDataPath"].format(ROOT=sample_data_root)

        with open(out_config_data_path, "w") as f:
            json.dump(data, f, indent=4)

    replace_paths()

    # golden image configs
    def gen_golden_image_configs():
        with open(out_config_data_path, "r") as f:
            data = json.load(f)

        OUTPUT_PATH = opj(sample_data_root, f"golden_{actor}_config.json")
        data["face_params"]["prediction_delay"] = 0.0
        data["face_params"]["upper_face_smoothing"] = 0.0
        data["face_params"]["lower_face_smoothing"] = 0.0
        data["blendshape_params"]["strengthTemporalSmoothing"] = 0.0
        with open(OUTPUT_PATH, "w") as f:
            json.dump(data, f, indent=4)

        # make a copy of the default tongue config for the golden image
        shutil.copy(
            opj(sample_data_root, "bs_tongue_config.json"), opj(sample_data_root, f"golden_{actor}_tongue_config.json")
        )

    gen_golden_image_configs()


if __name__ == "__main__":
    """
    Note: By default only latest models are tests
    add a custom function call to test different models or a local model.
    """
    # generate samples from audio2face-nets

    regression_gen_data(source_model=opj("audio2face-3d-v2.3.1-claire"), custom_folder_name="claire")
    regression_gen_data(source_model=opj("audio2face-3d-v2.3.1-james"), custom_folder_name="james")
    regression_gen_data(source_model=opj("audio2face-3d-v2.3-mark"), custom_folder_name="mark")
    # test unicode path
    regression_gen_data(source_model=opj("audio2face-3d-v2.3-mark"), custom_folder_name="🌎")

    diffusion_gen_data(source_model=opj("audio2face-3d-v3.0"), custom_folder_name="multi-diffusion")
