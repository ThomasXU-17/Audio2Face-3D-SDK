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
import shutil
from os.path import join as opj

from audio2x.data_utils import export_to_bin, int2bytes


def merge_json_files(file_paths, output_file, **extra_data):
    merged_data = {}
    for path in file_paths:
        with open(path, "r") as file:
            merged_data |= json.load(file)

    # add extra data, avoid overwriting at the top level keys
    for key, data in extra_data.items():
        if key in merged_data:
            if isinstance(merged_data[key], dict):
                merged_data[key].update(data)
            elif isinstance(merged_data[key], list):
                merged_data[key].extend(data)
            else:
                merged_data[key] = data
        else:
            merged_data[key] = data

    with open(output_file, "w") as outfile:
        json.dump(merged_data, outfile, indent=4)


def copy_files(list_of_files, in_dir, out_dir):
    for in_path_suffix, out_path_suffix in list_of_files:
        in_path = os.path.normpath(opj(in_dir, in_path_suffix))
        out_path = os.path.normpath(opj(out_dir, out_path_suffix))
        print("{} --> {}".format(in_path, out_path))
        if os.path.isfile(in_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copy(in_path, out_path)
        else:
            os.makedirs(out_path, exist_ok=True)
            for in_fpath in glob.glob(in_path):
                if os.path.isfile(in_fpath):
                    shutil.copy(in_fpath, out_path)
                elif os.path.exists(in_fpath):
                    shutil.copytree(in_fpath, out_path, dirs_exist_ok=True)
                else:
                    print(f"[WARNING] {in_fpath} does not exist")
