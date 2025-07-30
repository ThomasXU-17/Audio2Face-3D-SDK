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
import random
import shutil
from os.path import join as opj

import audio2x.trt
import numpy as np
from common import AUDIO_TRACKS_DATA_ROOT, BENCHMARK_DATA_ROOT, ONNX_MODEL_FOLDER_PATH
from pydub import AudioSegment
from utils import gen_config_data

TARGET_SAMPLE_RATE = 16000


def convert_and_normalize_audio_int_to_float(audio_data, scale_factor=1.0):
    # Normalize audio data to the range of -1.0 to 1.0
    max_amplitude = np.max(np.abs(audio_data))
    normalized_audio = audio_data.astype(np.float32) / max_amplitude * scale_factor
    return normalized_audio, max_amplitude


def convert_and_denormalize_audio_float_to_int(normalized_audio, max_amplitude):
    # Convert normalized audio data back to the original scale
    denormalized_audio = normalized_audio * max_amplitude
    return denormalized_audio.astype(np.int16)


def concatenate_wav_files(folder_path, output_path, target_duration, target_sample_rate, silence_duration, seed=None):
    # List all WAV files in the folder in alphabetical order
    wav_files = sorted([file for file in os.listdir(folder_path) if file.endswith(".wav")])
    random.seed(seed)
    random.shuffle(wav_files)

    # Initialize variables
    output_audio = AudioSegment.silent(duration=0)
    total_duration = 0

    while total_duration < target_duration * 60:  # Convert target duration to seconds
        for wav_file in wav_files:
            # Read the WAV file
            file_path = os.path.join(folder_path, wav_file)
            audio_data = AudioSegment.from_wav(file_path)

            # Convert to mono
            audio_data = audio_data.set_channels(1)

            audio_data = audio_data.set_sample_width(2)

            # Resample audio data to the target sample rate
            audio_data = audio_data.set_frame_rate(target_sample_rate)

            # Concatenate audio data with silence
            output_audio += audio_data + AudioSegment.silent(duration=int(silence_duration * 1000))

            # Update total duration
            total_duration = len(output_audio) / 1000  # Convert milliseconds to seconds

            # Break if the target duration is reached
            if total_duration >= target_duration * 60:
                break

    # Export the final concatenated audio to a new WAV file
    output_audio = output_audio.set_channels(1)
    output_audio = output_audio.set_sample_width(2)
    output_audio.export(output_path, format="wav", bitrate="16k")


if __name__ == "__main__":
    os.makedirs(BENCHMARK_DATA_ROOT, exist_ok=True)

    NUM_STREAMS = 10
    for i in range(NUM_STREAMS):
        print(f"Generating 5 minutes audio {i+1}/{NUM_STREAMS}")
        concatenate_wav_files(
            folder_path=AUDIO_TRACKS_DATA_ROOT,
            output_path=opj(BENCHMARK_DATA_ROOT, f"5_minutes_{i}.wav"),
            target_duration=5,  # 5 min
            target_sample_rate=TARGET_SAMPLE_RATE,
            silence_duration=2,  # 2 sec
            seed=i,  # seed for consistent randomized results
        )

    for filename in ["model_config.json", "network_info.json", "trt_info.json", "model.json"]:
        shutil.copy(opj(ONNX_MODEL_FOLDER_PATH, filename), opj(BENCHMARK_DATA_ROOT, filename))
    audio2x.trt.convert_onnx_to_trt_from_folders(ONNX_MODEL_FOLDER_PATH, BENCHMARK_DATA_ROOT)
    trt_info_fpath = opj(BENCHMARK_DATA_ROOT, "trt_info.json")
    gen_config_data(opj(BENCHMARK_DATA_ROOT, "network.trt"), trt_info_fpath, BENCHMARK_DATA_ROOT)
