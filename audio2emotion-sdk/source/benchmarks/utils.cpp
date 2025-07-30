// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
#include "utils.h"

#include <cstdint>
#include <AudioFile.h>

std::vector<float> upsample(const std::vector<float>& input, int targetSampleRate, int originalSampleRate) {
    std::vector<float> output;
    float ratio = static_cast<float>(targetSampleRate) / originalSampleRate;

    for (size_t i = 0; i < input.size(); ++i) {
        output.push_back(input[i]);
        if (i < input.size() - 1) {
            float nextSample = input[i + 1];
            for (float t = 1.0f; t < ratio; t += 1.0f) {
                float interpolatedSample = input[i] + (nextSample - input[i]) * (t / ratio);
                output.push_back(interpolatedSample);
            }
        }
    }
    return output;
}

std::vector<float> downsample(const std::vector<float>& input, int targetSampleRate, int originalSampleRate) { //decimate
    std::vector<float> output;
    float ratio = static_cast<float>(originalSampleRate) / targetSampleRate;

    for (size_t i = 0; i < input.size(); i += static_cast<size_t>(ratio)) {
        output.push_back(input[i]);
    }

    return output;
}

std::vector<float> readAudio(const std::string& filename) {
    AudioFile<float> audio(filename);
    if(audio.getNumChannels() == 0 || audio.getLengthInSeconds() == 0) return {};
    const auto sr = audio.getSampleRate();
    // FIXME: Hard-coded number of samples, we should use audio_params.samplerate from the network info.
    if(sr == 16000) return audio.samples[0];
    const auto original = audio.samples[0];

    if(sr < 16000) {
        std::cerr << "Unsupported sample rate " << sr << std::endl;
        return {};
    }

        //really bad resampling, let's use matx poly_resample, which is the sampe implementation of scipy poly resample
    const int multiple = sr/16000;
    if(multiple * 16000 == sr) // multiple of 16khz khz
    {
        return downsample(original, 16000, sr);
    }
    if(audio.getSampleRate() == 24000) // 44.1 khz
    {
        const int lcm  = 48000;
        return downsample(upsample(original,  lcm, sr), 16000, lcm);
    }
    if(sr == 44100 || sr == 88200) // 44.1 khz 88.2khz
    {
        const int lcm  = 7056000;
        return downsample(upsample(original,  lcm, sr), 16000, lcm);
    }

    std::cerr << "Unsupported sample rate " << sr << std::endl;
    return {}; //not supported
}

std::vector<float> loadAudio() {
    // OPTME: allow for switching audio track
    return readAudio(TEST_DATA_DIR "sample-data/audio_4sec_16k_s16le.wav");
}
