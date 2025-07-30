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
#pragma once

#include "audio2face/blendshape_solver.h"
#include "audio2x/executor.h"
#include "audio2x/audio_accumulator.h"
#include "audio2x/emotion_accumulator.h"

namespace nva2f {

// Base interface for face executors.
//
// It abstracts the model being computed to generate results.
class IFaceExecutor : public nva2x::IExecutor {
public:
    // Structure to receive emotions used for a given frame.
    struct Emotions {
        std::size_t trackIndex{0};
        timestamp_t timeStampCurrentFrame{0};
        timestamp_t timeStampNextFrame{0};
        cudaStream_t cudaStream{nullptr};
        nva2x::DeviceTensorFloatConstView emotions;
    };
    // The emotions are given as a callback because a single execution can
    // provide 1 or multiple frames.
    using emotions_callback_t = void (*)(void* userdata, const Emotions& emotions);
    // Set the callback to use to collect emotions.  Setting this callback is optional.
    virtual std::error_code SetEmotionsCallback(emotions_callback_t callback, void* userdata) = 0;

    // Get the timestamp of the next emotion to be read, so that all emotions
    // up to that timestamp can be dropped.
    virtual timestamp_t GetNextEmotionTimestampToRead(std::size_t trackIndex) const = 0;
    // Get the next audio sample to be read, so that all samples up to that
    // sample can be dropped.
    virtual std::size_t GetNextAudioSampleToRead(std::size_t trackIndex) const = 0;
};

// Interface for executors which provide geometry.
//
// It abstracts whether it's computed by the regression, diffusion or any other
// model.
class IGeometryExecutor : public IFaceExecutor {
public:
    // Control which parts get executed.
    enum class ExecutionOption : std::uint32_t {
        None = 0b0000,
        Skin = 0b0001,
        Tongue = 0b0010,
        SkinTongue = Skin | Tongue,
        Jaw = 0b0100,
        Eyes = 0b1000,
        All = Skin | Tongue | Jaw | Eyes,
    };
    virtual std::error_code SetExecutionOption(ExecutionOption executionOption) = 0;
    virtual ExecutionOption GetExecutionOption() const = 0;

    // Get the size of the skin geometry output.
    virtual std::size_t GetSkinGeometrySize() const = 0;
    // Get the size of the tongue geometry output.
    virtual std::size_t GetTongueGeometrySize() const = 0;
    // Get the size of the jaw transform output.
    virtual std::size_t GetJawTransformSize() const = 0;
    // Get the size of the eyes rotation output.
    virtual std::size_t GetEyesRotationSize() const = 0;

    // Structure to receive results of a given execution.
    // Some fields are optionally set, depending on if they were activated or not.
    struct Results {
        std::size_t trackIndex{0};
        timestamp_t timeStampCurrentFrame{0};
        timestamp_t timeStampNextFrame{0};
        cudaStream_t skinCudaStream{nullptr};
        nva2x::DeviceTensorFloatConstView skinGeometry;
        cudaStream_t tongueCudaStream{nullptr};
        nva2x::DeviceTensorFloatConstView tongueGeometry;
        cudaStream_t jawCudaStream{nullptr};
        nva2x::DeviceTensorFloatConstView jawTransform;
        cudaStream_t eyesCudaStream{nullptr};
        nva2x::DeviceTensorFloatConstView eyesRotation;
    };
    // The results are given as a callback because a single execution can
    // provide 1 or multiple frames.
    // The user can return false to stop computation (in case of multi-frame execution).
    using results_callback_t = bool (*)(void* userdata, const Results& results);
    // Set the callback to use to collect results.  Setting this callback is mandatory before executing.
    // An error will be returned when running the execution if no callback is set.
    virtual std::error_code SetResultsCallback(results_callback_t callback, void* userdata) = 0;
};

// Interface for executors which provide blendshape weights.
//
// It abstracts whether it's computed by the regression, diffusion or any other
// model.
class IBlendshapeExecutor : public IFaceExecutor {
public:
    // Get the number of weights for output.
    virtual std::size_t GetWeightCount() const = 0;

    // Type of results returned by the executor.
    enum class ResultsType {
        UNKNOWN, HOST, DEVICE,
    };

    // Return whether this executor returns GPU or CPU results.
    virtual ResultsType GetResultType() const = 0;

    // Structure to receive results of a given execution when generated on the host.
    // Some fields are optionally set, depending on if they were activated or not.
    struct HostResults {
        std::size_t trackIndex{0};
        timestamp_t timeStampCurrentFrame{0};
        timestamp_t timeStampNextFrame{0};
        nva2x::HostTensorFloatConstView weights;
    };
    // The results are given as a callback because a single execution can
    // provide 1 or multiple frames.
    // This callback can be called after Execute() returns and from any thread.
    using host_results_callback_t = void (*)(void* userdata, const HostResults& results, std::error_code errorCode);
    // Set the callback to use to collect results.  Setting this callback is mandatory if the
    // underlying executor returns results on the host.
    // An error will be returned when running the execution if no callback is set.
    virtual std::error_code SetResultsCallback(host_results_callback_t callback, void* userdata) = 0;

    // Structure to receive results of a given execution when generated on the device.
    // Some fields are optionally set, depending on if they were activated or not.
    struct DeviceResults {
        std::size_t trackIndex{0};
        timestamp_t timeStampCurrentFrame{0};
        timestamp_t timeStampNextFrame{0};
        cudaStream_t cudaStream{nullptr};
        nva2x::DeviceTensorFloatConstView weights;
    };
    // The results are given as a callback because a single execution can
    // provide 1 or multiple frames.
    // This callback will be called before the Execute() function returns in the same thread
    // that called Execute().
    // The user can return false to stop computation (in case of multi-frame execution).
    using device_results_callback_t = bool (*)(void* userdata, const DeviceResults& results);
    // Set the callback to use to collect results.  Setting this callback is mandatory if the
    // underlying executor returns results on the device.
    // An error will be returned when running the execution if not callback is not set.
    virtual std::error_code SetResultsCallback(device_results_callback_t callback, void* userdata) = 0;

    // Wait for all asynchronously scheduled tasks to be done for a given track.
    virtual std::error_code Wait(std::size_t trackIndex) = 0;
};

// Geometry executor creation parameters, common to all geometry executors.
struct GeometryExecutorCreationParameters {
    // CUDA stream to use for the executor.
    cudaStream_t cudaStream{nullptr};
    // Number of tracks to execute.
    std::size_t nbTracks{0};
    // Array of shared audio accumulators to sample from.
    // The number of accumulators is given by nbTracks.
    const nva2x::IAudioAccumulator* const* sharedAudioAccumulators{nullptr};
    // Array of shared emotion accumulators to sample from.
    // The number of accumulators is given by nbTracks.
    const nva2x::IEmotionAccumulator* const* sharedEmotionAccumulators{nullptr};
};

} // namespace nva2f
