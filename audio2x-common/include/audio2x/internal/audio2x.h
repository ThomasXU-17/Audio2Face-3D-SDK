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

// The purpose of this file is to completely wrap the implementation of the
// public API, so that the .cpp file that defines the exported symbols can
// include this file and pretty much nothing else, making it easier to export
// the symbols by simply (re-)compiling the .cpp file with the exported symbols
// in the final DLL.  This is to allow a static library to define symbols to
// be exported when linked later.
//
// All of this gymnastic would not be required if we were using something like
// CMake's object libs.

#include "audio2x/audio_accumulator.h"
#include "audio2x/float_accumulator.h"
#include "audio2x/executor.h"
#include "audio2x/cuda_stream.h"
#include "audio2x/cuda_utils.h"
#include "audio2x/emotion_accumulator.h"
#include "audio2x/inference_engine.h"
#include "audio2x/io.h"
#include "audio2x/tensor.h"
#include "audio2x/tensor_dict.h"


namespace nva2x {

class DeviceTensorVoidView::Accessor {
public:
    static inline DeviceTensorVoidView Build(void* data, std::size_t size) { return {data, size}; }
};

class DeviceTensorVoidConstView::Accessor {
public:
    static inline DeviceTensorVoidConstView Build(const void* data, std::size_t size) { return {data, size}; }
};

} // namespace nva2x


namespace nva2x::internal {

// float_accumulator.h
IFloatAccumulator* CreateFloatAccumulator(std::size_t tensorSize, std::size_t tensorCount);

// audio_accumulator.h
IAudioAccumulator* CreateAudioAccumulator(std::size_t tensorSize, std::size_t tensorCount);

// cuda_stream.h
ICudaStream* CreateCudaStream();
ICudaStream* CreateDefaultCudaStream();

// cuda_utils.h
std::error_code SetCudaDeviceIfNeeded(int device);

// emotion_accumulator.h
IEmotionAccumulator* CreateEmotionAccumulator(
  std::size_t emotionSize, std::size_t emotionCountPerBuffer, std::size_t preallocatedBufferCount
);

// executor.h
bool HasExecutionStarted(const IExecutor& executor);
std::size_t GetNbReadyTracks(const IExecutor& executor);

// inference_engine.h
IInferenceEngine* CreateInferenceEngine();

// io.h
IDataBytes* CreateDataBytes();

// tensor_dict.h
IHostTensorDict* CreateHostTensorDict();

// tensor_float.h
IDeviceTensorFloat* CreateDeviceTensorFloat(std::size_t size);
IDeviceTensorFloat* CreateDeviceTensorFloat(HostTensorFloatConstView source, cudaStream_t cudaStream);
IHostTensorFloat* CreateHostTensorFloat(std::size_t size);
IHostTensorFloat* CreateHostPinnedTensorFloat(std::size_t size);
DeviceTensorFloatView GetDeviceTensorFloatView(float* data, std::size_t size);
DeviceTensorFloatConstView GetDeviceTensorFloatConstView(const float* data, std::size_t size);
std::error_code CopyDeviceToDevice(DeviceTensorFloatView destination, DeviceTensorFloatConstView source, cudaStream_t cudaStream);
std::error_code CopyHostToDevice(DeviceTensorFloatView destination, HostTensorFloatConstView source, cudaStream_t cudaStream);
std::error_code CopyDeviceToHost(HostTensorFloatView destination, DeviceTensorFloatConstView source, cudaStream_t cudaStream);
std::error_code CopyHostToHost(HostTensorFloatView destination, HostTensorFloatConstView source, cudaStream_t cudaStream);
std::error_code CopyDeviceToDevice(DeviceTensorFloatView destination, DeviceTensorFloatConstView source);
std::error_code CopyHostToDevice(DeviceTensorFloatView destination, HostTensorFloatConstView source);
std::error_code CopyDeviceToHost(HostTensorFloatView destination, DeviceTensorFloatConstView source);
std::error_code CopyHostToHost(HostTensorFloatView destination, HostTensorFloatConstView source);
std::error_code FillOnDevice(DeviceTensorFloatView destination, float value, cudaStream_t cudaStream);
std::error_code FillOnHost(HostTensorFloatView destination, float value);

// tensor_bool.h
IDeviceTensorBool* CreateDeviceTensorBool(HostTensorBoolConstView source, cudaStream_t cudaStream);
IDeviceTensorBool* CreateDeviceTensorBool(std::size_t size);
IHostTensorBool* CreateHostTensorBool(std::size_t size);
IHostTensorBool* CreateHostPinnedTensorBool(std::size_t size);
DeviceTensorBoolView GetDeviceTensorBoolView(bool* data, std::size_t size);
DeviceTensorBoolConstView GetDeviceTensorBoolConstView(const bool* data, std::size_t size);
std::error_code CopyDeviceToDevice(DeviceTensorBoolView destination, DeviceTensorBoolConstView source, cudaStream_t cudaStream);
std::error_code CopyHostToDevice(DeviceTensorBoolView destination, HostTensorBoolConstView source, cudaStream_t cudaStream);
std::error_code CopyDeviceToHost(HostTensorBoolView destination, DeviceTensorBoolConstView source, cudaStream_t cudaStream);
std::error_code CopyHostToHost(HostTensorBoolView destination, HostTensorBoolConstView source, cudaStream_t cudaStream);
std::error_code CopyDeviceToDevice(DeviceTensorBoolView destination, DeviceTensorBoolConstView source);
std::error_code CopyHostToDevice(DeviceTensorBoolView destination, HostTensorBoolConstView source);
std::error_code CopyDeviceToHost(HostTensorBoolView destination, DeviceTensorBoolConstView source);
std::error_code CopyHostToHost(HostTensorBoolView destination, HostTensorBoolConstView source);
std::error_code FillOnDevice(DeviceTensorBoolView destination, bool value, cudaStream_t cudaStream);
std::error_code FillOnHost(HostTensorBoolView destination, bool value);

// tensor_int64.h
IDeviceTensorInt64* CreateDeviceTensorInt64(std::size_t size);
IDeviceTensorInt64* CreateDeviceTensorInt64(HostTensorInt64ConstView source, cudaStream_t cudaStream);
IHostTensorInt64* CreateHostTensorInt64(std::size_t size);
IHostTensorInt64* CreateHostPinnedTensorInt64(std::size_t size);
DeviceTensorInt64View GetDeviceTensorInt64View(int64_t* data, std::size_t size);
DeviceTensorInt64ConstView GetDeviceTensorInt64ConstView(const int64_t* data, std::size_t size);
std::error_code CopyDeviceToDevice(DeviceTensorInt64View destination, DeviceTensorInt64ConstView source, cudaStream_t cudaStream);
std::error_code CopyHostToDevice(DeviceTensorInt64View destination, HostTensorInt64ConstView source, cudaStream_t cudaStream);
std::error_code CopyDeviceToHost(HostTensorInt64View destination, DeviceTensorInt64ConstView source, cudaStream_t cudaStream);
std::error_code CopyHostToHost(HostTensorInt64View destination, HostTensorInt64ConstView source, cudaStream_t cudaStream);
std::error_code CopyDeviceToDevice(DeviceTensorInt64View destination, DeviceTensorInt64ConstView source);
std::error_code CopyHostToDevice(DeviceTensorInt64View destination, HostTensorInt64ConstView source);
std::error_code CopyDeviceToHost(HostTensorInt64View destination, DeviceTensorInt64ConstView source);
std::error_code CopyHostToHost(HostTensorInt64View destination, HostTensorInt64ConstView source);
std::error_code FillOnDevice(DeviceTensorInt64View destination, int64_t value, cudaStream_t cudaStream);
std::error_code FillOnHost(HostTensorInt64View destination, int64_t value);

// tensor_uint64.h
IDeviceTensorUInt64* CreateDeviceTensorUInt64(std::size_t size);
IDeviceTensorUInt64* CreateDeviceTensorUInt64(HostTensorUInt64ConstView source, cudaStream_t cudaStream);
IHostTensorUInt64* CreateHostTensorUInt64(std::size_t size);
IHostTensorUInt64* CreateHostPinnedTensorUInt64(std::size_t size);
DeviceTensorUInt64View GetDeviceTensorUInt64View(uint64_t* data, std::size_t size);
DeviceTensorUInt64ConstView GetDeviceTensorUInt64ConstView(const uint64_t* data, std::size_t size);
std::error_code CopyDeviceToDevice(DeviceTensorUInt64View destination, DeviceTensorUInt64ConstView source, cudaStream_t cudaStream);
std::error_code CopyHostToDevice(DeviceTensorUInt64View destination, HostTensorUInt64ConstView source, cudaStream_t cudaStream);
std::error_code CopyDeviceToHost(HostTensorUInt64View destination, DeviceTensorUInt64ConstView source, cudaStream_t cudaStream);
std::error_code CopyHostToHost(HostTensorUInt64View destination, HostTensorUInt64ConstView source, cudaStream_t cudaStream);
std::error_code CopyDeviceToDevice(DeviceTensorUInt64View destination, DeviceTensorUInt64ConstView source);
std::error_code CopyHostToDevice(DeviceTensorUInt64View destination, HostTensorUInt64ConstView source);
std::error_code CopyDeviceToHost(HostTensorUInt64View destination, DeviceTensorUInt64ConstView source);
std::error_code CopyHostToHost(HostTensorUInt64View destination, HostTensorUInt64ConstView source);
std::error_code FillOnDevice(DeviceTensorUInt64View destination, uint64_t value, cudaStream_t cudaStream);
std::error_code FillOnHost(HostTensorUInt64View destination, uint64_t value);

} // namespace nva2x::internal
