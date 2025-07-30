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

#include "audio2x/internal/audio2x.h"

#include <cassert>


// audio_accumulator.h
nva2x::IAudioAccumulator* nva2x::CreateAudioAccumulator(std::size_t tensorSize, std::size_t tensorCount) {
  return internal::CreateAudioAccumulator(tensorSize, tensorCount);
}


// float_accumulator.h
nva2x::IFloatAccumulator* nva2x::CreateFloatAccumulator(std::size_t tensorSize, std::size_t tensorCount) {
  return internal::CreateFloatAccumulator(tensorSize, tensorCount);
}


// cuda_stream.h
nva2x::ICudaStream* nva2x::CreateCudaStream() {
  return internal::CreateCudaStream();
}

nva2x::ICudaStream* nva2x::CreateDefaultCudaStream() {
  return internal::CreateDefaultCudaStream();
}


// cuda_utils.h
std::error_code nva2x::SetCudaDeviceIfNeeded(int device) {
  return internal::SetCudaDeviceIfNeeded(device);
}


// emotion_accumulator.h
nva2x::IEmotionAccumulator* nva2x::CreateEmotionAccumulator(
  std::size_t emotionSize, std::size_t emotionCountPerBuffer, std::size_t preallocatedBufferCount
  ) {
  return internal::CreateEmotionAccumulator(emotionSize, emotionCountPerBuffer, preallocatedBufferCount);
}


// executor.h
bool nva2x::HasExecutionStarted(const IExecutor& executor) {
  return internal::HasExecutionStarted(executor);
}

std::size_t nva2x::GetNbReadyTracks(const IExecutor& executor) {
  return internal::GetNbReadyTracks(executor);
}


// inference_engine.h
nva2x::IInferenceEngine* nva2x::CreateInferenceEngine() {
  return internal::CreateInferenceEngine();
}


// io.h
nva2x::IDataBytes* nva2x::CreateDataBytes() {
  return internal::CreateDataBytes();
}


// tensor_dict.h
nva2x::IHostTensorDict* nva2x::CreateHostTensorDict() {
  return internal::CreateHostTensorDict();
}


// tensor_float.h

//
// DeviceTensorFloatView publicly defined class.
//

nva2x::DeviceTensorFloatView::DeviceTensorFloatView()
: _data{nullptr}, _size{0}
{
}

nva2x::DeviceTensorFloatView nva2x::DeviceTensorFloatView::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

float* nva2x::DeviceTensorFloatView::Data() const {
    return _data;
}

std::size_t nva2x::DeviceTensorFloatView::Size() const {
    return _size;
}

nva2x::DeviceTensorFloatView::DeviceTensorFloatView(float* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::DeviceTensorFloatView::operator nva2x::DeviceTensorVoidView()  {
    return nva2x::DeviceTensorVoidView::Accessor::Build(_data, _size);
}

nva2x::DeviceTensorFloatView::operator nva2x::DeviceTensorVoidConstView() const {
  return nva2x::DeviceTensorVoidConstView::Accessor::Build(_data, _size);
}

//
// DeviceTensorFloatConstView publicly defined class.
//

nva2x::DeviceTensorFloatConstView::DeviceTensorFloatConstView()
: _data{nullptr}, _size{0}
{
}

nva2x::DeviceTensorFloatConstView::DeviceTensorFloatConstView(const DeviceTensorFloatView& other)
: _data{other.Data()}, _size{other.Size()}
{
}

nva2x::DeviceTensorFloatConstView nva2x::DeviceTensorFloatConstView::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

const float* nva2x::DeviceTensorFloatConstView::Data() const {
    return _data;
}

std::size_t nva2x::DeviceTensorFloatConstView::Size() const {
    return _size;
}

nva2x::DeviceTensorFloatConstView::DeviceTensorFloatConstView(const float* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::DeviceTensorFloatConstView::operator nva2x::DeviceTensorVoidConstView() const {
  return nva2x::DeviceTensorVoidConstView::Accessor::Build(_data, _size);
}

//
// HostTensorFloatView publicly defined class.
//

nva2x::HostTensorFloatView::HostTensorFloatView()
: _data{nullptr}, _size{0}
{
}

nva2x::HostTensorFloatView::HostTensorFloatView(float* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::HostTensorFloatView nva2x::HostTensorFloatView::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

float* nva2x::HostTensorFloatView::Data() const {
    return _data;
}

std::size_t nva2x::HostTensorFloatView::Size() const {
    return _size;
}

//
// HostTensorFloatConstView publicly defined class.
//

nva2x::HostTensorFloatConstView::HostTensorFloatConstView()
: _data{nullptr}, _size{0}
{
}

nva2x::HostTensorFloatConstView::HostTensorFloatConstView(const float* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::HostTensorFloatConstView::HostTensorFloatConstView(const HostTensorFloatView& other)
: _data{other.Data()}, _size{other.Size()}
{
}

nva2x::HostTensorFloatConstView nva2x::HostTensorFloatConstView::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

const float* nva2x::HostTensorFloatConstView::Data() const {
    return _data;
}

std::size_t nva2x::HostTensorFloatConstView::Size() const {
    return _size;
}

//
// Creation functions.
//

nva2x::IDeviceTensorFloat* nva2x::CreateDeviceTensorFloat(nva2x::HostTensorFloatConstView source, cudaStream_t cudaStream) {
  return nva2x::internal::CreateDeviceTensorFloat(source, cudaStream);
}

nva2x::IDeviceTensorFloat* nva2x::CreateDeviceTensorFloat(std::size_t size) {
  return nva2x::internal::CreateDeviceTensorFloat(size);
}

nva2x::IHostTensorFloat* nva2x::CreateHostTensorFloat(std::size_t size) {
  return nva2x::internal::CreateHostTensorFloat(size);
}

nva2x::IHostTensorFloat* nva2x::CreateHostPinnedTensorFloat(std::size_t size) {
  return nva2x::internal::CreateHostPinnedTensorFloat(size);
}

nva2x::DeviceTensorFloatView nva2x::GetDeviceTensorFloatView(float* data, std::size_t size) {
  return nva2x::internal::GetDeviceTensorFloatView(data, size);
}

nva2x::DeviceTensorFloatConstView nva2x::GetDeviceTensorFloatConstView(const float* data, std::size_t size) {
  return nva2x::internal::GetDeviceTensorFloatConstView(data, size);
}

//
// Copy functions.
//

std::error_code nva2x::CopyDeviceToDevice(
  DeviceTensorFloatView destination, DeviceTensorFloatConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyDeviceToDevice(destination, source, cudaStream);
}

std::error_code nva2x::CopyHostToDevice(
  DeviceTensorFloatView destination, HostTensorFloatConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyHostToDevice(destination, source, cudaStream);
}

std::error_code nva2x::CopyDeviceToHost(
  HostTensorFloatView destination, DeviceTensorFloatConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyDeviceToHost(destination, source, cudaStream);
}

std::error_code nva2x::CopyHostToHost(
  HostTensorFloatView destination, HostTensorFloatConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyHostToHost(destination, source, cudaStream);
}

std::error_code nva2x::CopyDeviceToDevice(
  DeviceTensorFloatView destination, DeviceTensorFloatConstView source
  ) {
  return nva2x::internal::CopyDeviceToDevice(destination, source);
}

std::error_code nva2x::CopyHostToDevice(
  DeviceTensorFloatView destination, HostTensorFloatConstView source
  ) {
  return nva2x::internal::CopyHostToDevice(destination, source);
}

std::error_code nva2x::CopyDeviceToHost(
  HostTensorFloatView destination, DeviceTensorFloatConstView source
  ) {
  return nva2x::internal::CopyDeviceToHost(destination, source);
}

std::error_code nva2x::CopyHostToHost(
  HostTensorFloatView destination, HostTensorFloatConstView source
  ) {
  return nva2x::internal::CopyHostToHost(destination, source);
}

//
// Fill functions.
//

std::error_code nva2x::FillOnDevice(
  DeviceTensorFloatView destination, float value, cudaStream_t cudaStream
  ) {
  return nva2x::internal::FillOnDevice(destination, value, cudaStream);
}

std::error_code nva2x::FillOnHost(
  HostTensorFloatView destination, float value
  ) {
  return nva2x::internal::FillOnHost(destination, value);
}


// tensor_bool.h

//
// DeviceTensorBoolView publicly defined class.
//

nva2x::DeviceTensorBoolView::DeviceTensorBoolView()
: _data{nullptr}, _size{0}
{
}

nva2x::DeviceTensorBoolView nva2x::DeviceTensorBoolView::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

bool* nva2x::DeviceTensorBoolView::Data() const {
    return _data;
}

std::size_t nva2x::DeviceTensorBoolView::Size() const {
    return _size;
}

nva2x::DeviceTensorBoolView::DeviceTensorBoolView(bool* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::DeviceTensorBoolView::operator nva2x::DeviceTensorVoidView()  {
  return nva2x::DeviceTensorVoidView::Accessor::Build(_data, _size);
}

nva2x::DeviceTensorBoolView::operator nva2x::DeviceTensorVoidConstView() const {
  return nva2x::DeviceTensorVoidConstView::Accessor::Build(_data, _size);
}

//
// DeviceTensorBoolConstView publicly defined class.
//

nva2x::DeviceTensorBoolConstView::DeviceTensorBoolConstView()
: _data{nullptr}, _size{0}
{
}

nva2x::DeviceTensorBoolConstView::DeviceTensorBoolConstView(const DeviceTensorBoolView& other)
: _data{other.Data()}, _size{other.Size()}
{
}

nva2x::DeviceTensorBoolConstView nva2x::DeviceTensorBoolConstView::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

const bool* nva2x::DeviceTensorBoolConstView::Data() const {
    return _data;
}

std::size_t nva2x::DeviceTensorBoolConstView::Size() const {
    return _size;
}

nva2x::DeviceTensorBoolConstView::DeviceTensorBoolConstView(const bool* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::DeviceTensorBoolConstView::operator nva2x::DeviceTensorVoidConstView() const {
  return nva2x::DeviceTensorVoidConstView::Accessor::Build(_data, _size);
}

//
// HostTensorBoolView publicly defined class.
//

nva2x::HostTensorBoolView::HostTensorBoolView()
: _data{nullptr}, _size{0}
{
}

nva2x::HostTensorBoolView::HostTensorBoolView(bool* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::HostTensorBoolView nva2x::HostTensorBoolView::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

bool* nva2x::HostTensorBoolView::Data() const {
    return _data;
}

std::size_t nva2x::HostTensorBoolView::Size() const {
    return _size;
}

//
// HostTensorBoolConstView publicly defined class.
//

nva2x::HostTensorBoolConstView::HostTensorBoolConstView()
: _data{nullptr}, _size{0}
{
}

nva2x::HostTensorBoolConstView::HostTensorBoolConstView(const bool* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::HostTensorBoolConstView::HostTensorBoolConstView(const HostTensorBoolView& other)
: _data{other.Data()}, _size{other.Size()}
{
}

nva2x::HostTensorBoolConstView nva2x::HostTensorBoolConstView::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

const bool* nva2x::HostTensorBoolConstView::Data() const {
    return _data;
}

std::size_t nva2x::HostTensorBoolConstView::Size() const {
    return _size;
}

//
// Creation functions.
//

nva2x::IDeviceTensorBool* nva2x::CreateDeviceTensorBool(std::size_t size) {
  return nva2x::internal::CreateDeviceTensorBool(size);
}

nva2x::IDeviceTensorBool* nva2x::CreateDeviceTensorBool(HostTensorBoolConstView source, cudaStream_t cudaStream) {
  return nva2x::internal::CreateDeviceTensorBool(source, cudaStream);
}

nva2x::IHostTensorBool* nva2x::CreateHostTensorBool(std::size_t size) {
  return nva2x::internal::CreateHostTensorBool(size);
}

nva2x::IHostTensorBool* nva2x::CreateHostPinnedTensorBool(std::size_t size) {
  return nva2x::internal::CreateHostPinnedTensorBool(size);
}

nva2x::DeviceTensorBoolView nva2x::GetDeviceTensorBoolView(bool* data, std::size_t size) {
  return nva2x::internal::GetDeviceTensorBoolView(data, size);
}

nva2x::DeviceTensorBoolConstView nva2x::GetDeviceTensorBoolConstView(const bool* data, std::size_t size) {
  return nva2x::internal::GetDeviceTensorBoolConstView(data, size);
}

//
// Copy functions.
//

std::error_code nva2x::CopyDeviceToDevice(
  DeviceTensorBoolView destination, DeviceTensorBoolConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyDeviceToDevice(destination, source, cudaStream);
}

std::error_code nva2x::CopyHostToDevice(
  DeviceTensorBoolView destination, HostTensorBoolConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyHostToDevice(destination, source, cudaStream);
}

std::error_code nva2x::CopyDeviceToHost(
  HostTensorBoolView destination, DeviceTensorBoolConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyDeviceToHost(destination, source, cudaStream);
}

std::error_code nva2x::CopyHostToHost(
  HostTensorBoolView destination, HostTensorBoolConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyHostToHost(destination, source, cudaStream);
}

std::error_code nva2x::CopyDeviceToDevice(
  DeviceTensorBoolView destination, DeviceTensorBoolConstView source
  ) {
  return nva2x::internal::CopyDeviceToDevice(destination, source);
}

std::error_code nva2x::CopyHostToDevice(
  DeviceTensorBoolView destination, HostTensorBoolConstView source
  ) {
  return nva2x::internal::CopyHostToDevice(destination, source);
}

std::error_code nva2x::CopyDeviceToHost(
  HostTensorBoolView destination, DeviceTensorBoolConstView source
  ) {
  return nva2x::internal::CopyDeviceToHost(destination, source);
}

std::error_code nva2x::CopyHostToHost(
  HostTensorBoolView destination, HostTensorBoolConstView source
  ) {
  return nva2x::internal::CopyHostToHost(destination, source);
}

//
// Fill functions.
//

std::error_code nva2x::FillOnDevice(
  DeviceTensorBoolView destination, bool value, cudaStream_t cudaStream
  ) {
  return nva2x::internal::FillOnDevice(destination, value, cudaStream);
}

std::error_code nva2x::FillOnHost(
  HostTensorBoolView destination, bool value
  ) {
  return nva2x::internal::FillOnHost(destination, value);
}


// tensor_int64.h

//
// DeviceTensorInt64View publicly defined class.
//

nva2x::DeviceTensorInt64View::DeviceTensorInt64View()
: _data{nullptr}, _size{0}
{
}

nva2x::DeviceTensorInt64View nva2x::DeviceTensorInt64View::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

int64_t* nva2x::DeviceTensorInt64View::Data() const {
    return _data;
}

std::size_t nva2x::DeviceTensorInt64View::Size() const {
    return _size;
}

nva2x::DeviceTensorInt64View::DeviceTensorInt64View(int64_t* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::DeviceTensorInt64View::operator nva2x::DeviceTensorVoidView()  {
  return nva2x::DeviceTensorVoidView::Accessor::Build(_data, _size);
}

nva2x::DeviceTensorInt64View::operator nva2x::DeviceTensorVoidConstView() const {
  return nva2x::DeviceTensorVoidConstView::Accessor::Build(_data, _size);
}

//
// DeviceTensorInt64ConstView publicly defined class.
//

nva2x::DeviceTensorInt64ConstView::DeviceTensorInt64ConstView()
: _data{nullptr}, _size{0}
{
}

nva2x::DeviceTensorInt64ConstView::DeviceTensorInt64ConstView(const DeviceTensorInt64View& other)
: _data{other.Data()}, _size{other.Size()}
{
}

nva2x::DeviceTensorInt64ConstView nva2x::DeviceTensorInt64ConstView::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

const int64_t* nva2x::DeviceTensorInt64ConstView::Data() const {
    return _data;
}

std::size_t nva2x::DeviceTensorInt64ConstView::Size() const {
    return _size;
}

nva2x::DeviceTensorInt64ConstView::DeviceTensorInt64ConstView(const int64_t* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::DeviceTensorInt64ConstView::operator nva2x::DeviceTensorVoidConstView() const {
  return nva2x::DeviceTensorVoidConstView::Accessor::Build(_data, _size);
}

//
// HostTensorInt64View publicly defined class.
//

nva2x::HostTensorInt64View::HostTensorInt64View()
: _data{nullptr}, _size{0}
{
}

nva2x::HostTensorInt64View::HostTensorInt64View(int64_t* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::HostTensorInt64View nva2x::HostTensorInt64View::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

int64_t* nva2x::HostTensorInt64View::Data() const {
    return _data;
}

std::size_t nva2x::HostTensorInt64View::Size() const {
    return _size;
}

//
// HostTensorInt64ConstView publicly defined class.
//

nva2x::HostTensorInt64ConstView::HostTensorInt64ConstView()
: _data{nullptr}, _size{0}
{
}

nva2x::HostTensorInt64ConstView::HostTensorInt64ConstView(const int64_t* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::HostTensorInt64ConstView::HostTensorInt64ConstView(const HostTensorInt64View& other)
: _data{other.Data()}, _size{other.Size()}
{
}

nva2x::HostTensorInt64ConstView nva2x::HostTensorInt64ConstView::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

const int64_t* nva2x::HostTensorInt64ConstView::Data() const {
    return _data;
}

std::size_t nva2x::HostTensorInt64ConstView::Size() const {
    return _size;
}

//
// Creation functions.
//

nva2x::IDeviceTensorInt64* nva2x::CreateDeviceTensorInt64(std::size_t size) {
  return nva2x::internal::CreateDeviceTensorInt64(size);
}

nva2x::IDeviceTensorInt64* nva2x::CreateDeviceTensorInt64(HostTensorInt64ConstView source, cudaStream_t cudaStream) {
  return nva2x::internal::CreateDeviceTensorInt64(source, cudaStream);
}

nva2x::IHostTensorInt64* nva2x::CreateHostTensorInt64(std::size_t size) {
  return nva2x::internal::CreateHostTensorInt64(size);
}

nva2x::IHostTensorInt64* nva2x::CreateHostPinnedTensorInt64(std::size_t size) {
  return nva2x::internal::CreateHostPinnedTensorInt64(size);
}

nva2x::DeviceTensorInt64View nva2x::GetDeviceTensorInt64View(int64_t* data, std::size_t size) {
  return nva2x::internal::GetDeviceTensorInt64View(data, size);
}

nva2x::DeviceTensorInt64ConstView nva2x::GetDeviceTensorInt64ConstView(const int64_t* data, std::size_t size) {
  return nva2x::internal::GetDeviceTensorInt64ConstView(data, size);
}

//
// Copy functions.
//

std::error_code nva2x::CopyDeviceToDevice(
  DeviceTensorInt64View destination, DeviceTensorInt64ConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyDeviceToDevice(destination, source, cudaStream);
}

std::error_code nva2x::CopyHostToDevice(
  DeviceTensorInt64View destination, HostTensorInt64ConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyHostToDevice(destination, source, cudaStream);
}

std::error_code nva2x::CopyDeviceToHost(
  HostTensorInt64View destination, DeviceTensorInt64ConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyDeviceToHost(destination, source, cudaStream);
}

std::error_code nva2x::CopyHostToHost(
  HostTensorInt64View destination, HostTensorInt64ConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyHostToHost(destination, source, cudaStream);
}

std::error_code nva2x::CopyDeviceToDevice(
  DeviceTensorInt64View destination, DeviceTensorInt64ConstView source
  ) {
  return nva2x::internal::CopyDeviceToDevice(destination, source);
}

std::error_code nva2x::CopyHostToDevice(
  DeviceTensorInt64View destination, HostTensorInt64ConstView source
  ) {
  return nva2x::internal::CopyHostToDevice(destination, source);
}

std::error_code nva2x::CopyDeviceToHost(
  HostTensorInt64View destination, DeviceTensorInt64ConstView source
  ) {
  return nva2x::internal::CopyDeviceToHost(destination, source);
}

std::error_code nva2x::CopyHostToHost(
  HostTensorInt64View destination, HostTensorInt64ConstView source
  ) {
  return nva2x::internal::CopyHostToHost(destination, source);
}

//
// Fill functions.
//

std::error_code nva2x::FillOnDevice(
  DeviceTensorInt64View destination, int64_t value, cudaStream_t cudaStream
  ) {
  return nva2x::internal::FillOnDevice(destination, value, cudaStream);
}

std::error_code nva2x::FillOnHost(
  HostTensorInt64View destination, int64_t value
  ) {
  return nva2x::internal::FillOnHost(destination, value);
}


// tensor_uint64.h

//
// DeviceTensorUInt64View publicly defined class.
//

nva2x::DeviceTensorUInt64View::DeviceTensorUInt64View()
: _data{nullptr}, _size{0}
{
}

nva2x::DeviceTensorUInt64View nva2x::DeviceTensorUInt64View::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

uint64_t* nva2x::DeviceTensorUInt64View::Data() const {
    return _data;
}

std::size_t nva2x::DeviceTensorUInt64View::Size() const {
    return _size;
}

nva2x::DeviceTensorUInt64View::DeviceTensorUInt64View(uint64_t* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::DeviceTensorUInt64View::operator nva2x::DeviceTensorVoidView()  {
  return nva2x::DeviceTensorVoidView::Accessor::Build(_data, _size);
}

nva2x::DeviceTensorUInt64View::operator nva2x::DeviceTensorVoidConstView() const {
  return nva2x::DeviceTensorVoidConstView::Accessor::Build(_data, _size);
}

//
// DeviceTensorUInt64ConstView publicly defined class.
//

nva2x::DeviceTensorUInt64ConstView::DeviceTensorUInt64ConstView()
: _data{nullptr}, _size{0}
{
}

nva2x::DeviceTensorUInt64ConstView::DeviceTensorUInt64ConstView(const DeviceTensorUInt64View& other)
: _data{other.Data()}, _size{other.Size()}
{
}

nva2x::DeviceTensorUInt64ConstView nva2x::DeviceTensorUInt64ConstView::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

const uint64_t* nva2x::DeviceTensorUInt64ConstView::Data() const {
    return _data;
}

std::size_t nva2x::DeviceTensorUInt64ConstView::Size() const {
    return _size;
}

nva2x::DeviceTensorUInt64ConstView::DeviceTensorUInt64ConstView(const uint64_t* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::DeviceTensorUInt64ConstView::operator nva2x::DeviceTensorVoidConstView() const {
  return nva2x::DeviceTensorVoidConstView::Accessor::Build(_data, _size);
}

//
// HostTensorUInt64View publicly defined class.
//

nva2x::HostTensorUInt64View::HostTensorUInt64View()
: _data{nullptr}, _size{0}
{
}

nva2x::HostTensorUInt64View::HostTensorUInt64View(uint64_t* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::HostTensorUInt64View nva2x::HostTensorUInt64View::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

uint64_t* nva2x::HostTensorUInt64View::Data() const {
    return _data;
}

std::size_t nva2x::HostTensorUInt64View::Size() const {
    return _size;
}

//
// HostTensorInt64ConstView publicly defined class.
//

nva2x::HostTensorUInt64ConstView::HostTensorUInt64ConstView()
: _data{nullptr}, _size{0}
{
}

nva2x::HostTensorUInt64ConstView::HostTensorUInt64ConstView(const uint64_t* data, std::size_t size)
: _data{data}, _size{size}
{
}

nva2x::HostTensorUInt64ConstView::HostTensorUInt64ConstView(const HostTensorUInt64View& other)
: _data{other.Data()}, _size{other.Size()}
{
}

nva2x::HostTensorUInt64ConstView nva2x::HostTensorUInt64ConstView::View(std::size_t viewOffset, std::size_t viewSize) const {
  assert(viewOffset + viewSize <= _size);
  return {_data + viewOffset, viewSize};
}

const uint64_t* nva2x::HostTensorUInt64ConstView::Data() const {
    return _data;
}

std::size_t nva2x::HostTensorUInt64ConstView::Size() const {
    return _size;
}

//
// Creation functions.
//

nva2x::IDeviceTensorUInt64* nva2x::CreateDeviceTensorUInt64(std::size_t size) {
  return nva2x::internal::CreateDeviceTensorUInt64(size);
}

nva2x::IDeviceTensorUInt64* nva2x::CreateDeviceTensorUInt64(HostTensorUInt64ConstView source, cudaStream_t cudaStream) {
  return nva2x::internal::CreateDeviceTensorUInt64(source, cudaStream);
}

nva2x::IHostTensorUInt64* nva2x::CreateHostTensorUInt64(std::size_t size) {
  return nva2x::internal::CreateHostTensorUInt64(size);
}

nva2x::IHostTensorUInt64* nva2x::CreateHostPinnedTensorUInt64(std::size_t size) {
  return nva2x::internal::CreateHostPinnedTensorUInt64(size);
}

nva2x::DeviceTensorUInt64View nva2x::GetDeviceTensorUInt64View(uint64_t* data, std::size_t size) {
  return nva2x::internal::GetDeviceTensorUInt64View(data, size);
}

nva2x::DeviceTensorUInt64ConstView nva2x::GetDeviceTensorUInt64ConstView(const uint64_t* data, std::size_t size) {
  return nva2x::internal::GetDeviceTensorUInt64ConstView(data, size);
}

//
// Copy functions.
//

std::error_code nva2x::CopyDeviceToDevice(
  DeviceTensorUInt64View destination, DeviceTensorUInt64ConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyDeviceToDevice(destination, source, cudaStream);
}

std::error_code nva2x::CopyHostToDevice(
  DeviceTensorUInt64View destination, HostTensorUInt64ConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyHostToDevice(destination, source, cudaStream);
}

std::error_code nva2x::CopyDeviceToHost(
  HostTensorUInt64View destination, DeviceTensorUInt64ConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyDeviceToHost(destination, source, cudaStream);
}

std::error_code nva2x::CopyHostToHost(
  HostTensorUInt64View destination, HostTensorUInt64ConstView source, cudaStream_t cudaStream
  ) {
  return nva2x::internal::CopyHostToHost(destination, source, cudaStream);
}

std::error_code nva2x::CopyDeviceToDevice(
  DeviceTensorUInt64View destination, DeviceTensorUInt64ConstView source
  ) {
  return nva2x::internal::CopyDeviceToDevice(destination, source);
}

std::error_code nva2x::CopyHostToDevice(
  DeviceTensorUInt64View destination, HostTensorUInt64ConstView source
  ) {
  return nva2x::internal::CopyHostToDevice(destination, source);
}

std::error_code nva2x::CopyDeviceToHost(
  HostTensorUInt64View destination, DeviceTensorUInt64ConstView source
  ) {
  return nva2x::internal::CopyDeviceToHost(destination, source);
}

std::error_code nva2x::CopyHostToHost(
  HostTensorUInt64View destination, HostTensorUInt64ConstView source
  ) {
  return nva2x::internal::CopyHostToHost(destination, source);
}

//
// Fill functions.
//

std::error_code nva2x::FillOnDevice(
  DeviceTensorUInt64View destination, uint64_t value, cudaStream_t cudaStream
  ) {
  return nva2x::internal::FillOnDevice(destination, value, cudaStream);
}

std::error_code nva2x::FillOnHost(
  HostTensorUInt64View destination, uint64_t value
  ) {
  return nva2x::internal::FillOnHost(destination, value);
}
