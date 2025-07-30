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
#include "audio2x/internal/io.h"
#include "audio2x/internal/audio2x.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/error.h"

#include <cstring>
#include <filesystem>
#include <fstream>


namespace {

std::error_code SafeResize(std::vector<char>& data, std::size_t size) {
  try {
    data.resize(size);
  } catch (std::bad_alloc &) {
    A2X_LOG_ERROR("Unable to resize DataBytes; size = " << size);
    return nva2x::ErrorCode::eInvalidValue;
  }
  return nva2x::ErrorCode::eSuccess;
}

}


namespace nva2x {

IDataBytes::~IDataBytes() = default;

DataBytes::DataBytes() {
  A2X_LOG_DEBUG("DataBytes::DataBytes()");
}

DataBytes::~DataBytes() {
  A2X_LOG_DEBUG("DataBytes::~DataBytes()");
}

std::error_code DataBytes::ReadFromFile(const char* filePath) {
  auto u8filePath = std::filesystem::u8path(filePath); // interpret as UTF-8
  A2X_LOG_INFO("Reading data file: " << filePath);
  std::ifstream f(u8filePath, std::ifstream::binary);
  A2X_CHECK_ERROR_WITH_MSG(f && f.good() && f.is_open(),
             "Unable to open file: " << filePath
                                     << " ; Message: " << std::strerror(errno), ErrorCode::eOpenFileFailed);
  f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    f.seekg(0, std::ios::end);
    std::streamsize len = f.tellg();
    size_t dataSize = static_cast<size_t>(len);
    A2X_CHECK_RESULT_WITH_MSG(SafeResize(_data, dataSize),
               "Unable to allocate data bytes; size=" << dataSize);
    f.seekg(0, std::ios::beg);
    f.read(_data.data(), len);
  } catch (std::ifstream::failure &) {
    A2X_LOG_ERROR("Invalid file: " << filePath);
    f.close();
    return ErrorCode::eReadFileFailed;
  }
  f.close();
  return ErrorCode::eSuccess;
}

const void* DataBytes::Data() const {
  return _data.data();
}

std::size_t DataBytes::Size() const {
  return _data.size();
}

void DataBytes::Destroy() {
  A2X_LOG_DEBUG("DataBytes::Destroy()");
  delete this;
}


DataReader::DataReader(const void* data, std::size_t dataSize)
: _data(data), _dataSize(dataSize), _dataPos(0) {
  A2X_LOG_DEBUG("DataReader::DataReader()");
}

DataReader::~DataReader() {
  A2X_LOG_DEBUG("DataReader::~DataReader()");
}

std::error_code DataReader::Read(void* dst, size_t size) {
  A2X_CHECK_ERROR_WITH_MSG(_dataPos + size <= _dataSize,
             "Unable to read data chunk: out of bounds", ErrorCode::eOutOfBounds)
  std::memcpy(dst, static_cast<const std::byte*>(_data) + _dataPos, size);
  _dataPos += size;
  return ErrorCode::eSuccess;
}

void DataReader::Reset() {
  _dataPos = 0;
}

} // namespace nva2x


nva2x::IDataBytes* nva2x::internal::CreateDataBytes() {
  A2X_LOG_DEBUG("CreateDataBytes()");
  return new DataBytes();
}
