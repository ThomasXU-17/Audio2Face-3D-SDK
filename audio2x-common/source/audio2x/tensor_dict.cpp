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
#include "audio2x/internal/tensor_dict.h"
#include "audio2x/internal/audio2x.h"
#include "audio2x/internal/io.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/internal/npz_utils.h"
#include "audio2x/error.h"
#include "cnpy.h"

#include <algorithm>


namespace {

std::error_code ReadFromNpz(std::unordered_map<std::string, nva2x::HostTensorFloat>& tensors, const cnpy::npz_t& npzInputData) {
  tensors.clear();
  for (const auto& [name, npyArray] : npzInputData) {
    nva2x::HostTensorFloat& tensor = tensors[name];
    A2X_CHECK_RESULT_WITH_MSG(tensor.Allocate(npyArray.num_vals), "Unable to allocate tensor of size: " << npyArray.num_vals);
    std::copy_n(npyArray.data<float>(), npyArray.num_vals, tensor.Data());
  }
  return nva2x::ErrorCode::eSuccess;
}

}


namespace nva2x {

IHostTensorDict::~IHostTensorDict() = default;

HostTensorDict::HostTensorDict() {
  A2X_LOG_DEBUG("HostTensorDict::HostTensorDict()");
}

HostTensorDict::~HostTensorDict() {
  A2X_LOG_DEBUG("HostTensorDict::~HostTensorDict()");
}

// FIXME: currently works only with Little-endian format for uint32_t (which
// comes from Python writer)
std::error_code HostTensorDict::ReadFromBuffer(const void *data, size_t size) {
  _tensors.clear();

  uint32_t numberOfTensors;
  uint32_t nameLength;
  uint32_t tensorLength;
  constexpr const unsigned int MAX_TENSOR_NAME_LEN = 256;
  char nameBuffer[MAX_TENSOR_NAME_LEN + 1];
  DataReader reader(data, size);

  A2X_CHECK_RESULT_WITH_MSG(reader.Read(&numberOfTensors, sizeof(uint32_t)), "Unable to get number of tensors");
  A2X_LOG_DEBUG("Number of tensors: " << numberOfTensors);

  while (numberOfTensors--) {
    A2X_CHECK_RESULT_WITH_MSG(reader.Read(&nameLength, sizeof(uint32_t)), "Unable to get length of tensor name");
    A2X_CHECK_ERROR_WITH_MSG(nameLength <= MAX_TENSOR_NAME_LEN,
               "Tensor name length exceeds " << MAX_TENSOR_NAME_LEN, ErrorCode::eOutOfBounds);
    A2X_CHECK_RESULT_WITH_MSG(reader.Read(nameBuffer, nameLength), "Unable to get tensor name");
    nameBuffer[nameLength] = '\0';
    std::string tensorName(nameBuffer);

    A2X_CHECK_RESULT_WITH_MSG(reader.Read(&tensorLength, sizeof(uint32_t)), "Unable to get length of a tensor");

    nva2x::HostTensorFloat& tensor = _tensors[tensorName];
    A2X_CHECK_RESULT_WITH_MSG(tensor.Allocate(tensorLength),
               "Unable to allocate tensor of size: " << tensorLength);
    A2X_CHECK_RESULT_WITH_MSG(reader.Read(tensor.Data(), sizeof(float) * tensorLength), "Unable to get tensor data");

    A2X_LOG_DEBUG("Adding tensor: " << tensorName << " -> " << tensorLength);
  }

  return ErrorCode::eSuccess;
}

std::error_code HostTensorDict::ReadFromFile(const char *filePath) {
  std::string strPath = filePath;
  if (strPath.size() >= 4 && strPath.substr(strPath.size() - 4) == ".npz") {
    // npz file
    cnpy::npz_t npzInputData;
    try {
      npzInputData = npz_load(strPath);
    } catch(const std::exception& e [[maybe_unused]]) {
      A2X_LOG_ERROR("Unable to read npz file: " << filePath);
      return ErrorCode::eReadFileFailed;
    }
    A2X_CHECK_RESULT_WITH_MSG(ReadFromNpz(_tensors, npzInputData), "Unable to read tensors from npz file: " << filePath);
    return ErrorCode::eSuccess;
  }
  // bin file
  DataBytes dataBytes;
  A2X_CHECK_RESULT_WITH_MSG(dataBytes.ReadFromFile(filePath),
             "Unable to read file: " << filePath);
  A2X_CHECK_RESULT_WITH_MSG(ReadFromBuffer(dataBytes.Data(), dataBytes.Size()),
             "Unable to read tensors from file: " << filePath);
  return ErrorCode::eSuccess;
}

const HostTensorFloat* HostTensorDict::At(const char *tensorName) const {
  try {
    const HostTensorFloat& tensorHost = _tensors.at(tensorName);
    return &tensorHost;
  } catch (std::out_of_range &) {
    A2X_LOG_ERROR("Unable to get tensor with name: " << tensorName);
    return nullptr;
  }
}

size_t HostTensorDict::Size() const {
  return _tensors.size();
}

void HostTensorDict::Destroy() {
  A2X_LOG_DEBUG("HostTensorDict::Destroy()");
  delete this;
}

} // namespace nva2x


nva2x::IHostTensorDict* nva2x::internal::CreateHostTensorDict() {
  A2X_LOG_DEBUG("CreateHostTensorDict()");
  return new HostTensorDict();
}
