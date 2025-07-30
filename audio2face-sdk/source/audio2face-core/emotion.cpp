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
#include "audio2face/internal/emotion.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2face/error.h"
#include "audio2x/cuda_utils.h"
#include "audio2x/error.h"
#include "audio2x/internal/io.h"
#include "audio2x/internal/npz_utils.h"
#include "audio2x/internal/tensor.h"

#include <cstring>
#include <cassert>
#include <filesystem>

namespace nva2f {

IEmotionDatabase::~IEmotionDatabase() = default;

EmotionDatabase::EmotionDatabase()
    : _cudaStream(nullptr), _emotionLength(0),
      _initialized(false) {
  LOG_DEBUG("EmotionDatabase::EmotionDatabase()");
}

EmotionDatabase::~EmotionDatabase() {
  LOG_DEBUG("EmotionDatabase::~EmotionDatabase()");
}

std::error_code EmotionDatabase::SetCudaStream(cudaStream_t cudaStream) {
  _cudaStream = cudaStream;
  return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionDatabase::InitFromFile(const char *emotionDatabaseFilePath) {
  auto u8EmotionDatabaseFilePath = std::filesystem::u8path(emotionDatabaseFilePath);
  if (u8EmotionDatabaseFilePath.extension() == std::filesystem::u8path(".npz")) {
    CHECK_RESULT_WITH_MSG(InitFromNpz(emotionDatabaseFilePath),
             "Unable to initialize emotion database");
  } else {
    LOG_ERROR("Unsupported file extension in inputDataFilePath: " << emotionDatabaseFilePath);
    return nva2x::ErrorCode::eOpenFileFailed;
  }
  return nva2x::ErrorCode::eSuccess;
}

std::error_code EmotionDatabase::InitFromNpz(const char *emotionDBPath) {
  _initialized = false;
  _shotList.clear();

  cnpy::npz_t npzInputData;
  try {
    npzInputData = nva2x::npz_load(emotionDBPath);
  } catch(const std::exception& e [[maybe_unused]]) {
    return nva2x::ErrorCode::eReadFileFailed;
  }

  cnpy::NpyArray emo_spec_names_npy_arr = npzInputData["emo_spec_names"];
  cnpy::NpyArray emo_spec_start_npy_arr = npzInputData["emo_spec_start"];
  cnpy::NpyArray emo_spec_size_npy_arr = npzInputData["emo_spec_size"];
  CHECK_ERROR_WITH_MSG(emo_spec_names_npy_arr.shape[0] == emo_spec_start_npy_arr.shape[0],
    "The length of 'emo_spec_start' does not match 'emo_spec_names'.", nva2x::ErrorCode::eNotInitialized)
  CHECK_ERROR_WITH_MSG(emo_spec_names_npy_arr.shape[0] == emo_spec_size_npy_arr.shape[0],
    "The length of 'emo_spec_size' does not match 'emo_spec_names'.", nva2x::ErrorCode::eNotInitialized)
  std::vector<std::string> emo_spec_names = nva2x::parse_string_array_from_npy_array(emo_spec_names_npy_arr);
  for(int i=0;i<emo_spec_names.size();++i) {
    std::string shotName = emo_spec_names[i];
    int shotStart = emo_spec_start_npy_arr.data<int>()[i];
    int shotSize = emo_spec_size_npy_arr.data<int>()[i];
    _shotList[shotName] = {static_cast<uint32_t>(shotStart), static_cast<uint32_t>(shotSize)};
  }

  {
    cnpy::NpyArray emo_db_npy_arr = npzInputData["emo_db"];
    size_t size = std::accumulate(emo_db_npy_arr.shape.begin(), emo_db_npy_arr.shape.end(), 1, std::multiplies<size_t>());
    CHECK_ERROR_WITH_MSG(emo_db_npy_arr.shape.size() == 2, "emo_db dimension is not 2", nva2x::ErrorCode::eNotInitialized)
    _emotionLength = emo_db_npy_arr.shape[1];

    CHECK_RESULT_WITH_MSG(_emotionData.Init({emo_db_npy_arr.data<float>(), size}, _cudaStream),
              "Unable to initialize emotion database matrix");
  }


  _initialized = true;
  return nva2x::ErrorCode::eSuccess;
}

std::size_t EmotionDatabase::GetEmotionLength() const {
  return _emotionLength;
}

std::error_code EmotionDatabase::GetEmotion(const char *emotionShot,
                                 unsigned int emotionFrame,
                                 nva2x::DeviceTensorFloatView emotion) const {
  LOG_INFO("Getting emotion vector for (" << emotionShot << ", " << emotionFrame
                                          << ")");
  CHECK_ERROR_WITH_MSG(_initialized, "EmotionDatabase is not initialized", nva2x::ErrorCode::eNotInitialized);
  CHECK_ERROR_WITH_MSG(emotion.Size() == _emotionLength, "Mismatched size for emotion in EmotionDatabase", nva2x::ErrorCode::eMismatch);

  assert(emotion.Data() != nullptr);

  try {
    auto emoShotInfo = _shotList.at(emotionShot);
    size_t emoShotStart = emoShotInfo.start;
    size_t emoShotSize = emoShotInfo.size;
    CHECK_ERROR_WITH_MSG(emotionFrame >= 0 && emotionFrame < emoShotSize,
               "emotion frame is out of bounds: " << emotionFrame, nva2x::ErrorCode::eOutOfBounds);
    size_t globalEmotionFrame = emoShotStart + emotionFrame;
    CHECK_RESULT(nva2x::CopyDeviceToDevice(
      emotion, _emotionData.View(globalEmotionFrame * _emotionLength, _emotionLength), _cudaStream));
  } catch (std::out_of_range &) {
    LOG_ERROR("Unable to get emotion shot with name: " << emotionShot);
    return ErrorCode::eOutOfRange;
  }
  return nva2x::ErrorCode::eSuccess;
}

const char* EmotionDatabase::GetEmotionShotName(std::size_t index) const {
  if (!_initialized) {
    LOG_ERROR("EmotionDatabase is not initialized");
    return nullptr;
  }
  if (index >= _shotList.size()) {
    LOG_ERROR("Index out of range");
    return nullptr;
  }

  auto it = _shotList.begin();
  std::advance(it, index);
  return it->first.c_str();
}

std::size_t EmotionDatabase::GetNbEmotionShots() const {
  if (!_initialized) {
    LOG_ERROR("EmotionDatabase is not initialized");
    return 0;
  }
  return _shotList.size();
}

uint32_t EmotionDatabase::GetEmotionShotSize(const char* emotionShotName) const {
  if (!_initialized) {
    LOG_ERROR("EmotionDatabase is not initialized");
    return 0;
  }

  try {
    return _shotList.at(emotionShotName).size;
  } catch (std::out_of_range &) {
    LOG_ERROR("Unable to get emotion shot info with name: " << emotionShotName);
    return 0;
  }
}

void EmotionDatabase::Destroy() {
  LOG_DEBUG("EmotionDatabase::Destroy()");
  delete this;
}

IEmotionDatabase *CreateEmotionDatabase_INTERNAL() {
  LOG_DEBUG("CreateEmotionDatabase_INTERNAL()");
  return new EmotionDatabase();
}

} // namespace nva2f
