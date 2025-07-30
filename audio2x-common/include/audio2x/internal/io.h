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

#include "audio2x/io.h"

#include <vector>

namespace nva2x {

class DataBytes : public IDataBytes {
public:
  DataBytes();
  ~DataBytes();

  std::error_code ReadFromFile(const char* filePath) override;
  const void* Data() const override;
  std::size_t Size() const override;
  void Destroy() override;

private:
  std::vector<char> _data;
};

class DataReader {
public:
  DataReader(const void* data, std::size_t dataSize);
  ~DataReader();
  std::error_code Read(void* dst, std::size_t size);
  void Reset();

private:
  const void* _data;
  std::size_t _dataSize;
  std::size_t _dataPos;
};

} // namespace nva2x
