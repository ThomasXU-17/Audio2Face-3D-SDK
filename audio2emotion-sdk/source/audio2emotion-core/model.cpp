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
#include "audio2emotion/internal/model.h"
#include "audio2emotion/model.h"

#include <array>

namespace nva2e {

const nva2x::BufferBindingsDescription& IClassifierModel::GetBindingsDescription() {
  using IOType = nva2x::IBufferBindingsDescription::IOType;
  using DimensionType = nva2x::IBufferBindingsDescription::DimensionType;
  static constexpr std::array<nva2x::BindingDescription, 2> kDescriptions = {{
    {"input_values", IOType::INPUT, {{DimensionType::BATCH, DimensionType::DYNAMIC}}},
    {"output", IOType::OUTPUT, {{DimensionType::BATCH, DimensionType::FIXED}}},
  }};
  // Validate everything is as expected at compile-time.
  static_assert(nva2x::IsSorted(kDescriptions.data(), kDescriptions.size()));
  static_assert(0 == nva2x::CompareCStr("input_values", kDescriptions[kInputTensorIndex].name));
  static_assert(0 == nva2x::CompareCStr("output", kDescriptions[kResultTensorIndex].name));
  static_assert(1 == nva2x::GetInputCount(kDescriptions.data(), kDescriptions.size()));
  static_assert(1 == nva2x::GetBatchIndexCount(kDescriptions[kInputTensorIndex]));
  static_assert(1 == nva2x::GetBatchIndexCount(kDescriptions[kResultTensorIndex]));

  static const nva2x::BufferBindingsDescription descriptions({kDescriptions.begin(), kDescriptions.end()});
  return descriptions;
}

nva2x::BufferBindings* IClassifierModel::CreateBindings() {
  return new nva2x::BufferBindings(GetBindingsDescription());
}

const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForClassifierModel_INTERNAL() {
  return IClassifierModel::GetBindingsDescription();
}

nva2x::IBufferBindings* CreateBindingsForClassifierModel_INTERNAL() {
  return IClassifierModel::CreateBindings();
}

} // namespace nva2e
