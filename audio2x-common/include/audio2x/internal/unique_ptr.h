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

#include <memory>

namespace nva2x {

namespace internal {

struct Destroyer {
    template <typename T>
    void operator()(T* obj) const {
        obj->Destroy();
    }
};

} // namespace internal

template <typename T>
using UniquePtr = std::unique_ptr<T, internal::Destroyer>;

template <typename T>
UniquePtr<T> ToUniquePtr(T* ptr) { return UniquePtr<T>(ptr); }

template <typename T>
using SharedPtr = std::shared_ptr<T>;

template <typename T>
std::shared_ptr<T> ToSharedPtr(T* ptr) {
  return std::shared_ptr<T>(ptr, [](T* p) { p->Destroy(); });
}

} // namespace nva2x
