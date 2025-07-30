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

namespace nva2x {

class NvtxTraceScope {
public:
    NvtxTraceScope(const char* name);
    ~NvtxTraceScope();

    NvtxTraceScope(const NvtxTraceScope&) = delete;
    NvtxTraceScope(NvtxTraceScope&&) = delete;
    NvtxTraceScope& operator=(const NvtxTraceScope&) = delete;
    NvtxTraceScope& operator=(NvtxTraceScope&&) = delete;
};

} // namespace nva2x

#ifdef USE_NVTX
#define A2X_CONCAT_(x, y) x##y
#define A2X_CONCAT(x, y) A2X_CONCAT_(x, y)
#define A2X_UNIQUE_NAME A2X_CONCAT(var, __COUNTER__)
#define NVTX_TRACE(name) ::nva2x::NvtxTraceScope A2X_UNIQUE_NAME(name)
#else
#define NVTX_TRACE(name)
#endif
