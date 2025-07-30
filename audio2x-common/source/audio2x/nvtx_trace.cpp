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
#include "audio2x/internal/nvtx_trace.h"

#include <mutex>

#include <nvtx3/nvToolsExt.h>

namespace {

    nvtxDomainHandle_t a2xDomain = nullptr;
    std::once_flag a2xDomainFlag;

}

namespace nva2x {

NvtxTraceScope::NvtxTraceScope(const char* name) {
    // TODO: call nvtxDomainDestroy(a2xDomain)
    //       Maybe from a global Finalize() function...?
    std::call_once(a2xDomainFlag, [](){ a2xDomain = nvtxDomainCreateA("Audio2X SDK"); });

    nvtxEventAttributes_t eventAttrib;
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.category = 0;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0xFF87CEEB;
    eventAttrib.payloadType = NVTX_PAYLOAD_UNKNOWN;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    nvtxDomainRangePushEx(a2xDomain, &eventAttrib);
}

NvtxTraceScope::~NvtxTraceScope() {
    nvtxDomainRangePop(a2xDomain);
}

} // namespace nva2x
