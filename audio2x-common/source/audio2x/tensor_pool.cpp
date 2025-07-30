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
#include "audio2x/internal/tensor_pool.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/error.h"

#include <cassert>
#include <numeric>

namespace nva2x {


std::error_code DeviceTensorPool::Allocate(std::size_t tensorSize, std::size_t tensorCount) {
    if (tensorSize != _tensorSize) {
        A2X_CHECK_RESULT_WITH_MSG(Deallocate(), "Unable to deallocate tensor before allocation");
        _tensorSize = tensorSize;
    }

    assert(_tensorSize == tensorSize);

    const auto oldCount = _pool.size();
    _pool.resize(tensorCount);
    for (auto i = oldCount; i < tensorCount; ++i) {
        assert(_pool[i].get() == nullptr);
        _pool[i] = std::make_unique<DeviceTensorFloat>();
        A2X_CHECK_RESULT_WITH_MSG(_pool[i]->Allocate(_tensorSize), "Unable to allocate tensor");
    }

    return ErrorCode::eSuccess;
}

std::error_code DeviceTensorPool::Deallocate() {
    _tensorSize = 0;
    _pool.clear();
    return ErrorCode::eSuccess;
}

std::unique_ptr<DeviceTensorFloat> DeviceTensorPool::Obtain() {
    if (!_pool.empty()) {
        auto tensor = std::move(_pool.back());
        _pool.pop_back();
        return tensor;
    }

    auto tensor =  std::make_unique<DeviceTensorFloat>();
    A2X_CHECK_ERROR_WITH_MSG(!tensor->Allocate(_tensorSize), "Unable to allocate tensor", {});
    return tensor;
}

std::error_code DeviceTensorPool::Return(std::unique_ptr<DeviceTensorFloat> tensor) {
    A2X_CHECK_ERROR_WITH_MSG(tensor, "Empty tensor returned to pool", ErrorCode::eNullPointer);
    A2X_CHECK_ERROR_WITH_MSG(tensor->Size() == _tensorSize, "Wrong size of tensor returned to pool", ErrorCode::eMismatch);

    _pool.emplace_back(std::move(tensor));

    return ErrorCode::eSuccess;
}


} // namespace nva2x
