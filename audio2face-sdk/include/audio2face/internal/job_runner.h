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
#include <thread>
#include <mutex>
#include <future>
#include <condition_variable>
#include <functional>
#include <vector>
#include <deque>
#include <type_traits>

#include "audio2face/job_runner.h"

namespace nva2f {

class ThreadPoolJobRunner : public IJobRunner {
public:
    ThreadPoolJobRunner(size_t numThreads);
    ~ThreadPoolJobRunner();
    void Enqueue(JobRunnerTask task, void* taskData) override;
    void Destroy() override;
private:
    std::atomic_bool m_isActive{ true };
    std::vector<std::thread> m_pool;
    std::condition_variable m_cv;
    std::mutex m_guard;
    std::deque<std::packaged_task<void()>> m_pendingJobs;

    void _workerFunction();
    void _terminate();
};


IJobRunner *CreateThreadPoolJobRunner_INTERNAL(size_t numThreads);

} // namespace nva2f
