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

#include "audio2face/internal/job_runner.h"
#include "audio2face/internal/logger.h"

namespace nva2f {

IJobRunner::~IJobRunner() = default;

ThreadPoolJobRunner::ThreadPoolJobRunner(size_t numThreads) {
    for (int i=0; i<numThreads; ++i) {
        m_pool.emplace_back(&ThreadPoolJobRunner::_workerFunction, this);
    }
}

ThreadPoolJobRunner::~ThreadPoolJobRunner() {
    LOG_DEBUG("ThreadPoolJobRunner::~ThreadPoolJobRunner()");
    _terminate();
}

void ThreadPoolJobRunner::Destroy() {
    delete this;
}

void ThreadPoolJobRunner::Enqueue(JobRunnerTask task, void* taskData) {
    std::unique_lock lock(m_guard);
    m_pendingJobs.emplace_back(std::bind(task, taskData));
    m_cv.notify_one();
}

void ThreadPoolJobRunner::_workerFunction() {
    while (m_isActive) {
        std::packaged_task<void()> job;
        {
            std::unique_lock lock(m_guard);
            m_cv.wait(lock, [&] { return !m_pendingJobs.empty() || !m_isActive; });
            if (!m_isActive) break;
            job.swap(m_pendingJobs.front());
            m_pendingJobs.pop_front();
        }
        job();
    }
}

void ThreadPoolJobRunner::_terminate() {
    m_isActive = false;
    m_cv.notify_all();
    for (auto& th : m_pool) {
        th.join();
    }
}

IJobRunner *CreateThreadPoolJobRunner_INTERNAL(size_t numThreads) {
    return new ThreadPoolJobRunner(numThreads);
}

} // namespace nva2f
