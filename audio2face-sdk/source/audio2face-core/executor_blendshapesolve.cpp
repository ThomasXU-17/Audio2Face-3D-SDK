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
#include "audio2face/internal/executor_blendshapesolve.h"
#include "audio2face/internal/macros.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/job_runner.h"
#include "audio2face/internal/blendshape_solver.h"
#include "audio2face/error.h"
#include "audio2x/error.h"
#include "audio2x/internal/nvtx_trace.h"

#include <cassert>
#include <thread>

namespace nva2f {

// For IGeometryExecutor::ExecutionOption operators.
using namespace ::nva2f::internal;

std::size_t BlendshapeSolveExecutor::GetNbTracks() const {
    return _trackData.size();
}

std::error_code BlendshapeSolveExecutor::Reset(std::size_t trackIndex) {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);

    if (_geometryExecutor) {
        A2F_CHECK_RESULT_WITH_MSG(_geometryExecutor->Reset(trackIndex), "Unable to reset geometry executor");
    }

    auto& trackData = _trackData[trackIndex];
    if (trackData.skinSolver) {
        A2F_CHECK_RESULT_WITH_MSG(trackData.skinSolver->Reset(), "Unable to reset skin blendshape solver");
    }
    if (trackData.tongueSolver) {
        A2F_CHECK_RESULT_WITH_MSG(trackData.tongueSolver->Reset(), "Unable to reset tongue blendshape solver");
    }

    return nva2x::ErrorCode::eSuccess;
}

void BlendshapeSolveExecutor::Destroy() {
    delete this;
}

bool BlendshapeSolveExecutor::HasExecutionStarted(std::size_t trackIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", false);
    return _geometryExecutor->HasExecutionStarted(trackIndex);
}

std::size_t BlendshapeSolveExecutor::GetNbAvailableExecutions(std::size_t trackIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", 0);
    return _geometryExecutor->GetNbAvailableExecutions(trackIndex);
}

std::size_t BlendshapeSolveExecutor::GetTotalNbFrames(std::size_t trackIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", 0);
    return _geometryExecutor->GetTotalNbFrames(trackIndex);
}

std::size_t BlendshapeSolveExecutor::GetSamplingRate() const {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", 0);
    return _geometryExecutor->GetSamplingRate();
}

void BlendshapeSolveExecutor::GetFrameRate(std::size_t& numerator, std::size_t& denominator) const {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", );
    _geometryExecutor->GetFrameRate(numerator, denominator);
}

BlendshapeSolveExecutor::timestamp_t BlendshapeSolveExecutor::GetFrameTimestamp(std::size_t frameIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", std::numeric_limits<timestamp_t>::min());
    return _geometryExecutor->GetFrameTimestamp(frameIndex);
}

std::error_code BlendshapeSolveExecutor::SetEmotionsCallback(emotions_callback_t callback, void* userdata) {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", nva2x::ErrorCode::eNullPointer);
    return _geometryExecutor->SetEmotionsCallback(callback, userdata);
}

BlendshapeSolveExecutor::timestamp_t BlendshapeSolveExecutor::GetNextEmotionTimestampToRead(std::size_t trackIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", std::numeric_limits<timestamp_t>::min());
    return _geometryExecutor->GetNextEmotionTimestampToRead(trackIndex);
}

std::size_t BlendshapeSolveExecutor::GetNextAudioSampleToRead(std::size_t trackIndex) const {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", std::numeric_limits<std::size_t>::min());
    return _geometryExecutor->GetNextAudioSampleToRead(trackIndex);
}

std::size_t BlendshapeSolveExecutor::GetWeightCount() const {
    if (_trackData.empty()) {
        return 0;
    }
    const auto& trackData = _trackData[0];
    std::size_t count = 0;
    if (trackData.skinSolver) {
        count += trackData.skinSolver->NumBlendshapePoses();
    }
    if (trackData.tongueSolver) {
        count += trackData.tongueSolver->NumBlendshapePoses();
    }
    return count;
}

std::error_code BlendshapeSolveExecutor::SetResultsCallback(host_results_callback_t, void*) {
    return nva2x::ErrorCode::eUnsupported;
}

std::error_code BlendshapeSolveExecutor::SetResultsCallback(device_results_callback_t, void*) {
    return nva2x::ErrorCode::eUnsupported;
}

std::error_code BlendshapeSolveExecutor::Wait(std::size_t trackIndex) {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);

    auto& trackData = _trackData[trackIndex];
    if (trackData.skinSolver) {
        A2F_CHECK_RESULT_WITH_MSG(trackData.skinSolver->Wait(), "Unable to wait for skin solver");
        }
    if (trackData.tongueSolver) {
        A2F_CHECK_RESULT_WITH_MSG(trackData.tongueSolver->Wait(), "Unable to wait for tongue solver");
    }

    return nva2x::ErrorCode::eSuccess;
}


std::error_code BlendshapeSolveExecutor::GetInputStrength(float& inputStrength) const {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", nva2x::ErrorCode::eNullPointer);
    return GetExecutorInputStrength_INTERNAL(*_geometryExecutor, inputStrength);
}

std::error_code BlendshapeSolveExecutor::SetInputStrength(float inputStrength) {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", nva2x::ErrorCode::eNullPointer);
    return SetExecutorInputStrength_INTERNAL(*_geometryExecutor, inputStrength);
}

std::error_code BlendshapeSolveExecutor::GetGeometryExecutor(IGeometryExecutor** geometryExecutor) const {
    A2F_CHECK_ERROR_WITH_MSG(geometryExecutor, "Geometry executor cannot be null", nva2x::ErrorCode::eNullPointer);
    *geometryExecutor = _geometryExecutor.get();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveExecutor::GetSkinSolver(
    std::size_t trackIndex, IBlendshapeSolver** skinSolver
    ) const {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
    *skinSolver = _trackData[trackIndex].skinSolver.get();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveExecutor::GetTongueSolver(
    std::size_t trackIndex, IBlendshapeSolver** tongueSolver
    ) const {
    A2F_CHECK_ERROR_WITH_MSG(trackIndex < _trackData.size(), "Track index out of bounds", nva2x::ErrorCode::eOutOfBounds);
    *tongueSolver = _trackData[trackIndex].tongueSolver.get();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveExecutor::SetGeometryResultsCallback(
    IGeometryExecutor::results_callback_t callback, void* userdata
    ) {
    A2F_CHECK_ERROR_WITH_MSG(
        !nva2x::HasExecutionStarted(*this),
        "Results callback can only be set before execution is started",
        nva2x::ErrorCode::eExecutionAlreadyStarted
        );
    _geometryResultsCallback = callback;
    _geometryResultsUserdata = userdata;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveExecutor::Get(std::size_t trackIndex, AnimatorSkinParams& params) const {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", nva2x::ErrorCode::eNullPointer);
    return GetExecutorSkinParameters_INTERNAL(*_geometryExecutor, trackIndex, params);
}

std::error_code BlendshapeSolveExecutor::Set(std::size_t trackIndex, const AnimatorSkinParams& params) {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", nva2x::ErrorCode::eNullPointer);
    return SetExecutorSkinParameters_INTERNAL(*_geometryExecutor, trackIndex, params);
}

std::error_code BlendshapeSolveExecutor::Get(std::size_t trackIndex, AnimatorTongueParams& params) const {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", nva2x::ErrorCode::eNullPointer);
    return GetExecutorTongueParameters_INTERNAL(*_geometryExecutor, trackIndex, params);
}

std::error_code BlendshapeSolveExecutor::Set(std::size_t trackIndex, const AnimatorTongueParams& params) {
    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", nva2x::ErrorCode::eNullPointer);
    return SetExecutorTongueParameters_INTERNAL(*_geometryExecutor, trackIndex, params);
}

std::error_code BlendshapeSolveExecutor::Init(
    nva2x::UniquePtr<IGeometryExecutor> transferredGeometryExecutor,
    const BlendshapeSolveExecutorCreationParameters& params,
    bool gpuSolver
    ) {
    A2F_CHECK_ERROR_WITH_MSG(transferredGeometryExecutor, "Geometry executor cannot be null", nva2x::ErrorCode::eNullPointer);

    _geometryExecutor = std::move(transferredGeometryExecutor);

    _trackData.resize(_geometryExecutor->GetNbTracks());

    IGeometryExecutor::ExecutionOption executionOption = IGeometryExecutor::ExecutionOption::None;
    if (params.initializationSkinParams) {
        for (auto& trackData : _trackData) {
            auto& skinSolver = trackData.skinSolver;
            skinSolver.reset(nva2f::CreateBlendshapeSolver_INTERNAL(gpuSolver));
            A2F_CHECK_ERROR_WITH_MSG(skinSolver, "Unable to load skin blendshape solver", nva2x::ErrorCode::eNullPointer);
            A2F_CHECK_RESULT_WITH_MSG(skinSolver->SetBlendshapeData(params.initializationSkinParams->data), "Unable to set blendshape data for skin");
            A2F_CHECK_RESULT_WITH_MSG(skinSolver->SetParameters(params.initializationSkinParams->params), "Unable to set parameters for skin");
            A2F_CHECK_RESULT_WITH_MSG(skinSolver->SetBlendshapeConfig(params.initializationSkinParams->config), "Unable to set config for skin");
            A2F_CHECK_RESULT_WITH_MSG(skinSolver->Prepare(), "Unable to prepare skin blendshape solver");
        }
        executionOption |= IGeometryExecutor::ExecutionOption::Skin;
    }

    if (params.initializationTongueParams) {
        for (auto& trackData : _trackData) {
            auto& tongueSolver = trackData.tongueSolver;
            tongueSolver.reset(nva2f::CreateBlendshapeSolver_INTERNAL(gpuSolver));
            A2F_CHECK_ERROR_WITH_MSG(tongueSolver, "Unable to load tongue blendshape solver", nva2x::ErrorCode::eNullPointer);
            A2F_CHECK_RESULT_WITH_MSG(tongueSolver->SetBlendshapeData(params.initializationTongueParams->data), "Unable to set blendshape data for tongue");
            A2F_CHECK_RESULT_WITH_MSG(tongueSolver->SetParameters(params.initializationTongueParams->params), "Unable to set parameters for tongue");
            A2F_CHECK_RESULT_WITH_MSG(tongueSolver->SetBlendshapeConfig(params.initializationTongueParams->config), "Unable to set config for tongue");
            A2F_CHECK_RESULT_WITH_MSG(tongueSolver->Prepare(), "Unable to prepare tongue blendshape solver");
            }
        executionOption |= IGeometryExecutor::ExecutionOption::Tongue;
    }

    A2F_CHECK_RESULT_WITH_MSG(
        _geometryExecutor->SetExecutionOption(executionOption),
        "Unable to set execution option on geometry executor"
        );

    return nva2x::ErrorCode::eSuccess;
}


HostBlendshapeSolveExecutor::ResultsType HostBlendshapeSolveExecutor::GetResultType() const {
    return ResultsType::HOST;
}

std::error_code HostBlendshapeSolveExecutor::Execute(std::size_t* pNbExecutedTracks) {
    NVTX_TRACE("IBlendshapeSolveExecutor::Execute (host)");

    A2F_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);

    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", nva2x::ErrorCode::eNullPointer);
    return _geometryExecutor->Execute(pNbExecutedTracks);
}

std::error_code HostBlendshapeSolveExecutor::SetResultsCallback(
    host_results_callback_t callback, void* userdata
    ) {
    A2F_CHECK_ERROR_WITH_MSG(
        !nva2x::HasExecutionStarted(*this),
        "Results callback can only be set before execution is started",
        nva2x::ErrorCode::eExecutionAlreadyStarted
        );
    A2F_CHECK_ERROR_WITH_MSG(callback || !userdata, "User data must be null if host callback is null", nva2x::ErrorCode::eNullPointer);
    _resultsCallback = callback;
    _resultsUserdata = userdata;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code HostBlendshapeSolveExecutor::Init(
    nva2x::UniquePtr<IGeometryExecutor> transferredGeometryExecutor,
    const HostBlendshapeSolveExecutorCreationParameters& params
    ) {
    A2F_CHECK_RESULT(
        BlendshapeSolveExecutor::Init(std::move(transferredGeometryExecutor), params, false)
        );

    auto jobRunner = params.sharedJobRunner;
    if (!jobRunner) {
        std::size_t nbThreads = 0;
        if (params.initializationSkinParams) {
            ++nbThreads;
        }
        if (params.initializationTongueParams) {
            ++nbThreads;
        }
        nbThreads *= _geometryExecutor->GetNbTracks();

        // Get hardware concurrency and limit thread count
        const std::size_t maxHardwareThreads = std::thread::hardware_concurrency();
        if (maxHardwareThreads != 0) {
            nbThreads = std::min(nbThreads, maxHardwareThreads);
        }

        _jobRunner.reset(nva2f::CreateThreadPoolJobRunner_INTERNAL(nbThreads));
        jobRunner = _jobRunner.get();
    }

    for (auto& trackData : _trackData) {
        if (trackData.skinSolver) {
            A2F_CHECK_RESULT_WITH_MSG(trackData.skinSolver->SetJobRunner(jobRunner), "Unable to set job runner for skin");
        }
        if (trackData.tongueSolver) {
            A2F_CHECK_RESULT_WITH_MSG(trackData.tongueSolver->SetJobRunner(jobRunner), "Unable to set job runner for tongue");
        }
    }

    A2F_CHECK_RESULT_WITH_MSG(
        _geometryExecutor->SetResultsCallback(callbackForGeometryExecutor, this),
        "Unable to set geometry executor results callaback"
        );

    return nva2x::ErrorCode::eSuccess;
}

bool HostBlendshapeSolveExecutor::callbackHelperForGeometryExecutor(
    IBlendshapeSolver* skinSolver,
    IBlendshapeSolver* tongueSolver,
    IGeometryExecutor::results_callback_t geometryCallback,
    void* geometryUserdata,
    host_results_callback_t resultsCallback,
    void* resultsUserdata,
    const IGeometryExecutor::Results& results
    ) {
    if (geometryCallback) {
        geometryCallback(geometryUserdata, results);
    }

    struct BlendshapeSolverCallbackData {
        host_results_callback_t resultsCallback;
        void* resultsUserdata;
        std::size_t trackIndex;
        timestamp_t timeStampCurrentFrame;
        timestamp_t timeStampNextFrame;
        nva2x::HostTensorFloat results;
        std::mutex mutex;
        std::error_code errorCode;
        std::size_t refCount;
    };

    nva2f::BlendshapeSolverCallback blendshapeSolverCallback = [](void* callbackdata, std::error_code errorCode) {
        assert(callbackdata);
        auto* data = static_cast<BlendshapeSolverCallbackData*>(callbackdata);

        std::unique_lock<std::mutex> lock(data->mutex);

        // Accumulate the error code.
        if (!data->errorCode) {
            data->errorCode = errorCode;
        }

        // Check if all callbacks have been received.
        if (--data->refCount == 0) {
            // Last solve callback received, call the user callback.
            HostResults results;
            results.trackIndex = data->trackIndex;
            results.timeStampCurrentFrame = data->timeStampCurrentFrame;
            results.timeStampNextFrame = data->timeStampNextFrame;
            results.weights = data->results;
            data->resultsCallback(data->resultsUserdata, results, data->errorCode);

            // Unlock before deleting.
            lock.unlock();
            delete data;
        }
    };

    // The blendshape solve CPU readback enqueued on the same stream as the one provided the results.
    // Note that SolveAsync is actually blocking in the case of the CPU solver in the sense
    // that it needs to block until the previous frame is done reading / executing.
    auto wrapperEnqueueSolve = [&]() -> std::error_code {
        const std::size_t skinWeightCount = skinSolver ? skinSolver->NumBlendshapePoses() : 0;
        const std::size_t tongueWeightCount = tongueSolver ? tongueSolver->NumBlendshapePoses() : 0;

        const auto totalWeightCount = skinWeightCount + tongueWeightCount;
        if (totalWeightCount == 0) {
            // Nothing to solve, call with empty results.
            HostResults hostResults;
            hostResults.trackIndex = results.trackIndex;
            hostResults.timeStampCurrentFrame = results.timeStampCurrentFrame;
            hostResults.timeStampNextFrame = results.timeStampNextFrame;
            resultsCallback(resultsUserdata, hostResults, nva2x::ErrorCode::eSuccess);
            return nva2x::ErrorCode::eSuccess;
        }

        auto callbackData = std::make_unique<BlendshapeSolverCallbackData>();
        callbackData->resultsCallback = resultsCallback;
        callbackData->resultsUserdata = resultsUserdata;
        callbackData->trackIndex = results.trackIndex;
        callbackData->timeStampCurrentFrame = results.timeStampCurrentFrame;
        callbackData->timeStampNextFrame = results.timeStampNextFrame;
        A2F_CHECK_RESULT_WITH_MSG(
            callbackData->results.Allocate(totalWeightCount),
            "Unable to allocate results buffer"
            );

        // Lock to make sure we are done enqueuing the solves to properly
        // handle error cases.
        std::unique_lock<std::mutex> lock(callbackData->mutex);
        callbackData->errorCode = nva2x::ErrorCode::eSuccess;
        callbackData->refCount = 0;

        const auto skinWeights = callbackData->results.View(0, skinWeightCount);
        if (results.skinGeometry.Size() > 0) {
            if (skinSolver) {
                NVTX_TRACE("SkinSolveAsyncHost");
                const auto errorCode = skinSolver->SolveAsync(
                    results.skinGeometry,
                    skinWeights,
                    blendshapeSolverCallback,
                    callbackData.get()
                    );
                if (errorCode) {
                    // Unable to launch solve, callback will not be called.
                    if (!callbackData->errorCode) {
                        callbackData->errorCode = errorCode;
                    }
                }
                else {
                    ++callbackData->refCount;
                }
            }
            else {
                assert(skinWeights.Size() == 0);
            }
        }
        else {
            A2F_CHECK_RESULT_WITH_MSG(
                nva2x::FillOnHost(skinWeights, 0.0f),
                "Unable to zero skin results buffer"
                );
        }

        const auto tongueWeights = callbackData->results.View(skinWeightCount, tongueWeightCount);
        if (results.tongueGeometry.Size() > 0) {
            if (tongueSolver) {
                NVTX_TRACE("TongueSolveAsyncHost");
                const auto errorCode = tongueSolver->SolveAsync(
                    results.tongueGeometry,
                    tongueWeights,
                    blendshapeSolverCallback,
                    callbackData.get()
                );
                if (errorCode) {
                    // Unable to launch solve, callback will not be called.
                    if (!callbackData->errorCode) {
                        callbackData->errorCode = errorCode;
                    }
                }
                else {
                    ++callbackData->refCount;
                }
            }
            else {
                assert(tongueWeights.Size() == 0);
            }
        }
        else {
            A2F_CHECK_RESULT_WITH_MSG(
                nva2x::FillOnHost(tongueWeights, 0.0f),
                "Unable to zero tongue results buffer"
                );
        }

        if (callbackData->refCount == 0) {
            // No solve launched, return the error right away, if any.
            A2F_CHECK_RESULT_WITH_MSG(
                callbackData->errorCode,
                "Unable to run launch host blendshape solve on skin or tongue"
                );
        }
        else {
            callbackData.release();
        }

        // At least one solve was launched, return success.
        // If there was an error, it will be returned by the callback.
        return nva2x::ErrorCode::eSuccess;
    };

    if (auto enqueueErrorCode = wrapperEnqueueSolve()) {
        // If the wrapper did not return an error, the callback will eventually be called.
        // Otherwise, the callback will not be called, so call it with an error.
        HostResults errorResults;
        errorResults.timeStampCurrentFrame = results.timeStampCurrentFrame;
        errorResults.timeStampNextFrame = results.timeStampNextFrame;
        resultsCallback(resultsUserdata, errorResults, enqueueErrorCode);
        return false;
    }

    return true;
}

bool HostBlendshapeSolveExecutor::callbackForGeometryExecutor(
    void* userdata, const IGeometryExecutor::Results& results
    ) {
    assert(userdata);
    auto& executor = *static_cast<HostBlendshapeSolveExecutor*>(userdata);
    assert(executor.GetResultType() == ResultsType::HOST);

    return callbackHelperForGeometryExecutor(
        executor._trackData[results.trackIndex].skinSolver.get(),
        executor._trackData[results.trackIndex].tongueSolver.get(),
        executor._geometryResultsCallback,
        executor._geometryResultsUserdata,
        executor._resultsCallback,
        executor._resultsUserdata,
        results
        );
}


DeviceBlendshapeSolveExecutor::ResultsType DeviceBlendshapeSolveExecutor::GetResultType() const {
    return ResultsType::DEVICE;
}

std::error_code DeviceBlendshapeSolveExecutor::Execute(std::size_t* pNbExecutedTracks) {
    NVTX_TRACE("IBlendshapeSolveExecutor::Execute (device)");

    A2F_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);

    A2F_CHECK_ERROR_WITH_MSG(_geometryExecutor, "Geometry executor cannot be null", nva2x::ErrorCode::eNullPointer);
    return _geometryExecutor->Execute(pNbExecutedTracks);
}

std::error_code DeviceBlendshapeSolveExecutor::SetResultsCallback(
    device_results_callback_t callback, void* userdata
    ) {
    A2F_CHECK_ERROR_WITH_MSG(
        !nva2x::HasExecutionStarted(*this),
        "Results callback can only be set before execution is started",
        nva2x::ErrorCode::eExecutionAlreadyStarted
        );
    A2F_CHECK_ERROR_WITH_MSG(callback || !userdata, "User data must be null if device callback is null", nva2x::ErrorCode::eNullPointer);
    _resultsCallback = callback;
    _resultsUserdata = userdata;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code DeviceBlendshapeSolveExecutor::Init(
    nva2x::UniquePtr<IGeometryExecutor> transferredGeometryExecutor,
    const DeviceBlendshapeSolveExecutorCreationParameters& params
    ) {
    A2F_CHECK_RESULT(
        BlendshapeSolveExecutor::Init(std::move(transferredGeometryExecutor), params, true)
        );

    A2F_CHECK_RESULT_WITH_MSG(
        _geometryExecutor->SetResultsCallback(callbackForGeometryExecutor, this),
        "Unable to set geometry executor results callaback"
        );

    A2F_CHECK_RESULT_WITH_MSG(
        _weights.Allocate(GetWeightCount() * _trackData.size()),
        "Unable to allocate weights buffer"
        );

    return nva2x::ErrorCode::eSuccess;
}

bool DeviceBlendshapeSolveExecutor::callbackHelperForGeometryExecutor(
    IBlendshapeSolver* skinSolver,
    IBlendshapeSolver* tongueSolver,
    IGeometryExecutor::results_callback_t geometryCallback,
    void* geometryUserdata,
    device_results_callback_t resultsCallback,
    void* resultsUserdata,
    nva2x::DeviceTensorFloatView weights,
    const IGeometryExecutor::Results& results
    ) {
    if (geometryCallback) {
        geometryCallback(geometryUserdata, results);
    }

    const std::size_t skinWeightCount = skinSolver ? skinSolver->NumBlendshapePoses() : 0;
    const std::size_t tongueWeightCount = tongueSolver ? tongueSolver->NumBlendshapePoses() : 0;

    const auto totalWeightCount = skinWeightCount + tongueWeightCount;
    assert(totalWeightCount == weights.Size());
    if (totalWeightCount == 0) {
        // Nothing to solve, call with empty results.
        DeviceResults deviceResults;
        deviceResults.trackIndex = results.trackIndex;
        deviceResults.timeStampCurrentFrame = results.timeStampCurrentFrame;
        deviceResults.timeStampNextFrame = results.timeStampNextFrame;
        return resultsCallback(resultsUserdata, deviceResults);
    }

    DeviceResults deviceResults;
    deviceResults.trackIndex = results.trackIndex;
    deviceResults.timeStampCurrentFrame = results.timeStampCurrentFrame;
    deviceResults.timeStampNextFrame = results.timeStampNextFrame;
    // Since blendshape results are returned in a single array with a single stream as a result,
    // we assume that skin and tongue are using the same stream, otherwise extra synchronization
    // would need to be added.
    deviceResults.cudaStream = results.skinCudaStream ? results.skinCudaStream : results.tongueCudaStream;
    const auto trackWeights = weights;
    deviceResults.weights = trackWeights;

    auto wrapperEnqueueSolve = [&]() -> std::error_code {
        const auto skinWeights = trackWeights.View(0, skinWeightCount);
        if (skinSolver && results.skinGeometry.Size() > 0) {
            NVTX_TRACE("SkinSolveAsyncDevice");
            A2F_CHECK_RESULT_WITH_MSG(
                skinSolver->SetCudaStream(results.skinCudaStream),
                "Unable to set CUDA stream on skin blendshape solver"
                );
            A2F_CHECK_RESULT_WITH_MSG(
                skinSolver->SolveAsync(results.skinGeometry, skinWeights),
                "Unable to run blendshape solve on skin"
                );
            if (results.skinCudaStream != deviceResults.cudaStream) {
                NVTX_TRACE("SkinSolveSyncDevice");
                assert(!"Implement better stream synchronization");
                A2F_CUDA_CHECK_ERROR(
                    cudaStreamSynchronize(results.skinCudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError
                    );
            }
        }
        else {
            A2F_CHECK_RESULT_WITH_MSG(
                nva2x::FillOnDevice(skinWeights, 0.0f, deviceResults.cudaStream),
                "Unable to zero skin results buffer"
                );
        }

        const auto tongueWeights = trackWeights.View(skinWeightCount, tongueWeightCount);
        if (tongueSolver && results.tongueGeometry.Size() > 0) {
            NVTX_TRACE("TongueSolveAsyncDevice");
            A2F_CHECK_RESULT_WITH_MSG(
                tongueSolver->SetCudaStream(results.tongueCudaStream),
                "Unable to set CUDA stream on tongue blendshape solver"
                );
            A2F_CHECK_RESULT_WITH_MSG(
                tongueSolver->SolveAsync(results.tongueGeometry, tongueWeights),
                "Unable to run blendshape solve on tongue"
                );
            if (results.tongueCudaStream != deviceResults.cudaStream) {
                NVTX_TRACE("TongueSolveSyncDevice");
                assert(!"Implement better stream synchronization");
                A2F_CUDA_CHECK_ERROR(
                    cudaStreamSynchronize(results.tongueCudaStream), nva2x::ErrorCode::eCudaStreamSynchronizeError
                    );
            }
        }
        else {
            A2F_CHECK_RESULT_WITH_MSG(
                nva2x::FillOnDevice(tongueWeights, 0.0f, deviceResults.cudaStream),
                "Unable to zero tongue results buffer"
                );
        }

        return nva2x::ErrorCode::eSuccess;
    };

    if (auto enqueueErrorCode = wrapperEnqueueSolve()) {
        // We don't really have a mechanism to report the error other than logging it.
        A2F_CHECK_ERROR_WITH_MSG(!enqueueErrorCode, "Unable to enqueue device blendshape solve", false);
    }

    return resultsCallback(resultsUserdata, deviceResults);
}

bool DeviceBlendshapeSolveExecutor::callbackForGeometryExecutor(
    void* userdata, const IGeometryExecutor::Results& results
    ) {
    assert(userdata);
    auto& executor = *static_cast<DeviceBlendshapeSolveExecutor*>(userdata);
    assert(executor.GetResultType() == ResultsType::DEVICE);

    const auto totalWeightCount = executor.GetWeightCount();
    const auto weights = executor._weights.View(results.trackIndex * totalWeightCount, totalWeightCount);

    return callbackHelperForGeometryExecutor(
        executor._trackData[results.trackIndex].skinSolver.get(),
        executor._trackData[results.trackIndex].tongueSolver.get(),
        executor._geometryResultsCallback,
        executor._geometryResultsUserdata,
        executor._resultsCallback,
        executor._resultsUserdata,
        weights,
        results
        );
}

} // namespace nva2f

nva2f::IBlendshapeExecutor* nva2f::CreateHostBlendshapeSolveExecutor_INTERNAL(
    IGeometryExecutor* transferredGeometryExecutor,
    const HostBlendshapeSolveExecutorCreationParameters& params
    ) {
  LOG_DEBUG("CreateHostBlendshapeSolveExecutor()");
  auto geometryExecutor = nva2x::ToUniquePtr(transferredGeometryExecutor);
  auto executor = std::make_unique<HostBlendshapeSolveExecutor>();
  if (executor->Init(std::move(geometryExecutor), params)) {
    LOG_ERROR("Unable to create blendshape solve executor");
    return nullptr;
  }
  return executor.release();
}

nva2f::IBlendshapeExecutor* nva2f::CreateDeviceBlendshapeSolveExecutor_INTERNAL(
    IGeometryExecutor* transferredGeometryExecutor,
    const DeviceBlendshapeSolveExecutorCreationParameters& params
    ) {
  LOG_DEBUG("CreateDeviceBlendshapeSolveExecutor()");
  auto geometryExecutor = nva2x::ToUniquePtr(transferredGeometryExecutor);
  auto executor = std::make_unique<DeviceBlendshapeSolveExecutor>();
  if (executor->Init(std::move(geometryExecutor), params)) {
    LOG_ERROR("Unable to create blendshape solve executor");
    return nullptr;
  }
  return executor.release();
}
