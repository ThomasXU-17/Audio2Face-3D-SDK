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
#include "audio2face/internal/interactive_executor_blendshapesolve.h"
#include "audio2face/internal/blendshape_solver_base.h"
#include "audio2face/internal/executor_blendshapesolve.h"
#include "audio2face/internal/job_runner.h"
#include "audio2face/internal/macros.h"
#include "audio2face/internal/logger.h"
#include "audio2x/error.h"

namespace {

    // This class disables smoothing of the weights so that all evaluations
    // don't require state from previous frames and therefore are only a
    // function of the current frame.
    class StatelessScope {
    public:
        StatelessScope(nva2f::IBlendshapeSolver* solver) : _solver(solver) {
            if (_solver) {
                const auto params = _solver->GetParameters();
                _params = params;

                auto statelessParams = params;
                statelessParams.TemporalReg = 0.0f;
                [[maybe_unused]] const auto result = _solver->SetParameters(statelessParams);
                assert(!result);
            }
        }

        ~StatelessScope() {
            if (_solver) {
                [[maybe_unused]] const auto result = _solver->SetParameters(_params);
                assert(!result);
            }
        }

    private:
        nva2f::IBlendshapeSolver* _solver;
        nva2f::BlendshapeSolverParams _params;
    };
}

namespace nva2f {

std::error_code BlendshapeSolveInteractiveExecutor::Invalidate(invalidation_layer_t layer) {
    // Forward to the geometry executor.
    switch (layer) {
        case IGeometryInteractiveExecutor::kLayerNone: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerAll: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerInference: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerSkin: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerTongue: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerTeeth: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerEyes:
            A2F_CHECK_RESULT(_geometryInteractiveExecutor->Invalidate(layer));
    }

    switch (layer) {
        case IGeometryInteractiveExecutor::kLayerNone:
            break;
        case IGeometryInteractiveExecutor::kLayerAll:
            _skinSolverPreparedValid = false;
            _tongueSolverPreparedValid = false;
            _weightsResultsValid = false;
            break;
        case kLayerSkinSolverPrepare:
            _skinSolverPreparedValid = false;
            _weightsResultsValid = false;
            break;
        case kLayerTongueSolverPrepare:
            _tongueSolverPreparedValid = false;
            _weightsResultsValid = false;
            break;
        case IGeometryInteractiveExecutor::kLayerInference: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerSkin: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerTongue: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerTeeth: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerEyes: [[fallthrough]];
        case kLayerBlendshapeWeights:
            _weightsResultsValid = false;
            break;
        default:
            return nva2x::ErrorCode::eInvalidValue;
    }

    return nva2x::ErrorCode::eSuccess;
}

bool BlendshapeSolveInteractiveExecutor::IsValid(invalidation_layer_t layer) const {
    switch (layer) {
        case IGeometryInteractiveExecutor::kLayerNone:
            return true;
        case IGeometryInteractiveExecutor::kLayerAll:
            return _geometryInteractiveExecutor->IsValid(layer)
                && _skinSolverPreparedValid
                && _tongueSolverPreparedValid
                && _weightsResultsValid;
        case kLayerSkinSolverPrepare:
            return _skinSolverPreparedValid;
        case kLayerTongueSolverPrepare:
            return _tongueSolverPreparedValid;
        case kLayerBlendshapeWeights:
            return _weightsResultsValid;
        case IGeometryInteractiveExecutor::kLayerInference: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerSkin: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerTongue: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerTeeth: [[fallthrough]];
        case IGeometryInteractiveExecutor::kLayerEyes:
            return _geometryInteractiveExecutor->IsValid(layer);
        default:
            return false;
    }
}

void BlendshapeSolveInteractiveExecutor::Destroy() {
    delete this;
}

std::size_t BlendshapeSolveInteractiveExecutor::GetTotalNbFrames() const {
    return _geometryInteractiveExecutor->GetTotalNbFrames();
}

std::size_t BlendshapeSolveInteractiveExecutor::GetSamplingRate() const {
    return _geometryInteractiveExecutor->GetSamplingRate();
}

void BlendshapeSolveInteractiveExecutor::GetFrameRate(std::size_t& numerator, std::size_t& denominator) const {
    return _geometryInteractiveExecutor->GetFrameRate(numerator, denominator);
}

BlendshapeSolveInteractiveExecutor::timestamp_t BlendshapeSolveInteractiveExecutor::GetFrameTimestamp(std::size_t frameIndex) const {
    return _geometryInteractiveExecutor->GetFrameTimestamp(frameIndex);
}

std::error_code BlendshapeSolveInteractiveExecutor::ComputeFrame(std::size_t frameIndex) {
    // Changing the parameters of the skin invalidates the solver prepare.
    StatelessScope skinStatelessScope(_skinSolver.get());
    _skinSolverPreparedValid = false;

    StatelessScope tongueStatelessScope(_tongueSolver.get());
    _tongueSolverPreparedValid = false;

    if (_skinSolver) {
        A2F_CHECK_RESULT_WITH_MSG(_skinSolver->Prepare(), "Unable to prepare skin solver");
    }
    if (_tongueSolver) {
        A2F_CHECK_RESULT_WITH_MSG(_tongueSolver->Prepare(), "Unable to prepare tongue solver");
    }

    const auto result = _geometryInteractiveExecutor->ComputeFrame(frameIndex);

    if (_skinSolver) {
        A2F_CHECK_RESULT_WITH_MSG(_skinSolver->Wait(), "Unable to wait for skin solver");
    }
    if (_tongueSolver) {
        A2F_CHECK_RESULT_WITH_MSG(_tongueSolver->Wait(), "Unable to wait for tongue solver");
    }

    A2F_CHECK_RESULT_WITH_MSG(result, "Unable to compute frame");

    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveInteractiveExecutor::ComputeAllFrames() {
    if (_skinSolver) {
        A2F_CHECK_RESULT_WITH_MSG(_skinSolver->Reset(), "Unable to reset skin solver");
        if (!_skinSolverPreparedValid) {
            A2F_CHECK_RESULT_WITH_MSG(_skinSolver->Prepare(), "Unable to prepare skin solver");
            _skinSolverPreparedValid = true;
        }
    }
    if (_tongueSolver) {
        A2F_CHECK_RESULT_WITH_MSG(_tongueSolver->Reset(), "Unable to reset tongue solver");
        if (!_tongueSolverPreparedValid) {
            A2F_CHECK_RESULT_WITH_MSG(_tongueSolver->Prepare(), "Unable to prepare tongue solver");
            _tongueSolverPreparedValid = true;
        }
    }

    const auto result = _geometryInteractiveExecutor->ComputeAllFrames();

    if (_skinSolver) {
        A2F_CHECK_RESULT_WITH_MSG(_skinSolver->Wait(), "Unable to wait for skin solver");
    }
    if (_tongueSolver) {
        A2F_CHECK_RESULT_WITH_MSG(_tongueSolver->Wait(), "Unable to wait for tongue solver");
    }

    A2F_CHECK_RESULT_WITH_MSG(result, "Unable to compute all frames");

    _weightsResultsValid = true;

    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveInteractiveExecutor::Interrupt() {
    return _geometryInteractiveExecutor->Interrupt();
}

std::size_t BlendshapeSolveInteractiveExecutor::GetWeightCount() const {
    std::size_t weightCount = 0;
    if (_skinSolver) {
        weightCount += _skinSolver->NumBlendshapePoses();
    }
    if (_tongueSolver) {
        weightCount += _tongueSolver->NumBlendshapePoses();
    }
    return weightCount;
}

std::error_code BlendshapeSolveInteractiveExecutor::SetResultsCallback(host_results_callback_t, void*) {
    return nva2x::ErrorCode::eUnsupported;
}

std::error_code BlendshapeSolveInteractiveExecutor::SetResultsCallback(device_results_callback_t, void*) {
    return nva2x::ErrorCode::eUnsupported;
}

std::error_code BlendshapeSolveInteractiveExecutor::GetInputStrength(float& inputStrength) const {
    return GetInteractiveExecutorInputStrength_INTERNAL(*_geometryInteractiveExecutor, inputStrength);
}

std::error_code BlendshapeSolveInteractiveExecutor::SetInputStrength(float inputStrength) {
    const auto result = SetInteractiveExecutorInputStrength_INTERNAL(*_geometryInteractiveExecutor, inputStrength);
    if (!_geometryInteractiveExecutor->IsValid(kLayerAll)) {
        _weightsResultsValid = false;
    }
    return result;
}

std::error_code BlendshapeSolveInteractiveExecutor::Get(AnimatorSkinParams& params) const {
    return GetInteractiveExecutorSkinParameters_INTERNAL(*_geometryInteractiveExecutor, params);
}

std::error_code BlendshapeSolveInteractiveExecutor::Set(const AnimatorSkinParams& params) {
    const auto result = SetInteractiveExecutorSkinParameters_INTERNAL(*_geometryInteractiveExecutor, params);
    if (!_geometryInteractiveExecutor->IsValid(kLayerAll)) {
        _weightsResultsValid = false;
    }
    return result;
}

std::error_code BlendshapeSolveInteractiveExecutor::Get(AnimatorTongueParams& params) const {
    return GetInteractiveExecutorTongueParameters_INTERNAL(*_geometryInteractiveExecutor, params);
}

std::error_code BlendshapeSolveInteractiveExecutor::Set(const AnimatorTongueParams& params) {
    const auto result = SetInteractiveExecutorTongueParameters_INTERNAL(*_geometryInteractiveExecutor, params);
    if (!_geometryInteractiveExecutor->IsValid(kLayerAll)) {
        _weightsResultsValid = false;
    }
    return result;
}

std::error_code BlendshapeSolveInteractiveExecutor::Get(AnimatorTeethParams& params) const {
    return GetInteractiveExecutorTeethParameters_INTERNAL(*_geometryInteractiveExecutor, params);
}

std::error_code BlendshapeSolveInteractiveExecutor::Set(const AnimatorTeethParams& params) {
    const auto result = SetInteractiveExecutorTeethParameters_INTERNAL(*_geometryInteractiveExecutor, params);
    if (!_geometryInteractiveExecutor->IsValid(kLayerAll)) {
        _weightsResultsValid = false;
    }
    return result;
}

std::error_code BlendshapeSolveInteractiveExecutor::Get(AnimatorEyesParams& params) const {
    return GetInteractiveExecutorEyesParameters_INTERNAL(*_geometryInteractiveExecutor, params);
}

std::error_code BlendshapeSolveInteractiveExecutor::Set(const AnimatorEyesParams& params) {
    const auto result = SetInteractiveExecutorEyesParameters_INTERNAL(*_geometryInteractiveExecutor, params);
    if (!_geometryInteractiveExecutor->IsValid(kLayerAll)) {
        _weightsResultsValid = false;
    }
    return result;
}

std::error_code BlendshapeSolveInteractiveExecutor::GetGeometryInteractiveExecutor(
    IGeometryInteractiveExecutor** geometryInteractiveExecutor
    ) const {
    A2F_CHECK_ERROR_WITH_MSG(geometryInteractiveExecutor, "Geometry executor cannot be null", nva2x::ErrorCode::eNullPointer);
    *geometryInteractiveExecutor = _geometryInteractiveExecutor.get();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveInteractiveExecutor::SetGeometryResultsCallback(
    IGeometryInteractiveExecutor::results_callback_t callback, void* userdata
    ) {
    _geometryResultsCallback = callback;
    _geometryResultsUserdata = userdata;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveInteractiveExecutor::SetSkinConfig(const BlendshapeSolverConfig& config) {
    A2F_CHECK_ERROR_WITH_MSG(_skinSolver, "Skin solver cannot be null", nva2x::ErrorCode::eNullPointer);
    const auto skinConfig = _skinSolver->GetBlendshapeConfig();
    if (!AreEqual_INTERNAL(skinConfig, config)) {
        A2F_CHECK_RESULT_WITH_MSG(Invalidate(kLayerSkinSolverPrepare), "Unable to invalidate skin solver prepare");
        return _skinSolver->SetBlendshapeConfig(config);
    }
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveInteractiveExecutor::GetSkinConfig(BlendshapeSolverConfig& config) const {
    A2F_CHECK_ERROR_WITH_MSG(_skinSolver, "Skin solver cannot be null", nva2x::ErrorCode::eNullPointer);
    config = _skinSolver->GetBlendshapeConfig();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveInteractiveExecutor::SetSkinParameters(const BlendshapeSolverParams& params) {
    A2F_CHECK_ERROR_WITH_MSG(_skinSolver, "Skin solver cannot be null", nva2x::ErrorCode::eNullPointer);
    const auto& skinParams = _skinSolver->GetParameters();
    if (!AreEqual_INTERNAL(skinParams, params)) {
        A2F_CHECK_RESULT_WITH_MSG(Invalidate(kLayerSkinSolverPrepare), "Unable to invalidate blendshape weights");
        return _skinSolver->SetParameters(params);
    }
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveInteractiveExecutor::GetSkinParameters(BlendshapeSolverParams& params) const {
    A2F_CHECK_ERROR_WITH_MSG(_skinSolver, "Skin solver cannot be null", nva2x::ErrorCode::eNullPointer);
    params = _skinSolver->GetParameters();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveInteractiveExecutor::SetTongueConfig(const BlendshapeSolverConfig& config) {
    A2F_CHECK_ERROR_WITH_MSG(_tongueSolver, "Tongue solver cannot be null", nva2x::ErrorCode::eNullPointer);
    const auto tongueConfig = _tongueSolver->GetBlendshapeConfig();
    if (!AreEqual_INTERNAL(tongueConfig, config)) {
        A2F_CHECK_RESULT_WITH_MSG(Invalidate(kLayerTongueSolverPrepare), "Unable to invalidate tongue solver prepare");
        return _tongueSolver->SetBlendshapeConfig(config);
    }
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveInteractiveExecutor::GetTongueConfig(BlendshapeSolverConfig& config) const {
    A2F_CHECK_ERROR_WITH_MSG(_tongueSolver, "Tongue solver cannot be null", nva2x::ErrorCode::eNullPointer);
    config = _tongueSolver->GetBlendshapeConfig();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveInteractiveExecutor::SetTongueParameters(const BlendshapeSolverParams& params) {
    A2F_CHECK_ERROR_WITH_MSG(_tongueSolver, "Tongue solver cannot be null", nva2x::ErrorCode::eNullPointer);
    const auto& tongueParams = _tongueSolver->GetParameters();
    if (!AreEqual_INTERNAL(tongueParams, params)) {
        A2F_CHECK_RESULT_WITH_MSG(Invalidate(kLayerTongueSolverPrepare), "Unable to invalidate blendshape weights");
        return _tongueSolver->SetParameters(params);
    }
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveInteractiveExecutor::GetTongueParameters(BlendshapeSolverParams& params) const {
    A2F_CHECK_ERROR_WITH_MSG(_tongueSolver, "Tongue solver cannot be null", nva2x::ErrorCode::eNullPointer);
    params = _tongueSolver->GetParameters();
    return nva2x::ErrorCode::eSuccess;
}

std::error_code BlendshapeSolveInteractiveExecutor::Init(
    nva2x::UniquePtr<IGeometryInteractiveExecutor> transferredGeometryInteractiveExecutor,
    const BlendshapeSolveExecutorCreationParameters& params,
    bool useGpu
    ) {
    A2F_CHECK_ERROR_WITH_MSG(transferredGeometryInteractiveExecutor, "Geometry interactive executor cannot be null", nva2x::ErrorCode::eNullPointer);

    _geometryInteractiveExecutor = std::move(transferredGeometryInteractiveExecutor);

    if (params.initializationSkinParams) {
        auto& skinSolver = _skinSolver;
        skinSolver.reset(nva2f::CreateBlendshapeSolver_INTERNAL(useGpu));
        A2F_CHECK_ERROR_WITH_MSG(skinSolver, "Unable to load skin blendshape solver", nva2x::ErrorCode::eNullPointer);
        A2F_CHECK_RESULT_WITH_MSG(skinSolver->SetBlendshapeData(params.initializationSkinParams->data), "Unable to set blendshape data for skin");
        A2F_CHECK_RESULT_WITH_MSG(skinSolver->SetParameters(params.initializationSkinParams->params), "Unable to set parameters for skin");
        A2F_CHECK_RESULT_WITH_MSG(skinSolver->SetBlendshapeConfig(params.initializationSkinParams->config), "Unable to set config for skin");
        A2F_CHECK_RESULT_WITH_MSG(skinSolver->Prepare(), "Unable to prepare skin blendshape solver");
        _skinSolverPreparedValid = true;
    }

    if (params.initializationTongueParams) {
        auto& tongueSolver = _tongueSolver;
        tongueSolver.reset(nva2f::CreateBlendshapeSolver_INTERNAL(useGpu));
        A2F_CHECK_ERROR_WITH_MSG(tongueSolver, "Unable to load tongue blendshape solver", nva2x::ErrorCode::eNullPointer);
        A2F_CHECK_RESULT_WITH_MSG(tongueSolver->SetBlendshapeData(params.initializationTongueParams->data), "Unable to set blendshape data for tongue");
        A2F_CHECK_RESULT_WITH_MSG(tongueSolver->SetParameters(params.initializationTongueParams->params), "Unable to set parameters for tongue");
        A2F_CHECK_RESULT_WITH_MSG(tongueSolver->SetBlendshapeConfig(params.initializationTongueParams->config), "Unable to set config for tongue");
        A2F_CHECK_RESULT_WITH_MSG(tongueSolver->Prepare(), "Unable to prepare tongue blendshape solver");
        _tongueSolverPreparedValid = true;
    }

    return nva2x::ErrorCode::eSuccess;
}


HostBlendshapeSolveInteractiveExecutor::ResultsType HostBlendshapeSolveInteractiveExecutor::GetResultType() const {
    return ResultsType::HOST;
}

std::error_code HostBlendshapeSolveInteractiveExecutor::ComputeFrame(std::size_t frameIndex) {
    A2F_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);
    return BlendshapeSolveInteractiveExecutor::ComputeFrame(frameIndex);
}

std::error_code HostBlendshapeSolveInteractiveExecutor::ComputeAllFrames() {
    A2F_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);
    return BlendshapeSolveInteractiveExecutor::ComputeAllFrames();
}

std::error_code HostBlendshapeSolveInteractiveExecutor::SetResultsCallback(
    host_results_callback_t callback, void* userdata
    ) {
    A2F_CHECK_ERROR_WITH_MSG(callback || !userdata, "User data must be null if host callback is null", nva2x::ErrorCode::eNullPointer);
    _resultsCallback = callback;
    _resultsUserdata = userdata;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code HostBlendshapeSolveInteractiveExecutor::Init(
    nva2x::UniquePtr<IGeometryInteractiveExecutor> transferredGeometryInteractiveExecutor,
    const HostBlendshapeSolveExecutorCreationParameters& params
    ) {
    A2F_CHECK_RESULT(
        BlendshapeSolveInteractiveExecutor::Init(std::move(transferredGeometryInteractiveExecutor), params, false)
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

        // Get hardware concurrency and limit thread count
        const std::size_t maxHardwareThreads = std::thread::hardware_concurrency();
        if (maxHardwareThreads != 0) {
            nbThreads = std::min(nbThreads, maxHardwareThreads);
        }

        _jobRunner.reset(nva2f::CreateThreadPoolJobRunner_INTERNAL(nbThreads));
        jobRunner = _jobRunner.get();
    }

    if (_skinSolver) {
        A2F_CHECK_RESULT_WITH_MSG(_skinSolver->SetJobRunner(jobRunner), "Unable to set job runner for skin");
    }
    if (_tongueSolver) {
        A2F_CHECK_RESULT_WITH_MSG(_tongueSolver->SetJobRunner(jobRunner), "Unable to set job runner for tongue");
    }

    A2F_CHECK_RESULT_WITH_MSG(
        _geometryInteractiveExecutor->SetResultsCallback(callbackForGeometryInteractiveExecutor, this),
        "Unable to set geometry interactive executor results callaback"
        );

    return nva2x::ErrorCode::eSuccess;
}

bool HostBlendshapeSolveInteractiveExecutor::callbackForGeometryInteractiveExecutor(
    void* userdata, const IGeometryInteractiveExecutor::Results& results
    ) {
    assert(userdata);
    auto& executor = *static_cast<HostBlendshapeSolveInteractiveExecutor*>(userdata);
    assert(executor.GetResultType() == ResultsType::HOST);

    return HostBlendshapeSolveExecutor::callbackHelperForGeometryExecutor(
        executor._skinSolver.get(),
        executor._tongueSolver.get(),
        executor._geometryResultsCallback,
        executor._geometryResultsUserdata,
        executor._resultsCallback,
        executor._resultsUserdata,
        results
        );
}


DeviceBlendshapeSolveInteractiveExecutor::ResultsType DeviceBlendshapeSolveInteractiveExecutor::GetResultType() const {
    return ResultsType::DEVICE;
}

std::error_code DeviceBlendshapeSolveInteractiveExecutor::ComputeFrame(std::size_t frameIndex) {
    A2F_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);
    return BlendshapeSolveInteractiveExecutor::ComputeFrame(frameIndex);
}

std::error_code DeviceBlendshapeSolveInteractiveExecutor::ComputeAllFrames() {
    A2F_CHECK_ERROR_WITH_MSG(_resultsCallback, "Results callback cannot be null", nva2x::ErrorCode::eNullPointer);
    return BlendshapeSolveInteractiveExecutor::ComputeAllFrames();
}

std::error_code DeviceBlendshapeSolveInteractiveExecutor::SetResultsCallback(
    device_results_callback_t callback, void* userdata
    ) {
    A2F_CHECK_ERROR_WITH_MSG(callback || !userdata, "User data must be null if device callback is null", nva2x::ErrorCode::eNullPointer);
    _resultsCallback = callback;
    _resultsUserdata = userdata;
    return nva2x::ErrorCode::eSuccess;
}

std::error_code DeviceBlendshapeSolveInteractiveExecutor::Init(
    nva2x::UniquePtr<IGeometryInteractiveExecutor> transferredGeometryInteractiveExecutor,
    const DeviceBlendshapeSolveExecutorCreationParameters& params
    ) {
    A2F_CHECK_RESULT(
        BlendshapeSolveInteractiveExecutor::Init(std::move(transferredGeometryInteractiveExecutor), params, true)
        );

    A2F_CHECK_RESULT_WITH_MSG(
        _geometryInteractiveExecutor->SetResultsCallback(callbackForGeometryInteractiveExecutor, this),
        "Unable to set geometry interactive executor results callaback"
        );

    A2F_CHECK_RESULT_WITH_MSG(
        _weights.Allocate(GetWeightCount()),
        "Unable to allocate weights buffer"
        );

    return nva2x::ErrorCode::eSuccess;
}

bool DeviceBlendshapeSolveInteractiveExecutor::callbackForGeometryInteractiveExecutor(
    void* userdata, const IGeometryInteractiveExecutor::Results& results
    ) {
    assert(userdata);
    auto& executor = *static_cast<DeviceBlendshapeSolveInteractiveExecutor*>(userdata);
    assert(executor.GetResultType() == ResultsType::DEVICE);

    const auto totalWeightCount = executor.GetWeightCount();
    const auto weights = executor._weights.View(results.trackIndex * totalWeightCount, totalWeightCount);

    return DeviceBlendshapeSolveExecutor::callbackHelperForGeometryExecutor(
        executor._skinSolver.get(),
        executor._tongueSolver.get(),
        executor._geometryResultsCallback,
        executor._geometryResultsUserdata,
        executor._resultsCallback,
        executor._resultsUserdata,
        weights,
        results
        );
}

} // namespace nva2f

nva2f::IBlendshapeInteractiveExecutor* nva2f::CreateHostBlendshapeSolveInteractiveExecutor_INTERNAL(
    IGeometryInteractiveExecutor* transferredGeometryExecutor,
    const HostBlendshapeSolveExecutorCreationParameters& params
    ) {
  LOG_DEBUG("CreateHostBlendshapeSolveInteractiveExecutor()");
  auto geometryExecutor = nva2x::ToUniquePtr(transferredGeometryExecutor);
  auto executor = std::make_unique<HostBlendshapeSolveInteractiveExecutor>();
  if (executor->Init(std::move(geometryExecutor), params)) {
    LOG_ERROR("Unable to create blendshape solve interactive executor");
    return nullptr;
  }
  return executor.release();
}

nva2f::IBlendshapeInteractiveExecutor* nva2f::CreateDeviceBlendshapeSolveInteractiveExecutor_INTERNAL(
    IGeometryInteractiveExecutor* transferredGeometryExecutor,
    const DeviceBlendshapeSolveExecutorCreationParameters& params
    ) {
  LOG_DEBUG("CreateDeviceBlendshapeSolveInteractiveExecutor()");
  auto geometryExecutor = nva2x::ToUniquePtr(transferredGeometryExecutor);
  auto executor = std::make_unique<DeviceBlendshapeSolveInteractiveExecutor>();
  if (executor->Init(std::move(geometryExecutor), params)) {
    LOG_ERROR("Unable to create blendshape solve interactive executor");
    return nullptr;
  }
  return executor.release();
}
