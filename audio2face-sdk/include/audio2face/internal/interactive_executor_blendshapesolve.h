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

#include "audio2face/executor_blendshapesolve.h"
#include "audio2face/blendshape_solver.h"
#include "audio2face/internal/interactive_executor.h"
#include "audio2x/internal/unique_ptr.h"

namespace nva2f {

class BlendshapeSolveInteractiveExecutor : public IBlendshapeInteractiveExecutor
                                         , public IFaceInteractiveExecutorAccessorInputStrength
                                         , public IFaceInteractiveExecutorAccessorSkinParameters
                                         , public IFaceInteractiveExecutorAccessorTongueParameters
                                         , public IFaceInteractiveExecutorAccessorTeethParameters
                                         , public IFaceInteractiveExecutorAccessorEyesParameters
                                         , public IFaceInteractiveExecutorAccessorGeometryInteractiveExecutor
                                         , public IFaceInteractiveExecutorAccessorGeometryResultsCallback
                                         , public IFaceInteractiveExecutorAccessorBlendshapeParameters {
public:
    std::error_code Invalidate(invalidation_layer_t layer) override;
    bool IsValid(invalidation_layer_t layer) const override;
    void Destroy() override;

    std::size_t GetTotalNbFrames() const override;
    std::size_t GetSamplingRate() const override;
    void GetFrameRate(std::size_t& numerator, std::size_t& denominator) const override;

    timestamp_t GetFrameTimestamp(std::size_t frameIndex) const override;

    std::error_code ComputeFrame(std::size_t frameIndex) override;
    std::error_code ComputeAllFrames() override;

    std::error_code Interrupt() override;

    std::size_t GetWeightCount() const override;

    std::error_code SetResultsCallback(host_results_callback_t callback, void* userdata) override;
    std::error_code SetResultsCallback(device_results_callback_t callback, void* userdata) override;

    std::error_code GetInputStrength(float& inputStrength) const override;
    std::error_code SetInputStrength(float inputStrength) override;
    std::error_code Get(AnimatorSkinParams& params) const override;
    std::error_code Set(const AnimatorSkinParams& params) override;
    std::error_code Get(AnimatorTongueParams& params) const override;
    std::error_code Set(const AnimatorTongueParams& params) override;
    std::error_code Get(AnimatorTeethParams& params) const override;
    std::error_code Set(const AnimatorTeethParams& params) override;
    std::error_code Get(AnimatorEyesParams& params) const override;
    std::error_code Set(const AnimatorEyesParams& params) override;

    std::error_code GetGeometryInteractiveExecutor(IGeometryInteractiveExecutor** geometryInteractiveExecutor) const override;
    std::error_code SetGeometryResultsCallback(IGeometryInteractiveExecutor::results_callback_t callback, void* userdata) override;
    std::error_code SetSkinConfig(const BlendshapeSolverConfig& config) override;
    std::error_code GetSkinConfig(BlendshapeSolverConfig& config) const override;
    std::error_code SetSkinParameters(const BlendshapeSolverParams& params) override;
    std::error_code GetSkinParameters(BlendshapeSolverParams& params) const override;
    std::error_code SetTongueConfig(const BlendshapeSolverConfig& config) override;
    std::error_code GetTongueConfig(BlendshapeSolverConfig& config) const override;
    std::error_code SetTongueParameters(const BlendshapeSolverParams& params) override;
    std::error_code GetTongueParameters(BlendshapeSolverParams& params) const override;

    std::error_code Init(
        nva2x::UniquePtr<IGeometryInteractiveExecutor> transferredGeometryInteractiveExecutor,
        const BlendshapeSolveExecutorCreationParameters& params,
        bool useGpu
        );

protected:
    nva2x::UniquePtr<IGeometryInteractiveExecutor> _geometryInteractiveExecutor;
    IGeometryInteractiveExecutor::results_callback_t _geometryResultsCallback{};
    void *_geometryResultsUserdata{};

    nva2x::UniquePtr<IBlendshapeSolver> _skinSolver;
    nva2x::UniquePtr<IBlendshapeSolver> _tongueSolver;

    bool _skinSolverPreparedValid{false};
    bool _tongueSolverPreparedValid{false};
    bool _weightsResultsValid{false};
};

class HostBlendshapeSolveInteractiveExecutor : public BlendshapeSolveInteractiveExecutor {
public:
    ResultsType GetResultType() const override;

    std::error_code ComputeFrame(std::size_t frameIndex) override;
    std::error_code ComputeAllFrames() override;

    std::error_code SetResultsCallback(host_results_callback_t callback, void* userdata) override;

    std::error_code Init(
        nva2x::UniquePtr<IGeometryInteractiveExecutor> transferredGeometryInteractiveExecutor,
        const HostBlendshapeSolveExecutorCreationParameters& params
        );

private:
    static bool callbackForGeometryInteractiveExecutor(void* userdata, const IGeometryInteractiveExecutor::Results& results);

    nva2x::UniquePtr<IJobRunner> _jobRunner;

    host_results_callback_t _resultsCallback{};
    void* _resultsUserdata{};
};

class DeviceBlendshapeSolveInteractiveExecutor : public BlendshapeSolveInteractiveExecutor {
public:
    ResultsType GetResultType() const override;

    std::error_code ComputeFrame(std::size_t frameIndex) override;
    std::error_code ComputeAllFrames() override;

    std::error_code SetResultsCallback(device_results_callback_t callback, void* userdata) override;

    std::error_code Init(
        nva2x::UniquePtr<IGeometryInteractiveExecutor> transferredGeometryInteractiveExecutor,
        const DeviceBlendshapeSolveExecutorCreationParameters& params
        );

private:
    static bool callbackForGeometryInteractiveExecutor(void* userdata, const IGeometryInteractiveExecutor::Results& results);

    device_results_callback_t _resultsCallback{};
    void* _resultsUserdata{};

    nva2x::DeviceTensorFloat _weights;
};

IBlendshapeInteractiveExecutor* CreateHostBlendshapeSolveInteractiveExecutor_INTERNAL(
    IGeometryInteractiveExecutor* transferredGeometryInteractiveExecutor,
    const HostBlendshapeSolveExecutorCreationParameters& params
    );

IBlendshapeInteractiveExecutor* CreateDeviceBlendshapeSolveInteractiveExecutor_INTERNAL(
    IGeometryInteractiveExecutor* transferredGeometryInteractiveExecutor,
    const DeviceBlendshapeSolveExecutorCreationParameters& params
    );

} // namespace nva2f
