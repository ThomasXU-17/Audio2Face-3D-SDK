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
#include "audio2face/internal/executor.h"
#include "audio2x/internal/unique_ptr.h"

namespace nva2f {

class BlendshapeSolveExecutor : public IBlendshapeExecutor,
                                public IFaceExecutorAccessorInputStrength,
                                public IFaceExecutorAccessorGeometryExecutor,
                                public IFaceExecutorAccessorGeometryResultsCallback,
                                public IFaceExecutorAccessorBlendshapeSolvers,
                                public IFaceExecutorAccessorSkinParameters,
                                public IFaceExecutorAccessorTongueParameters {

public:
    std::size_t GetNbTracks() const override;

    std::error_code Reset(std::size_t trackIndex) override;
    void Destroy() override;

    bool HasExecutionStarted(std::size_t trackIndex) const override;
    std::size_t GetNbAvailableExecutions(std::size_t trackIndex) const override;
    std::size_t GetTotalNbFrames(std::size_t trackIndex) const override;
    std::size_t GetSamplingRate() const;
    void GetFrameRate(std::size_t& numerator, std::size_t& denominator) const override;

    timestamp_t GetFrameTimestamp(std::size_t frameIndex) const override;

    std::error_code SetEmotionsCallback(emotions_callback_t callback, void* userdata) override;

    timestamp_t GetNextEmotionTimestampToRead(std::size_t trackIndex) const override;
    std::size_t GetNextAudioSampleToRead(std::size_t trackIndex) const override;

    std::size_t GetWeightCount() const override;

    std::error_code SetResultsCallback(host_results_callback_t callback, void* userdata) override;
    std::error_code SetResultsCallback(device_results_callback_t callback, void* userdata) override;

    std::error_code Wait(std::size_t trackIndex) override;

    std::error_code GetInputStrength(float& inputStrength) const override;
    std::error_code SetInputStrength(float inputStrength) override;
    std::error_code GetGeometryExecutor(IGeometryExecutor** geometryExecutor) const override;
    std::error_code SetGeometryResultsCallback(IGeometryExecutor::results_callback_t callback, void* userdata) override;
    std::error_code GetSkinSolver(std::size_t trackIndex, IBlendshapeSolver** skinSolver) const override;
    std::error_code GetTongueSolver(std::size_t trackIndex, IBlendshapeSolver** tongueSolver) const override;
    std::error_code Get(std::size_t trackIndex, AnimatorSkinParams& params) const override;
    std::error_code Set(std::size_t trackIndex, const AnimatorSkinParams& params) override;
    std::error_code Get(std::size_t trackIndex, AnimatorTongueParams& params) const override;
    std::error_code Set(std::size_t trackIndex, const AnimatorTongueParams& params) override;

    std::error_code Init(
        nva2x::UniquePtr<IGeometryExecutor> transferredGeometryExecutor,
        const BlendshapeSolveExecutorCreationParameters& params,
        bool useGpu
        );

protected:
    nva2x::UniquePtr<IGeometryExecutor> _geometryExecutor;
    IGeometryExecutor::results_callback_t _geometryResultsCallback{};
    void *_geometryResultsUserdata{};

    struct TrackData {
        nva2x::UniquePtr<IBlendshapeSolver> skinSolver;
        nva2x::UniquePtr<IBlendshapeSolver> tongueSolver;
    };
    std::vector<TrackData> _trackData;
};

class HostBlendshapeSolveExecutor : public BlendshapeSolveExecutor {
public:
    ResultsType GetResultType() const override;

    std::error_code Execute(std::size_t* pNbExecutedTracks) override;

    std::error_code SetResultsCallback(host_results_callback_t callback, void* userdata) override;

    std::error_code Init(
        nva2x::UniquePtr<IGeometryExecutor> transferredGeometryExecutor,
        const HostBlendshapeSolveExecutorCreationParameters& params
        );

    static bool callbackHelperForGeometryExecutor(
        IBlendshapeSolver* skinSolver,
        IBlendshapeSolver* tongueSolver,
        IGeometryExecutor::results_callback_t geometryCallback,
        void* geometryUserdata,
        host_results_callback_t resultsCallback,
        void* resultsUserdata,
        const IGeometryExecutor::Results& results
        );

private:
    static bool callbackForGeometryExecutor(void* userdata, const IGeometryExecutor::Results& results);

    nva2x::UniquePtr<IJobRunner> _jobRunner;

    host_results_callback_t _resultsCallback{};
    void* _resultsUserdata{};
};

class DeviceBlendshapeSolveExecutor : public BlendshapeSolveExecutor {
public:
    ResultsType GetResultType() const override;

    std::error_code Execute(std::size_t* pNbExecutedTracks) override;

    std::error_code SetResultsCallback(device_results_callback_t callback, void* userdata) override;

    std::error_code Init(
        nva2x::UniquePtr<IGeometryExecutor> transferredGeometryExecutor,
        const DeviceBlendshapeSolveExecutorCreationParameters& params
        );

    static bool callbackHelperForGeometryExecutor(
        IBlendshapeSolver* skinSolver,
        IBlendshapeSolver* tongueSolver,
        IGeometryExecutor::results_callback_t geometryCallback,
        void* geometryUserdata,
        device_results_callback_t resultsCallback,
        void* resultsUserdata,
        nva2x::DeviceTensorFloatView weights,
        const IGeometryExecutor::Results& results
        );

private:
    static bool callbackForGeometryExecutor(void* userdata, const IGeometryExecutor::Results& results);

    device_results_callback_t _resultsCallback{};
    void* _resultsUserdata{};

    nva2x::DeviceTensorFloat _weights;
};

IBlendshapeExecutor* CreateHostBlendshapeSolveExecutor_INTERNAL(
    IGeometryExecutor* transferredGeometryExecutor,
    const HostBlendshapeSolveExecutorCreationParameters& params
    );

IBlendshapeExecutor* CreateDeviceBlendshapeSolveExecutor_INTERNAL(
    IGeometryExecutor* transferredGeometryExecutor,
    const DeviceBlendshapeSolveExecutorCreationParameters& params
    );

} // namespace nva2f
