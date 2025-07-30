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

#include "audio2face/dll_export.h"
#include "audio2face/animator.h"
#include "audio2face/blendshape_solver.h"
#include "audio2face/emotion.h"
#include "audio2face/executor.h"
#include "audio2face/executor_regression.h"
#include "audio2face/executor_diffusion.h"
#include "audio2face/executor_blendshapesolve.h"
#include "audio2face/interactive_executor.h"
#include "audio2face/job_runner.h"
#include "audio2face/error.h"
#include "audio2face/model_regression.h"
#include "audio2face/model_diffusion.h"
#include "audio2face/multitrack_animator.h"
#include "audio2face/parse_helper.h"
#include "audio2face/noise.h"
#include "audio2x/inference_engine.h"

namespace nva2f {

// Create a PCA reconstruction animator.
AUDIO2FACE_DLL_API IAnimatorPcaReconstruction *CreateAnimatorPcaReconstruction();
// Create a skin animator.
AUDIO2FACE_DLL_API IAnimatorSkin *CreateAnimatorSkin();
// Create a tongue animator.
AUDIO2FACE_DLL_API IAnimatorTongue *CreateAnimatorTongue();
// Create a teeth animator.
AUDIO2FACE_DLL_API IAnimatorTeeth *CreateAnimatorTeeth();
// Create an eyes animator.
AUDIO2FACE_DLL_API IAnimatorEyes *CreateAnimatorEyes();
// Create a multi-track skin animator.
AUDIO2FACE_DLL_API IMultiTrackAnimatorSkin *CreateMultiTrackAnimatorSkin();
// Create a multi-track tongue animator.
AUDIO2FACE_DLL_API IMultiTrackAnimatorTongue *CreateMultiTrackAnimatorTongue();
// Create a multi-track teeth animator.
AUDIO2FACE_DLL_API IMultiTrackAnimatorTeeth *CreateMultiTrackAnimatorTeeth();
// Create a multi-track eyes animator.
AUDIO2FACE_DLL_API IMultiTrackAnimatorEyes *CreateMultiTrackAnimatorEyes();
// Create an emotion database.
AUDIO2FACE_DLL_API IEmotionDatabase *CreateEmotionDatabase();
// Create a thread pool job runner with specified number of threads.
AUDIO2FACE_DLL_API IJobRunner *CreateThreadPoolJobRunner(size_t numThreads);

// Create inference input buffers for regression model.
AUDIO2FACE_DLL_API IRegressionModel::IInferenceInputBuffers* CreateInferenceInputBuffersForRegressionModel(
    const IRegressionModel::NetworkInfo& networkInfo, std::size_t count = 1
    );
// Create inference output buffers for regression model.
AUDIO2FACE_DLL_API IRegressionModel::IInferenceOutputBuffers* CreateInferenceOutputBuffersForRegressionModel(
    const IRegressionModel::NetworkInfo& networkInfo, std::size_t count = 1
    );
// Create result buffers for regression model.
AUDIO2FACE_DLL_API IRegressionModel::IResultBuffers* CreateResultBuffersForRegressionModel(
    const IRegressionModel::NetworkInfo& networkInfo, std::size_t count = 1
    );

// Create inference input buffers for diffusion model.
AUDIO2FACE_DLL_API IDiffusionModel::IInferenceInputBuffers* CreateInferenceInputBuffersForDiffusionModel(
    const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count = 1
    );
// Create inference state buffers for diffusion model.
AUDIO2FACE_DLL_API IDiffusionModel::IInferenceStateBuffers* CreateInferenceStateBuffersForDiffusionModel(
    const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count = 1
    );
// Create inference output buffers for diffusion model.
AUDIO2FACE_DLL_API IDiffusionModel::IInferenceOutputBuffers* CreateInferenceOutputBuffersForDiffusionModel(
    const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count = 1
    );
// Create result buffers for diffusion model.
AUDIO2FACE_DLL_API IDiffusionModel::IResultBuffers* CreateResultBuffersForDiffusionModel(
    const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count = 1
    );

// Create a noise generator for specified number of tracks and size.
AUDIO2FACE_DLL_API INoiseGenerator* CreateNoiseGenerator(std::size_t nbTracks, std::size_t sizeToGenerate);

// Get buffer bindings description for regression model.
AUDIO2FACE_DLL_API const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForRegressionModel();
// Get buffer bindings description for diffusion model.
AUDIO2FACE_DLL_API const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForDiffusionModel();

// Create buffer bindings for regression model.
AUDIO2FACE_DLL_API nva2x::IBufferBindings* CreateBindingsForRegressionModel();
// Create buffer bindings for diffusion model.
AUDIO2FACE_DLL_API nva2x::IBufferBindings* CreateBindingsForDiffusionModel();

// Create a blendshape solver using CPU or GPU solve.
AUDIO2FACE_DLL_API IBlendshapeSolver* CreateBlendshapeSolver(bool gpuSolver);


// Read animator parameters from file.
AUDIO2FACE_DLL_API std::error_code ReadAnimatorParams(
    const char* path, float& inputStrength, AnimatorParams& params
    );

// Read regression network info from file.
AUDIO2FACE_DLL_API IRegressionModel::INetworkInfo* ReadRegressionNetworkInfo(const char* path);
// Read regression animator data from file.
AUDIO2FACE_DLL_API IRegressionModel::IAnimatorData* ReadRegressionAnimatorData(const char* path);
// Read regression model info from single model.json file.
AUDIO2FACE_DLL_API IRegressionModel::IGeometryModelInfo* ReadRegressionModelInfo(const char* path);
// Read regression model info from individual files.
AUDIO2FACE_DLL_API IRegressionModel::IGeometryModelInfo* ReadRegressionModelInfo(
    const char* networkInfoPath,
    const char* networkPath,
    const char* emotionDatabasePath,
    const char* modelConfigPath,
    const char* modelDataPath
    );

// Read diffusion network info from file.
AUDIO2FACE_DLL_API IDiffusionModel::INetworkInfo* ReadDiffusionNetworkInfo(const char* path);
// Read diffusion animator data from file.
AUDIO2FACE_DLL_API IDiffusionModel::IAnimatorData* ReadDiffusionAnimatorData(const char* path);
// Read diffusion model info from single model.json file.
AUDIO2FACE_DLL_API IDiffusionModel::IGeometryModelInfo* ReadDiffusionModelInfo(const char* path);
// Read diffusion model info from individual files.
AUDIO2FACE_DLL_API IDiffusionModel::IGeometryModelInfo* ReadDiffusionModelInfo(
    const char* networkInfoPath,
    const char* networkPath,
    std::size_t identityCount,
    const char* const* modelConfigPaths,
    const char* const* modelDataPaths
    );

// Read blendshape solver config from file.
AUDIO2FACE_DLL_API IBlendshapeSolverConfig* ReadBlendshapeSolverConfig(const char* configPath);
// Read blendshape solver data from file.
AUDIO2FACE_DLL_API IBlendshapeSolverData* ReadBlendshapeSolverData(const char* dataPath);

// Read regression blendshape solve model info from single model.json file.
AUDIO2FACE_DLL_API IRegressionModel::IBlendshapeSolveModelInfo* ReadRegressionBlendshapeSolveModelInfo(const char* path);
// Read regression blendshape solve model info from individual files.
AUDIO2FACE_DLL_API IRegressionModel::IBlendshapeSolveModelInfo* ReadRegressionBlendshapeSolveModelInfo(
    const char* skinConfigPath,
    const char* skinDataPath,
    const char* tongueConfigPath,
    const char* tongueDataPath
    );

// Read diffusion blendshape solve model info from single model.json file.
AUDIO2FACE_DLL_API IDiffusionModel::IBlendshapeSolveModelInfo* ReadDiffusionBlendshapeSolveModelInfo(const char* path);
// Read diffusion blendshape solve model info from individual files.
AUDIO2FACE_DLL_API IDiffusionModel::IBlendshapeSolveModelInfo* ReadDiffusionBlendshapeSolveModelInfo(
    std::size_t identityCount,
    const char* const* skinConfigPaths,
    const char* const* skinDataPaths,
    const char* const* tongueConfigPaths,
    const char* const* tongueDataPaths
    );

// Get input strength from face executor.
AUDIO2FACE_DLL_API std::error_code GetExecutorInputStrength(
    const IFaceExecutor& executor, float& inputStrength
    );
// Set input strength for face executor.
AUDIO2FACE_DLL_API std::error_code SetExecutorInputStrength(
    IFaceExecutor& executor, float inputStrength
    );
// Get geometry executor from face executor.
AUDIO2FACE_DLL_API std::error_code GetExecutorGeometryExecutor(
    const IFaceExecutor& executor, IGeometryExecutor** geometryExecutor
    );
// Set geometry results callback for face executor.
AUDIO2FACE_DLL_API std::error_code SetExecutorGeometryResultsCallback(
    IFaceExecutor& executor, IGeometryExecutor::results_callback_t callback, void* userdata
    );
// Get skin solver from face executor for specified track.
AUDIO2FACE_DLL_API std::error_code GetExecutorSkinSolver(
    const IFaceExecutor& executor, std::size_t trackIndex, IBlendshapeSolver** skinSolver
    );
// Get tongue solver from face executor for specified track.
AUDIO2FACE_DLL_API std::error_code GetExecutorTongueSolver(
    const IFaceExecutor& executor, std::size_t trackIndex, IBlendshapeSolver** tongueSolver
    );
// Get skin parameters from face executor for specified track.
AUDIO2FACE_DLL_API std::error_code GetExecutorSkinParameters(
    const IFaceExecutor& executor, std::size_t trackIndex, AnimatorSkinParams& params
    );
// Set skin parameters for face executor for specified track.
AUDIO2FACE_DLL_API std::error_code SetExecutorSkinParameters(
    IFaceExecutor& executor, std::size_t trackIndex, const AnimatorSkinParams& params
    );
// Get tongue parameters from face executor for specified track.
AUDIO2FACE_DLL_API std::error_code GetExecutorTongueParameters(
    const IFaceExecutor& executor, std::size_t trackIndex, AnimatorTongueParams& params
    );
// Set tongue parameters for face executor for specified track.
AUDIO2FACE_DLL_API std::error_code SetExecutorTongueParameters(
    IFaceExecutor& executor, std::size_t trackIndex, const AnimatorTongueParams& params
    );
// Get teeth parameters from face executor for specified track.
AUDIO2FACE_DLL_API std::error_code GetExecutorTeethParameters(
    const IFaceExecutor& executor, std::size_t trackIndex, AnimatorTeethParams& params
    );
// Set teeth parameters for face executor for specified track.
AUDIO2FACE_DLL_API std::error_code SetExecutorTeethParameters(
    IFaceExecutor& executor, std::size_t trackIndex, const AnimatorTeethParams& params
    );
// Get eyes parameters from face executor for specified track.
AUDIO2FACE_DLL_API std::error_code GetExecutorEyesParameters(
    const IFaceExecutor& executor, std::size_t trackIndex, AnimatorEyesParams& params
    );
// Set eyes parameters for face executor for specified track.
AUDIO2FACE_DLL_API std::error_code SetExecutorEyesParameters(
    IFaceExecutor& executor, std::size_t trackIndex, const AnimatorEyesParams& params
    );

// Compare two skin parameter sets for equality.
AUDIO2FACE_DLL_API bool AreEqual(const AnimatorSkinParams& a, const AnimatorSkinParams& b);
// Compare two tongue parameter sets for equality.
AUDIO2FACE_DLL_API bool AreEqual(const AnimatorTongueParams& a, const AnimatorTongueParams& b);
// Compare two teeth parameter sets for equality.
AUDIO2FACE_DLL_API bool AreEqual(const AnimatorTeethParams& a, const AnimatorTeethParams& b);
// Compare two eyes parameter sets for equality.
AUDIO2FACE_DLL_API bool AreEqual(const AnimatorEyesParams& a, const AnimatorEyesParams& b);
// Compare two blendshape solver parameter sets for equality.
AUDIO2FACE_DLL_API bool AreEqual(const BlendshapeSolverParams& a, const BlendshapeSolverParams& b);
// Compare two blendshape solver config sets for equality.
AUDIO2FACE_DLL_API bool AreEqual(const BlendshapeSolverConfig& a, const BlendshapeSolverConfig& b);

// Create regression geometry executor with specified parameters.
AUDIO2FACE_DLL_API IGeometryExecutor* CreateRegressionGeometryExecutor(
    const GeometryExecutorCreationParameters& params,
    const IRegressionModel::GeometryExecutorCreationParameters& regressionParams
    );
// Create diffusion geometry executor with specified parameters.
AUDIO2FACE_DLL_API IGeometryExecutor* CreateDiffusionGeometryExecutor(
    const GeometryExecutorCreationParameters& params,
    const IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams
    );
// Create host blendshape solve executor with specified parameters.
// Ownership of the geometry executor is transferred to the blendshape executor.
AUDIO2FACE_DLL_API IBlendshapeExecutor* CreateHostBlendshapeSolveExecutor(
    IGeometryExecutor* transferredGeometryExecutor,
    const HostBlendshapeSolveExecutorCreationParameters& params
    );
// Create device blendshape solve executor with specified parameters.
// Ownership of the geometry executor is transferred to the blendshape executor.
AUDIO2FACE_DLL_API IBlendshapeExecutor* CreateDeviceBlendshapeSolveExecutor(
    IGeometryExecutor* transferredGeometryExecutor,
    const DeviceBlendshapeSolveExecutorCreationParameters& params
    );

// Bitwise OR operator for execution options.
AUDIO2FACE_DLL_API IGeometryExecutor::ExecutionOption operator|(IGeometryExecutor::ExecutionOption a, IGeometryExecutor::ExecutionOption b);
// Bitwise OR assignment operator for execution options.
AUDIO2FACE_DLL_API IGeometryExecutor::ExecutionOption& operator|=(IGeometryExecutor::ExecutionOption& a, IGeometryExecutor::ExecutionOption b);
// Bitwise AND operator for execution options.
AUDIO2FACE_DLL_API IGeometryExecutor::ExecutionOption operator&(IGeometryExecutor::ExecutionOption a, IGeometryExecutor::ExecutionOption b);
// Bitwise AND assignment operator for execution options.
AUDIO2FACE_DLL_API IGeometryExecutor::ExecutionOption& operator&=(IGeometryExecutor::ExecutionOption& a, IGeometryExecutor::ExecutionOption b);
// Check if any of the specified flags are set.
AUDIO2FACE_DLL_API bool IsAnySet(IGeometryExecutor::ExecutionOption flags, IGeometryExecutor::ExecutionOption flagsToCheck);


// Get input strength from interactive face executor.
AUDIO2FACE_DLL_API std::error_code GetInteractiveExecutorInputStrength(
    const IFaceInteractiveExecutor& executor, float& inputStrength
    );
// Set input strength for interactive face executor.
AUDIO2FACE_DLL_API std::error_code SetInteractiveExecutorInputStrength(
    IFaceInteractiveExecutor& executor, float inputStrength
    );
// Get skin parameters from interactive face executor.
AUDIO2FACE_DLL_API std::error_code GetInteractiveExecutorSkinParameters(
    const IFaceInteractiveExecutor& executor, AnimatorSkinParams& params
    );
// Set skin parameters for interactive face executor.
AUDIO2FACE_DLL_API std::error_code SetInteractiveExecutorSkinParameters(
    IFaceInteractiveExecutor& executor, const AnimatorSkinParams& params
    );
// Get tongue parameters from interactive face executor.
AUDIO2FACE_DLL_API std::error_code GetInteractiveExecutorTongueParameters(
    const IFaceInteractiveExecutor& executor, AnimatorTongueParams& params
    );
// Set tongue parameters for interactive face executor.
AUDIO2FACE_DLL_API std::error_code SetInteractiveExecutorTongueParameters(
    IFaceInteractiveExecutor& executor, const AnimatorTongueParams& params
    );
// Get teeth parameters from interactive face executor.
AUDIO2FACE_DLL_API std::error_code GetInteractiveExecutorTeethParameters(
    const IFaceInteractiveExecutor& executor, AnimatorTeethParams& params
    );
// Set teeth parameters for interactive face executor.
AUDIO2FACE_DLL_API std::error_code SetInteractiveExecutorTeethParameters(
    IFaceInteractiveExecutor& executor, const AnimatorTeethParams& params
    );
// Get eyes parameters from interactive face executor.
AUDIO2FACE_DLL_API std::error_code GetInteractiveExecutorEyesParameters(
    const IFaceInteractiveExecutor& executor, AnimatorEyesParams& params
    );
// Set eyes parameters for interactive face executor.
AUDIO2FACE_DLL_API std::error_code SetInteractiveExecutorEyesParameters(
    IFaceInteractiveExecutor& executor, const AnimatorEyesParams& params
    );

// Get interactive geometry executor from interactive face executor.
AUDIO2FACE_DLL_API std::error_code GetInteractiveExecutorGeometryExecutor(
    const IFaceInteractiveExecutor& executor, IGeometryInteractiveExecutor** geometryExecutor
    );
// Set geometry results callback for interactive face executor.
AUDIO2FACE_DLL_API std::error_code SetInteractiveExecutorGeometryResultsCallback(
    IFaceInteractiveExecutor& executor, IGeometryInteractiveExecutor::results_callback_t callback, void* userdata
    );
// Get blendshape solve skin config from interactive face executor.
AUDIO2FACE_DLL_API std::error_code GetInteractiveExecutorBlendshapeSkinConfig(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverConfig& config
    );
// Set blendshape solve skin config for interactive face executor.
AUDIO2FACE_DLL_API std::error_code SetInteractiveExecutorBlendshapeSkinConfig(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverConfig& config
    );
// Get blendshape solve skin parameters from interactive face executor.
AUDIO2FACE_DLL_API std::error_code GetInteractiveExecutorBlendshapeSkinParameters(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverParams& params
    );
// Set blendshape solve skin parameters for interactive face executor.
AUDIO2FACE_DLL_API std::error_code SetInteractiveExecutorBlendshapeSkinParameters(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverParams& params
    );
// Get blendshape solve tongue config from interactive face executor.
AUDIO2FACE_DLL_API std::error_code GetInteractiveExecutorBlendshapeTongueConfig(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverConfig& config
    );
// Set blendshape solve tongue config for interactive face executor.
AUDIO2FACE_DLL_API std::error_code SetInteractiveExecutorBlendshapeTongueConfig(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverConfig& config
    );
// Get blendshape solve tongue parameters from interactive face executor.
AUDIO2FACE_DLL_API std::error_code GetInteractiveExecutorBlendshapeTongueParameters(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverParams& params
    );
// Set blendshape solve tongue parameters for interactive face executor.
AUDIO2FACE_DLL_API std::error_code SetInteractiveExecutorBlendshapeTongueParameters(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverParams& params
    );

// Create regression geometry interactive executor with specified parameters.
AUDIO2FACE_DLL_API IGeometryInteractiveExecutor* CreateRegressionGeometryInteractiveExecutor(
    const GeometryExecutorCreationParameters& params,
    const IRegressionModel::GeometryExecutorCreationParameters& regressionParams,
    std::size_t batchSize
    );
// Create diffusion geometry interactive executor with specified parameters.
AUDIO2FACE_DLL_API IGeometryInteractiveExecutor* CreateDiffusionGeometryInteractiveExecutor(
    const GeometryExecutorCreationParameters& params,
    const IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams,
    std::size_t nbInferencesForPreview
    );
// Create host blendshape solve interactive executor with specified parameters.
AUDIO2FACE_DLL_API IBlendshapeInteractiveExecutor* CreateHostBlendshapeSolveInteractiveExecutor(
    IGeometryInteractiveExecutor* transferredGeometryInteractiveExecutor,
    const HostBlendshapeSolveExecutorCreationParameters& params
    );
// Create device blendshape solve interactive executor with specified parameters.
AUDIO2FACE_DLL_API IBlendshapeInteractiveExecutor* CreateDeviceBlendshapeSolveInteractiveExecutor(
    IGeometryInteractiveExecutor* transferredGeometryInteractiveExecutor,
    const DeviceBlendshapeSolveExecutorCreationParameters& params
    );


// Read regression geometry executor bundle from single model.json file.
AUDIO2FACE_DLL_API IGeometryExecutorBundle* ReadRegressionGeometryExecutorBundle(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IRegressionModel::IGeometryModelInfo** outModelInfo
    );
// Read diffusion geometry executor bundle from single model.json file.
AUDIO2FACE_DLL_API IGeometryExecutorBundle* ReadDiffusionGeometryExecutorBundle(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t identityIndex,
    bool constantNoise,
    IDiffusionModel::IGeometryModelInfo** outModelInfo
    );

// Read regression blendshape solve executor bundle from single model.json file.
AUDIO2FACE_DLL_API IBlendshapeExecutorBundle* ReadRegressionBlendshapeSolveExecutorBundle(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    bool useGpuSolver,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IRegressionModel::IGeometryModelInfo** outModelInfo,
    IRegressionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
    );
// Read diffusion blendshape solve executor bundle from single model.json file.
AUDIO2FACE_DLL_API IBlendshapeExecutorBundle* ReadDiffusionBlendshapeSolveExecutorBundle(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    bool useGpuSolver,
    std::size_t identityIndex,
    bool constantNoise,
    IDiffusionModel::IGeometryModelInfo** outModelInfo,
    IDiffusionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
    );

} // namespace nva2f
