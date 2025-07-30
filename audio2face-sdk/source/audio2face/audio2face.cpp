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
#include "audio2face/audio2face.h"

namespace nva2f {

// Implementations are inside audio2face-core.lib
IAnimatorPcaReconstruction *CreateAnimatorPcaReconstruction_INTERNAL();
IAnimatorSkin *CreateAnimatorSkin_INTERNAL();
IAnimatorTongue *CreateAnimatorTongue_INTERNAL();
IAnimatorTeeth *CreateAnimatorTeeth_INTERNAL();
IAnimatorEyes *CreateAnimatorEyes_INTERNAL();
IMultiTrackAnimatorSkin *CreateMultiTrackAnimatorSkin_INTERNAL();
IMultiTrackAnimatorTongue *CreateMultiTrackAnimatorTongue_INTERNAL();
IMultiTrackAnimatorTeeth *CreateMultiTrackAnimatorTeeth_INTERNAL();
IMultiTrackAnimatorEyes *CreateMultiTrackAnimatorEyes_INTERNAL();
IEmotionDatabase *CreateEmotionDatabase_INTERNAL();
IJobRunner *CreateThreadPoolJobRunner_INTERNAL(size_t numThreads);

IRegressionModel::IInferenceInputBuffers* CreateInferenceInputBuffersForRegressionModel_INTERNAL(
  const IRegressionModel::NetworkInfo& networkInfo, std::size_t count
  );
IRegressionModel::IInferenceOutputBuffers* CreateInferenceOutputBuffersForRegressionModel_INTERNAL(
  const IRegressionModel::NetworkInfo& networkInfo, std::size_t count
  );
IRegressionModel::IResultBuffers* CreateResultBuffersForRegressionModel_INTERNAL(
  const IRegressionModel::NetworkInfo& networkInfo, std::size_t count
  );

IDiffusionModel::IInferenceInputBuffers* CreateInferenceInputBuffersForDiffusionModel_INTERNAL(
  const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count
  );
IDiffusionModel::IInferenceStateBuffers* CreateInferenceStateBuffersForDiffusionModel_INTERNAL(
  const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count
  );
IDiffusionModel::IInferenceOutputBuffers* CreateInferenceOutputBuffersForDiffusionModel_INTERNAL(
  const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count
  );
IDiffusionModel::IResultBuffers* CreateResultBuffersForDiffusionModel_INTERNAL(
  const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count
  );

INoiseGenerator* CreateNoiseGenerator_INTERNAL(std::size_t nbTracks, std::size_t sizeToGenerate);


const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForRegressionModel_INTERNAL();
const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForDiffusionModel_INTERNAL();

nva2x::IBufferBindings* CreateBindingsForRegressionModel_INTERNAL();
nva2x::IBufferBindings* CreateBindingsForDiffusionModel_INTERNAL();

std::error_code ReadAnimatorParams_INTERNAL(const char* path, float& inputStrength, AnimatorParams& params);

IRegressionModel::INetworkInfo* ReadRegressionNetworkInfo_INTERNAL(const char* path);
IRegressionModel::IAnimatorData* ReadRegressionAnimatorData_INTERNAL(const char* path);

IRegressionModel::IGeometryModelInfo* ReadRegressionModelInfo_INTERNAL(const char* path);
IRegressionModel::IGeometryModelInfo* ReadRegressionModelInfo_INTERNAL(
    const char* networkInfoPath,
    const char* networkPath,
    const char* emotionDatabasePath,
    const char* modelConfigPath,
    const char* modelDataPath
    );

IDiffusionModel::INetworkInfo* ReadDiffusionNetworkInfo_INTERNAL(const char* path);
IDiffusionModel::IAnimatorData* ReadDiffusionAnimatorData_INTERNAL(const char* path);

IDiffusionModel::IGeometryModelInfo* ReadDiffusionModelInfo_INTERNAL(const char* path);
IDiffusionModel::IGeometryModelInfo* ReadDiffusionModelInfo_INTERNAL(
    const char* networkInfoPath,
    const char* networkPath,
    std::size_t identityCount,
    const char* const* modelConfigPaths,
    const char* const* modelDataPaths
    );

IBlendshapeSolverConfig* ReadBlendshapeSolverConfig_INTERNAL(const char* configPath);
IBlendshapeSolverData* ReadBlendshapeSolverData_INTERNAL(const char* dataPath);

IRegressionModel::IBlendshapeSolveModelInfo* ReadRegressionBlendshapeSolveModelInfo_INTERNAL(const char* path);
IRegressionModel::IBlendshapeSolveModelInfo* ReadRegressionBlendshapeSolveModelInfo_INTERNAL(
    const char* skinConfigPath,
    const char* skinDataPath,
    const char* tongueConfigPath,
    const char* tongueDataPath
    );

IDiffusionModel::IBlendshapeSolveModelInfo* ReadDiffusionBlendshapeSolveModelInfo_INTERNAL(const char* path);
IDiffusionModel::IBlendshapeSolveModelInfo* ReadDiffusionBlendshapeSolveModelInfo_INTERNAL(
    std::size_t identityCount,
    const char* const* skinConfigPaths,
    const char* const* skinDataPaths,
    const char* const* tongueConfigPaths,
    const char* const* tongueDataPaths
    );

std::error_code GetExecutorInputStrength_INTERNAL(
  const IFaceExecutor& executor, float& inputStrength
  );
std::error_code SetExecutorInputStrength_INTERNAL(
  IFaceExecutor& executor, float inputStrength
  );
std::error_code GetExecutorGeometryExecutor_INTERNAL(
    const IFaceExecutor& executor, IGeometryExecutor** geometryExecutor
    );
std::error_code SetExecutorGeometryResultsCallback_INTERNAL(
    IFaceExecutor& executor, IGeometryExecutor::results_callback_t callback, void* userdata
    );
std::error_code GetExecutorSkinSolver_INTERNAL(
  const IFaceExecutor& executor, std::size_t trackIndex, IBlendshapeSolver** skinSolver
  );
std::error_code GetExecutorTongueSolver_INTERNAL(
  const IFaceExecutor& executor, std::size_t trackIndex, IBlendshapeSolver** tongueSolver
  );
std::error_code GetExecutorSkinParameters_INTERNAL(
  const IFaceExecutor& executor, std::size_t trackIndex, AnimatorSkinParams& params
  );
std::error_code SetExecutorSkinParameters_INTERNAL(
  IFaceExecutor& executor, std::size_t trackIndex, const AnimatorSkinParams& params
  );
std::error_code GetExecutorTongueParameters_INTERNAL(
  const IFaceExecutor& executor, std::size_t trackIndex, AnimatorTongueParams& params
  );
std::error_code SetExecutorTongueParameters_INTERNAL(
  IFaceExecutor& executor, std::size_t trackIndex, const AnimatorTongueParams& params
  );
std::error_code GetExecutorTeethParameters_INTERNAL(
  const IFaceExecutor& executor, std::size_t trackIndex, AnimatorTeethParams& params
  );
std::error_code SetExecutorTeethParameters_INTERNAL(
  IFaceExecutor& executor, std::size_t trackIndex, const AnimatorTeethParams& params
  );
std::error_code GetExecutorEyesParameters_INTERNAL(
  const IFaceExecutor& executor, std::size_t trackIndex, AnimatorEyesParams& params
  );
std::error_code SetExecutorEyesParameters_INTERNAL(
  IFaceExecutor& executor, std::size_t trackIndex, const AnimatorEyesParams& params
  );

bool AreEqual_INTERNAL(const AnimatorSkinParams& a, const AnimatorSkinParams& b);
bool AreEqual_INTERNAL(const AnimatorTongueParams& a, const AnimatorTongueParams& b);
bool AreEqual_INTERNAL(const AnimatorTeethParams& a, const AnimatorTeethParams& b);
bool AreEqual_INTERNAL(const AnimatorEyesParams& a, const AnimatorEyesParams& b);
bool AreEqual_INTERNAL(const BlendshapeSolverParams& a, const BlendshapeSolverParams& b);
bool AreEqual_INTERNAL(const BlendshapeSolverConfig& a, const BlendshapeSolverConfig& b);

IGeometryExecutor* CreateRegressionGeometryExecutor_INTERNAL(
    const GeometryExecutorCreationParameters& params,
    const IRegressionModel::GeometryExecutorCreationParameters& regressionParams
    );
IGeometryExecutor* CreateDiffusionGeometryExecutor_INTERNAL(
    const GeometryExecutorCreationParameters& params,
    const IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams
    );
IBlendshapeExecutor* CreateHostBlendshapeSolveExecutor_INTERNAL(
    IGeometryExecutor* transferredGeometryExecutor,
    const HostBlendshapeSolveExecutorCreationParameters& params
    );
IBlendshapeExecutor* CreateDeviceBlendshapeSolveExecutor_INTERNAL(
    IGeometryExecutor* transferredGeometryExecutor,
    const DeviceBlendshapeSolveExecutorCreationParameters& params
    );

namespace internal {

IGeometryExecutor::ExecutionOption operator|(IGeometryExecutor::ExecutionOption a, IGeometryExecutor::ExecutionOption b);
IGeometryExecutor::ExecutionOption& operator|=(IGeometryExecutor::ExecutionOption& a, IGeometryExecutor::ExecutionOption b);
IGeometryExecutor::ExecutionOption operator&(IGeometryExecutor::ExecutionOption a, IGeometryExecutor::ExecutionOption b);
IGeometryExecutor::ExecutionOption& operator&=(IGeometryExecutor::ExecutionOption& a, IGeometryExecutor::ExecutionOption b);
bool IsAnySet(IGeometryExecutor::ExecutionOption flags, IGeometryExecutor::ExecutionOption flagsToCheck);

}

std::error_code GetInteractiveExecutorInputStrength_INTERNAL(
    const IFaceInteractiveExecutor& executor, float& inputStrength
    );
std::error_code SetInteractiveExecutorInputStrength_INTERNAL(
    IFaceInteractiveExecutor& executor, float inputStrength
    );
std::error_code GetInteractiveExecutorSkinParameters_INTERNAL(
    const IFaceInteractiveExecutor& executor, AnimatorSkinParams& params
    );
std::error_code SetInteractiveExecutorSkinParameters_INTERNAL(
    IFaceInteractiveExecutor& executor, const AnimatorSkinParams& params
    );
std::error_code GetInteractiveExecutorTongueParameters_INTERNAL(
    const IFaceInteractiveExecutor& executor, AnimatorTongueParams& params
    );
std::error_code SetInteractiveExecutorTongueParameters_INTERNAL(
    IFaceInteractiveExecutor& executor, const AnimatorTongueParams& params
    );
std::error_code GetInteractiveExecutorTeethParameters_INTERNAL(
    const IFaceInteractiveExecutor& executor, AnimatorTeethParams& params
    );
std::error_code SetInteractiveExecutorTeethParameters_INTERNAL(
    IFaceInteractiveExecutor& executor, const AnimatorTeethParams& params
    );
std::error_code GetInteractiveExecutorEyesParameters_INTERNAL(
    const IFaceInteractiveExecutor& executor, AnimatorEyesParams& params
    );
std::error_code SetInteractiveExecutorEyesParameters_INTERNAL(
    IFaceInteractiveExecutor& executor, const AnimatorEyesParams& params
    );

std::error_code GetInteractiveExecutorGeometryExecutor_INTERNAL(
    const IFaceInteractiveExecutor& executor, IGeometryInteractiveExecutor** geometryExecutor
    );
std::error_code SetInteractiveExecutorGeometryResultsCallback_INTERNAL(
    IFaceInteractiveExecutor& executor, IGeometryInteractiveExecutor::results_callback_t callback, void* userdata
    );
std::error_code GetInteractiveExecutorBlendshapeSkinConfig_INTERNAL(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverConfig& config
    );
std::error_code SetInteractiveExecutorBlendshapeSkinConfig_INTERNAL(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverConfig& config
    );
std::error_code GetInteractiveExecutorBlendshapeSkinParameters_INTERNAL(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverParams& params
    );
std::error_code SetInteractiveExecutorBlendshapeSkinParameters_INTERNAL(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverParams& params
    );
std::error_code GetInteractiveExecutorBlendshapeTongueConfig_INTERNAL(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverConfig& config
    );
std::error_code SetInteractiveExecutorBlendshapeTongueConfig_INTERNAL(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverConfig& config
    );
std::error_code GetInteractiveExecutorBlendshapeTongueParameters_INTERNAL(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverParams& params
    );
std::error_code SetInteractiveExecutorBlendshapeTongueParameters_INTERNAL(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverParams& params
    );

IGeometryInteractiveExecutor* CreateRegressionGeometryInteractiveExecutor_INTERNAL(
    const GeometryExecutorCreationParameters& params,
    const IRegressionModel::GeometryExecutorCreationParameters& regressionParams,
    std::size_t batchSize
    );
IGeometryInteractiveExecutor* CreateDiffusionGeometryInteractiveExecutor_INTERNAL(
    const GeometryExecutorCreationParameters& params,
    const IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams,
    std::size_t nbInferencesForPreview
    );
IBlendshapeInteractiveExecutor* CreateHostBlendshapeSolveInteractiveExecutor_INTERNAL(
    IGeometryInteractiveExecutor* transferredGeometryInteractiveExecutor,
    const HostBlendshapeSolveExecutorCreationParameters& params
    );
IBlendshapeInteractiveExecutor* CreateDeviceBlendshapeSolveInteractiveExecutor_INTERNAL(
    IGeometryInteractiveExecutor* transferredGeometryInteractiveExecutor,
    const DeviceBlendshapeSolveExecutorCreationParameters& params
    );


IGeometryExecutorBundle* ReadRegressionGeometryExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IRegressionModel::IGeometryModelInfo** outModelInfo
    );
IGeometryExecutorBundle* ReadDiffusionGeometryExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t identityIndex,
    bool constantNoise,
    IDiffusionModel::IGeometryModelInfo** outModelInfo
    );

IBlendshapeExecutorBundle* ReadRegressionBlendshapeSolveExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    bool useGpuSolver,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IRegressionModel::IGeometryModelInfo** outModelInfo,
    IRegressionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
    );
IBlendshapeExecutorBundle* ReadDiffusionBlendshapeSolveExecutorBundle_INTERNAL(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    bool useGpuSolver,
    std::size_t identityIndex,
    bool constantNoise,
    IDiffusionModel::IGeometryModelInfo** outModelInfo,
    IDiffusionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
    );

//////////////////////////////////////////////

IAnimatorPcaReconstruction *CreateAnimatorPcaReconstruction() {
  return CreateAnimatorPcaReconstruction_INTERNAL();
}

IAnimatorSkin *CreateAnimatorSkin() {
  return CreateAnimatorSkin_INTERNAL();
}

IAnimatorTongue *CreateAnimatorTongue() {
  return CreateAnimatorTongue_INTERNAL();
}

IAnimatorTeeth *CreateAnimatorTeeth() {
  return CreateAnimatorTeeth_INTERNAL();
}

IAnimatorEyes *CreateAnimatorEyes() {
  return CreateAnimatorEyes_INTERNAL();
}

IMultiTrackAnimatorSkin *CreateMultiTrackAnimatorSkin() {
  return CreateMultiTrackAnimatorSkin_INTERNAL();
}

IMultiTrackAnimatorTongue *CreateMultiTrackAnimatorTongue() {
  return CreateMultiTrackAnimatorTongue_INTERNAL();
}

IMultiTrackAnimatorTeeth *CreateMultiTrackAnimatorTeeth() {
  return CreateMultiTrackAnimatorTeeth_INTERNAL();
}

IMultiTrackAnimatorEyes *CreateMultiTrackAnimatorEyes() {
  return CreateMultiTrackAnimatorEyes_INTERNAL();
}

IEmotionDatabase *CreateEmotionDatabase() {
  return CreateEmotionDatabase_INTERNAL();
}

IJobRunner *CreateThreadPoolJobRunner(size_t numThreads) {
  return CreateThreadPoolJobRunner_INTERNAL(numThreads);
}

IRegressionModel::IInferenceInputBuffers* CreateInferenceInputBuffersForRegressionModel(
    const IRegressionModel::NetworkInfo& networkInfo, std::size_t count
    ) {
  return CreateInferenceInputBuffersForRegressionModel_INTERNAL(networkInfo, count);
}

IRegressionModel::IInferenceOutputBuffers* CreateInferenceOutputBuffersForRegressionModel(
    const IRegressionModel::NetworkInfo& networkInfo, std::size_t count
    ) {
  return CreateInferenceOutputBuffersForRegressionModel_INTERNAL(networkInfo, count);
}

IRegressionModel::IResultBuffers* CreateResultBuffersForRegressionModel(
    const IRegressionModel::NetworkInfo& networkInfo, std::size_t count
    ) {
  return CreateResultBuffersForRegressionModel_INTERNAL(networkInfo, count);
}

IDiffusionModel::IInferenceInputBuffers* CreateInferenceInputBuffersForDiffusionModel(
    const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count
    ) {
  return CreateInferenceInputBuffersForDiffusionModel_INTERNAL(networkInfo, count);
}

IDiffusionModel::IInferenceStateBuffers* CreateInferenceStateBuffersForDiffusionModel(
    const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count
    ) {
  return CreateInferenceStateBuffersForDiffusionModel_INTERNAL(networkInfo, count);
}

IDiffusionModel::IInferenceOutputBuffers* CreateInferenceOutputBuffersForDiffusionModel(
    const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count
    ) {
  return CreateInferenceOutputBuffersForDiffusionModel_INTERNAL(networkInfo, count);
}

IDiffusionModel::IResultBuffers* CreateResultBuffersForDiffusionModel(
    const IDiffusionModel::NetworkInfo& networkInfo, std::size_t count
    ) {
  return CreateResultBuffersForDiffusionModel_INTERNAL(networkInfo, count);
}

INoiseGenerator* CreateNoiseGenerator(std::size_t nbTracks, std::size_t sizeToGenerate) {
  return CreateNoiseGenerator_INTERNAL(nbTracks, sizeToGenerate);
}

const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForRegressionModel() {
  return GetBindingsDescriptionForRegressionModel_INTERNAL();
}

const nva2x::IBufferBindingsDescription& GetBindingsDescriptionForDiffusionModel() {
  return GetBindingsDescriptionForDiffusionModel_INTERNAL();
}

nva2x::IBufferBindings* CreateBindingsForRegressionModel() {
  return CreateBindingsForRegressionModel_INTERNAL();
}

nva2x::IBufferBindings* CreateBindingsForDiffusionModel() {
  return CreateBindingsForDiffusionModel_INTERNAL();
}

// bs solver
IBlendshapeSolver* CreateBlendshapeSolver_INTERNAL(bool gpuSolver);

IBlendshapeSolver* CreateBlendshapeSolver(bool gpuSolver) {
  return CreateBlendshapeSolver_INTERNAL(gpuSolver);
}

// Parser helper
std::error_code ReadAnimatorParams(const char* path, float& inputStrength, AnimatorParams& params) {
    return ReadAnimatorParams_INTERNAL(path, inputStrength, params);
}

IRegressionModel::INetworkInfo* ReadRegressionNetworkInfo(const char* path) {
  return ReadRegressionNetworkInfo_INTERNAL(path);
}

IRegressionModel::IAnimatorData* ReadRegressionAnimatorData(const char* path) {
  return ReadRegressionAnimatorData_INTERNAL(path);
}

IRegressionModel::IGeometryModelInfo* ReadRegressionModelInfo(const char* path) {
  return ReadRegressionModelInfo_INTERNAL(path);
}

IRegressionModel::IGeometryModelInfo* ReadRegressionModelInfo(
    const char* networkInfoPath,
    const char* networkPath,
    const char* emotionDatabasePath,
    const char* modelConfigPath,
    const char* modelDataPath
    ) {
  return ReadRegressionModelInfo_INTERNAL(
      networkInfoPath, networkPath, emotionDatabasePath, modelConfigPath, modelDataPath
      );
}

IDiffusionModel::INetworkInfo* ReadDiffusionNetworkInfo(const char* path) {
  return ReadDiffusionNetworkInfo_INTERNAL(path);
}

IDiffusionModel::IAnimatorData* ReadDiffusionAnimatorData(const char* path) {
  return ReadDiffusionAnimatorData_INTERNAL(path);
}

IDiffusionModel::IGeometryModelInfo* ReadDiffusionModelInfo(const char* path) {
  return ReadDiffusionModelInfo_INTERNAL(path);
}

IDiffusionModel::IGeometryModelInfo* ReadDiffusionModelInfo(
    const char* networkInfoPath,
    const char* networkPath,
    std::size_t identityCount,
    const char* const* modelConfigPaths,
    const char* const* modelDataPaths
    ) {
  return ReadDiffusionModelInfo_INTERNAL(
      networkInfoPath, networkPath, identityCount, modelConfigPaths, modelDataPaths
      );
}

IBlendshapeSolverConfig* ReadBlendshapeSolverConfig(const char* configPath) {
  return ReadBlendshapeSolverConfig_INTERNAL(configPath);
}

IBlendshapeSolverData* ReadBlendshapeSolverData(const char* dataPath) {
  return ReadBlendshapeSolverData_INTERNAL(dataPath);
}

IRegressionModel::IBlendshapeSolveModelInfo* ReadRegressionBlendshapeSolveModelInfo(const char* path) {
  return ReadRegressionBlendshapeSolveModelInfo_INTERNAL(path);
}

IRegressionModel::IBlendshapeSolveModelInfo* ReadRegressionBlendshapeSolveModelInfo(
    const char* skinConfigPath,
    const char* skinDataPath,
    const char* tongueConfigPath,
    const char* tongueDataPath
    ) {
  return ReadRegressionBlendshapeSolveModelInfo_INTERNAL(
    skinConfigPath, skinDataPath, tongueConfigPath, tongueDataPath
    );
}

IDiffusionModel::IBlendshapeSolveModelInfo* ReadDiffusionBlendshapeSolveModelInfo(const char* path) {
  return ReadDiffusionBlendshapeSolveModelInfo_INTERNAL(path);
}

IDiffusionModel::IBlendshapeSolveModelInfo* ReadDiffusionBlendshapeSolveModelInfo(
    std::size_t identityCount,
    const char* const* skinConfigPaths,
    const char* const* skinDataPaths,
    const char* const* tongueConfigPaths,
    const char* const* tongueDataPaths
    ) {
  return ReadDiffusionBlendshapeSolveModelInfo_INTERNAL(
    identityCount, skinConfigPaths, skinDataPaths, tongueConfigPaths, tongueDataPaths
    );
}

std::error_code GetExecutorInputStrength(const IFaceExecutor& executor, float& inputStrength) {
  return GetExecutorInputStrength_INTERNAL(executor, inputStrength);
}

std::error_code SetExecutorInputStrength(IFaceExecutor& executor, float inputStrength) {
  return SetExecutorInputStrength_INTERNAL(executor, inputStrength);
}

std::error_code GetExecutorGeometryExecutor(
  const IFaceExecutor& executor, IGeometryExecutor** geometryExecutor
  ) {
  return GetExecutorGeometryExecutor_INTERNAL(executor, geometryExecutor);
}

std::error_code SetExecutorGeometryResultsCallback(
  IFaceExecutor& executor, IGeometryExecutor::results_callback_t callback, void* userdata
  ) {
  return SetExecutorGeometryResultsCallback_INTERNAL(executor, callback, userdata);
}

std::error_code GetExecutorSkinSolver(
  const IFaceExecutor& executor, std::size_t trackIndex, IBlendshapeSolver** skinSolver
  ) {
  return GetExecutorSkinSolver_INTERNAL(executor, trackIndex, skinSolver);
}

std::error_code GetExecutorTongueSolver(
  const IFaceExecutor& executor, std::size_t trackIndex, IBlendshapeSolver** tongueSolver
  ) {
  return GetExecutorTongueSolver_INTERNAL(executor, trackIndex, tongueSolver);
}

std::error_code GetExecutorSkinParameters(
  const IFaceExecutor& executor, std::size_t trackIndex, AnimatorSkinParams& params
  ) {
  return GetExecutorSkinParameters_INTERNAL(executor, trackIndex, params);
}

std::error_code SetExecutorSkinParameters(
  IFaceExecutor& executor, std::size_t trackIndex, const AnimatorSkinParams& params
  ) {
  return SetExecutorSkinParameters_INTERNAL(executor, trackIndex, params);
}

std::error_code GetExecutorTongueParameters(
  const IFaceExecutor& executor, std::size_t trackIndex, AnimatorTongueParams& params
  ) {
  return GetExecutorTongueParameters_INTERNAL(executor, trackIndex, params);
}

std::error_code SetExecutorTongueParameters(
  IFaceExecutor& executor, std::size_t trackIndex, const AnimatorTongueParams& params
  ) {
  return SetExecutorTongueParameters_INTERNAL(executor, trackIndex, params);
}

std::error_code GetExecutorTeethParameters(
  const IFaceExecutor& executor, std::size_t trackIndex, AnimatorTeethParams& params
  ) {
  return GetExecutorTeethParameters_INTERNAL(executor, trackIndex, params);
}

std::error_code SetExecutorTeethParameters(
  IFaceExecutor& executor, std::size_t trackIndex, const AnimatorTeethParams& params
  ) {
  return SetExecutorTeethParameters_INTERNAL(executor, trackIndex, params);
}

std::error_code GetExecutorEyesParameters(
  const IFaceExecutor& executor, std::size_t trackIndex, AnimatorEyesParams& params
  ) {
  return GetExecutorEyesParameters_INTERNAL(executor, trackIndex, params);
}

std::error_code SetExecutorEyesParameters(
  IFaceExecutor& executor, std::size_t trackIndex, const AnimatorEyesParams& params
  ) {
  return SetExecutorEyesParameters_INTERNAL(executor, trackIndex, params);
}

bool AreEqual(const AnimatorSkinParams& a, const AnimatorSkinParams& b) {
  return AreEqual_INTERNAL(a, b);
}

bool AreEqual(const AnimatorTongueParams& a, const AnimatorTongueParams& b) {
  return AreEqual_INTERNAL(a, b);
}

bool AreEqual(const AnimatorTeethParams& a, const AnimatorTeethParams& b) {
  return AreEqual_INTERNAL(a, b);
}

bool AreEqual(const AnimatorEyesParams& a, const AnimatorEyesParams& b) {
  return AreEqual_INTERNAL(a, b);
}

bool AreEqual(const BlendshapeSolverParams& a, const BlendshapeSolverParams& b) {
  return AreEqual_INTERNAL(a, b);
}

bool AreEqual(const BlendshapeSolverConfig& a, const BlendshapeSolverConfig& b) {
  return AreEqual_INTERNAL(a, b);
}

IGeometryExecutor* CreateRegressionGeometryExecutor(
    const GeometryExecutorCreationParameters& params,
    const IRegressionModel::GeometryExecutorCreationParameters& regressionParams
    ) {
    return CreateRegressionGeometryExecutor_INTERNAL(params, regressionParams);
}

IGeometryExecutor* CreateDiffusionGeometryExecutor(
    const GeometryExecutorCreationParameters& params,
    const IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams
    ) {
    return CreateDiffusionGeometryExecutor_INTERNAL(params, diffusionParams);
}

IBlendshapeExecutor* CreateHostBlendshapeSolveExecutor(
    IGeometryExecutor* transferredGeometryExecutor,
    const HostBlendshapeSolveExecutorCreationParameters& params
    ) {
    return CreateHostBlendshapeSolveExecutor_INTERNAL(transferredGeometryExecutor, params);
}

IBlendshapeExecutor* CreateDeviceBlendshapeSolveExecutor(
    IGeometryExecutor* transferredGeometryExecutor,
    const DeviceBlendshapeSolveExecutorCreationParameters& params
    ) {
    return CreateDeviceBlendshapeSolveExecutor_INTERNAL(transferredGeometryExecutor, params);
}

IGeometryExecutor::ExecutionOption operator|(IGeometryExecutor::ExecutionOption a, IGeometryExecutor::ExecutionOption b) {
    return internal::operator|(a,b);
}

IGeometryExecutor::ExecutionOption& operator|=(IGeometryExecutor::ExecutionOption& a, IGeometryExecutor::ExecutionOption b) {
    return internal::operator|=(a,b);
}

IGeometryExecutor::ExecutionOption operator&(IGeometryExecutor::ExecutionOption a, IGeometryExecutor::ExecutionOption b) {
    return internal::operator&(a,b);
}

IGeometryExecutor::ExecutionOption& operator&=(IGeometryExecutor::ExecutionOption& a, IGeometryExecutor::ExecutionOption b) {
    return internal::operator&=(a,b);
}

bool IsAnySet(IGeometryExecutor::ExecutionOption flags, IGeometryExecutor::ExecutionOption flagsToCheck) {
  return internal::IsAnySet(flags, flagsToCheck);
}

std::error_code GetInteractiveExecutorInputStrength(
    const IFaceInteractiveExecutor& executor, float& inputStrength
    ) {
    return GetInteractiveExecutorInputStrength_INTERNAL(executor, inputStrength);
}

std::error_code SetInteractiveExecutorInputStrength(
    IFaceInteractiveExecutor& executor, float inputStrength
    ) {
    return SetInteractiveExecutorInputStrength_INTERNAL(executor, inputStrength);
}

std::error_code GetInteractiveExecutorSkinParameters(
    const IFaceInteractiveExecutor& executor, AnimatorSkinParams& params
    ) {
    return GetInteractiveExecutorSkinParameters_INTERNAL(executor, params);
}

std::error_code SetInteractiveExecutorSkinParameters(
    IFaceInteractiveExecutor& executor, const AnimatorSkinParams& params
    ) {
    return SetInteractiveExecutorSkinParameters_INTERNAL(executor, params);
}

std::error_code GetInteractiveExecutorTongueParameters(
    const IFaceInteractiveExecutor& executor, AnimatorTongueParams& params
    ) {
    return GetInteractiveExecutorTongueParameters_INTERNAL(executor, params);
}

std::error_code SetInteractiveExecutorTongueParameters(
    IFaceInteractiveExecutor& executor, const AnimatorTongueParams& params
    ) {
    return SetInteractiveExecutorTongueParameters_INTERNAL(executor, params);
}

std::error_code GetInteractiveExecutorTeethParameters(
    const IFaceInteractiveExecutor& executor, AnimatorTeethParams& params
    ) {
    return GetInteractiveExecutorTeethParameters_INTERNAL(executor, params);
}

std::error_code SetInteractiveExecutorTeethParameters(
    IFaceInteractiveExecutor& executor, const AnimatorTeethParams& params
    ) {
    return SetInteractiveExecutorTeethParameters_INTERNAL(executor, params);
}

std::error_code GetInteractiveExecutorEyesParameters(
    const IFaceInteractiveExecutor& executor, AnimatorEyesParams& params
    ) {
    return GetInteractiveExecutorEyesParameters_INTERNAL(executor, params);
}

std::error_code SetInteractiveExecutorEyesParameters(
    IFaceInteractiveExecutor& executor, const AnimatorEyesParams& params
    ) {
    return SetInteractiveExecutorEyesParameters_INTERNAL(executor, params);
}

std::error_code GetInteractiveExecutorGeometryExecutor(
    const IFaceInteractiveExecutor& executor, IGeometryInteractiveExecutor** geometryExecutor
    ) {
    return GetInteractiveExecutorGeometryExecutor_INTERNAL(executor, geometryExecutor);
}

std::error_code SetInteractiveExecutorGeometryResultsCallback(
    IFaceInteractiveExecutor& executor, IGeometryInteractiveExecutor::results_callback_t callback, void* userdata
    ) {
    return SetInteractiveExecutorGeometryResultsCallback_INTERNAL(executor, callback, userdata);
}

std::error_code GetInteractiveExecutorBlendshapeSkinConfig(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverConfig& config
    ) {
    return GetInteractiveExecutorBlendshapeSkinConfig_INTERNAL(executor, config);
}

std::error_code SetInteractiveExecutorBlendshapeSkinConfig(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverConfig& config
    ) {
    return SetInteractiveExecutorBlendshapeSkinConfig_INTERNAL(executor, config);
}

std::error_code GetInteractiveExecutorBlendshapeSkinParameters(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverParams& params
    ) {
    return GetInteractiveExecutorBlendshapeSkinParameters_INTERNAL(executor, params);
}

std::error_code SetInteractiveExecutorBlendshapeSkinParameters(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverParams& params
    ) {
    return SetInteractiveExecutorBlendshapeSkinParameters_INTERNAL(executor, params);
}

std::error_code GetInteractiveExecutorBlendshapeTongueConfig(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverConfig& config
    ) {
    return GetInteractiveExecutorBlendshapeTongueConfig_INTERNAL(executor, config);
}

std::error_code SetInteractiveExecutorBlendshapeTongueConfig(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverConfig& config
    ) {
    return SetInteractiveExecutorBlendshapeTongueConfig_INTERNAL(executor, config);
}

std::error_code GetInteractiveExecutorBlendshapeTongueParameters(
    const IFaceInteractiveExecutor& executor, BlendshapeSolverParams& params
    ) {
    return GetInteractiveExecutorBlendshapeTongueParameters_INTERNAL(executor, params);
}

std::error_code SetInteractiveExecutorBlendshapeTongueParameters(
    IFaceInteractiveExecutor& executor, const BlendshapeSolverParams& params
    ) {
    return SetInteractiveExecutorBlendshapeTongueParameters_INTERNAL(executor, params);
}

IGeometryInteractiveExecutor* CreateRegressionGeometryInteractiveExecutor(
    const GeometryExecutorCreationParameters& params,
    const IRegressionModel::GeometryExecutorCreationParameters& regressionParams,
    std::size_t batchSize
    ) {
    return CreateRegressionGeometryInteractiveExecutor_INTERNAL(params, regressionParams, batchSize);
}

IGeometryInteractiveExecutor* CreateDiffusionGeometryInteractiveExecutor(
    const GeometryExecutorCreationParameters& params,
    const IDiffusionModel::GeometryExecutorCreationParameters& diffusionParams,
    std::size_t nbInferencesForPreview
    ) {
    return CreateDiffusionGeometryInteractiveExecutor_INTERNAL(params, diffusionParams, nbInferencesForPreview);
}

IBlendshapeInteractiveExecutor* CreateHostBlendshapeSolveInteractiveExecutor(
    IGeometryInteractiveExecutor* transferredGeometryInteractiveExecutor,
    const HostBlendshapeSolveExecutorCreationParameters& params
    ) {
    return CreateHostBlendshapeSolveInteractiveExecutor_INTERNAL(transferredGeometryInteractiveExecutor, params);
}

IBlendshapeInteractiveExecutor* CreateDeviceBlendshapeSolveInteractiveExecutor(
    IGeometryInteractiveExecutor* transferredGeometryInteractiveExecutor,
    const DeviceBlendshapeSolveExecutorCreationParameters& params
    ) {
    return CreateDeviceBlendshapeSolveInteractiveExecutor_INTERNAL(transferredGeometryInteractiveExecutor, params);
}

IGeometryExecutorBundle* ReadRegressionGeometryExecutorBundle(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IRegressionModel::IGeometryModelInfo** outModelInfo
    ) {
    return ReadRegressionGeometryExecutorBundle_INTERNAL(
        nbTracks, path, executionOption, frameRateNumerator, frameRateDenominator, outModelInfo
        );
}

IGeometryExecutorBundle* ReadDiffusionGeometryExecutorBundle(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    std::size_t identityIndex,
    bool constantNoise,
    IDiffusionModel::IGeometryModelInfo** outModelInfo
    ) {
    return ReadDiffusionGeometryExecutorBundle_INTERNAL(
        nbTracks, path, executionOption, identityIndex, constantNoise, outModelInfo
        );
}

IBlendshapeExecutorBundle* ReadRegressionBlendshapeSolveExecutorBundle(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    bool useGpuSolver,
    std::size_t frameRateNumerator,
    std::size_t frameRateDenominator,
    IRegressionModel::IGeometryModelInfo** outModelInfo,
    IRegressionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
    ) {
    return ReadRegressionBlendshapeSolveExecutorBundle_INTERNAL(
        nbTracks,
        path,
        executionOption,
        useGpuSolver,
        frameRateNumerator,
        frameRateDenominator,
        outModelInfo,
        outBlendshapeSolveModelInfo
        );
}

IBlendshapeExecutorBundle* ReadDiffusionBlendshapeSolveExecutorBundle(
    std::size_t nbTracks,
    const char* path,
    IGeometryExecutor::ExecutionOption executionOption,
    bool useGpuSolver,
    std::size_t identityIndex,
    bool constantNoise,
    IDiffusionModel::IGeometryModelInfo** outModelInfo,
    IDiffusionModel::IBlendshapeSolveModelInfo** outBlendshapeSolveModelInfo
    ) {
    return ReadDiffusionBlendshapeSolveExecutorBundle_INTERNAL(
        nbTracks,
        path,
        executionOption,
        useGpuSolver,
        identityIndex,
        constantNoise,
        outModelInfo,
        outBlendshapeSolveModelInfo
        );
}

} // namespace nva2f
