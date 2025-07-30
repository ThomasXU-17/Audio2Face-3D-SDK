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
#include "audio2face/internal/multitrack_animator_cuda.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2face/error.h"
#include "audio2x/error.h"

#include <cassert>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <cuda_runtime_api.h>

#include "tbtSVD/SVD.h"


namespace {

__global__ void TracksSetKernel(
  std::uint64_t* deviceBits, std::size_t size, std::uint64_t value
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  deviceBits[idx] = value;
}


__global__ void AnimatorSkinGetYKernelPacked(float* poseY, const float* animatorData, size_t animatorDataStride, size_t numVertices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices)
    {
        poseY[idx] = animatorData[(1 + idx * 3) * animatorDataStride + 2];
    }
}

__global__ void AnimatorSkinGetFaceMaskLowerKernel(
    float* faceMaskLower, size_t numVertices, float min, float maxSubMin, float faceMaskLevel, float faceMaskSoftness)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices)
    {
        faceMaskLower[idx] =
            1.0f / (1.0f + expf(-(faceMaskLevel - (faceMaskLower[idx] - min) / maxSubMin) / faceMaskSoftness));
    }
}


__device__ void ComputeSkinPostProcessing(
  float& result,
  float inputDelta,
  float eyeClosePoseDelta, float lipOpenPoseDelta, float neutralPose, float faceMaskLower,
  float& lower1, float& lower2,
  float& upper1, float& upper2,
  float skinStrength, float eyelidOpenOffset, float blinkOffset, float blinkStrength, float lipOpenOffset,
  float lowerFaceAlpha, float upperFaceAlpha,
  float lowerFaceStrength, float upperFaceStrength,
  bool initialized
  ) {
  // Step 1.
  const float delta =
    skinStrength * inputDelta +
    eyeClosePoseDelta * (-eyelidOpenOffset + blinkOffset * blinkStrength) +
    lipOpenPoseDelta * lipOpenOffset;

  // Smoothing.
  if (initialized && lowerFaceAlpha > 0.0f) {
    lower1 += (delta - lower1) * lowerFaceAlpha;
    lower2 += (lower1 - lower2) * lowerFaceAlpha;
  }
  else {
    lower1 = delta;
    lower2 = delta;
  }

  if (initialized && upperFaceAlpha > 0.0f) {
    upper1 += (delta - upper1) * upperFaceAlpha;
    upper2 += (upper1 - upper2) * upperFaceAlpha;
  }
  else {
    upper1 = delta;
    upper2 = delta;
  }

  // Step 2.
  result =
    neutralPose +
    upper2 * upperFaceStrength * (1.0f - faceMaskLower) +
    lower2 * lowerFaceStrength * faceMaskLower;
}

__global__ void AnimatorSkinEverythingKernel(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= poseSize * nbTracks) {
    return;
  }

  const std::size_t trackIdx = idx / poseSize;

  const std::size_t elemIdx = idx % poseSize;
  const bool initialized = true;

  float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
  const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

  const float eyeClosePoseDeltaValue = animatorData[elemIdx * animatorDataStride + 0];
  const float lipOpenPoseDeltaValue = animatorData[elemIdx * animatorDataStride + 1];
  const float neutralPoseValue = animatorData[elemIdx * animatorDataStride + 2];
  const float faceMaskLowerValue = faceMaskLower[idx / 3];

  float& lower1 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 0];
  float& lower2 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 1];
  float& upper1 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 2];
  float& upper2 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 3];

  const float skinStrength = params[trackIdx * paramsStride + 0];
  const float eyelidOpenOffset = params[trackIdx * paramsStride + 1];
  const float blinkOffset = params[trackIdx * paramsStride + 2];
  const float blinkStrength = params[trackIdx * paramsStride + 3];
  const float lipOpenOffset = params[trackIdx * paramsStride + 4];
  const float lowerFaceAlpha = params[trackIdx * paramsStride + 5];
  const float upperFaceAlpha = params[trackIdx * paramsStride + 6];
  const float lowerFaceStrength = params[trackIdx * paramsStride + 7];
  const float upperFaceStrength = params[trackIdx * paramsStride + 8];

  ComputeSkinPostProcessing(
    result,
    inputDelta,
    eyeClosePoseDeltaValue, lipOpenPoseDeltaValue, neutralPoseValue, faceMaskLowerValue,
    lower1, lower2,
    upper1, upper2,
    skinStrength, eyelidOpenOffset, blinkOffset, blinkStrength, lipOpenOffset,
    lowerFaceAlpha, upperFaceAlpha, lowerFaceStrength, upperFaceStrength,
    initialized
  );
}

__global__ void AnimatorSkinControlKernel(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const std::uint64_t* activeTracks,
  const std::uint64_t* initializedTracks,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= poseSize * nbTracks) {
    return;
  }

  const std::size_t trackIdx = idx / poseSize;

  const bool active = (activeTracks[trackIdx / 64] >> (trackIdx % 64)) & 0b1;
  if (!active) {
    return;
  }

  const bool initialized = (initializedTracks[trackIdx / 64] >> (trackIdx % 64)) & 0b1;

  const std::size_t elemIdx = idx % poseSize;
  float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
  const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

  const float eyeClosePoseDeltaValue = animatorData[elemIdx * animatorDataStride + 0];
  const float lipOpenPoseDeltaValue = animatorData[elemIdx * animatorDataStride + 1];
  const float neutralPoseValue = animatorData[elemIdx * animatorDataStride + 2];
  const float faceMaskLowerValue = faceMaskLower[idx / 3];

  float& lower1 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 0];
  float& lower2 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 1];
  float& upper1 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 2];
  float& upper2 = interpData[(poseSize * trackIdx + elemIdx) * interpDataStride + 3];

  const float skinStrength = params[trackIdx * paramsStride + 0];
  const float eyelidOpenOffset = params[trackIdx * paramsStride + 1];
  const float blinkOffset = params[trackIdx * paramsStride + 2];
  const float blinkStrength = params[trackIdx * paramsStride + 3];
  const float lipOpenOffset = params[trackIdx * paramsStride + 4];
  const float lowerFaceAlpha = params[trackIdx * paramsStride + 5];
  const float upperFaceAlpha = params[trackIdx * paramsStride + 6];
  const float lowerFaceStrength = params[trackIdx * paramsStride + 7];
  const float upperFaceStrength = params[trackIdx * paramsStride + 8];

  ComputeSkinPostProcessing(
    result,
    inputDelta,
    eyeClosePoseDeltaValue, lipOpenPoseDeltaValue, neutralPoseValue, faceMaskLowerValue,
    lower1, lower2,
    upper1, upper2,
    skinStrength, eyelidOpenOffset, blinkOffset, blinkStrength, lipOpenOffset,
    lowerFaceAlpha, upperFaceAlpha, lowerFaceStrength, upperFaceStrength,
    initialized
  );
}


__global__ void AnimateTongueKernel(
  float* results, std::size_t resultsOffset, std::size_t resultsStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t poseSize, std::size_t nbTracks
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < poseSize * nbTracks) {
    const std::size_t trackIdx = idx / poseSize;
    const std::size_t elemIdx = idx % poseSize;

    float& result = results[trackIdx * resultsStride + resultsOffset + elemIdx];
    const float inputDelta = inputDeltas[trackIdx * inputDeltasStride + inputDeltasOffset + elemIdx];

    const float neutralPoseValue = neutralPose[elemIdx];

    const float tongueStrength = params[trackIdx * paramsStride + 0];
    const float tongueHeightOffset = params[trackIdx * paramsStride + 1];
    const float tongueDepthOffset = params[trackIdx * paramsStride + 2];

    result = neutralPoseValue + inputDelta * tongueStrength;
    if (elemIdx % 3 == 1) {
      result += tongueHeightOffset;
    }
    if (elemIdx % 3 == 2) {
      result += tongueDepthOffset;
    }
  }
}


// This kernel could be optimized to use more parallelism, registers, etc.
// But it did not seem to make a difference on measured performance given
// the small amount of memory and computation involved.
__global__ void ComputeJawTransformKernel(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t nbPoints,
  std::size_t nbTracks
 ) {
  const auto track_idx = blockIdx.x;
  // Keep 1 thread per block.
  if (threadIdx.x != 0) {
    return;
  }

  extern __shared__ float sharedMemory[];
  std::size_t sharedMemoryOffset = 0;

  // Compute the pose of the jaw (the target pose).
  const auto lowerTeethStrength = params[track_idx * paramsStride + 0];
  const auto offsets = params + track_idx * paramsStride + 1;

  inputDeltas += track_idx * inputDeltasStride + inputDeltasOffset;
  float* targetPose = sharedMemory + sharedMemoryOffset;
  sharedMemoryOffset += nbPoints * 3;
  for (std::size_t i = 0; i < nbPoints; ++i) {
    for (int componentIdx = 0; componentIdx < 3; ++componentIdx) {
      targetPose[i*3 + componentIdx] = neutralPose[i*3 + componentIdx] +
        inputDeltas[i*3 + componentIdx] * lowerTeethStrength + offsets[componentIdx];
    }
  }

  // Compute the mean of the target pose.
  float* targetMean = sharedMemory + sharedMemoryOffset;
  sharedMemoryOffset += 3;
  for (int componentIdx = 0; componentIdx < 3; ++componentIdx) {
    float average = targetPose[componentIdx];
    for (std::size_t vertexIdx = 1; vertexIdx < nbPoints; ++vertexIdx) {
      average += targetPose[vertexIdx * 3 + componentIdx];
    }
    average /= nbPoints;
    targetMean[componentIdx] = average;
  }

  // The source pose is actually the neutral pose, we already have the average, it's constant,
  // it's stored after the neutral pose.
  const float* sourcePose = neutralPose;
  const float* sourceMean = neutralPose + nbPoints * 3;

  // Compute the 3x3 H matrix.
  float* H = sharedMemory + sharedMemoryOffset;
  sharedMemoryOffset += 9;
  for (int row_idx = 0; row_idx < 3; ++row_idx) {
    for (int col_idx = 0; col_idx < 3; ++col_idx) {
      float value = 0;
      for (std::size_t vertex_idx = 0; vertex_idx < nbPoints; ++vertex_idx) {
        const float targetDelta = targetPose[vertex_idx * 3 + col_idx] - targetMean[col_idx];
        const float sourceDelta = sourcePose[vertex_idx * 3 + row_idx] - sourceMean[row_idx];
        value += sourceDelta * targetDelta;
      }
      // Store matrix in row major since it will help with dot products used
      // to reconstruct the rotation matrix.
      H[row_idx * 3 + col_idx] = value;
    }
  }

  // Compute SVD.
  float* U = sharedMemory + sharedMemoryOffset;
  sharedMemoryOffset += 9;
  float* V = sharedMemory + sharedMemoryOffset;
  sharedMemoryOffset += 9;
  const auto A = SVD::Mat3x3::fromPtr(H, 0, 1);
  const auto usv = SVD::svd(A);
  usv.U.toPtr(U, 0, 1);
  usv.V.toPtr(V, 0, 1);

  // Re-use H to store rotation results.
  for (int row_idx = 0; row_idx < 3; ++row_idx) {
    for (int col_idx = 0; col_idx < 3; ++col_idx) {
      const auto u = U + 3 * col_idx;
      const auto v = V + 3 * row_idx;
      // Store the rotation matrix in column major order since that will match the output.
      H[col_idx * 3 + row_idx] = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
    }
  }

  // Compute determinant of H
  const float det = H[0*3 + 0] * (H[1*3 + 1] * H[2*3 + 2] - H[1*3 + 2] * H[2*3 + 1])
                  - H[0*3 + 1] * (H[1*3 + 0] * H[2*3 + 2] - H[1*3 + 2] * H[2*3 + 0])
                  + H[0*3 + 2] * (H[1*3 + 0] * H[2*3 + 1] - H[1*3 + 1] * H[2*3 + 0]);
  H[3*2 + 0] /= det;
  H[3*2 + 1] /= det;
  H[3*2 + 2] /= det;

  float* tt = sharedMemory + sharedMemoryOffset;
  sharedMemoryOffset += 3;
  tt[0] = targetMean[0] - (sourceMean[0] * H[3*0 + 0] + sourceMean[1] * H[3*1 + 0] + sourceMean[2] * H[3*2 + 0]);
  tt[1] = targetMean[1] - (sourceMean[0] * H[3*0 + 1] + sourceMean[1] * H[3*1 + 1] + sourceMean[2] * H[3*2 + 1]);
  tt[2] = targetMean[2] - (sourceMean[0] * H[3*0 + 2] + sourceMean[1] * H[3*1 + 2] + sourceMean[2] * H[3*2 + 2]);

  // Store the result.
  const auto output = result + track_idx * resultStride + resultOffset;
  output[0*4 + 0] = H[0*3 + 0];
  output[0*4 + 1] = H[0*3 + 1];
  output[0*4 + 2] = H[0*3 + 2];
  output[0*4 + 3] = 0;
  output[1*4 + 0] = H[1*3 + 0];
  output[1*4 + 1] = H[1*3 + 1];
  output[1*4 + 2] = H[1*3 + 2];
  output[1*4 + 3] = 0;
  output[2*4 + 0] = H[2*3 + 0];
  output[2*4 + 1] = H[2*3 + 1];
  output[2*4 + 2] = H[2*3 + 2];
  output[2*4 + 3] = 0;
  output[3*4 + 0] = tt[0];
  output[3*4 + 1] = tt[1];
  output[3*4 + 2] = tt[2];
  output[3*4 + 3] = 1;
}


__device__ inline void ComputeEyesRotation(
  float* outputEyesRotation, std::size_t outputEyesRotationOffset, std::size_t outputEyesRotationStride,
  const float* inputEyesRotationResult, std::size_t inputEyesRotationResultOffset, std::size_t inputEyesRotationResultStride,
  const float* params, std::size_t paramsStride,
  const float* saccadeRot, std::size_t saccadeRotSize,
  float dt,
  float* liveTime,
  const std::uint64_t* activeTracks,
  std::size_t nbTracks
 ) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto trackIdx = idx / 6;
  if (trackIdx >= nbTracks) {
    return;
  }
  const bool active = !activeTracks || ((activeTracks[trackIdx / 64] >> (trackIdx % 64)) & 0b1);
  if (!active) {
    return;
  }

  const auto output = outputEyesRotation + trackIdx * outputEyesRotationStride + outputEyesRotationOffset;

  const auto elementIdx = idx % 6;
  const auto eyeIdx = elementIdx / 3;
  const auto axisIdx = elementIdx % 3;

  float value;
  if (axisIdx < 2) {
    const auto inputEyes = inputEyesRotationResult + trackIdx * inputEyesRotationResultStride + inputEyesRotationResultOffset;
    const auto input = inputEyes[eyeIdx*2 + axisIdx];

    const auto eyeParams = params + trackIdx * paramsStride;
    const auto eyeballsStrength = eyeParams[0];
    const auto saccadeStrength = eyeParams[1];
    const auto offsets = eyeParams + 2;
    const auto eyeOffset = offsets[eyeIdx*2 + axisIdx];
    const auto saccadeSeed = eyeParams[6];

    auto& outLiveTime = liveTime[trackIdx];

    const auto maxNbFrames = saccadeRotSize / 2;
    auto wrap = [max=static_cast<float>(maxNbFrames)](float value) {
      float result = fmod(value, max);
      if (result < 0.0f) {
        result += max;
      }
      return result;
    };

    // Compute frame index from live time and seed
    float inLiveTime = outLiveTime;
    float totalTime = saccadeSeed + inLiveTime;
    totalTime = wrap(totalTime);
    const auto frameIdx = static_cast<int>(totalTime);
    assert(frameIdx >= 0);
    assert(frameIdx < maxNbFrames);
    const float saccade = saccadeRot[2 * frameIdx + axisIdx];

    // Using explicit intrinsics to prevent fma and get the same results as the CPU version
    value = eyeOffset + __fmul_rn(input, eyeballsStrength) + __fmul_rn(saccadeStrength, saccade);

    if (elementIdx == 0) {
      // Increment live time
      // Assume the saccade information was meant for 30 FPS, otherwise there would be more
      // movement with a higher FPS.
      constexpr float fps = 30.0f;
      const float increment = dt * fps;

      // Increment live time and wrap so it doesn't grow indefinitely
      inLiveTime += increment;
      inLiveTime = wrap(inLiveTime);
      outLiveTime = inLiveTime;
    }
  }
  else {
    assert(axisIdx == 2);
    value = 0.0f;
  }

  output[elementIdx] = value;
}

__global__ void ComputeEyesRotationEverythingKernel(
  float* outputEyesRotation, std::size_t outputEyesRotationOffset, std::size_t outputEyesRotationStride,
  const float* inputEyesRotationResult, std::size_t inputEyesRotationResultOffset, std::size_t inputEyesRotationResultStride,
  const float* params, std::size_t paramsStride,
  const float* saccadeRot, std::size_t saccadeRotSize,
  float dt,
  float* liveTime,
  std::size_t nbTracks
 ) {
  // This kernel could be optimized not to check if the track is active.
  // But it doesn't seem to make a difference in performance, the compiler might even be able
  // to optimize the check away.
  ComputeEyesRotation(
    outputEyesRotation, outputEyesRotationOffset, outputEyesRotationStride,
    inputEyesRotationResult, inputEyesRotationResultOffset, inputEyesRotationResultStride,
    params, paramsStride,
    saccadeRot, saccadeRotSize,
    dt,
    liveTime,
    nullptr,
    nbTracks
  );
}

__global__ void ComputeEyesRotationControlKernel(
  float* outputEyesRotation, std::size_t outputEyesRotationOffset, std::size_t outputEyesRotationStride,
  const float* inputEyesRotationResult, std::size_t inputEyesRotationResultOffset, std::size_t inputEyesRotationResultStride,
  const float* params, std::size_t paramsStride,
  const float* saccadeRot, std::size_t saccadeRotSize,
  float dt,
  float* liveTime,
  const std::uint64_t* activeTracks,
  std::size_t nbTracks
 ) {
  ComputeEyesRotation(
    outputEyesRotation, outputEyesRotationOffset, outputEyesRotationStride,
    inputEyesRotationResult, inputEyesRotationResultOffset, inputEyesRotationResultStride,
    params, paramsStride,
    saccadeRot, saccadeRotSize,
    dt,
    liveTime,
    activeTracks,
    nbTracks
  );
}


} // End of anonymous namespace


namespace nva2f::cuda {

std::error_code Tracks_Set(
  std::uint64_t* deviceBits, const std::uint64_t* hostBits, std::size_t size,
  cudaStream_t cudaStream
  ) {
  // This function does something very similar to a memcpy, but does it using kernels
  // so it doesn't have to synchronize CPU / GPU and we can keep enqueuing stuff without
  // keeping the memory untouched on the CPU side.
  //
  // We could "unroll" this to have even fewer kernel launches.
  dim3 numBlocks(IDIVUP(1u, 32u));
  dim3 numThreads(32u);

  for (std::size_t i = 0; i < size; ++i) {
    TracksSetKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
      deviceBits + i, 1, hostBits[i]
      );
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);
  }

  return nva2x::ErrorCode::eSuccess;
}


std::error_code CalculateFaceMaskLowerFromPackedNeutralPose(
  float* faceMaskLower,
  const float* animatorData,
  std::size_t animatorDataStride,
  std::size_t poseSize,
  float faceMaskLevel,
  float faceMaskSoftness,
  cudaStream_t cudaStream
) {
    assert(poseSize % 3 == 0);
    A2F_CHECK_ERROR_WITH_MSG(faceMaskSoftness > 0, "Face Mask Softness should be greater than zero", ErrorCode::eOutOfRange);

    const size_t numVertices = poseSize / 3;

    dim3 numBlocks(IDIVUP(static_cast<unsigned int>(numVertices), 1024u));
    dim3 numThreads(1024u);

    AnimatorSkinGetYKernelPacked<<<numBlocks, numThreads, 0, cudaStream>>>(
      faceMaskLower, animatorData, animatorDataStride, numVertices
      );
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    float min, max;
    try
    {
        const thrust::device_ptr<float> faceMaskLowerPtr = thrust::device_pointer_cast<float>(faceMaskLower);
        thrust::pair<const thrust::device_ptr<float>, const thrust::device_ptr<float>> minmax =
            thrust::minmax_element(thrust::cuda::par.on(cudaStream), faceMaskLowerPtr, faceMaskLowerPtr + numVertices);
        A2F_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaThrustError);
        min = *(minmax.first);
        max = *(minmax.second);
    }
    catch (...)
    {
        LOG_ERROR("Unable to calculate minmax for neutral pose Y values");
        return nva2x::ErrorCode::eCudaThrustError;
    }

    CHECK_ERROR_WITH_MSG(max - min > 0, "Neutral pose: max Y value should be greater than min Y value", ErrorCode::eOutOfRange);

    AnimatorSkinGetFaceMaskLowerKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
        faceMaskLower, numVertices, min, max - min, faceMaskLevel, faceMaskSoftness);
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    return nva2x::ErrorCode::eSuccess;
}


std::error_code AnimateSkin_Everything(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorSkinEverythingKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        animatorData, animatorDataStride,
        faceMaskLower,
        interpData, interpDataStride,
        params, paramsStride,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code AnimateSkin_Control(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* animatorData, std::size_t animatorDataStride,
  const float* faceMaskLower,
  float* interpData, std::size_t interpDataStride,
  const float* params, std::size_t paramsStride,
  const std::uint64_t* activeTracks,
  const std::uint64_t* initializedTracks,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimatorSkinControlKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        animatorData, animatorDataStride,
        faceMaskLower,
        interpData, interpDataStride,
        params, paramsStride,
        activeTracks, initializedTracks,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code AnimateTongue(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralPose,
  const float* params, std::size_t paramsStride,
  std::size_t poseSize, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(poseSize * nbTracks), 1024u));
  dim3 numThreads(1024u);

  AnimateTongueKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        neutralPose,
        params, paramsStride,
        poseSize, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code ComputeJawTransform(
  float* result, std::size_t resultOffset, std::size_t resultStride,
  const float* inputDeltas, std::size_t inputDeltasOffset, std::size_t inputDeltasStride,
  const float* neutralJaw,
  const float* params, std::size_t paramsStride,
  std::size_t nbPoints, std::size_t nbTracks,
  cudaStream_t cudaStream
  ) {
  dim3 numBlocks(static_cast<unsigned int>(nbTracks));
  dim3 numThreads(32u);

  const std::size_t sharedMemorySize = (nbPoints * (3) + (3 + 9 + 9 + 9 + 3)) * sizeof(float);

  ComputeJawTransformKernel<<<numBlocks, numThreads, sharedMemorySize, cudaStream>>>(
        result, resultOffset, resultStride,
        inputDeltas, inputDeltasOffset, inputDeltasStride,
        neutralJaw,
        params, paramsStride,
        nbPoints, nbTracks
        );
  CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}


std::error_code ComputeEyesRotation_Everything(
    float* outputEyesRotation, std::size_t outputEyesRotationOffset, std::size_t outputEyesRotationStride,
    const float* inputEyesRotationResult, std::size_t inputEyesRotationResultOffset, std::size_t inputEyesRotationResultStride,
    const float* params, std::size_t paramsStride,
    const float* saccadeRot, std::size_t saccadeRotSize,
    float dt,
    float* liveTime,
    std::size_t nbTracks,
    cudaStream_t cudaStream
) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(nbTracks * 6), 1024u));
  dim3 numThreads(1024u);

  ComputeEyesRotationEverythingKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
    outputEyesRotation, outputEyesRotationOffset, outputEyesRotationStride,
    inputEyesRotationResult, inputEyesRotationResultOffset, inputEyesRotationResultStride,
    params, paramsStride,
    saccadeRot, saccadeRotSize,
    dt,
    liveTime,
    nbTracks
    );
  A2F_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}

std::error_code ComputeEyesRotation_Control(
    float* outputEyesRotation, std::size_t outputEyesRotationOffset, std::size_t outputEyesRotationStride,
    const float* inputEyesRotationResult, std::size_t inputEyesRotationResultOffset, std::size_t inputEyesRotationResultStride,
    const float* params, std::size_t paramsStride,
    const float* saccadeRot, std::size_t saccadeRotSize,
    float dt,
    float* liveTime,
    const std::uint64_t* activeTracks,
    std::size_t nbTracks,
    cudaStream_t cudaStream
) {
  dim3 numBlocks(IDIVUP(static_cast<unsigned int>(nbTracks * 6), 1024u));
  dim3 numThreads(1024u);

  ComputeEyesRotationControlKernel<<<numBlocks, numThreads, 0, cudaStream>>>(
    outputEyesRotation, outputEyesRotationOffset, outputEyesRotationStride,
    inputEyesRotationResult, inputEyesRotationResultOffset, inputEyesRotationResultStride,
    params, paramsStride,
    saccadeRot, saccadeRotSize,
    dt,
    liveTime,
    activeTracks,
    nbTracks
    );
  A2F_CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

  return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f::cuda
