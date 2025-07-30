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
#include "audio2face/internal/animator.h"
#include "audio2face/internal/logger.h"
#include "audio2face/internal/macros.h"
#include "audio2x/cuda_utils.h"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <cassert>

#include <cuda_runtime_api.h>

namespace nva2f
{

std::error_code AnimatorPcaReconstruction::Animate(
    nva2x::DeviceTensorFloatConstView inputPcaCoefs, nva2x::DeviceTensorFloatView outputVertices)
{
    CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "AnimatorPcaReconstruction: Animator Data is not set", ErrorCode::eDataNotSet);

    assert(inputPcaCoefs.Data() != nullptr);
    assert(outputVertices.Data() != nullptr);

    // outputVertices is both an input and output, fill it with 0 to avoid wrong values (NaN, INF).
    CHECK_RESULT(nva2x::FillOnDevice(outputVertices, 0.0f, _cudaStream));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasSgemv(
        _cublasHandle, CUBLAS_OP_N,
        static_cast<int>(_dataView.shapeSize), static_cast<int>(_numShapes),
        &alpha, _dataView.shapesMatrix.Data(), static_cast<int>(_dataView.shapeSize),
        inputPcaCoefs.Data(), 1,
        &beta, outputVertices.Data(), 1
        );
    CHECK_ERROR_WITH_MSG(status == CUBLAS_STATUS_SUCCESS, "Unable to run matrix multiplication", ErrorCode::eCublasExecutionError);

    return nva2x::ErrorCode::eSuccess;
}

//////////////////////////////////////////////

__global__ void InterpolatorInitDataKernel(const float* raw, float* dataArr, size_t size, unsigned int degree)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        for (int i = 0; i < degree + 1; ++i)
        {
            // Pseudocode: dataArr[i] = raw
            dataArr[idx + size * i] = raw[idx];
        }
    }
}

__global__ void InterpolatorUpdateKernel(
    const float* raw, float* dataArr, float* smoothed, size_t size, unsigned int degree, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Pseudocode: dataArr[0] = raw
        dataArr[idx] = raw[idx];
        for (int i = 1; i < degree + 1; ++i)
        {
            // Pseudocode: dataArr[i] += (dataArr[i-1] - dataArr[i]) * alpha;
            dataArr[idx + size * i] += (dataArr[idx + size * (i - 1)] - dataArr[idx + size * i]) * alpha;
        }
        // Pseudocode: smoothed = dataArr[-1];
        smoothed[idx] = dataArr[idx + size * degree];
    }
}

__global__ void InterpolatorUpdateKernelNoOp(
    const float* raw, float* dataArr, float* smoothed, size_t size, unsigned int degree)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Pseudocode: dataArr[0] = raw
        dataArr[idx] = raw[idx];
        for (int i = 1; i < degree + 1; ++i)
        {
            // Pseudocode: dataArr[i] = dataArr[i-1]
            dataArr[idx + size * i] = dataArr[idx + size * (i - 1)];
        }
        // Pseudocode: smoothed = dataArr[-1];
        smoothed[idx] = dataArr[idx + size * degree];
    }
}

//////////////////////////////////////////////

std::error_code Interpolator::Update(
    nva2x::DeviceTensorFloatConstView raw, float dt, nva2x::DeviceTensorFloatView smoothed)
{
    CHECK_ERROR_WITH_MSG(_initialized, "Interpolator is not initialized", nva2x::ErrorCode::eNotInitialized);
    CHECK_ERROR_WITH_MSG(raw.Size() == _size, "Mismatched sizes in Interpolator", nva2x::ErrorCode::eMismatch);
    CHECK_ERROR_WITH_MSG(smoothed.Size() == _size, "Mismatched sizes in Interpolator", nva2x::ErrorCode::eMismatch);

    assert(raw.Data() != nullptr);
    assert(smoothed.Data() != nullptr);

    dim3 numBlocks(IDIVUP(static_cast<unsigned int>(_size), 1024u));
    dim3 numThreads(1024u);

    if (!_dataArrInitialized) // first call
    {
        InterpolatorInitDataKernel<<<numBlocks, numThreads, 0, _cudaStream>>>(raw.Data(), _dataArr.Data(), _size, _degree);
        CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);
    }

    if (_smoothing > 0.0f) {
        float alpha = 1.0f - std::pow(0.5f, dt / _smoothing);
        InterpolatorUpdateKernel<<<numBlocks, numThreads, 0, _cudaStream>>>(
            raw.Data(), _dataArr.Data(), smoothed.Data(), _size, _degree, alpha);
        CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);
    }
    else {
        InterpolatorUpdateKernelNoOp<<<numBlocks, numThreads, 0, _cudaStream>>>(
            raw.Data(), _dataArr.Data(), smoothed.Data(), _size, _degree);
        CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);
    }

    if (!_dataArrInitialized)
    {
        _dataArrInitialized = true;
    }

    return nva2x::ErrorCode::eSuccess;
}

//////////////////////////////////////////////

__global__ void AnimatorSkinGetYKernel(const float* pose, float* poseY, size_t numVertices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices)
    {
        poseY[idx] = pose[1 + idx * 3];
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

__global__ void AnimatorSkinComposeKernelStep2(const float* smoothedLower,
                                          const float* smoothedUpper,
                                          const float* faceMaskLower,
                                          const float* neutralPose,
                                          float* pose,
                                          float lowerFaceStrength,
                                          float upperFaceStrength,
                                          size_t poseSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < poseSize)
    {
        int maskIdx = idx / 3;
        pose[idx] = neutralPose[idx] + smoothedUpper[idx] * upperFaceStrength * (1.0f - faceMaskLower[maskIdx]) +
                    smoothedLower[idx] * lowerFaceStrength * faceMaskLower[maskIdx];
    }
}

__global__ void AnimatorSkinComposeKernelStep1(const float* result,
                                            const float* eyeClosePoseDelta,
                                            const float* lipOpenPoseDelta,
                                            float* pose,
                                            float skinStrength,
                                            float eyelidOpenOffset,
                                            float blinkOffset,
                                            float blinkStrength,
                                            float lipOpenOffset,
                                            size_t poseSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < poseSize)
    {
        pose[idx] = skinStrength * result[idx] + eyeClosePoseDelta[idx] * (-eyelidOpenOffset + blinkOffset * blinkStrength) +
                    lipOpenPoseDelta[idx] * lipOpenOffset;
    }
}

__global__ void AnimatorTongueOffsetKernel(
    float* resultPos, const float* neutralPose, size_t poseSize, float strength, float heightOffset, float depthOffset
    )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < poseSize)
    {
        resultPos[idx] = neutralPose[idx] + resultPos[idx] * strength;
        if (idx % 3 == 1)
        {
            resultPos[idx] += heightOffset;
        }
        else if (idx % 3 == 2)
        {
            resultPos[idx] += depthOffset;
        }
    }
}

std::error_code AnimatorSkinCalculateFaceMaskLower(const float* neutralPoseData,
                                        float* faceMaskLowerData,
                                        size_t poseSize,
                                        float faceMaskLevel,
                                        float faceMaskSoftness,
                                        cudaStream_t cudaStream)
{
    CHECK_ERROR_WITH_MSG(faceMaskSoftness > 0, "Face Mask Softness should be greater than zero", ErrorCode::eOutOfRange);

    size_t numVertices = poseSize / 3;

    dim3 numBlocks(IDIVUP(static_cast<unsigned int>(numVertices), 1024u));
    dim3 numThreads(1024u);

    AnimatorSkinGetYKernel<<<numBlocks, numThreads, 0, cudaStream>>>(neutralPoseData, faceMaskLowerData, numVertices);
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    float min, max;
    try
    {
        const thrust::device_ptr<float> faceMaskLowerPtr = thrust::device_pointer_cast<float>(faceMaskLowerData);
        thrust::pair<const thrust::device_ptr<float>, const thrust::device_ptr<float>> minmax =
            thrust::minmax_element(thrust::cuda::par.on(cudaStream), faceMaskLowerPtr, faceMaskLowerPtr + numVertices);
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
        faceMaskLowerData, numVertices, min, max - min, faceMaskLevel, faceMaskSoftness);
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    return nva2x::ErrorCode::eSuccess;
}

//////////////////////////////////////////////

std::error_code AnimatorSkin::CalculateFaceMaskLower()
{
    return AnimatorSkinCalculateFaceMaskLower(_dataView.neutralPose.Data(), _faceMaskLower.Data(), _dataView.neutralPose.Size(),
                                              _params.faceMaskLevel, _params.faceMaskSoftness, _cudaStream);
}

std::error_code AnimatorSkin::Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, float dt, nva2x::DeviceTensorFloatView outputVertices)
{
    // _animatorDataIsSet guarantees readiness of _interpLower, _interpUpper, _smoothedLower, _smoothedUpper,
    // _faceMaskLower, _lipOpenPoseDelta, _eyeClosePoseDelta
    CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "AnimatorSkin: Animator Data is not set", ErrorCode::eDataNotSet);
    CHECK_ERROR_WITH_MSG(inputDeltas.Size() == _dataView.neutralPose.Size(), "Mismatched size for input deltas in AnimatorSkin", nva2x::ErrorCode::eMismatch);
    CHECK_ERROR_WITH_MSG(outputVertices.Size() == _dataView.neutralPose.Size(), "Mismatched size for output vertices in AnimatorSkin", nva2x::ErrorCode::eMismatch);

    assert(inputDeltas.Data() != nullptr);
    assert(outputVertices.Data() != nullptr);

    // Note that this function supports having input and output point to the same place.
    if (inputDeltas.Data() != outputVertices.Data()) {
        CHECK_RESULT(nva2x::CopyDeviceToDevice(outputVertices, inputDeltas, _cudaStream));
    }

    dim3 numBlocks(IDIVUP(static_cast<unsigned int>(_dataView.neutralPose.Size()), 1024u));
    dim3 numThreads(1024u);

    AnimatorSkinComposeKernelStep1<<<numBlocks, numThreads, 0, _cudaStream>>>(
        outputVertices.Data(), _dataView.eyeClosePoseDelta.Data(), _dataView.lipOpenPoseDelta.Data(), outputVertices.Data(),
        _params.skinStrength, _params.eyelidOpenOffset, _params.blinkOffset,
        _params.blinkStrength, _params.lipOpenOffset, _dataView.neutralPose.Size());
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    // original animator that blend upper and lower.
    CHECK_RESULT_WITH_MSG(_interpLower.Update(outputVertices, dt, _smoothedLower), "Unable to smooth lower pose");
    CHECK_RESULT_WITH_MSG(_interpUpper.Update(outputVertices, dt, _smoothedUpper), "Unable to smooth upper pose");

    AnimatorSkinComposeKernelStep2<<<numBlocks, numThreads, 0, _cudaStream>>>(
        _smoothedLower.Data(), _smoothedUpper.Data(), _faceMaskLower.Data(), _dataView.neutralPose.Data(), outputVertices.Data(),
        _params.lowerFaceStrength, _params.upperFaceStrength, _dataView.neutralPose.Size());
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    return nva2x::ErrorCode::eSuccess;
}

//////////////////////////////////////////////

std::error_code AnimatorTongue::Animate(
    nva2x::DeviceTensorFloatConstView inputDeltas, float dt, nva2x::DeviceTensorFloatView outputVertices)
{
    // _animatorDataIsSet guarantees readiness of _shapesMatrix, _shapesMean
    CHECK_ERROR_WITH_MSG(_animatorDataIsSet, "AnimatorTongue: Animator Data is not set", ErrorCode::eDataNotSet);
    CHECK_ERROR_WITH_MSG(inputDeltas.Size() == _dataView.neutralPose.Size(), "Mismatched size for input deltas in AnimatorTongue", nva2x::ErrorCode::eMismatch);
    CHECK_ERROR_WITH_MSG(outputVertices.Size() == _dataView.neutralPose.Size(), "Mismatched size for output vertices in AnimatorTongue", nva2x::ErrorCode::eMismatch);

    assert(inputDeltas.Data() != nullptr);
    assert(outputVertices.Data() != nullptr);

    // Note that this function supports having input and output point to the same place.
    if (inputDeltas.Data() != outputVertices.Data()) {
        CHECK_RESULT(nva2x::CopyDeviceToDevice(outputVertices, inputDeltas, _cudaStream));
    }

    dim3 numBlocks(IDIVUP(static_cast<unsigned int>(_dataView.neutralPose.Size()), 1024u));
    dim3 numThreads(1024u);

    AnimatorTongueOffsetKernel<<<numBlocks, numThreads, 0, _cudaStream>>>(
        outputVertices.Data(), _dataView.neutralPose.Data(), _dataView.neutralPose.Size(),
        _params.tongueStrength, _params.tongueHeightOffset, _params.tongueDepthOffset);
    CUDA_CHECK_ERROR(cudaGetLastError(), nva2x::ErrorCode::eCudaKernelError);

    return nva2x::ErrorCode::eSuccess;
}

} // namespace nva2f
