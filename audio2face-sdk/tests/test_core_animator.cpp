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
#include "audio2x/internal/cuda_stream.h"
#include "audio2x/internal/io.h"
#include "audio2x/internal/tensor_dict.h"
#include "utils.h"
#include <gtest/gtest.h>

class TestCoreInterpolator : public ::testing::Test {};

class TestCoreAnimatorSkin : public ::testing::Test {};

class TestCoreAnimatorTongue : public ::testing::Test {};

TEST_F(TestCoreInterpolator, TestUpdate) {
  nva2x::HostTensorDict testData;
  ASSERT_TRUE(!testData.ReadFromFile(TEST_DATA_DIR "_data/generated/audio2face-sdk/tests/data/test_data_interpolator.bin"));

  auto rawArrHostPtr = testData.At("raw_arr");
  auto dtArrPtr = testData.At("dt_arr");
  auto smoothedHostTruePtr = testData.At("smoothed");
  ASSERT_NE(rawArrHostPtr, nullptr);
  ASSERT_NE(dtArrPtr, nullptr);
  ASSERT_NE(smoothedHostTruePtr, nullptr);
  const nva2x::HostTensorFloat &rawArrHost = *rawArrHostPtr;
  const nva2x::HostTensorFloat &dtArr = *dtArrPtr;
  const nva2x::HostTensorFloat &smoothedHostTrue = *smoothedHostTruePtr;
  const size_t tensorSize = smoothedHostTrue.Size();

  nva2x::HostTensorFloat smoothedHost;
  ASSERT_TRUE(!smoothedHost.Allocate(tensorSize));

  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2x::DeviceTensorFloat rawDevice, smoothedDevice;
  ASSERT_TRUE(!rawDevice.Allocate(tensorSize));
  ASSERT_TRUE(!smoothedDevice.Allocate(tensorSize));

  nva2f::Interpolator interpolator;
  float smoothing = 0.2f;
  ASSERT_TRUE(!interpolator.Init(smoothing, tensorSize));
  ASSERT_TRUE(!interpolator.SetCudaStream(cudaStream.Data()));

  for (unsigned int t = 0; t < dtArr.Size(); ++t) {
    float dt = dtArr.Data()[t];
    ASSERT_TRUE(!nva2x::CopyHostToDevice(rawDevice, rawArrHost.View(t * tensorSize, tensorSize), cudaStream.Data()));
    ASSERT_TRUE(!interpolator.Update(rawDevice, dt, smoothedDevice));
    ASSERT_TRUE(!nva2x::CopyDeviceToHost(smoothedHost, smoothedDevice, cudaStream.Data()));
    ASSERT_TRUE(!cudaStream.Synchronize());
  }

  for (unsigned int i = 0; i < smoothedHost.Size(); ++i) {
    ASSERT_NEAR(smoothedHostTrue.Data()[i], smoothedHost.Data()[i], 1e-6)
        << "i = " << i;
  }
}

TEST_F(TestCoreAnimatorSkin, TestSetter) {
  std::vector<float> data(1234);
  FillRandom(data);

  const nva2x::HostTensorFloatConstView dataView{data.data(), data.size()};
  const nva2f::AnimatorSkin::HostData hostData = {dataView, dataView, dataView};

  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2f::AnimatorSkinParams animatorParams = {
      0.1f, 0.2f, 1.2f, 0.7f, 0.5f, 0.13f, 0.9f, 1.1f, 0.15f, -0.1f};
  nva2f::AnimatorSkin animator;
  ASSERT_TRUE(!animator.SetCudaStream(cudaStream.Data()));
  ASSERT_TRUE(!animator.Init(animatorParams));
  ASSERT_TRUE(!animator.SetAnimatorData(hostData));
  ASSERT_TRUE(!cudaStream.Synchronize());
  {
    nva2f::AnimatorSkinParams params = {
      1.0f, 0.2f, 1.2f, 0.7f, 0.5f, 0.13f, 0.9f, 1.1f, 0.15f, -0.1f};
    auto err = animator.SetParameters(params);
    LOG_INFO(err.message());
    ASSERT_TRUE(nva2f::ErrorCode::eOutOfRange == nva2f::get_error_code(err));
  }
  {
    auto params = animator.GetParameters();
    ASSERT_TRUE(0.1f == params.lowerFaceSmoothing);
  }
  {
    auto err = animator.SetLowerFaceSmoothing(NAN);
    LOG_INFO(err.message());

    ASSERT_TRUE(nva2f::ErrorCode::eNotANumber == nva2f::get_error_code(err));
  }
  {
    auto err = animator.SetUpperFaceSmoothing(INFINITY);
    LOG_INFO(err.message());
    ASSERT_TRUE(nva2f::ErrorCode::eOutOfRange == nva2f::get_error_code(err));
  }
  {
    auto err = animator.SetLowerFaceStrength(1);
    LOG_INFO(err.message());
    ASSERT_TRUE(!err);
  }
  {
    auto err = animator.SetUpperFaceStrength(-1);
    LOG_INFO(err.message());
    ASSERT_TRUE(nva2f::ErrorCode::eOutOfRange == nva2f::get_error_code(err));
  }
  {
    auto err = animator.SetFaceMaskLevel(1.1f);
    LOG_INFO(err.message());
    ASSERT_TRUE(nva2f::ErrorCode::eOutOfRange == nva2f::get_error_code(err));
  }
  {
    auto err = animator.SetFaceMaskSoftness(0.6f);
    LOG_INFO(err.message());
    ASSERT_TRUE(nva2f::ErrorCode::eOutOfRange == nva2f::get_error_code(err));
  }
  {
    auto err = animator.SetSkinStrength(2.1f);
    LOG_INFO(err.message());
    ASSERT_TRUE(nva2f::ErrorCode::eOutOfRange == nva2f::get_error_code(err));
  }
  {
    auto err = animator.SetBlinkStrength(2.1f);
    LOG_INFO(err.message());
    ASSERT_TRUE(nva2f::ErrorCode::eOutOfRange == nva2f::get_error_code(err));
  }
  {
    auto err = animator.SetEyelidOpenOffset(1.1f);
    LOG_INFO(err.message());
    ASSERT_TRUE(nva2f::ErrorCode::eOutOfRange == nva2f::get_error_code(err));;
  }
  {
    auto err = animator.SetLipOpenOffset(0.3f);
    LOG_INFO(err.message());
    ASSERT_TRUE(nva2f::ErrorCode::eOutOfRange == nva2f::get_error_code(err));
  }
  {
    auto err = animator.SetBlinkOffset(1.3f);
    LOG_INFO(err.message());
    ASSERT_TRUE(nva2f::ErrorCode::eOutOfRange == nva2f::get_error_code(err));
  }
  {
    auto err = animator.SetBlinkOffset(0.3f);
    LOG_INFO(err.message());
    ASSERT_TRUE(!err);
  }
}

TEST_F(TestCoreAnimatorSkin, TestRangeConfig) {
  ASSERT_TRUE((nva2f::RangeConfig<float>{0.0023f, 0, 0.1f} == nva2f::IAnimatorSkin::GetRangeConfig(&nva2f::AnimatorSkinParams::lowerFaceSmoothing)));
  ASSERT_TRUE((nva2f::RangeConfig<float>{0.001f, 0, 0.1f} == nva2f::IAnimatorSkin::GetRangeConfig(&nva2f::AnimatorSkinParams::upperFaceSmoothing)));
  ASSERT_TRUE((nva2f::RangeConfig<float>{1.3f, 0, 2.f} == nva2f::IAnimatorSkin::GetRangeConfig(&nva2f::AnimatorSkinParams::lowerFaceStrength)));
  ASSERT_TRUE((nva2f::RangeConfig<float>{1.f, 0, 2.f} == nva2f::IAnimatorSkin::GetRangeConfig(&nva2f::AnimatorSkinParams::upperFaceStrength)));
  ASSERT_TRUE((nva2f::RangeConfig<float>{0.6f, 0, 1.f} == nva2f::IAnimatorSkin::GetRangeConfig(&nva2f::AnimatorSkinParams::faceMaskLevel)));
  ASSERT_TRUE((nva2f::RangeConfig<float>{0.0085f, 0.001f, 0.5f} == nva2f::IAnimatorSkin::GetRangeConfig(&nva2f::AnimatorSkinParams::faceMaskSoftness)));
  ASSERT_TRUE((nva2f::RangeConfig<float>{1.f, 0, 2.f} == nva2f::IAnimatorSkin::GetRangeConfig(&nva2f::AnimatorSkinParams::skinStrength)));
  ASSERT_TRUE((nva2f::RangeConfig<float>{1.f, 0, 2.f} == nva2f::IAnimatorSkin::GetRangeConfig(&nva2f::AnimatorSkinParams::blinkStrength)));
  ASSERT_TRUE((nva2f::RangeConfig<float>{0.06f, -1.f, 1.f} == nva2f::IAnimatorSkin::GetRangeConfig(&nva2f::AnimatorSkinParams::eyelidOpenOffset)));
  ASSERT_TRUE((nva2f::RangeConfig<float>{-0.03f, -0.2f, 0.2f} == nva2f::IAnimatorSkin::GetRangeConfig(&nva2f::AnimatorSkinParams::lipOpenOffset)));
  ASSERT_TRUE((nva2f::RangeConfig<float>{0.f, 0.f, 1.f} == nva2f::IAnimatorSkin::GetRangeConfig(&nva2f::AnimatorSkinParams::blinkOffset)));
}

TEST_F(TestCoreAnimatorTongue, TestRangeConfig) {
  ASSERT_TRUE((nva2f::RangeConfig<float>{1.5f, 0, 3.f} == nva2f::IAnimatorTongue::GetRangeConfig(&nva2f::AnimatorTongueParams::tongueStrength)));
  ASSERT_TRUE((nva2f::RangeConfig<float>{0.2f, -3.f, 3.f} == nva2f::IAnimatorTongue::GetRangeConfig(&nva2f::AnimatorTongueParams::tongueHeightOffset)));
  ASSERT_TRUE((nva2f::RangeConfig<float>{0.13f, -3.f, 3.f} == nva2f::IAnimatorTongue::GetRangeConfig(&nva2f::AnimatorTongueParams::tongueDepthOffset)));
}

TEST_F(TestCoreAnimatorTongue, TestSetter) {
  std::vector<float> data(1234);
  FillRandom(data);

  const nva2x::HostTensorFloatConstView dataView{data.data(), data.size()};
  const nva2f::AnimatorTongue::HostData hostData = {dataView};

  nva2x::CudaStream cudaStream;
  ASSERT_TRUE(!cudaStream.Init());

  nva2f::AnimatorTongueParams animatorParams = {1.5f, 0.39f, 0.13f};
  nva2f::AnimatorTongue animator;
  ASSERT_TRUE(!animator.SetCudaStream(cudaStream.Data()));
  ASSERT_TRUE(!animator.Init(animatorParams));
  ASSERT_TRUE(!animator.SetAnimatorData(hostData));
  ASSERT_TRUE(!cudaStream.Synchronize());
  {
    nva2f::AnimatorTongueParams params = {15.0f, 0.39f, 0.13f};
    auto err = animator.SetParameters(params);
    LOG_INFO(err.message());
    ASSERT_TRUE(nva2f::ErrorCode::eOutOfRange == nva2f::get_error_code(err));
  }
  {
    auto params = animator.GetParameters();
    ASSERT_TRUE(1.5f == params.tongueStrength);
    ASSERT_TRUE(0.39f == params.tongueHeightOffset);
    ASSERT_TRUE(0.13f == params.tongueDepthOffset);
  }
  {
    auto err = animator.SetTongueStrength(2.0f);
    LOG_INFO(err.message());
    ASSERT_TRUE(!err);
  }
  {
    auto err = animator.SetTongueHeightOffset(4.0f);
    LOG_INFO(err.message());
    ASSERT_TRUE(nva2f::ErrorCode::eOutOfRange == nva2f::get_error_code(err));
  }
  {
    auto err = animator.SetTongueDepthOffset(3.0f);
    LOG_INFO(err.message());
    ASSERT_TRUE(!err);
  }
}
