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
#include "audio2x/internal/unique_ptr.h"
#include <gtest/gtest.h>

#include <array>
#include <random>
#include <vector>

TEST(TestCoreAnimatorEyes, TestInit) {
  // Default values.
  {
    nva2f::AnimatorEyes animator;
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(!animator.Init(animatorEyesParams));
    ASSERT_EQ(1.0f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(1.0f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().saccadeSeed);
  }

  // Valid non-default values.
  {
    nva2f::AnimatorEyes animator;
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.1f, 0.9f, 0.1f, -2.0f, -0.1f, 2.0f, 5.0f};
    ASSERT_TRUE(!animator.Init(animatorEyesParams));
    ASSERT_EQ(1.1f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(0.9f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(0.1f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(-2.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(-0.1f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(2.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(5.0f, animator.GetParameters().saccadeSeed);
  }

  // Invalid values.
  // NOTE: Init() does NOT perform validation, therefore it is a way to set
  // invalid values.
  {
    nva2f::AnimatorEyes animator;
    const nva2f::AnimatorEyesParams animatorEyesParams = {-1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(!animator.Init(animatorEyesParams));
    ASSERT_EQ(-1.0f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(1.0f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().saccadeSeed);
  }
  {
    nva2f::AnimatorEyes animator;
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(!animator.Init(animatorEyesParams));
    ASSERT_EQ(1.0f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(-1.0f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().saccadeSeed);
  }
  {
    nva2f::AnimatorEyes animator;
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 100.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(!animator.Init(animatorEyesParams));
    ASSERT_EQ(1.0f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(1.0f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(100.0f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().saccadeSeed);
  }
  {
    nva2f::AnimatorEyes animator;
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, -100.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(!animator.Init(animatorEyesParams));
    ASSERT_EQ(1.0f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(1.0f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(-100.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().saccadeSeed);
  }
  {
    nva2f::AnimatorEyes animator;
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, 0.0f, 100.0f, 0.0f, 0.0f};
    ASSERT_TRUE(!animator.Init(animatorEyesParams));
    ASSERT_EQ(1.0f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(1.0f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(100.0f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().saccadeSeed);
  }
  {
    nva2f::AnimatorEyes animator;
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, -100.0f, 0.0f};
    ASSERT_TRUE(!animator.Init(animatorEyesParams));
    ASSERT_EQ(1.0f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(1.0f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(-100.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().saccadeSeed);
  }
  {
    nva2f::AnimatorEyes animator;
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -10.0f};
    ASSERT_TRUE(!animator.Init(animatorEyesParams));
    ASSERT_EQ(1.0f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(1.0f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(-10.0f, animator.GetParameters().saccadeSeed);
  }
}

TEST(TestCoreAnimatorEyes, TestParameters) {
  nva2f::AnimatorEyes animator;

  {
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(!animator.Init(animatorEyesParams));
  }

  // Individual functions.
  // Default values.
  {
    ASSERT_TRUE(!animator.SetEyeballsStrength(1.0f));
    ASSERT_TRUE(!animator.SetSaccadeStrength(1.0f));
    ASSERT_TRUE(!animator.SetRightEyeballRotationOffsetX(0.0f));
    ASSERT_TRUE(!animator.SetRightEyeballRotationOffsetY(0.0f));
    ASSERT_TRUE(!animator.SetLeftEyeballRotationOffsetX(0.0f));
    ASSERT_TRUE(!animator.SetLeftEyeballRotationOffsetY(0.0f));
    ASSERT_TRUE(!animator.SetSaccadeSeed(0.0f));
    ASSERT_EQ(1.0f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(1.0f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().saccadeSeed);
  }

  // Valid non-default values.
  {
    ASSERT_TRUE(!animator.SetEyeballsStrength(1.1f));
    ASSERT_TRUE(!animator.SetSaccadeStrength(0.9f));
    ASSERT_TRUE(!animator.SetRightEyeballRotationOffsetX(0.1f));
    ASSERT_TRUE(!animator.SetRightEyeballRotationOffsetY(-2.0f));
    ASSERT_TRUE(!animator.SetLeftEyeballRotationOffsetX(-0.1f));
    ASSERT_TRUE(!animator.SetLeftEyeballRotationOffsetY(2.0f));
    ASSERT_TRUE(!animator.SetSaccadeSeed(5.0f));
    ASSERT_EQ(1.1f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(0.9f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(0.1f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(-2.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(-0.1f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(2.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(5.0f, animator.GetParameters().saccadeSeed);
  }

  // Invalid values.
  {
    ASSERT_FALSE(!animator.SetEyeballsStrength(-1.0f));
    ASSERT_EQ(1.1f, animator.GetParameters().eyeballsStrength);
    ASSERT_FALSE(!animator.SetSaccadeStrength(-1.0f));
    ASSERT_EQ(0.9f, animator.GetParameters().saccadeStrength);
    ASSERT_FALSE(!animator.SetRightEyeballRotationOffsetX(100.0f));
    ASSERT_EQ(0.1f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_FALSE(!animator.SetRightEyeballRotationOffsetY(-100.0f));
    ASSERT_EQ(-2.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_FALSE(!animator.SetLeftEyeballRotationOffsetX(100.0f));
    ASSERT_EQ(-0.1f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_FALSE(!animator.SetLeftEyeballRotationOffsetY(-100.0f));
    ASSERT_EQ(2.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_FALSE(!animator.SetSaccadeSeed(-10.0f));
    ASSERT_EQ(5.0f, animator.GetParameters().saccadeSeed);
  }

  // Function for all parameters at once.
  // Default values.
  {
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(!animator.SetParameters(animatorEyesParams));
    ASSERT_EQ(1.0f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(1.0f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(0.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(0.0f, animator.GetParameters().saccadeSeed);
  }

  // Valid non-default values.
  {
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.1f, 0.9f, 0.1f, -2.0f, -0.1f, 2.0f, 5.0f};
    ASSERT_TRUE(!animator.SetParameters(animatorEyesParams));
    ASSERT_EQ(1.1f, animator.GetParameters().eyeballsStrength);
    ASSERT_EQ(0.9f, animator.GetParameters().saccadeStrength);
    ASSERT_EQ(0.1f, animator.GetParameters().rightEyeballRotationOffsetX);
    ASSERT_EQ(-2.0f, animator.GetParameters().rightEyeballRotationOffsetY);
    ASSERT_EQ(-0.1f, animator.GetParameters().leftEyeballRotationOffsetX);
    ASSERT_EQ(2.0f, animator.GetParameters().leftEyeballRotationOffsetY);
    ASSERT_EQ(5.0f, animator.GetParameters().saccadeSeed);
  }

  // Invalid values.
  {
    const nva2f::AnimatorEyesParams animatorEyesParams = {-1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_FALSE(!animator.SetParameters(animatorEyesParams));
  }
  {
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_FALSE(!animator.SetParameters(animatorEyesParams));
  }
  {
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 100.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_FALSE(!animator.SetParameters(animatorEyesParams));
  }
  {
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, -100.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_FALSE(!animator.SetParameters(animatorEyesParams));
  }
  {
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, 0.0f, 100.0f, 0.0f, 0.0f};
    ASSERT_FALSE(!animator.SetParameters(animatorEyesParams));
  }
  {
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, -100.0f, 0.0f};
    ASSERT_FALSE(!animator.SetParameters(animatorEyesParams));
  }
  {
    const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -10.0f};
    ASSERT_FALSE(!animator.SetParameters(animatorEyesParams));
  }
}

TEST(TestCoreAnimatorEyes, TestExtraction) {
  auto animator = nva2x::ToUniquePtr(nva2f::CreateAnimatorEyes_INTERNAL());

  const nva2f::AnimatorEyesParams animatorEyesParams = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_TRUE(!animator->Init(animatorEyesParams));

  std::mt19937 gen;
  static constexpr int kNbSaccadeRuns = 100;
  for (int n = 0; n < kNbSaccadeRuns; ++n) {
    // Create a random set of saccade.
    const int nbSaccades = std::uniform_int_distribution<int>{4000, 6000}(gen);
    std::vector<float> saccadeRot(nbSaccades * 2);
    for (float& rot : saccadeRot) {
      rot = std::uniform_real_distribution<float>{-1.0f, 1.0f}(gen);
    }

    ASSERT_TRUE(!animator->SetAnimatorData({nva2x::ToConstView(saccadeRot)}));

    static constexpr int kNbRotationRuns = 100;
    for (int m = 0; m < kNbRotationRuns; ++m) {
      // Create a random eye result.
      const std::array<float, 4> eyesRotation = {
        std::uniform_real_distribution<float>{-10.0f, 10.0f}(gen),
        std::uniform_real_distribution<float>{-10.0f, 10.0f}(gen),
        std::uniform_real_distribution<float>{-10.0f, 10.0f}(gen),
        std::uniform_real_distribution<float>{-10.0f, 10.0f}(gen),
      };
      // Create a random saccade index.
      const int frameIndex = std::uniform_int_distribution<int>{0000, nbSaccades - 1}(gen);
      ASSERT_TRUE(!animator->SetFrameIndex(frameIndex));

      // Create the destination rotations.
      const std::array<float, 3> rightResultRotations {
        eyesRotation[0] + saccadeRot[2*frameIndex + 0],
        eyesRotation[1] + saccadeRot[2*frameIndex + 1],
        0.0f,
      };
      const std::array<float, 3> leftResultRotations {
        eyesRotation[2] + saccadeRot[2*frameIndex + 0],
        eyesRotation[3] + saccadeRot[2*frameIndex + 1],
        0.0f,
      };


      std::array<float, 3> rightEyeRotation;
      std::array<float, 3> leftEyeRotation;
      ASSERT_TRUE(!animator->ComputeEyesRotation(
        nva2x::ToView(rightEyeRotation), nva2x::ToView(leftEyeRotation), nva2x::ToConstView(eyesRotation)));


      // Check that they are equal.
      {
        ASSERT_NEAR(rightResultRotations[0], rightEyeRotation[0], 1e-4f);
        ASSERT_NEAR(rightResultRotations[1], rightEyeRotation[1], 1e-4f);
        ASSERT_NEAR(rightResultRotations[2], rightEyeRotation[2], 1e-4f);
        ASSERT_NEAR(leftResultRotations[0], leftEyeRotation[0], 1e-4f);
        ASSERT_NEAR(leftResultRotations[1], leftEyeRotation[1], 1e-4f);
        ASSERT_NEAR(leftResultRotations[2], leftEyeRotation[2], 1e-4f);
      }


      // Check that strength can disable the rotations.
      {
        ASSERT_TRUE(!animator->SetEyeballsStrength(0.0f));
        ASSERT_TRUE(!animator->SetSaccadeStrength(0.0f));
        ASSERT_TRUE(!animator->ComputeEyesRotation(
        nva2x::ToView(rightEyeRotation), nva2x::ToView(leftEyeRotation), nva2x::ToConstView(eyesRotation)));

        ASSERT_EQ(0.0f, rightEyeRotation[0]);
        ASSERT_EQ(0.0f, rightEyeRotation[1]);
        ASSERT_EQ(0.0f, rightEyeRotation[2]);
        ASSERT_EQ(0.0f, leftEyeRotation[0]);
        ASSERT_EQ(0.0f, leftEyeRotation[1]);
        ASSERT_EQ(0.0f, leftEyeRotation[2]);
      }

      // Check that offsets work.
      {
        const float rightEyeballRotationOffsetX = std::uniform_real_distribution<float>{-10.0f, 10.0f}(gen);
        const float rightEyeballRotationOffsetY = std::uniform_real_distribution<float>{-10.0f, 10.0f}(gen);
        const float leftEyeballRotationOffsetX = std::uniform_real_distribution<float>{-10.0f, 10.0f}(gen);
        const float leftEyeballRotationOffsetY = std::uniform_real_distribution<float>{-10.0f, 10.0f}(gen);

        ASSERT_TRUE(!animator->SetRightEyeballRotationOffsetX(rightEyeballRotationOffsetX));
        ASSERT_TRUE(!animator->SetRightEyeballRotationOffsetY(rightEyeballRotationOffsetY));
        ASSERT_TRUE(!animator->SetLeftEyeballRotationOffsetX(leftEyeballRotationOffsetX));
        ASSERT_TRUE(!animator->SetLeftEyeballRotationOffsetY(leftEyeballRotationOffsetY));
        ASSERT_TRUE(!animator->ComputeEyesRotation(
        nva2x::ToView(rightEyeRotation), nva2x::ToView(leftEyeRotation), nva2x::ToConstView(eyesRotation)));

        ASSERT_EQ(rightEyeballRotationOffsetX, rightEyeRotation[0]);
        ASSERT_EQ(rightEyeballRotationOffsetY, rightEyeRotation[1]);
        ASSERT_EQ(0.0f, rightEyeRotation[2]);
        ASSERT_EQ(leftEyeballRotationOffsetX, leftEyeRotation[0]);
        ASSERT_EQ(leftEyeballRotationOffsetY, leftEyeRotation[1]);
        ASSERT_EQ(0.0f, leftEyeRotation[2]);
      }

      ASSERT_TRUE(!animator->SetEyeballsStrength(1.0f));
      ASSERT_TRUE(!animator->SetSaccadeStrength(1.0f));
      ASSERT_TRUE(!animator->SetRightEyeballRotationOffsetX(0.0f));
      ASSERT_TRUE(!animator->SetRightEyeballRotationOffsetY(0.0f));
      ASSERT_TRUE(!animator->SetLeftEyeballRotationOffsetX(0.0f));
      ASSERT_TRUE(!animator->SetLeftEyeballRotationOffsetY(0.0f));
    }
  }
}
