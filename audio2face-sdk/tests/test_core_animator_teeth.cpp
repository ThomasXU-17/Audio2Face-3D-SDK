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
#include "audio2face/internal/eigen_utils.h"
#include "audio2x/internal/unique_ptr.h"

#include <gtest/gtest.h>

#include <random>
#include <vector>

TEST(TestCoreAnimatorTeeth, TestInit) {
  // Default values.
  {
    nva2f::AnimatorTeeth animator;
    const nva2f::AnimatorTeethParams animatorTeethParams = {1.0f, 0.0f, 0.0f};
    ASSERT_TRUE(!animator.Init(animatorTeethParams));
    ASSERT_EQ(1.0f, animator.GetParameters().lowerTeethStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().lowerTeethHeightOffset);
    ASSERT_EQ(0.0f, animator.GetParameters().lowerTeethDepthOffset);
  }

  // Valid non-default values.
  {
    nva2f::AnimatorTeeth animator;
    const nva2f::AnimatorTeethParams animatorTeethParams = {1.3f, -0.1f, 0.25f};
    ASSERT_TRUE(!animator.Init(animatorTeethParams));
    ASSERT_EQ(1.3f, animator.GetParameters().lowerTeethStrength);
    ASSERT_EQ(-0.1f, animator.GetParameters().lowerTeethHeightOffset);
    ASSERT_EQ(0.25f, animator.GetParameters().lowerTeethDepthOffset);
  }

  // Invalid values.
  // NOTE: Init() does NOT perform validation, therefore it is a way to set
  // invalid values.
  {
    nva2f::AnimatorTeeth animator;
    const nva2f::AnimatorTeethParams animatorTeethParams = {-1.0f, 0.0f, 0.0f};
    ASSERT_TRUE(!animator.Init(animatorTeethParams));
    ASSERT_EQ(-1.0f, animator.GetParameters().lowerTeethStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().lowerTeethHeightOffset);
    ASSERT_EQ(0.0f, animator.GetParameters().lowerTeethDepthOffset);
  }
  {
    nva2f::AnimatorTeeth animator;
    const nva2f::AnimatorTeethParams animatorTeethParams = {1.0f, 100.0f, 0.0f};
    ASSERT_TRUE(!animator.Init(animatorTeethParams));
    ASSERT_EQ(1.0f, animator.GetParameters().lowerTeethStrength);
    ASSERT_EQ(100.0f, animator.GetParameters().lowerTeethHeightOffset);
    ASSERT_EQ(0.0f, animator.GetParameters().lowerTeethDepthOffset);
  }
  {
    nva2f::AnimatorTeeth animator;
    const nva2f::AnimatorTeethParams animatorTeethParams = {1.0f, 0.0f, -100.0f};
    ASSERT_TRUE(!animator.Init(animatorTeethParams));
    ASSERT_EQ(1.0f, animator.GetParameters().lowerTeethStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().lowerTeethHeightOffset);
    ASSERT_EQ(-100.0f, animator.GetParameters().lowerTeethDepthOffset);
  }
}

TEST(TestCoreAnimatorTeeth, TestParameters) {
  nva2f::AnimatorTeeth animator;

  {
    const nva2f::AnimatorTeethParams animatorTeethParams = {1.0f, 0.0f, 0.0f};
    ASSERT_TRUE(!animator.Init(animatorTeethParams));
  }

  // Individual functions.
  // Default values.
  {
    ASSERT_TRUE(!animator.SetLowerTeethStrength(1.0f));
    ASSERT_TRUE(!animator.SetLowerTeethHeightOffset(0.0f));
    ASSERT_TRUE(!animator.SetLowerTeethDepthOffset(0.0f));
    ASSERT_EQ(1.0f, animator.GetParameters().lowerTeethStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().lowerTeethHeightOffset);
    ASSERT_EQ(0.0f, animator.GetParameters().lowerTeethDepthOffset);
  }

  // Valid non-default values.
  {
    ASSERT_TRUE(!animator.SetLowerTeethStrength(1.3f));
    ASSERT_TRUE(!animator.SetLowerTeethHeightOffset(-0.1f));
    ASSERT_TRUE(!animator.SetLowerTeethDepthOffset(0.25f));
    ASSERT_EQ(1.3f, animator.GetParameters().lowerTeethStrength);
    ASSERT_EQ(-0.1f, animator.GetParameters().lowerTeethHeightOffset);
    ASSERT_EQ(0.25f, animator.GetParameters().lowerTeethDepthOffset);
  }

  // Invalid values.
  {
    ASSERT_FALSE(!animator.SetLowerTeethStrength(-1.0f));
    ASSERT_EQ(1.3f, animator.GetParameters().lowerTeethStrength);
    ASSERT_FALSE(!animator.SetLowerTeethHeightOffset(-100.0f));
    ASSERT_EQ(-0.1f, animator.GetParameters().lowerTeethHeightOffset);
    ASSERT_FALSE(!animator.SetLowerTeethDepthOffset(100.0f));
    ASSERT_EQ(0.25f, animator.GetParameters().lowerTeethDepthOffset);
  }

  // Function for all parameters at once.
  // Default values.
  {
    const nva2f::AnimatorTeethParams animatorTeethParams = {1.0f, 0.0f, 0.0f};
    ASSERT_TRUE(!animator.SetParameters(animatorTeethParams));
    ASSERT_EQ(1.0f, animator.GetParameters().lowerTeethStrength);
    ASSERT_EQ(0.0f, animator.GetParameters().lowerTeethHeightOffset);
    ASSERT_EQ(0.0f, animator.GetParameters().lowerTeethDepthOffset);
  }

  // Valid non-default values.
  {
    const nva2f::AnimatorTeethParams animatorTeethParams = {1.3f, -0.1f, 0.25f};
    ASSERT_TRUE(!animator.SetParameters(animatorTeethParams));
    ASSERT_EQ(1.3f, animator.GetParameters().lowerTeethStrength);
    ASSERT_EQ(-0.1f, animator.GetParameters().lowerTeethHeightOffset);
    ASSERT_EQ(0.25f, animator.GetParameters().lowerTeethDepthOffset);
  }

  // Invalid values.
  {
    const nva2f::AnimatorTeethParams animatorTeethParams = {-1.0f, 0.0f, 0.0f};
    ASSERT_FALSE(!animator.SetParameters(animatorTeethParams));
  }
  {
    const nva2f::AnimatorTeethParams animatorTeethParams = {1.0f, -100.0f, 0.0f};
    ASSERT_FALSE(!animator.SetParameters(animatorTeethParams));
  }
  {
    const nva2f::AnimatorTeethParams animatorTeethParams = {1.0f, 0.0f, -100.0f};
    ASSERT_FALSE(!animator.SetParameters(animatorTeethParams));
  }
}

TEST(TestCoreAnimatorTeeth, TestExtraction) {
  auto animator = nva2x::ToUniquePtr(nva2f::CreateAnimatorTeeth_INTERNAL());

  const nva2f::AnimatorTeethParams animatorTeethParams = {1.0f, 0.0f, 0.0f};
  ASSERT_TRUE(!animator->Init(animatorTeethParams));

  std::mt19937 gen;
  static constexpr int kNbNeutralPoseRuns = 100;
  for (int n = 0; n < kNbNeutralPoseRuns; ++n) {
    // Create a random set of points.
    const int nbPoints = std::uniform_int_distribution<int>{3, 10}(gen);
    std::vector<float> neutralPose(nbPoints * 3);
    for (float& coord : neutralPose) {
      coord = std::uniform_real_distribution<float>{-10.0f, 10.0f}(gen);
    }

    ASSERT_TRUE(!animator->SetAnimatorData({nva2x::ToConstView(neutralPose)}));

    static constexpr int kNbTransformRuns = 100;
    for (int m = 0; m < kNbTransformRuns; ++m) {
      // Create a random translation and rotation.
      const auto rotation =
        Eigen::AngleAxisf(
          std::uniform_real_distribution<float>{-180.0f, 180.0f}(gen),
          Eigen::Vector3f::UnitZ()
        ) *
        Eigen::AngleAxisf(
          std::uniform_real_distribution<float>{-180.0f, 180.0f}(gen),
          Eigen::Vector3f::UnitY()
        ) *
        Eigen::AngleAxisf(
          std::uniform_real_distribution<float>{-180.0f, 180.0f}(gen),
          Eigen::Vector3f::UnitX()
        );
      const Eigen::Matrix3f rotationMatrix = [rotation]() {
        Eigen::Matrix3f matrix;
        matrix = rotation;
        return matrix;
      }();
      const Eigen::Vector3f translation {
        std::uniform_real_distribution<float>{-10.0f, 10.0f}(gen),
        std::uniform_real_distribution<float>{-10.0f, 10.0f}(gen),
        std::uniform_real_distribution<float>{-10.0f, 10.0f}(gen)
      };

      // Create the destination points.
      std::vector<float> jawPose(nbPoints * 3);
      for (int i = 0; i < nbPoints; ++i)
      {
        const auto neutral = Eigen::Vector3f::Map(neutralPose.data() + i * 3);
        auto jaw = Eigen::Vector3f::Map(jawPose.data() + i * 3 );

        // jaw pose is relative to the neutral pose,
        jaw = rotationMatrix * neutral + translation - neutral;
      }

      Eigen::Matrix4f jawTransform;
      ASSERT_TRUE(!animator->ComputeJawTransform(
        nva2f::ToView(jawTransform), nva2x::ToConstView(jawPose)));


      // Transform the neutral pose.
      {
        auto verificationPose = neutralPose;
        const Eigen::Transform<float, 3, Eigen::Affine> transform(jawTransform);
        for (int i = 0; i < nbPoints; ++i)
        {
          auto point = Eigen::Vector3f::Map(verificationPose.data() + i * 3 );
          // jaw pose is relative to neutral pose.
          point = transform * point - point;
        }

        // Check that they are equal.
        ASSERT_EQ(verificationPose.size(), jawPose.size());
        for (size_t i = 0; i < verificationPose.size(); ++i)
        {
          ASSERT_NEAR(verificationPose[i], jawPose[i], 1e-4f);
        }
      }


      // Check that strength can disable the transform.
      {
        ASSERT_TRUE(!animator->SetLowerTeethStrength(0.0f));
        ASSERT_TRUE(!animator->ComputeJawTransform(
          nva2f::ToView(jawTransform), nva2x::ToConstView(jawPose)));

        auto verificationPose = neutralPose;
        const Eigen::Transform<float, 3, Eigen::Affine> transform(jawTransform);
        for (int i = 0; i < nbPoints; ++i)
        {
          auto point = Eigen::Vector3f::Map(verificationPose.data() + i * 3 );
          // jaw pose is relative to neutral pose.
          point = transform * point - point;
        }

        // Check that they are equal to the neutral pose (i.e. all offsets are 0).
        ASSERT_EQ(verificationPose.size(), jawPose.size());
        for (size_t i = 0; i < verificationPose.size(); ++i)
        {
          ASSERT_NEAR(verificationPose[i], 0.0f, 1e-4f);
        }
      }


      // Check that offsets work.
      {
        const float heightOffset = std::uniform_real_distribution<float>{-3.0f, 3.0f}(gen);
        const float depthOffset = std::uniform_real_distribution<float>{-3.0f, 3.0f}(gen);

        ASSERT_TRUE(!animator->SetLowerTeethStrength(0.0f));
        ASSERT_TRUE(!animator->SetLowerTeethHeightOffset(heightOffset));
        ASSERT_TRUE(!animator->SetLowerTeethDepthOffset(depthOffset));
        ASSERT_TRUE(!animator->ComputeJawTransform(
          nva2f::ToView(jawTransform), nva2x::ToConstView(jawPose)));

        auto verificationPose = neutralPose;
        const Eigen::Transform<float, 3, Eigen::Affine> transform(jawTransform);
        for (int i = 0; i < nbPoints; ++i)
        {
          auto point = Eigen::Vector3f::Map(verificationPose.data() + i * 3 );
          // jaw pose is relative to neutral pose.
          point = transform * point - point;

        // Check that they are equal to the offsets.
          ASSERT_NEAR(point[0], 0.0f, 1e-4f);
          ASSERT_NEAR(point[1], heightOffset, 1e-4f);
          ASSERT_NEAR(point[2], depthOffset, 1e-4f);
        }
      }

      ASSERT_TRUE(!animator->SetLowerTeethStrength(1.0f));
      ASSERT_TRUE(!animator->SetLowerTeethHeightOffset(0.0f));
      ASSERT_TRUE(!animator->SetLowerTeethDepthOffset(0.0f));
    }
  }
}
