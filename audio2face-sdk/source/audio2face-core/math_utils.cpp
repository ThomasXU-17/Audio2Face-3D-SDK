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

#include "audio2face/internal/math_utils.h"
#include "audio2x/error.h"

#include <Eigen/Dense>

#include <cassert>

std::error_code nva2f::rigidXform(
    float* output_transform,
    const float* to_pose,
    const float* from_pose,
    std::size_t nb_points
    )
{
    // Map the existing arrays to Eigen types.
    auto output_matrix = Eigen::Matrix<float, 4, 4, Eigen::ColMajor>::Map(output_transform, 4, 4);
    using PoseMatrix = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
    const auto a_matrix = PoseMatrix::Map(to_pose, nb_points, 3);
    const auto b_matrix = PoseMatrix::Map(from_pose, nb_points, 3);

    output_matrix.setIdentity();

    const Eigen::RowVector3f a_mean = a_matrix.colwise().mean();
    const Eigen::MatrixX3f a_delta = a_matrix.rowwise() - a_mean;

    const Eigen::RowVector3f b_mean = b_matrix.colwise().mean();
    const Eigen::MatrixX3f b_delta = b_matrix.rowwise() - b_mean;

    const Eigen::Matrix3f H = b_delta.transpose() * a_delta;

    // ANSME: The original implementation used numpy.linalg.svd which, according to
    // documentation, is using LAPACK routine _gesdd.
    // There might be numerical differences with the use of Eigen SVD (JacobiSVD or BDCSVD).
    // Note: We compute the full matrices because the dimensions are known at compile-time.
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (Eigen::Success != svd.info())
    {
        return nva2x::ErrorCode::eInvalidValue;
    }
    const Eigen::Matrix3f UT = svd.matrixU().transpose();
    const Eigen::Matrix3f VT = svd.matrixV();

    Eigen::Matrix3f eye = Eigen::Matrix3f::Identity();
    eye(2, 2) = (VT * UT).determinant();
    const Eigen::Matrix3f R = VT * eye * UT;
    const Eigen::Vector3f tt = (a_mean - b_mean * R.transpose());

    output_matrix.block<3, 3>(0, 0) = R;
    output_matrix.block<3, 1>(0, 3) = tt;

    return nva2x::ErrorCode::eSuccess;
}
