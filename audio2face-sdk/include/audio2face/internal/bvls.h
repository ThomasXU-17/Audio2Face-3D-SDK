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

#include <Eigen/Dense>

namespace nva2f {

namespace bvls {

void constrainedMin(
    Eigen::VectorXf &x,
    const Eigen::MatrixXf &A_mat,
    const Eigen::VectorXf &B_mat,
    const Eigen::VectorXi &on_bound
);

float costAndGradient(
    Eigen::VectorXf &g,
    const Eigen::VectorXf &x,
    const Eigen::MatrixXf &A_mat,
    const Eigen::VectorXf &B_mat
);

void initialFeasiblePoint(
    Eigen::VectorXf &x,
    const Eigen::MatrixXf &A_mat,
    const Eigen::VectorXf &B_mat,
    const Eigen::VectorXf &lb,
    const Eigen::VectorXf &ub
);

float computeKktOptimality(
  const Eigen::VectorXf &g,
  const Eigen::VectorXi &on_bound
);

int varToDetach(
    Eigen::VectorXf &x,
    const Eigen::MatrixXf &A_mat,
    const Eigen::VectorXf &B_mat,
    const Eigen::VectorXf &lb,
    const Eigen::VectorXf &ub
);

float makePointFeasible(
    Eigen::VectorXf &x,
    const Eigen::VectorXf &s,
    const Eigen::VectorXf &l,
    const Eigen::VectorXf &u
);

void updateOnBound(
  Eigen::VectorXi& on_bound,
  const Eigen::VectorXf& x,
  const Eigen::VectorXf& l,
  const Eigen::VectorXf& u
);

void solveSystem(
    Eigen::VectorXf &x,
    const Eigen::MatrixXf &A_mat,
    const Eigen::VectorXf &B_mat,
    const Eigen::VectorXf &l,
    const Eigen::VectorXf &u,
    float tolerance
);

} // namespace bvls

} // namespace nva2f
