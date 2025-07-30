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
#include "audio2face/internal/bvls.h"

#include <gtest/gtest.h>

using namespace Eigen;

class TestCoreBVLS : public ::testing::Test {};

TEST_F(TestCoreBVLS, TestConstrainedMin) {
    srand(12);
    MatrixXf A_mat = MatrixXf::Random(10, 5);
    VectorXf B_mat = VectorXf::Random(10);
    VectorXf x = VectorXf::Zero(5);

    // test with no constraints :
    VectorXi on_bound = VectorXi::Zero(5);
    nva2f::bvls::constrainedMin(x,A_mat,B_mat,on_bound);

    auto objective = [&A_mat, &B_mat](const VectorXf &x)
    {
        return 0.5f * (A_mat * x - B_mat).squaredNorm();
    };

    float dx = 0.001f;
    float E0 = objective(x);
    for(int i=0; i < x.rows(); ++i)
    {
        VectorXf x_plus = 1.0f * x;
        x_plus[i] += dx;
        ASSERT_TRUE(objective(x_plus) > E0);

        VectorXf x_minus = 1.0f * x;
        x_minus[i] -= dx;
        ASSERT_TRUE(objective(x_minus) > E0);
    }

    // test partially constrained :
    x = VectorXf::Random(5);
    VectorXf x_orig = 1.0f * x;
    on_bound[0] = 1;
    on_bound[3] = 1;
    nva2f::bvls::constrainedMin(x, A_mat, B_mat, on_bound);

    ASSERT_EQ(x[0], x_orig[0]);
    ASSERT_EQ(x[3], x_orig[3]);
    E0 = objective(x);
    for (int i = 0; i < x.rows(); ++i)
    {
        if(on_bound[i] != 0)
        {
            break;
        }
        VectorXf x_plus = 1.0f * x;
        x_plus[i] += dx;
        ASSERT_TRUE(objective(x_plus) > E0);

        VectorXf x_minus = 1.0f * x;
        x_minus[i] -= dx;
        ASSERT_TRUE(objective(x_minus) > E0);
    }

    // test fully constrained :
    x = VectorXf::Random(5);
    x_orig = 1.0f * x;
    on_bound = VectorXi::Ones(5);
    nva2f::bvls::constrainedMin(x, A_mat, B_mat, on_bound);
    ASSERT_EQ(x, x_orig);
}

TEST_F(TestCoreBVLS, TestCostAndGradient) {
    srand(122);
    MatrixXf A_mat = MatrixXf::Random(10, 5);
    VectorXf B_mat = VectorXf::Random(10);
    VectorXf x = VectorXf::Random(5);
    VectorXf g = VectorXf::Zero(5);

    float E = nva2f::bvls::costAndGradient(g, x, A_mat, B_mat);
    auto objective = [&A_mat, &B_mat](const VectorXf& x)
    {
        return 0.5 * (A_mat * x - B_mat).squaredNorm();
    };

    ASSERT_NEAR(E, objective(x), 1.e-6);
    float dx = 0.001f;
    for (int i = 0; i < x.rows(); ++i)
    {
        VectorXf x_plus = 1.0f * x;
        x_plus[i] += dx;

        VectorXf x_minus = 1.0f * x;
        x_minus[i] -= dx;

        ASSERT_NEAR((objective(x_plus) - objective(x_minus)) / (2 * dx), g[i], 5.e-4);
    }
}

TEST_F(TestCoreBVLS, TestInitialFeasiblePoint) {
    srand(4);
    for( int i=0; i < 100; ++i )
    {
        MatrixXf A_mat = MatrixXf::Random(10, 5);
        VectorXf B_mat = VectorXf::Random(10);
        VectorXf l = VectorXf::Random(5);
        VectorXf u = l + VectorXf::Random(5).cwiseAbs();

        VectorXf x = VectorXf::Zero(5);
        nva2f::bvls::initialFeasiblePoint(x, A_mat, B_mat, l, u);

        // not much to test here apart from feasibility :
        for(int j=0; j < x.rows(); ++j)
        {
            ASSERT_TRUE(x[j] >= l[j]);
            ASSERT_TRUE(x[j] <= u[j]);
        }
    }
}

TEST_F(TestCoreBVLS, TestDetachVar) {
    srand(66);
    for (int i = 0; i < 100; ++i)
    {
        MatrixXf A_mat = MatrixXf::Random(10, 5);
        VectorXf B_mat = VectorXf::Random(10);
        VectorXf l = VectorXf::Random(5);
        VectorXf u = l + VectorXf::Random(5).cwiseAbs();

        VectorXf x = VectorXf::Zero(5);
        nva2f::bvls::initialFeasiblePoint(x, A_mat, B_mat, l, u);

        int detach = nva2f::bvls::varToDetach(x, A_mat, B_mat, l, u);
        VectorXf g = VectorXf::Zero(5);
        nva2f::bvls::costAndGradient(g, x, A_mat, B_mat);

        VectorXi on_bound = VectorXi::Zero(5);
        nva2f::bvls::updateOnBound(on_bound, x, l, u);

        float kkt = nva2f::bvls::computeKktOptimality(g, on_bound);
        if(detach == -1)
        {
            ASSERT_LT(kkt, 1.e-6);
        }
        else
        {
            ASSERT_GE(kkt, 1.e-6);
            int detach_test;
            VectorXf prod = g.cwiseProduct(on_bound.cast<float>());
            prod.maxCoeff(&detach_test);
            ASSERT_EQ(detach_test, detach);
        }
    }
}

TEST_F(TestCoreBVLS, TestMakeFeasible) {
    srand(162);
    for( int n=0; n < 100; ++n )
    {
        VectorXf l = VectorXf::Random(5);
        VectorXf u = l + VectorXf::Random(5).cwiseAbs();

        // feasible point:
        VectorXf x = 0.5 * (l + u);
        for(int i=0; i < x.rows(); ++i)
        {
            float randval = abs(Eigen::VectorXf::Random(1)[0]);
            if (randval < 0.2)
            {
                x[i] = l[i];
            }
            else if (randval > 0.8)
            {
                x[i] = u[i];
            }
        }

        // possibly infeasible point:
        VectorXf s = 1.5 * VectorXf::Random(5);
        if(abs(Eigen::VectorXf::Random(1)[0]) < 0.1)
        {
            // force it to be feasible:
            s = 0.5 * (l + u);
        }

        VectorXf x_orig = 1.0f * x;
        float alpha = nva2f::bvls::makePointFeasible(x, s, l, u);
        // check result is feasible :
        for(int i=0; i < x.rows(); ++i)
        {
            ASSERT_TRUE(x[i] >= l[i]);
            ASSERT_TRUE(x[i] <= u[i]);
        }

        // check result is on the closed line segment between x_origand s:
        ASSERT_GE(alpha, 0.0f);
        ASSERT_LE(alpha, 1.0f);

        ASSERT_NEAR((x_orig + alpha * (s - x_orig) - x).norm(), 0.0f, 1.e-6f);

        auto isUnFeasible = [&l,&u](const VectorXf &x)
        {
            bool foundUnfeasible = false;
            for (int i = 0; i < x.rows(); ++i)
            {
                if (x[i] < l[i])
                {
                    foundUnfeasible = true;
                }
                if (x[i] > u[i])
                {
                    foundUnfeasible = true;
                }
            }
            return foundUnfeasible;
        };

        // If we're in the open line segment, walking forward towards s very slightly
        // should take us to an unfeasible point :
        if(alpha < 1 && alpha > 0)
        {
            VectorXf x_unfeasible = x_orig + (alpha + 0.001) * (s - x_orig);
            ASSERT_TRUE(isUnFeasible(x_unfeasible));
        }
    }
}

TEST_F(TestCoreBVLS, TestSolveLoop) {
    srand(129);
    for (int n = 0; n < 100; ++n)
    {
        int n_dofs = 20;
        int n_cols = 10 + int(21 * abs(VectorXf::Random(1)[0]));
        MatrixXf A_mat = MatrixXf::Random(n_cols, n_dofs);
        VectorXf B_mat = VectorXf::Random(n_cols);
        VectorXf l = VectorXf::Random(n_dofs);
        VectorXf u = l + VectorXf::Random(n_dofs).cwiseAbs();

        // select initial candidate solution :
        VectorXf x(n_dofs);
        nva2f::bvls::initialFeasiblePoint(
            x,
            A_mat,
            B_mat,
            l,
            u
        );
        for (int i = 0; i < x.rows(); ++i)
        {
            ASSERT_TRUE(x[i] >= l[i]);
            ASSERT_TRUE(x[i] <= u[i]);
        }

        // measure initial objective
        VectorXf g = VectorXf::Zero(n_dofs);
        float E = nva2f::bvls::costAndGradient(g, x, A_mat, B_mat);

        // initial kkt violation number:
        VectorXi on_bound = VectorXi::Zero(n_dofs);
        nva2f::bvls::updateOnBound(on_bound, x, l, u);
        float optimality = nva2f::bvls::computeKktOptimality(g, on_bound);

        float tolerance = 1.e-5f;
        while(true)
        {
            if(optimality < tolerance)
            {
                break;
            }

            // find active variable to detch:
            int move_to_free = nva2f::bvls::varToDetach(x, A_mat, B_mat, l, u);
            ASSERT_TRUE(move_to_free != -1);
            on_bound[move_to_free] = 0;
            while(true)
            {
                // minimize over new set of free variables :
                VectorXf s = 1.0f * x;
                nva2f::bvls::constrainedMin(
                    s,
                    A_mat,
                    B_mat,
                    on_bound
                );

                // walk from x to this point, as far as we can go, while
                // remaining feasible :
                float alpha = nva2f::bvls::makePointFeasible(x, s, l, u);
                if(alpha == 1.0f)
                {
                    // this means s was already feasible, so we can move on :
                    break;
                }

                // x will have stuck to a new face, so let's update on_bound:
                nva2f::bvls::updateOnBound(on_bound, x, l, u);
            }

            // recompute costand gradient
            float Enew = nva2f::bvls::costAndGradient(
                g,
                x,
                A_mat,
                B_mat
            );
            ASSERT_GT(E - Enew, -1.e-6);

            E = Enew;
            optimality = nva2f::bvls::computeKktOptimality(g, on_bound);

        }
        VectorXf x_lib = 1.0f * x;
        nva2f::bvls::solveSystem(x_lib, A_mat, B_mat, l, u, tolerance);

        ASSERT_NEAR((x - x_lib).norm(), 0, 1.e-6);
    }
}
