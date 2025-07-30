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
#include "audio2face/internal/logger.h"


namespace nva2f {

namespace bvls {

void constrainedMin(
    Eigen::VectorXf &x,
    const Eigen::MatrixXf &A_mat,
    const Eigen::VectorXf &B_mat,
    const Eigen::VectorXi &on_bound
)
{
  std::vector<int> free_vars;
  for( int i=0; i < on_bound.rows(); ++i )
  {
    if(on_bound[i] == 0)
    {
      free_vars.push_back(i);
    }
  }

  if(free_vars.empty())
  {
    // no free variables so nothing to do:
    return;
  }

  if((int)free_vars.size() == on_bound.rows())
  {
    // completely unconstrained so let's just solve things without messing around:
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    x = svd.solve(B_mat);
    return;
  }

  // x0 is basically the input value of x with all its free components zeroed out:
  Eigen::VectorXf x0 = x;
  for( auto i : free_vars )
  {
    x0[i] = 0;
  }


  // Ok, so S is a matrix that selects all free components of x, and:
  // x = x_0 + S^T x_free
  // so as to constrain the appropriate components.

  // Here's what we need to minimize with respect to x_free:
  // E = |A(x_0 + S^T x_free) - b|^2
  // E = (A(x_0 + S^T x_free) - b)^T (A(x_0 + S^T x_free) - b)

  // ...

  // How about A_free = A S^T, so:
  // E =  x_free^T A_free^T A_free x_free
  //      - 2 x_free^T A_free^T (b - A x_0 )
  //      + C
  // E =  |A_free x_free - (b - A x_0 )|^2 + C
  // Let's minimize that with an svd:

  Eigen::MatrixXf A_free = A_mat(Eigen::all, free_vars);
  Eigen::VectorXf b_free = B_mat - A_mat * x0;
  Eigen::VectorXf x_free = A_free.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_free);

  for( size_t i=0; i < free_vars.size(); ++i )
  {
    x[free_vars[i]] = x_free[i];
  }
}

float costAndGradient(
    Eigen::VectorXf &g,
    const Eigen::VectorXf &x,
    const Eigen::MatrixXf &A_mat,
    const Eigen::VectorXf &B_mat
)
{
  Eigen::VectorXf r = A_mat * x - B_mat;      // Calculate the residual
  g = A_mat.transpose() * r;  // Calculate the gradient
  return 0.5f * r.dot(r);
}

void initialFeasiblePoint(
    Eigen::VectorXf &x,
    const Eigen::MatrixXf &A_mat,
    const Eigen::VectorXf &B_mat,
    const Eigen::VectorXf &lb,
    const Eigen::VectorXf &ub
)
{
  // Start by assuming no vars are at their limits:
  Eigen::VectorXi on_bound = Eigen::VectorXi::Zero(x.rows());
  while (true) {

    // minimize objective subject to constraints, fixing variables at their limits:
    constrainedMin(x, A_mat, B_mat, on_bound);

    // Handling bounds
    int found_actives = 0;
    for (int i = 0; i < x.size(); ++i)
    {
      if(on_bound[i] != 0)
      {
        continue;
      }
      if (x(i) < lb[i]) // update lower bounds
      {
        x[i] = lb[i];
        on_bound[i] = -1;
        found_actives++;
      }
      else if (x(i) > ub[i]) // update upper bounds
      {
        x[i] = ub[i];
        on_bound[i] = 1;
        found_actives++;
      }
    }

    if (found_actives == 0)  // we found a feasible solution
       break;                // free_set is unchanged
  }

}

float computeKktOptimality(
  const Eigen::VectorXf &g,
  const Eigen::VectorXi &on_bound
)
{
  // Compute the maximum violation of KKT conditions.
  Eigen::VectorXf g_kkt = g.cwiseProduct(on_bound.cast<float>());

  for (int i = 0; i < g.size(); ++i) {
    if (on_bound[i] == 0) {
      g_kkt[i] = std::abs(g[i]);
    }
  }

  return g_kkt.maxCoeff();
}

int varToDetach(
    Eigen::VectorXf &x,
    const Eigen::MatrixXf &A_mat,
    const Eigen::VectorXf &B_mat,
    const Eigen::VectorXf &lb,
    const Eigen::VectorXf &ub
)
{
  Eigen::VectorXf g(x.rows());
  costAndGradient(
      g,
      x,
      A_mat,
      B_mat
  );

  int bestIdx = -1;
  float bestVal = 0.0;
  for( int i=0; i < x.rows(); ++i )
  {
    if(x[i] == lb[i])
    {
      // If the gradient is negative, the optimum is greater than x[i],
      // meaning this is a variable we might want to detach:
      if(g[i] < 0)
      {
        float val = -g[i];
        if( val > bestVal)
        {
          bestVal = val;
          bestIdx = i;
        }
      }
    }
    else if(x[i] == ub[i])
    {
      // If the gradient is positive, the optimum is less than x[i],
      // meaning this is a variable we might want to detach:
      if(g[i] > 0)
      {
        float val = g[i];
        if( val > bestVal)
        {
          bestVal = val;
          bestIdx = i;
        }
      }
    }
  }

  return bestIdx;
}

float makePointFeasible(
    Eigen::VectorXf &x,
    const Eigen::VectorXf &s,
    const Eigen::VectorXf &l,
    const Eigen::VectorXf &u
)
{
  // walk x along the line towards s as far as it will
  // go while still being feasible:
  float alpha = 1.0;
  int idx = -1;
  float boundvalue = 0.0;
  for(int i =0; i < x.rows(); ++i)
  {
    if(x[i] == s[i])
    {
      continue;
    }

    if(s[i] < l[i])
    {
      float alphatest = (x[i] - l[i]) / (x[i] - s[i]);
      if(alphatest < alpha)
      {
        idx=i;
        boundvalue = l[i];
        alpha = alphatest;
      }
    }
    else if(s[i] > u[i])
    {
      float alphatest = (u[i] - x[i]) / (s[i] - x[i]);
      if(alphatest < alpha)
      {
        idx=i;
        boundvalue = u[i];
        alpha = alphatest;
      }
    }
  }

  x += alpha * (s - x);
  if(idx != -1)
  {
    x[idx] = boundvalue;
  }
  return alpha;
}

void updateOnBound(
  Eigen::VectorXi& on_bound,
  const Eigen::VectorXf& x,
  const Eigen::VectorXf& l,
  const Eigen::VectorXf& u
)
{
    for (int i = 0; i < x.rows(); ++i)
    {
        if (x[i] == l[i])
        {
            on_bound[i] = -1;
        }
        else if (x[i] == u[i])
        {
            on_bound[i] = 1;
        }
        else
        {
            on_bound[i] = 0;
        }
    }
}

void solveSystem(
  Eigen::VectorXf &x,
  const Eigen::MatrixXf &A_mat,
  const Eigen::VectorXf &B_mat,
  const Eigen::VectorXf &l,
  const Eigen::VectorXf &u,
  float tolerance
)
{
  //select initial candidate solution:
  initialFeasiblePoint(x, A_mat, B_mat, l, u);

  // measure objective gradient at initial point:
  Eigen::VectorXf g(x.rows());
  float cost = costAndGradient(g, x, A_mat, B_mat);

  // calculate kkt violation number:
  Eigen::VectorXi on_bound(x.rows());
  updateOnBound(on_bound, x, l, u);

  float optimality = computeKktOptimality(g, on_bound);
  float prev_optimality = optimality;

  for( int iter=0; iter < x.rows(); ++iter )
  {
    if(optimality < tolerance)
    {
      break;
    }

    // find the best variable to free up:
    int move_to_free = varToDetach(x, A_mat, B_mat, l, u);
    if(move_to_free == -1)
    {
      // this will only happen when it's optimal anyway right?
      break;
    }
    on_bound[move_to_free] = 0;

    while(true)
    {
      // minimize cost over expanded set of free variables:
      Eigen::VectorXf s = 1.0 * x;
      constrainedMin(s, A_mat, B_mat, on_bound);

      // walk from x to this new point, as far as we can go, while
      // remaining feasible:
      float alpha = makePointFeasible(x, s, l, u);
      if(alpha == 1.0)
      {
          // this means s was already feasible, so we can move on:
          break;
      }

      // x will have stuck to a new face, so let's update on_bound:
      updateOnBound(on_bound, x, l, u);

    }

    float cost_new = costAndGradient(g, x, A_mat, B_mat);
    if(cost_new > cost)
    {
      LOG_DEBUG("solveSystem(): unexpected cost function increase in bounded value least squares");
      break;
    }

    cost = cost_new;
    optimality = computeKktOptimality(g, on_bound);
    if (fabs(prev_optimality - optimality) < 1e-6) {
      // break the loop when optimality stops improving
      break;
    }
    prev_optimality = optimality;

  }
}

} // namespace bvls

} // namespace nva2f
