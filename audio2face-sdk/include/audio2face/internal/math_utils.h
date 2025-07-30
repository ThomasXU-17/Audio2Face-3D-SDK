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
#include <cstddef>
#include <system_error>

namespace nva2f {

    // Compute the transform between two point sets.
    //
    // output_transform : a 4x4 matrix containing the best rigid transformation
    //                    (translate + rotate) transforming the points in the
    //                    from_pose array to the points in the to_pose array.
    //                    The matrix is stored in column-major.
    // to_pose          : destination point set after the transform to be computed
    //                    is applied, contains 3 x nb_points floats.
    //                    The points are stored [x0,y0,z0,x1,y1,z1,...].
    // from_pose        : source point set before the transform to be computed is
    //                    applied, contains 3 x nb_points floats.
    //                    The points are stored [x0,y0,z0,x1,y1,z1,...].
    // nb_points        : number of points in the to_pose and from_pose arrays.
    std::error_code rigidXform(
        float* output_transform,
        const float* to_pose,
        const float* from_pose,
        std::size_t nb_points
        );

} // namespace nva2f
