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

#include <iostream>

#define AUDIO2X_LOG_LEVEL_ERROR 1
#define AUDIO2X_LOG_LEVEL_INFO 2
#define AUDIO2X_LOG_LEVEL_DEBUG 3

#ifndef AUDIO2X_LOG_LEVEL
#define AUDIO2X_LOG_LEVEL AUDIO2X_LOG_LEVEL_DEBUG
#endif

// Generic macros that can be used to define SDK-specific ones.
#if AUDIO2X_LOG_LEVEL >= AUDIO2X_LOG_LEVEL_ERROR
#define A2X_BASE_LOG_ERROR(category, x) (std::cerr << "[" category "] [ERROR] " << x << std::endl)
#else
#define A2X_BASE_LOG_ERROR(category, x)
#endif

#if AUDIO2X_LOG_LEVEL >= AUDIO2X_LOG_LEVEL_INFO
#define A2X_BASE_LOG_INFO(category, x) (std::cout << "[" category "] [INFO] " << x << std::endl)
#else
#define A2X_BASE_LOG_INFO(category, x)
#endif

#if AUDIO2X_LOG_LEVEL >= AUDIO2X_LOG_LEVEL_DEBUG
#define A2X_BASE_LOG_DEBUG(category, x) (std::cout << "[" category "] [DEBUG] " << x << std::endl)
#else
#define A2X_BASE_LOG_DEBUG(category, x)
#endif

// A2X SDK specific macros.
#define A2X_LOG_ERROR(x)  A2X_BASE_LOG_ERROR("A2X SDK", x)
#define A2X_LOG_INFO(x)   A2X_BASE_LOG_INFO ("A2X SDK", x)
#define A2X_LOG_DEBUG(x)  A2X_BASE_LOG_DEBUG("A2X SDK", x)
