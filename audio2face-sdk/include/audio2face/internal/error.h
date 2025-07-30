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

#include "audio2face/error.h"
#include <refl.hpp>

namespace nva2f {

std::string Error2String(ErrorCode e);

template <typename T>
struct name_traits {
    constexpr static const size_t size = refl::member_list<T>{}.size;
    static const char* const name;
    static const std::array<std::string, size> fields;
};

template<typename T>
class FieldError : public std::error_category {
public:
    static const int kReservedBits = 16;
    static const int kReservedMask = (1<<kReservedBits) - 1;

    const char *name() const noexcept override {
        return name_traits<T>::name;
    }

    std::string message(int value) const override {
        const int idx = value >> kReservedBits;
        ErrorCode e = static_cast<ErrorCode>(value & kReservedMask);
        // For the default void type, we don't have any fields.
        if constexpr (!std::is_same_v<T, void>) {
           switch (e) {
            case ErrorCode::eNotANumber:
               return "Not A Number at " + name_traits<T>::fields[idx];
            case ErrorCode::eOutOfRange:
                return "Out Of Range at " + name_traits<T>::fields[idx];
            default:
                break;
            }
        }
        return Error2String(e);
    }

    static std::error_category const& instance() {
        // The category singleton
        static FieldError instance;
        return instance;
    }
};

template<typename T>
inline std::error_code make_error_code_with_info(ErrorCode e, size_t extra) {
    return {static_cast<int>(e) | (static_cast<int>(extra) << FieldError<T>::kReservedBits),
        FieldError<T>::instance()};
}

#define MAKE_ERROR_CATEGORY_NAME_TRAITS(type, category_name, members) \
template<> const char* const name_traits<type>::name = category_name; \
template<> const std::array<std::string, name_traits<type>::size> name_traits<type>::fields = \
refl::util::map_to_array<std::string>(members, [](auto member) { return get_name(member).str(); })

// default name traits for void
template<> inline const char* const name_traits<void>::name = "void";
template<> inline const std::array<std::string, name_traits<void>::size> name_traits<void>::fields = {};

}//namespace nva2f
