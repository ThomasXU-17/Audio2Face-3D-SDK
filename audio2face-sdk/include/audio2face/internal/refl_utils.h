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

#include <refl.hpp>

/////////////////////////////////////////////////////////////////////////////
// Access member descriptor from pointer-to-member
// https://github.com/veselink1/refl-cpp/issues/78
//
template <typename T, typename C, T C::*P>
constexpr auto find_field() {
    return find_one(refl::reflect<C>().members, [](auto m) { return m.pointer == P; });
}

template <typename>
struct mem_ptr_traits {};

template <typename T, typename C>
struct mem_ptr_traits<T C::*>{
    using declaring_type = C;
    using field_type = T;
};

template <typename T>
using field_descriptors_of = typename refl::trait::filter_t<refl::trait::is_field, refl::member_list<T>>;

template <typename T, typename C, T C::*P>
constexpr auto find_field_impl() {
    return refl::util::find_one(field_descriptors_of<C>{}, [](auto m) { return m.pointer == P; });
}

template <auto P>
constexpr auto find_field() {
    using traits = mem_ptr_traits<decltype(P)>;
    return find_field_impl<typename traits::field_type, typename traits::declaring_type, P>();
}

/////////////////////////////////////////////////////////////////////////////
// An example of a generic user-defined builder-style factory
// for named parameter idiom for struct
template <typename T>
class builder : public refl::runtime::proxy<builder<T>, T> {
public:
    template <typename... Args>
    builder(Args&&... args)
      : value_(std::forward<Args>(args)...) {
    }

    // Intercepts calls to T's members with
    // a mutable *this and a single argument
    template <typename Member, typename Value>
    static builder& invoke_impl(builder& self, Value&& value) {
        // Create instance of statically-determined member
        // descriptor to use helpers with ADL-lookup
        constexpr Member member;
        // Statically verify that the target member is writable
        static_assert(is_writable(member));
        // Set the value of the target field
        member(self.value_) = std::forward<Value>(value);
        // Return reference to builder
        return self;
    }

    T build() {
        return std::move(value_);
    }

private:
    T value_; // Backing object
};
