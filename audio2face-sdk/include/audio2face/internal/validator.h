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

#include <any>
#include "audio2face/animator.h"
#include "audio2face/internal/error.h"
#include "audio2face/internal/refl_utils.h"
#include "audio2x/error.h"

namespace nva2f {

/////////////////////////////////////////////////////////////////////////////
// range validator for built-in type
template<typename T>
struct Validator : refl::attr::usage::field {
    RangeConfig<T> config;

    constexpr Validator(T def)
        : config{def, std::numeric_limits<T>::min(), std::numeric_limits<T>::max()} {
    }

    constexpr Validator(T def, T min)
        : config{def, min, std::numeric_limits<T>::max()} {
    }

    constexpr Validator(T def, T min, T max)
        : config{def, min, max} {
    }

    constexpr Validator(T def, T min, T max, const char* desc)
        : config{def, min, max, desc} {
    }

    Validator(RangeConfig<T>& other)
        : config(other) {
    }

    template<typename P>
    std::error_code check(T value, size_t index) const {
        if (value < config.minimum) return make_error_code_with_info<P>(ErrorCode::eOutOfRange, index);
        if (value > config.maximum) return make_error_code_with_info<P>(ErrorCode::eOutOfRange, index);
        return nva2x::ErrorCode::eSuccess;
    }

};

template<>
template<typename P>
inline std::error_code Validator<float>::check(float value, size_t index) const {
    if (std::isnan(value)) return make_error_code_with_info<P>(ErrorCode::eNotANumber, index);
    if (value < config.minimum) return make_error_code_with_info<P>(ErrorCode::eOutOfRange, index);
    if (value > config.maximum) return make_error_code_with_info<P>(ErrorCode::eOutOfRange, index);
    return nva2x::ErrorCode::eSuccess;
}

/////////////////////////////////////////////////////////////////////////////

template <typename S>
struct ValidatorProxy : refl::runtime::proxy<ValidatorProxy<S>, S> {

    // Fields and property getters.
    static constexpr auto members = filter(refl::member_list<S>{},
        [](auto member) { return is_readable(member) && has_writer(member); });

    static constexpr auto member_size = refl::member_list<S>{}.size;

    // customed configs
    std::array<std::any, member_size> configs;

    // reference data
    S& target;

    // Provide constructors for value_proxy.
    ValidatorProxy(S& target)
        : target(target)
    {
    }

    ValidatorProxy(S&& target)
        : target(std::move(target))
    {
    }

    // Trap getter, return range configuation
    template <typename Member, typename Self>
    static constexpr auto invoke_impl(Self&& self) {
        // Create an instance of Member to support utility functions.
        constexpr Member member;
        static_assert(is_readable(member));
        using T = typename decltype(member)::value_type;

        if constexpr (refl::descriptor::has_attribute<Validator<T>>(member)) {
            constexpr auto member_index = refl::descriptor::detail::get_member_index(member);
            auto custom = self.configs[member_index];
            return custom.has_value() ?
                std::any_cast<Validator<T>>(custom).config :
                refl::descriptor::get_attribute<Validator<T>>(member).config;
        } else {
            return RangeConfig<T>::kDefault;
        }
    }

    // Trap setter
    template <typename Member, typename Self, typename Value>
    static auto invoke_impl(Self&& self, Value&& value) {
        // Create an instance of Member to support utility functions.
        constexpr Member member;
        static_assert(is_writable(member));
        using struct_type = typename decltype(member)::declaring_type;
        using value_type = typename decltype(member)::value_type;
        using T = refl::trait::remove_qualifiers_t<Value>;
        static_assert(std::is_same_v<value_type,T>);
        constexpr auto member_index = refl::descriptor::detail::get_member_index(member);

        if constexpr(refl::descriptor::has_attribute<Validator<T>>(member)) {
            auto custom = self.configs[member_index];
            auto& validator = custom.has_value() ?
                std::any_cast<Validator<T>&>(custom) :
                refl::descriptor::get_attribute<Validator<T>>(member);
            auto ret = validator.template check<struct_type>(value, member_index);
            if (!ret) {
                member(self.target, std::forward<Value>(value));
            }
            return ret;
        } else {
            member(self.target, std::forward<Value>(value));
            return nva2x::ErrorCode::eSuccess;
        }
    }

    // setup config
    template <typename Member, typename Self, typename T>
    static auto invoke_impl(Self&& self, RangeConfig<T>&& value) {
        // Create an instance of Member to support utility functions.
        constexpr Member member;
        static_assert(is_writable(member));
        using value_type = typename decltype(member)::value_type;
        static_assert(std::is_same_v<value_type, T>);

        if constexpr (refl::descriptor::has_attribute<Validator<T>>(member)) {
            constexpr auto member_index = refl::descriptor::detail::get_member_index(member);
            self.configs[member_index] = Validator<T>(value);
        }
        return nva2x::ErrorCode::eSuccess;
    }
};

/////////////////////////////////////////////////////////////////////////////

template<auto P>
std::error_code set_with_check(float& target, float value) {
    std::error_code ret;
    constexpr auto member = find_field<P>();
    if constexpr (refl::descriptor::has_attribute<Validator>(member)) {
        constexpr auto validator = refl::descriptor::get_attribute<Validator>(member);
        constexpr auto member_index = refl::descriptor::detail::get_member_index(member);
        using type = typename decltype(member)::declaring_type;
        ret = validator.template check<type>(value, member_index);
    }
    if (!ret) {
        target = value;
    }
    return ret;
}

template<typename T, typename C>
const RangeConfig<T>& GetRangeConfigImpl(T C::* P) {
    const Validator<T>* ret = nullptr;
    // run-time
    for_each(field_descriptors_of<C>{}, [&](auto m) {
        if constexpr (std::is_same_v<decltype(get_pointer(m)), T C::*>) {
            if (m.pointer == P) {
                if constexpr (refl::descriptor::has_attribute<Validator<T>>(m)) {
                    ret = &refl::descriptor::get_attribute<Validator<T>>(m);
                }
            }
        }
    });
    if (ret) {
        return ret->config;
    }
    return RangeConfig<T>::kDefault;
}

template<typename T, typename C>
RangeConfig<T> GetRangeConfigImpl(T C::* P, const ValidatorProxy<C>& proxy) {
    auto ret = RangeConfig<T>::kDefault;
    // run-time
    for_each(proxy.members, [&](auto m) {
        if constexpr (std::is_same_v<decltype(get_pointer(m)), T C::*>) {
            if (m.pointer == P) {
                using member_type = decltype(m);
                ret = ValidatorProxy<C>::template invoke_impl<member_type>(proxy);
            }
        }
    });
    return ret;
}

template<typename T, typename C>
std::error_code SetRangeConfigImpl(T C::* P, RangeConfig<T>&& config, ValidatorProxy<C>& proxy) {
    std::error_code ret = nva2x::ErrorCode::eSuccess;
    for_each(proxy.members, [&](auto m) {
        if constexpr (std::is_same_v<decltype(get_pointer(m)), T C::*>) {
            if (m.pointer == P) {
                using member_type = decltype(m);
                ret = ValidatorProxy<C>::template invoke_impl<member_type>(proxy, std::move(config));
            }
        }
    });
    return ret;
}

//////////////////////////////////////////////

}//namespace nv
