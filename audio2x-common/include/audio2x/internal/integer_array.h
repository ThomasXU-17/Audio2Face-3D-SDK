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

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

namespace nva2x {

// This class stores an array of integers in a integer storage type.
template <typename IntegerType, typename StorageType, std::size_t NbBitsPerElement>
class IntegerArray {
public:
  using value_type = IntegerType;
  using storage_type = StorageType;
  static constexpr std::size_t nb_bits_per_element = NbBitsPerElement;

  constexpr IntegerArray(std::initializer_list<IntegerType> args = {});

  constexpr std::size_t Size() const;
  constexpr std::size_t Capacity() const;

  constexpr IntegerType Get(std::size_t index) const;
  constexpr void Set(std::size_t index, IntegerType value);

  constexpr void Add(IntegerType value);
  constexpr void RemoveLast();
  constexpr void Clear();

private:
  static constexpr bool is_valid_integer_type();
  static constexpr std::size_t compute_nb_bits_for_count(std::size_t nb_bits);

  static_assert(is_valid_integer_type());
  static_assert(std::is_integral_v<StorageType>);
  static_assert(std::is_unsigned_v<StorageType>);

  static constexpr std::size_t nb_bits = sizeof(storage_type) * 8;
  static constexpr std::size_t nb_bits_for_count = compute_nb_bits_for_count(nb_bits);
  static constexpr std::size_t max_nb_elements = (nb_bits - nb_bits_for_count) / nb_bits_per_element;
  static constexpr StorageType element_mask = (1ULL << nb_bits_per_element) - 1;
  static constexpr StorageType count_mask = (1ULL << nb_bits_for_count) - 1;

  StorageType _storage{0};
};


// Implementation.

template <typename IntegerType, typename StorageType, std::size_t NbBitsPerElement>
constexpr IntegerArray<IntegerType, StorageType, NbBitsPerElement>::IntegerArray(
  std::initializer_list<IntegerType> args
  ) {
  for (const auto& arg : args) {
    Add(arg);
  }
}

template <typename IntegerType, typename StorageType, std::size_t NbBitsPerElement>
constexpr std::size_t IntegerArray<IntegerType, StorageType, NbBitsPerElement>::Size() const {
  return _storage & count_mask;
}

template <typename IntegerType, typename StorageType, std::size_t NbBitsPerElement>
constexpr std::size_t IntegerArray<IntegerType, StorageType, NbBitsPerElement>::Capacity() const {
  return max_nb_elements;
}

template <typename IntegerType, typename StorageType, std::size_t NbBitsPerElement>
constexpr IntegerType IntegerArray<IntegerType, StorageType, NbBitsPerElement>::Get(std::size_t index) const {
  assert(index < Size());
  return static_cast<IntegerType>(
    (_storage >> (index * nb_bits_per_element + nb_bits_for_count)) & element_mask
    );
}

template <typename IntegerType, typename StorageType, std::size_t NbBitsPerElement>
constexpr void IntegerArray<IntegerType, StorageType, NbBitsPerElement>::Set(std::size_t index, IntegerType value) {
  assert(index < Size());
  assert(static_cast<storage_type>(value) <= element_mask);
  const auto shift = index * nb_bits_per_element + nb_bits_for_count;
  // Clear the bits.
  _storage &= ~(element_mask << shift);
  // Set the new value.
    _storage |= (static_cast<storage_type>(value) << shift);
  }

template <typename IntegerType, typename StorageType, std::size_t NbBitsPerElement>
constexpr void IntegerArray<IntegerType, StorageType, NbBitsPerElement>::Add(IntegerType value) {
  assert(Size() < max_nb_elements);
  const auto index = Size();
  // Count is the lowest bits, so we can just increment it.
  ++_storage;
  Set(index, value);
}

template <typename IntegerType, typename StorageType, std::size_t NbBitsPerElement>
constexpr void IntegerArray<IntegerType, StorageType, NbBitsPerElement>::RemoveLast() {
  assert(Size() > 0);
  const auto index = Size() - 1;
  Set(index, 0);
  --_storage;
}

template <typename IntegerType, typename StorageType, std::size_t NbBitsPerElement>
constexpr void IntegerArray<IntegerType, StorageType, NbBitsPerElement>::Clear() {
  _storage = 0;
}

template <typename IntegerType, typename StorageType, std::size_t NbBitsPerElement>
constexpr bool IntegerArray<IntegerType, StorageType, NbBitsPerElement>::is_valid_integer_type() {
  if constexpr (std::is_integral_v<IntegerType>) {
    return std::is_unsigned_v<IntegerType>;
  } else if constexpr (std::is_enum_v<IntegerType>) {
    return std::is_integral_v<std::underlying_type_t<IntegerType>> && std::is_unsigned_v<std::underlying_type_t<IntegerType>>;
  } else {
    return false;
  }
}

template <typename IntegerType, typename StorageType, std::size_t NbBitsPerElement>
constexpr std::size_t IntegerArray<IntegerType, StorageType, NbBitsPerElement>::compute_nb_bits_for_count(std::size_t nb_bits) {
  for (std::size_t nb_bits_for_count = 0; ; ++nb_bits_for_count) {
    assert(nb_bits_for_count < nb_bits);
    const std::size_t max_count = (1ULL << nb_bits_for_count) - 1;
    const std::size_t max_nb_elements = (nb_bits - max_count) / nb_bits_per_element;
    if (max_count >= max_nb_elements) {
      return nb_bits_for_count;
    }
  }
}

} // namespace nva2x
