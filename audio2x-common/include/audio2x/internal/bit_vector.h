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
#include <type_traits>
#include <vector>

namespace nva2x {

// This class stores a vector of boolean values (bits) in a integer storage type.
template <typename Block>
class bit_vector {
public:
  using block_type = Block;

  inline std::size_t size() const;

  inline void resize(std::size_t size);

  inline bool operator[](std::size_t index) const;
  inline bool test(std::size_t index) const;
  inline void set(std::size_t index, bool value = true);
  inline void set_all();
  inline void reset_all();

  inline const block_type* block_data() const;
  inline std::size_t block_size() const;

private:
  static_assert(std::is_integral_v<block_type>);
  static_assert(std::is_unsigned_v<block_type>);

  static constexpr std::size_t nb_bits = sizeof(block_type) * 8;

  std::vector<block_type> _storage;
  std::size_t _size{0};
};


// Implementation.

template <typename Block>
inline std::size_t bit_vector<Block>::size() const {
  return _size;
}

template <typename Block>
inline void bit_vector<Block>::resize(std::size_t size) {
  _size = size;
  _storage.resize((size + nb_bits - 1) / nb_bits);
}

template <typename Block>
inline bool bit_vector<Block>::operator[](std::size_t index) const {
  return test(index);
}

template <typename Block>
inline bool bit_vector<Block>::test(std::size_t index) const {
  assert(index < _size);
  const auto block_index = index / nb_bits;
  const auto bit_index = index % nb_bits;
  return (_storage[block_index] & (static_cast<block_type>(1) << bit_index)) != 0;
}

template <typename Block>
inline void bit_vector<Block>::set(std::size_t index, bool value) {
  assert(index < _size);
  const auto block_index = index / nb_bits;
  const auto bit_index = index % nb_bits;
  if (value) {
    _storage[block_index] |= (static_cast<block_type>(1) << bit_index);
  } else {
    _storage[block_index] &= ~(static_cast<block_type>(1) << bit_index);
  }
}

template <typename Block>
inline void bit_vector<Block>::set_all() {
  std::fill(_storage.begin(), _storage.end(), ~static_cast<block_type>(0));
}

template <typename Block>
inline void bit_vector<Block>::reset_all() {
  std::fill(_storage.begin(), _storage.end(), 0);
}

template <typename Block>
inline const Block* bit_vector<Block>::block_data() const {
  return _storage.data();
}

template <typename Block>
inline std::size_t bit_vector<Block>::block_size() const {
  return _storage.size();
}

} // namespace nva2x
