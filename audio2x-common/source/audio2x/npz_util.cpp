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

#include "cnpy.h"
#include <filesystem>
#include <fstream>
#include <cstddef>
#include <cstring>

#ifdef _WIN32
#include <windows.h>

namespace {
    std::unique_ptr<wchar_t[]> ConvertToWideChar(const char* narrowString) {
        int wideStrLen = MultiByteToWideChar(CP_UTF8, 0, narrowString, -1, NULL, 0);
        if (wideStrLen == 0) {
            // Failed to get length of the wide string
            return nullptr;
        }

        // Allocate memory for the wide string
        std::unique_ptr<wchar_t[]> wideString(new wchar_t[wideStrLen]);
        if (!wideString) {
            // Failed to allocate memory
            return nullptr;
        }

        // Convert the narrow string to wide string
        if (!MultiByteToWideChar(CP_UTF8, 0, narrowString, -1, wideString.get(), wideStrLen)) {
            // Failed to convert narrow string to wide string
            return nullptr;
        }

        return wideString;
    }
}
#endif

cnpy::NpyArray load_the_npy_file(FILE* fp);
cnpy::NpyArray load_the_npz_array(FILE* fp, uint32_t compr_bytes, uint32_t uncompr_bytes);
namespace nva2x {

/*
    patched version of cnpy::npz_load to support utf8
    original implementation: https://github.com/rogersce/cnpy/blob/4e8810b1a8637695171ed346ce68f6984e585ef4/cnpy.cpp#L230
*/
cnpy::npz_t npz_load(std::string fname) {
#ifdef _WIN32
    std::unique_ptr<wchar_t[]> wfname = ConvertToWideChar(fname.c_str());
    FILE* fp = _wfopen(wfname.get(), L"rb");
#else
    FILE* fp = fopen(fname.c_str(), "r");
#endif

    if(!fp) {
        throw std::runtime_error("npz_load: Error! Unable to open file "+fname+"!");
    }

    cnpy::npz_t arrays;

    while(1) {
        std::vector<std::byte> local_header(30);
        size_t headerres = fread(local_header.data(), sizeof(std::byte), 30, fp);
        if(headerres != local_header.size())
            throw std::runtime_error("npz_load: failed fread");

        //if we've reached the global header, stop reading
        if(local_header[2] != std::byte(0x03) || local_header[3] != std::byte(0x04)) break;

        //read in the variable name
        uint16_t name_len = *reinterpret_cast<uint16_t*>(&local_header[26]);
        std::string varname(name_len,'\0');
        size_t vname_res = fread(varname.data(),sizeof(std::byte),name_len,fp);
        if(vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");

        //erase the lagging .npy
        varname.resize(varname.size() - 4);

        //read in the extra field
        uint16_t extra_field_len = *reinterpret_cast<uint16_t*>(&local_header[28]);
        if(extra_field_len > 0) {
            std::vector<std::byte> buff(extra_field_len);
            size_t efield_res = fread(buff.data(), sizeof(std::byte), extra_field_len, fp);
            if(efield_res != extra_field_len)
                throw std::runtime_error("npz_load: failed fread");
        }

        uint16_t compr_method = *reinterpret_cast<uint16_t*>(&local_header[8]);
        uint32_t compr_bytes = *reinterpret_cast<uint32_t*>(&local_header[18]);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t*>(&local_header[22]);

        if(compr_method == 0) {arrays[varname] = load_the_npy_file(fp);}
        else {arrays[varname] = load_the_npz_array(fp,compr_bytes,uncompr_bytes);}
    }

    fclose(fp);
    return arrays;
}

std::vector<std::string> parse_string_array_from_npy_array(const cnpy::NpyArray &str_npy_arr)
{
  std::vector<std::string> result;
  const char *const arr_buf = str_npy_arr.data<char>();
  /*
    Explanation:
    The function parses a string array stored in a NumPy array (str_npy_arr).
    The NumPy array is generated using numpy_arr.astype('S').
    Example input: ['apple', 'banana', 'cat']
    Serialized format: 'apple\0banana\0cat\0\0\0'
    The buffer has a fixed stride determined by str_npy_arr.word_size. Strings shorter than word_size are padded with '\0'.

    Note that when a string is exactly of word_size length, no '\0' is added, it immediately connects with the next string.
    Hence, `strnlen` is used to ensure the string length is at most word_size.
  */
  size_t word_size = str_npy_arr.word_size;
  for(size_t i=0;i<str_npy_arr.shape[0];++i) {
    const char *const str_buf = arr_buf + i * word_size;
    std::string str(str_buf, strnlen(str_buf, word_size));
    result.emplace_back(str);
  }
  return result;
}

} // namespace nva2x
