// Copyright (c) 2023 FlyCV Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <emscripten/bind.h>

#include "modules/core/mat/interface/mat.h"

using emscripten::class_;
using emscripten::allow_raw_pointers;
using emscripten::val;
using g_fcv_ns::Mat;
using g_fcv_ns::FCVImageType;

g_fcv_ns::Mat* create_mat(
        int width, 
        int height,
        FCVImageType type,
        uintptr_t data,
        int stride) {
    return new fcv::Mat(width, height, type, reinterpret_cast<void*>(data), stride);
}

template<class T>
emscripten::val mat_data(const g_fcv_ns::Mat& mat) {
    return emscripten::val(emscripten::memory_view<T>(mat.total_byte_size(), (T*)mat.data()));
}

EMSCRIPTEN_BINDINGS(class_mat) {
    class_<Mat>("Mat")
        .constructor<>()
        .constructor<int, int, FCVImageType>()
        .constructor(&create_mat, allow_raw_pointers())
        .function("width", &Mat::width)
        .function("height", &Mat::height)
        .function("channels", &Mat::channels)
        .function("clone", &Mat::clone)
        .function("convertTo", &Mat::convert_to)
        .function("type", &Mat::type)
        .function("data", emscripten::select_overload<emscripten::val(const Mat&)>(&mat_data<unsigned char>))
        ;
}