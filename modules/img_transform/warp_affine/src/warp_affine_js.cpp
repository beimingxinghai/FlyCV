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

#include "modules/img_transform/warp_affine/interface/warp_affine.h"

EMSCRIPTEN_BINDINGS(warp_affine) {
    //emscripten::function("getAffineTransform", &g_fcv_ns::get_affine_transform);
    emscripten::function("getRotationMatrix2D", &g_fcv_ns::get_rotation_matrix_2D);
    emscripten::function("warpAffine", &g_fcv_ns::warp_affine);
}