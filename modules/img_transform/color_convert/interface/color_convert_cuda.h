// Copyright (c) 2021 FlyCV Authors. All Rights Reserved.
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

#pragma once

#include <vector>

#include "modules/img_transform/color_convert/interface/color_convert.h"
#include "modules/core/base/interface/basic_types.h"
#include "modules/core/base/interface/macro_export.h"
#include "modules/core/base/interface/macro_ns.h"
#include "modules/core/mat/interface/cuda_mat.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

/**
 * @brief the implementation of perspective transformation, supported element types: u8 and f32
 * The function converts an input image from one color space to another
 * @param[in] src input image, supported image type:CudaMat, the number of channel: 1 ,3
 * @param[out] dst output image, supported image type:CudaMat, the number of channel: 1 ,3
 * @param[in] cvt_type color space conversion code (see #ColorConvertType, for example #CVT_I4202PA_BGR or #CVT_NV212PA_BGR or #CVT_NV122PA_BGR)
 * @param[in] stream cuda stream for bound,
 * default stream: Blocking call, not default stream: Non-Blocking call
 */
FCV_API int cvt_color(
        const CudaMat& src,
        CudaMat& dst,
        ColorConvertType cvt_type,
        Stream& stream = Stream::Null());

/**
 * @brief the implementation of perspective transformation, supported element types: u8 and f32
 * This function Converts an image from one color space to another where the source image is stored in
   three planes.This function only supports YUV420 to RGB conversion as of now.
 * @param[in] src_y 8-bit image (#GRAY_U8) of the Y plane
 * @param[in] src_u 8-bit image (#GRAY_U8) of the U plane
 * @param[in] src_v 8-bit image (#GRAY_U8) of the V plane
 * @param[out] dst image, supported image type:CudaMat, the number of channel: 1 ,3
 * @param[in] cvt_type color space conversion code (see #ColorConvertType, only supports #CVT_I4202PA_BGR)
 * @param[in] stream cuda stream for bound,
 * default stream: Blocking call, not default stream: Non-Blocking call
 */
FCV_API int cvt_color(
        const CudaMat& src_y,
        CudaMat& src_u,
        CudaMat& src_v,
        CudaMat& dst,
        ColorConvertType cvt_type,
        Stream& stream = Stream::Null());

G_FCV_NAMESPACE1_END()
