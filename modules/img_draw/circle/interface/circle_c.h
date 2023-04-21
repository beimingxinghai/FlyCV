// Copyright (c) 2022 FlyCV Authors. All Rights Reserved.
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

#include "modules/core/cmat/interface/cmat.h"
#include "modules/img_draw/line/interface/line_c.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

/** 
 * @brief Draws a circle. The function draws a simple or filled circle with a given center and radius.
 * @param[inout] img Image where the circle is drawn, only support PKG_BGR_U8 or PKG_RGB_U8 format now.
 * @param[in] center Center of the circle.
 * @param[in] radius Radius of the circle.
 * @param[in] color Circle color.
 * @param[in] thickness Thickness of the circle outline, if positive. Negative values, like #FILLED,
 * mean that a filled circle is to be drawn.
 * @param[in] line_type Type of the circle boundary. See #CLineType
 * @param[in] shift Number of fractional bits in the coordinates of the center and in the radius value.
 */
EXTERN_C FCV_API int fcvCircle(
        CMat* img,
        CPoint center,
        int radius,
        CScalar* color,
        int thickness,
        CLineType line_type, 
        int shift);

G_FCV_NAMESPACE1_END()
