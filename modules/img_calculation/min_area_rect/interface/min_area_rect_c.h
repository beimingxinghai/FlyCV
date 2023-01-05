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

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

/**
 * @brief Get the rotated rectangle with the smallest area enclosing all input points
 * @param[in] src Input points, only support GRAY_S32 currently.
 * @return CRotatedRect
 */
EXTERN_C FCV_API CRotatedRect fcvMinAreaRect(CPoint* pts, int pts_num);

G_FCV_NAMESPACE1_END()
