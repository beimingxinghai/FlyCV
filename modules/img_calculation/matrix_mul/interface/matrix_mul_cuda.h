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

#include "modules/core/mat/interface/cuda_mat.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

/**
 * @brief the implementation of perspective matrix_mul_cuda, supported element types: int, f32, double
 * The method returns a temporary object encoding per-element array multiplication,
   Computed with OpenCL
 * @param dst dst image, supported image type:Mat,the number of channel: 1
 * @param src0 source image, supported image type:Mat, the number of channel: 1.
 * @param src1 another source image, supported image type:Mat, the number of channel: 1.
 */
FCV_API int cuda_matrix_mul(CudaMat& dst, const CudaMat& src0, const CudaMat& src1);

G_FCV_NAMESPACE1_END()
