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
 * The method returns a temporary object encoding per-element array multiplication
 * @param[in] src0 source image, supported image type: CudaMat, the number of channel: 1.
 * @param[in] src1 another source image, supported image type: CudaMat, the number of channel: 1.
 * @param[in] stream cuda stream for bound,
 * default stream: Blocking call, not default stream: Non-Blocking call
 * @param[out] dst image, supported image type: CudaMat,the number of channel: 1
 */
FCV_API int matrix_mul(const CudaMat& src0, const CudaMat& src1, CudaMat& dst, Stream& stream = Stream::Null());

G_FCV_NAMESPACE1_END()
