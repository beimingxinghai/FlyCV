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

#include "modules/core/mat/interface/cuda_mat.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

/**
 * @brief combine split and memcpy
 *        the function split mat that channels in the multi-channel to
 *        multiple single channels, and the output is pointer.
 * @param[in] src input data Mat, which is package data type with the channel is 3 or 4, float data
 * @param[out] dst output data Mat, channel is 1, float data
 * @param[in] stream cuda stream for bound,
 * default stream: Blocking call, not default stream: Non-Blocking call
*/
FCV_API int split_to_memcpy(const CudaMat& src, CudaMat* dst, Stream& stream = Stream::Null());

G_FCV_NAMESPACE1_END()
