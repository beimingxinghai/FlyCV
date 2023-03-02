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

#include <cuda_runtime.h>

#include "flycv_namespace.h"
#include "modules/core/base/interface/macro_ns.h"
#include "modules/core/basic_math/interface/basic_math.h"


G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

/**
 * @brief Compute the round of double type data
 * @param[in] value，supported type:double,
 */
__device__ static inline int fcv_round_cuda(double value) {
    return static_cast<int>(value + (value >= 0 ? 0.5 : -0.5));
}

/**
 * @brief Compute the floor of double type data
 * @param[in] value，supported type:double,
 */
__device__ static inline int fcv_floor_cuda(double value) {
    int i = static_cast<int>(value);
    return (i - (i > value));
}

/**
 * @brief Compute the ceiling of double type data
 * @param[in] value，supported type:double,
 */
__device__ static inline int fcv_ceil_cuda(double value) {
    int i = static_cast<int>(value);
    return i + (i < value);
}

/**
 * @brief A value is bounded between an upper and lower bound
 * @param[in] val supported type:int,float,double...
 * @param[in] min supported type:int,float,double...
 * @param[in] max supported type:int,float,double...
 */
template<class T, class D>
__device__ static inline constexpr T fcv_clamp_cuda(
        const T& val,
        const D& min,
        const D& max) {
    return val < min ? min : (val > max ? max : val);
}

template<class T>
__device__ static inline constexpr short fcv_cast_s16_cuda(const T& val) {
    return static_cast<short>(fcv_clamp_cuda(val, S16_MIN_VAL, S16_MAX_VAL));
}

template<class T>
__device__ static inline constexpr unsigned short fcv_cast_u16_cuda(const T& val) {
    return static_cast<unsigned short>(fcv_clamp_cuda(val, U16_MIN_VAL, U16_MAX_VAL));
}

template<class T>
__device__ static inline constexpr signed char fcv_cast_s8_cuda(const T& val) {
    return static_cast<signed char>(fcv_clamp_cuda(val, S8_MIN_VAL, S8_MAX_VAL));
}

template<class T>
__device__ static inline constexpr unsigned char fcv_cast_u8_cuda(const T& val) {
    return static_cast<unsigned char>(fcv_clamp_cuda(val, U8_MIN_VAL, U8_MAX_VAL));
}

template<class T>
__device__ static inline constexpr T fcv_cast_cuda(const float& val) {
    return static_cast<T>(fcv_clamp_cuda(val, U8_MIN_VAL, U8_MAX_VAL));
}

/**
 * @brief Check if two numbers are equal
 * @param[in] a supported type:int,float,double...
 * @param[in] b supported type:int,float,double...
 */
template<class T, class D>
__device__ bool is_almost_equa_cuda(T a, D b) {
    return FCV_ABS(a - b) <  FCV_EPSILON ? true : false;
}

G_FCV_NAMESPACE1_END()
