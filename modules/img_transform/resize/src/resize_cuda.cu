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

#include <cmath>

#include "modules/core/base/include/type_info.h"
#include "modules/core/basic_math/interface/basic_math_cuda.h"
#include "modules/img_transform/resize/interface/resize_cuda.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

template <typename T>
__global__ void resize_nearest_kernel(const T* src_ptr,
                                      const int src_w,
                                      const int src_h,
                                      const int src_s,
                                      const float fx,
                                      const float fy,
                                      const int channel,
                                      const int dst_w,
                                      const int dst_h,
                                      const int dst_s,
                                      T* dst_ptr) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < dst_w && dst_y < dst_h) {
        for (int z = 0; z < channel; z++) {
            const int src_x = min(__float2int_rz(dst_x * fx), src_w - 1);
            const int src_y = min(__float2int_rz(dst_y * fy), src_h - 1);

            const int src_index = src_x * channel + src_y * src_s + z;
            const int dst_index = dst_x * channel + dst_y * dst_s + z;

            dst_ptr[dst_index] = src_ptr[src_index];
        }
    }
}

__device__ void calculate_bicubic_coeff(const int& src_in, const int& src_size, float& delta, int& src_out) {
    // Check for minimum out-of-bounds
    if (src_in < 0) {
        src_out = 0;
        delta = 0.f;
    }

    // Check for max out-of-bounds
    if (src_in >= src_size - 1) {
        src_out = src_size - 2;
        delta = 1.f;
    }
}

template <typename T>
__global__ void resize_linear_cn_kernel(const T* src_ptr,
                                        const int src_w,
                                        const int src_h,
                                        const int src_s,
                                        const float fx,
                                        const float fy,
                                        const int channel,
                                        const int dst_w,
                                        const int dst_h,
                                        const int dst_s,
                                        T* dst_ptr) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < dst_w && dst_y < dst_h) {
        float delta_x = (dst_x + 0.5) * fx - 0.5;
        float delta_y = (dst_y + 0.5) * fy - 0.5;

        int sx = __float2int_rd(delta_x);
        int sy = __float2int_rd(delta_y);

        delta_x -= sx;
        delta_y -= sy;

        calculate_bicubic_coeff(sx, src_w, delta_x, sx);
        calculate_bicubic_coeff(sy, src_h, delta_y, sy);

        for (int z = 0; z < channel; z++) {
            int src_index00 = sx * channel + sy * src_s + z;
            int src_index01 = (sx + 1) * channel + sy * src_s + z;
            int src_index10 = sx * channel + (sy + 1) * src_s + z;
            int src_index11 = (sx + 1) * channel + (sy + 1) * src_s + z;

            float out = src_ptr[src_index00] * ((1.0 - delta_x) * (1.0 - delta_y))
                        + src_ptr[src_index01] * ((delta_x) * (1.0 - delta_y))
                        + src_ptr[src_index10] * ((1.0 - delta_x) * (delta_y))
                        + src_ptr[src_index11] * ((delta_x) * (delta_y));

            int dst_index = dst_x * channel + dst_y * dst_s + z;

            dst_ptr[dst_index] = out;
        }
    }
}

template <typename T>
__global__ void resize_linear_yuv_kernel(const T* src_ptr,
                                         const int src_w,
                                         const int src_h,
                                         const int src_s,
                                         const float fx,
                                         const float fy,
                                         const int channel,
                                         const int dst_w,
                                         const int dst_h,
                                         const int dst_s,
                                         T* dst_ptr) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < dst_w && dst_y < dst_h) {
        float delta_x = (dst_x + 0.5) * fx - 0.5;
        float delta_y = (dst_y + 0.5) * fy - 0.5;

        int sx = __float2int_rd(delta_x);
        int sy = __float2int_rd(delta_y);

        delta_x -= sx;
        delta_y -= sy;

        calculate_bicubic_coeff(sx, src_w, delta_x, sx);
        calculate_bicubic_coeff(sy, src_h, delta_y, sy);

        int src_index00 = sx * channel + sy * src_s;
        int src_index01 = (sx + 1) * channel + sy * src_s;
        int src_index10 = sx * channel + (sy + 1) * src_s;
        int src_index11 = (sx + 1) * channel + (sy + 1) * src_s;

        float out = src_ptr[src_index00] * ((1.0 - delta_x) * (1.0 - delta_y))
                    + src_ptr[src_index01] * ((delta_x) * (1.0 - delta_y))
                    + src_ptr[src_index10] * ((1.0 - delta_x) * (delta_y))
                    + src_ptr[src_index11] * ((delta_x) * (delta_y));

        int dst_y_index = dst_x + dst_y * dst_s;

        dst_ptr[dst_y_index] = fcv_cast_cuda<T>(out);

        // 保证目标图像每4个y计算一次uv
        if (dst_x % 2 == 0 && dst_y % 2 == 0) {
            sx = (int)(dst_x * fx) / 2 * 2;
            sy = (int)((dst_y >> 1) * fy);

            int src_uv_index = (sx / 2) * 2 + sy * src_s + (src_h * src_s);
            int dst_uv_index = (dst_x / 2) * 2 + (dst_y >> 1) * dst_s + (dst_h * dst_s);

            dst_ptr[dst_uv_index] = src_ptr[src_uv_index];
            dst_ptr[dst_uv_index + 1] = src_ptr[src_uv_index + 1];
        }
    }
}

__device__ void calculate_bicubic_coeff(
        const int& src_in, const int& src_size, const float& delta, int& src_out, float* coeff) {
    const float A = -0.75f;
    float c[4];
    c[0] = ((A * (delta + 1.0f) - 5.0f * A) * (delta + 1.0f) + 8.0f * A) * (delta + 1.0f) - 4.0f * A;
    c[1] = ((A + 2.0f) * delta - (A + 3.0f)) * delta * delta + 1.0f;
    c[2] = ((A + 2.0f) * (1.0f - delta) - (A + 3.0f)) * (1.0f - delta) * (1.0f - delta) + 1.0f;
    c[3] = 1.0f - c[0] - c[1] - c[2];

    if (src_in >= 0 && src_in <= (src_size - 4)) {
        src_out = src_in;
        coeff[0] = c[0];
        coeff[1] = c[1];
        coeff[2] = c[2];
        coeff[3] = c[3];
    } else if ((-2) == src_in) {
        src_out = 0;
        coeff[0] = c[0] + c[1] + c[2];
        coeff[1] = c[3];
        coeff[2] = 0;
        coeff[3] = 0;
    } else if ((-1) == src_in) {
        src_out = 0;
        coeff[0] = c[0] + c[1];
        coeff[1] = c[2];
        coeff[2] = c[3];
        coeff[3] = 0;
    } else if ((src_size - 3) == src_in) {
        src_out = src_size - 4;
        coeff[0] = 0;
        coeff[1] = c[0];
        coeff[2] = c[1];
        coeff[3] = c[2] + c[3];
    } else if ((src_size - 2) == src_in) {
        src_out = src_size - 4;
        coeff[0] = 0;
        coeff[1] = 0;
        coeff[2] = c[0];
        coeff[3] = c[1] + c[2] + c[3];
    }
}

template <typename T>
__global__ void resize_bicubic_kernel(const T* src_ptr,
                                      const int src_w,
                                      const int src_h,
                                      const int src_s,
                                      const float fx,
                                      const float fy,
                                      const int channel,
                                      const int dst_w,
                                      const int dst_h,
                                      const int dst_s,
                                      T* dst_ptr) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < dst_w && dst_y < dst_h) {
        // x/y coordinate
        float delta_x = (dst_x + 0.5) * fx - 0.5;
        float delta_y = (dst_y + 0.5) * fy - 0.5;

        int sx = __float2int_rd(delta_x);
        int sy = __float2int_rd(delta_y);

        delta_x -= sx;
        delta_y -= sy;

        float cX[4];
        calculate_bicubic_coeff(sx - 1, src_w, delta_x, sx, cX);

        float cY[4];
        calculate_bicubic_coeff(sy - 1, src_h, delta_y, sy, cY);

        for (int z = 0; z < channel; z++) {
            float accum = 0;
            for (int row = 0; row < 4; ++row) {
                int src_index0 = sx * channel + (sy + row) * src_s + z;
                int src_index1 = (sx + 1) * channel + (sy + row) * src_s + z;
                int src_index2 = (sx + 2) * channel + (sy + row) * src_s + z;
                int src_index3 = (sx + 3) * channel + (sy + row) * src_s + z;
                accum += cY[row]
                         * (cX[0] * src_ptr[src_index0] + cX[1] * src_ptr[src_index1] + cX[2] * src_ptr[src_index2]
                            + cX[3] * src_ptr[src_index3]);
            }

            const int dst_index = dst_x * channel + dst_y * dst_s + z;
            dst_ptr[dst_index] = fcv_cast_cuda<T>(max(accum, 0.f));
        }
    }
}

template <typename T>
__device__ void resize_area_fast_device(const T* src_ptr,
                                        const int src_w,
                                        const int src_h,
                                        const int src_s,
                                        const float fx,
                                        const float fy,
                                        const int channel,
                                        const int dst_w,
                                        const int dst_h,
                                        const int dst_s,
                                        T* dst_ptr) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
    const float area_scale = 1.f / (fx * fy);

    if (dst_x < dst_w && dst_y < dst_h) {
        // x/y coordinate
        // 划定x轴方向区域
        float fsx1 = dst_x * fx;
        float fsx2 = fsx1 + fx;

        // 向上舍入
        int sx1 = __float2int_ru(fsx1);
        // 向下舍入
        int sx2 = __float2int_rd(fsx2);

        // 划定x轴方向区域
        float fsy1 = dst_y * fy;
        float fsy2 = fsy1 + fy;

        // 向上舍入
        int sy1 = __float2int_ru(fsy1);
        int sy2 = __float2int_rd(fsy2);

        for (int z = 0; z < channel; z++) {
            int dst_index = dst_x * channel + dst_y * dst_s + z;

            if (sx1 >= src_w || sy1 >= src_h) {
                dst_ptr[dst_index] = 0;
                return;
            }

            float out = 0.f;

            if (sy2 < src_h) {
                for (int sy = sy1; sy < sy2; ++sy) {
                    for (int sx = sx1; sx < sx2; ++sx) {
                        int src_index = sx * channel + sy * src_s + z;
                        out += src_ptr[src_index];
                    }
                }

                dst_ptr[dst_index] = fcv_cast_cuda<T>(out * area_scale);
            } else {
                int count = 0;
                for (int sy = sy1; sy < sy2; ++sy) {
                    if (sy >= src_h) {
                        break;
                    }
                    for (int sx = sx1; sx < sx2; ++sx) {
                        if (sx >= src_w) {
                            break;
                        }
                        int src_index = sx * channel + sy * src_s + z;
                        out += src_ptr[src_index];
                        count++;
                    }
                }
                dst_ptr[dst_index] = fcv_cast_cuda<T>(out / count);
            }
        }
    }
}

template <typename T>
__device__ void resize_area_fast2x2_device(const T* src_ptr,
                                           const int src_w,
                                           const int src_h,
                                           const int src_s,
                                           const int channel,
                                           const int dst_w,
                                           const int dst_h,
                                           const int dst_s,
                                           T* dst_ptr) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < dst_w && dst_y < dst_h) {
        // x/y coordinate
        int sx = dst_x * 2;
        int sy = dst_y * 2;

        for (int z = 0; z < channel; z++) {
            int dst_index = dst_x * channel + dst_y * dst_s + z;

            if (sx >= src_w || sy >= src_h) {
                dst_ptr[dst_index] = 0;
                return;
            }

            int src_index00 = sx * channel + sy * src_s + z;
            int src_index01 = (sx + 1) * channel + sy * src_s + z;
            int src_index10 = sx * channel + (sy + 1) * src_s + z;
            int src_index11 = (sx + 1) * channel + (sy + 1) * src_s + z;
            float out =
                    (src_ptr[src_index00] + src_ptr[src_index01] + src_ptr[src_index10] + src_ptr[src_index11] + 2) / 4;

            dst_ptr[dst_index] = fcv_cast_cuda<T>(out);
        }
    }
}

template <typename T>
__device__ void resize_area_in_device(const T* src_ptr,
                                      const int src_w,
                                      const int src_h,
                                      const int src_s,
                                      const float fx,
                                      const float fy,
                                      const int channel,
                                      const int dst_w,
                                      const int dst_h,
                                      const int dst_s,
                                      T* dst_ptr) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < dst_w && dst_y < dst_h) {
        // x/y coordinate
        // 划定x轴方向区域
        float fsx1 = dst_x * fx;
        float fsx2 = fsx1 + fx;

        // 向上舍入
        int sx1 = __float2int_ru(fsx1);
        // 向下舍入
        int sx2 = __float2int_rd(fsx2);

        sx2 = min(sx2, src_w - 1);
        sx1 = min(sx1, sx2);

        // 划定x轴方向区域
        float fsy1 = dst_y * fy;
        float fsy2 = fsy1 + fy;

        // 向上舍入
        int sy1 = __float2int_ru(fsy1);
        int sy2 = __float2int_rd(fsy2);

        sy2 = min(sy2, src_h - 1);
        sy1 = min(sy1, sy2);

        for (int z = 0; z < channel; z++) {
            int dst_index = dst_x * channel + dst_y * dst_s + z;
            if (sx1 >= src_w || sy1 >= src_h) {
                dst_ptr[dst_index] = 0;
                return;
            }

            float scale = 1.f / (fminf(fx, src_w - fsx1) * fminf(fy, src_h - fsy1));
            float out = 0.f;

            for (int sy = sy1; sy < sy2; ++sy) {
                for (int sx = sx1; sx < sx2; ++sx) {
                    int src_index = sx * channel + sy * src_s + z;
                    out += src_ptr[src_index] * scale;
                }

                if (sx1 > fsx1) {
                    int src_index = (sx1 - 1) * channel + sy * src_s + z;
                    out += src_ptr[src_index] * ((sx1 - fsx1) * scale);
                }

                if (sx2 < fsx2) {
                    int src_index = sx2 * channel + sy * src_s + z;
                    out += src_ptr[src_index] * ((fsx2 - sx2) * scale);
                }
            }

            if (sy1 > fsy1) {
                for (int sx = sx1; sx < sx2; ++sx) {
                    int src_index = sx * channel + (sy1 - 1) * src_s + z;
                    out += src_ptr[src_index] * ((sy1 - fsy1) * scale);
                }
            }

            if (sy2 < fsy2) {
                for (int sx = sx1; sx < sx2; ++sx) {
                    int src_index = sx * channel + sy2 * src_s + z;
                    out += src_ptr[src_index] * ((fsy2 - sy2) * scale);
                }
            }

            if ((sy1 > fsy1) && (sx1 > fsx1)) {
                int src_index = (sx1 - 1) * channel + (sy1 - 1) * src_s + z;
                out += src_ptr[src_index] * ((sy1 - fsy1) * (sx1 - fsx1) * scale);
            }

            if ((sy1 > fsy1) && (sx2 < fsx2)) {
                int src_index = sx2 * channel + (sy1 - 1) * src_s + z;
                out += src_ptr[src_index] * ((sy1 - fsy1) * (fsx2 - sx2) * scale);
            }

            if ((sy2 < fsy2) && (sx2 < fsx2)) {
                int src_index = sx2 * channel + sy2 * src_s + z;
                out += src_ptr[src_index] * ((fsy2 - sy2) * (fsx2 - sx2) * scale);
            }

            if ((sy2 < fsy2) && (sx1 > fsx1)) {
                int src_index = (sx1 - 1) * channel + sy2 * src_s + z;
                out += src_ptr[src_index] * ((fsy2 - sy2) * (sx1 - fsx1) * scale);
            }

            dst_ptr[dst_index] = fcv_cast_cuda<T>(out);
        }
    }
}

template <typename T>
__device__ void resize_area_out_device(const T* src_ptr,
                                       const int src_w,
                                       const int src_h,
                                       const int src_s,
                                       const float fx,
                                       const float fy,
                                       const int channel,
                                       const int dst_w,
                                       const int dst_h,
                                       const int dst_s,
                                       T* dst_ptr) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
    double inv_scale_x = 1. / fx;
    double inv_scale_y = 1. / fy;

    if (dst_x < dst_w && dst_y < dst_h) {
        int sx = __float2int_rd(dst_x * fx);
        int sy = __float2int_rd(dst_y * fy);

        float delta_x = (float)((dst_x + 1) - (sx + 1) * inv_scale_x);
        // 取小于1的权重
        delta_x = delta_x <= 0 ? 0.f : delta_x - __float2int_rd(delta_x);

        float delta_y = (float)((dst_y + 1) - (sy + 1) * inv_scale_y);
        // 取小于1的权重
        delta_y = delta_y <= 0 ? 0.f : delta_y - __float2int_rd(delta_y);

        // 条件判断
        if (sx < 0) {
            delta_x = 0, sx = 0;
        }

        if (sx >= src_w - 1) {
            delta_x = 1;
            sx = src_w - 2;
        }

        if (sy < 0) {
            delta_y = 0, sy = 0;
        }

        if (sy >= src_h - 1) {
            delta_y = 1;
            sy = src_h - 2;
        }

        for (int z = 0; z < channel; z++) {
            int src_index00 = sx * channel + sy * src_s + z;
            int src_index01 = (sx + 1) * channel + sy * src_s + z;
            int src_index10 = sx * channel + (sy + 1) * src_s + z;
            int src_index11 = (sx + 1) * channel + (sy + 1) * src_s + z;

            float out = src_ptr[src_index00] * ((1.0 - delta_x) * (1.0 - delta_y))
                        + src_ptr[src_index01] * ((delta_x) * (1.0 - delta_y))
                        + src_ptr[src_index10] * ((1.0 - delta_x) * (delta_y))
                        + src_ptr[src_index11] * ((delta_x) * (delta_y));

            const int dst_index = dst_x * channel + dst_y * dst_s + z;

            dst_ptr[dst_index] = fcv_cast_cuda<T>(out);
        }
    }
}

template <typename T>
__global__ void resize_area_kernel(const T* src_ptr,
                                   const int src_w,
                                   const int src_h,
                                   const int src_s,
                                   const float fx,
                                   const float fy,
                                   const int channel,
                                   const int dst_w,
                                   const int dst_h,
                                   const int dst_s,
                                   T* dst_ptr) {
    // 如果缩小的是整数倍，使用快速缩放
    // 整数倍缩放
    int iscale_x = roundf(fx);
    int iscale_y = roundf(fy);
    bool is_area_fast = abs(fx - iscale_x) < FCV_EPSILON && abs(fy - iscale_y) < FCV_EPSILON;
    // 缩小
    if (fx >= 1.0f && fy >= 1.0f) {
        if (is_area_fast) {
            bool fast_mode = (fx == 2 && fy == 2 && (channel == 1 || channel == 3 || channel == 4));
            if (fast_mode) {
                resize_area_fast2x2_device(src_ptr, src_w, src_h, src_s, channel, dst_w, dst_h, dst_s, dst_ptr);
                return;
            }
            resize_area_fast_device(src_ptr, src_w, src_h, src_s, fx, fy, channel, dst_w, dst_h, dst_s, dst_ptr);
            return;
        }

        resize_area_in_device(src_ptr, src_w, src_h, src_s, fx, fy, channel, dst_w, dst_h, dst_s, dst_ptr);
        return;
    }

    resize_area_out_device(src_ptr, src_w, src_h, src_s, fx, fy, channel, dst_w, dst_h, dst_s, dst_ptr);
}

static bool check_nearest_support(FCVImageType type) {
    if (!(type == FCVImageType::GRAY_U8 || type == FCVImageType::PKG_RGB_U8 || type == FCVImageType::PKG_BGR_U8
          || type == FCVImageType::PKG_RGBA_U8 || type == FCVImageType::PKG_BGRA_U8 || type == FCVImageType::GRAY_F32
          || type == FCVImageType::PKG_RGB_F32 || type == FCVImageType::PKG_BGR_F32
          || type == FCVImageType::PKG_RGBA_F32 || type == FCVImageType::PKG_BGRA_F32)) {
        LOG_ERR("Invalid DataType: %d ", int(type));
        return false;
    }
    return true;
}

static bool check_linear_support(FCVImageType type) {
    if (!(type == FCVImageType::GRAY_U8 || type == FCVImageType::PKG_RGB_U8 || type == FCVImageType::PKG_BGR_U8
          || type == FCVImageType::PKG_RGBA_U8 || type == FCVImageType::PKG_BGRA_U8 || type == FCVImageType::GRAY_F32
          || type == FCVImageType::PKG_RGB_F32 || type == FCVImageType::PKG_BGR_F32
          || type == FCVImageType::PKG_RGBA_F32 || type == FCVImageType::PKG_BGRA_F32 || type == FCVImageType::NV12
          || type == FCVImageType::NV21)) {
        LOG_ERR("Invalid DataType: %d ", int(type));
        return false;
    }
    return true;
}

static bool check_bicubic_support(FCVImageType type) {
    if (!(type == FCVImageType::GRAY_U8 || type == FCVImageType::PKG_RGB_U8 || type == FCVImageType::PKG_BGR_U8)) {
        LOG_ERR("Invalid DataType: %d ", int(type));
        return false;
    }
    return true;
}

static bool check_area_support(FCVImageType type) {
    if (!(type == FCVImageType::GRAY_U8 || type == FCVImageType::PKG_RGB_U8 || type == FCVImageType::PKG_BGR_U8
          || type == FCVImageType::PKG_RGBA_U8 || type == FCVImageType::PKG_BGRA_U8)) {
        LOG_ERR("Invalid DataType: %d ", int(type));
        return false;
    }
    return true;
}

int resize(CudaMat& src,
           CudaMat& dst,
           const Size& dsize,
           double fx,
           double fy,
           InterpolationType interpolation,
           Stream& stream) {
    if (src.empty()) {
        LOG_ERR("Input CudaMat of resize is empty!");
        return -1;
    }

    TypeInfo cur_type_info;
    if (get_type_info(src.type(), cur_type_info)) {
        LOG_ERR("failed to get type info from src mat while get_type_info");
        return -1;
    }

    if (dst.empty()) {
        if (dsize.width() > 0 && dsize.height() > 0) {
            dst = CudaMat(dsize.width(), dsize.height(), src.type());
        } else if (fx > 0 && fy > 0) {
            dst = CudaMat(std::round(src.width() * fx), std::round(src.height() * fy), src.type());
        } else {
            LOG_ERR("Dst CudaMat width or height is zero which is illegal!");
            return -1;
        }
    }

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag = CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(src.data(), src.total_byte_size(), device_id));
    }

    const int src_w = src.width();
    const int src_h = src.height();
    const int src_s = src.stride() / src.type_byte_size();
    int channel = src.channels();
    const int dst_w = dst.width();
    const int dst_h = dst.height();
    const int dst_s = dst.stride() / dst.type_byte_size();

    bool is_yuv = (src.type() == FCVImageType::NV12 || src.type() == FCVImageType::NV21);

    float inv_fx = float(src_w) / dst_w;
    float inv_fy = float(src_h) / dst_h;

    dim3 blocks(32, 8);
    int grid_x = fcv_ceil((dst_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((dst_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);

    switch (interpolation) {
    case InterpolationType::INTER_NEAREST:
        if (!check_nearest_support(src.type())) {
            return -1;
        }
        switch (cur_type_info.data_type) {
        case DataType::UINT8: {
            const unsigned char* src_data = (const unsigned char*)src.data();
            unsigned char* dst_data = (unsigned char*)dst.data();
            resize_nearest_kernel<<<grids, blocks>>>(
                    src_data, src_w, src_h, src_s, inv_fx, inv_fy, channel, dst_w, dst_h, dst_s, dst_data);
        } break;
        case DataType::F32: {
            const float* src_data = (const float*)src.data();
            float* dst_data = (float*)dst.data();
            resize_nearest_kernel<<<grids, blocks>>>(
                    src_data, src_w, src_h, src_s, inv_fx, inv_fy, channel, dst_w, dst_h, dst_s, dst_data);
        } break;
        default:
            LOG_ERR("subtract is not support this type, the current src element data type is %d",
                    int(cur_type_info.data_type));
            return -1;
        }
        break;
    case InterpolationType::INTER_LINEAR:
        if (!check_linear_support(src.type())) {
            return -1;
        }

        switch (cur_type_info.data_type) {
        case DataType::UINT8: {
            const unsigned char* src_data = (const unsigned char*)src.data();
            unsigned char* dst_data = (unsigned char*)dst.data();
            if (is_yuv) {
                resize_linear_yuv_kernel<<<grids, blocks>>>(
                        src_data, src_w, src_h, src_s, inv_fx, inv_fy, channel, dst_w, dst_h, dst_s, dst_data);
            } else {
                resize_linear_cn_kernel<<<grids, blocks>>>(
                        src_data, src_w, src_h, src_s, inv_fx, inv_fy, channel, dst_w, dst_h, dst_s, dst_data);
            }
        } break;
        case DataType::F32: {
            const float* src_data = (const float*)src.data();
            float* dst_data = (float*)dst.data();
            if (is_yuv) {
                resize_linear_yuv_kernel<<<grids, blocks>>>(
                        src_data, src_w, src_h, src_s, inv_fx, inv_fy, channel, dst_w, dst_h, dst_s, dst_data);
            } else {
                resize_linear_cn_kernel<<<grids, blocks>>>(
                        src_data, src_w, src_h, src_s, inv_fx, inv_fy, channel, dst_w, dst_h, dst_s, dst_data);
            }
        } break;
        default:
            LOG_ERR("subtract is not support this type, the current src element data type is %d",
                    int(cur_type_info.data_type));
            return -1;
        }
        break;
    case InterpolationType::INTER_CUBIC:
        if (!check_bicubic_support(src.type())) {
            return -1;
        }
        switch (cur_type_info.data_type) {
        case DataType::UINT8: {
            const unsigned char* src_data = (const unsigned char*)src.data();
            unsigned char* dst_data = (unsigned char*)dst.data();
            resize_bicubic_kernel<<<grids, blocks>>>(
                    src_data, src_w, src_h, src_s, inv_fx, inv_fy, channel, dst_w, dst_h, dst_s, dst_data);
        } break;
        case DataType::F32: {
            const float* src_data = (const float*)src.data();
            float* dst_data = (float*)dst.data();
            resize_bicubic_kernel<<<grids, blocks>>>(
                    src_data, src_w, src_h, src_s, inv_fx, inv_fy, channel, dst_w, dst_h, dst_s, dst_data);
        } break;
        default:
            LOG_ERR("subtract is not support this type, the current src element data type is %d",
                    int(cur_type_info.data_type));
            return -1;
        }
        break;
    case InterpolationType::INTER_AREA:
        if (!check_area_support(src.type())) {
            return -1;
        }
        switch (cur_type_info.data_type) {
        case DataType::UINT8: {
            const unsigned char* src_data = (const unsigned char*)src.data();
            unsigned char* dst_data = (unsigned char*)dst.data();
            resize_area_kernel<<<grids, blocks>>>(
                    src_data, src_w, src_h, src_s, inv_fx, inv_fy, channel, dst_w, dst_h, dst_s, dst_data);
        } break;
        case DataType::F32: {
            const float* src_data = (const float*)src.data();
            float* dst_data = (float*)dst.data();
            resize_area_kernel<<<grids, blocks>>>(
                    src_data, src_w, src_h, src_s, inv_fx, inv_fy, channel, dst_w, dst_h, dst_s, dst_data);
        } break;
        default:
            LOG_ERR("subtract is not support this type, the current src element data type is %d",
                    int(cur_type_info.data_type));
            return -1;
        }
        break;
    default:
        LOG_ERR("The resize interpolation %d is unsupported now", int(interpolation));
        return -1;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

G_FCV_NAMESPACE1_END()
