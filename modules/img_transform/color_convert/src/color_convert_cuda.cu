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

#include "modules/core/basic_math/interface/basic_math_cuda.h"
#include "modules/img_transform/color_convert/include/color_convert_common.h"
#include "modules/img_transform/color_convert/interface/color_convert_cuda.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

/****************************************************************************************\
*                                convert yuv to c3/c4 *
\****************************************************************************************/
/* R’ = 1.164(Y’– 16) + 1.596(Cr – 128)
 G’ = 1.164(Y’– 16) – 0.813(Cr – 128) – 0.392(Cb – 128)
 B’ = 1.164(Y’– 16) + 2.017(Cb – 128)
浮点乘法用 6位精度处理（即a*b = ((a << 6)*b )>>6
 R’ = 74.5(Y’– 16) + 102(Cr – 128) + 32= 74.5Y' + 102V - 14216
 G’ = 74.5(Y’– 16) – 52(Cr – 128) – 25(Cb – 128) + 32= 74.5Y' - 52V - 25U + 8696
 B’ = 74.5(Y’– 16) + 129(Cb – 128) + 32 = 74.5Y + 129U - 17672*/
__global__ void convert_from_yuv420sp_kernel(const unsigned char* y_ptr,
                                             const unsigned char* vu_ptr,
                                             unsigned char* dst_ptr,
                                             int src_h,
                                             int src_w,
                                             int src_stride,
                                             int dst_stride,
                                             bool is_nv12,
                                             int b_idx,
                                             int r_idx,
                                             int channel) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < src_w && y < src_h) {
        int Y_index = x + y * src_stride;
        int VU_index = fcv_floor_cuda(x / 2) * 2 + fcv_floor_cuda(y / 2) * src_stride;
        int dst_index = x * channel + y * dst_stride;
        int Y_value = FCV_MAX(y_ptr[Y_index], 16);
        int V_value = 0;
        int U_value = 0;

        if (is_nv12) {
            U_value = vu_ptr[VU_index];
            V_value = vu_ptr[VU_index + 1];
        } else {
            V_value = vu_ptr[VU_index];
            U_value = vu_ptr[VU_index + 1];
        }

        // R
        dst_ptr[dst_index + r_idx] = fcv_cast_u8_cuda(1.164 * (Y_value - 16) + 1.596 * (V_value - 128));
        // G
        dst_ptr[dst_index + 1] =
                fcv_cast_u8_cuda(1.164 * (Y_value - 16) - 0.813 * (V_value - 128) - 0.392 * (U_value - 128));
        // B
        dst_ptr[dst_index + b_idx] = fcv_cast_u8_cuda(1.164 * (Y_value - 16) + 2.017 * (U_value - 128));

        // A
        if (channel == 4) {
            dst_ptr[dst_index + 3] = 255;
        }

        // printf("x: %d y: %d Y_value: %d V_value: %d U_value:%d Y_index: %d VU_index: %d dst_index: %d ",
        //        x,
        //        y,
        //        Y_value,
        //        V_value,
        //        U_value,
        //        Y_index,
        //        VU_index,
        //        dst_index);
    }
}

static void convert_from_yuv420sp(const CudaMat& src, CudaMat& dst, bool is_nv12, int b_idx, int r_idx, int channel) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_c = src.channels();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    // printf("src_stride: %d dst_stride: %d \n", src_stride, dst_stride);

    CHECK_CVT_SIZE(((src_w % 2) == 0) && ((src_h % 2) == 0));

    const unsigned char* y_ptr = (const unsigned char*)src.data();
    const unsigned char* vu_ptr = y_ptr + src_w * src_h;

    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);

    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_from_yuv420sp_kernel<<<grids, blocks>>>(
            y_ptr, vu_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, is_nv12, b_idx, r_idx, channel);
}

/****************************************************************************************\
*                                convert c3/c4 to yuv                                   *
\****************************************************************************************/
/*
Y = 0.257R + 0.504G + 0.098B + 16 = (66R + 129G + 25B + 0x1080) >> 8
V = 0.439R - 0.368G - 0.071B + 128 = (112R - 94G - 18B + 128 << 8) >> 8
U = -0.148R - 0.291G + 0.439B + 128 = (112B - 74G - 38R + 128 << 8) >> 8
*/
__global__ void convert_to_yuv420sp_kernel(const unsigned char* src_ptr,
                                           unsigned char* y_ptr,
                                           unsigned char* vu_ptr,
                                           int src_h,
                                           int src_w,
                                           int src_stride,
                                           int dst_stride,
                                           bool is_nv12,
                                           int b_idx,
                                           int r_idx,
                                           int channel) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < src_w && y < src_h) {
        int BGR_index = x * channel + y * src_stride;
        int Y_index = x + y * dst_stride;

        int B_value = src_ptr[BGR_index + b_idx];
        int G_value = src_ptr[BGR_index + 1];
        int R_value = src_ptr[BGR_index + r_idx];

        double y_value = 0.257 * R_value + 0.504 * G_value + 0.098 * B_value + 16;

        // Y
        y_ptr[Y_index] = fcv_cast_u8_cuda(y_value);

        // VU
        if (x % 2 == 0 && y % 2 == 0) {
            int VU_index = x + (y / 2) * dst_stride;
            double v_value = 0.439 * R_value - 0.368 * G_value - 0.071 * B_value + 128;
            double u_value = -0.148 * R_value - 0.291 * G_value + 0.439 * B_value + 128;
            if (is_nv12) {
                vu_ptr[VU_index] = fcv_cast_u8_cuda(u_value);
                vu_ptr[VU_index + 1] = fcv_cast_u8_cuda(v_value);
            } else {
                vu_ptr[VU_index] = fcv_cast_u8_cuda(v_value);
                vu_ptr[VU_index + 1] = fcv_cast_u8_cuda(u_value);
            }
        }
    }
}

static void convert_to_yuv420sp(const CudaMat& src, CudaMat& dst, bool is_nv_12, int b_idx, int r_idx, int channel) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    // printf("src_stride: %d dst_stride: %d \n", src_stride, dst_stride);

    CHECK_CVT_SIZE(dst.height() != (src.height() * 3 / 2));

    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* y_ptr = (unsigned char*)dst.data();
    unsigned char* vu_ptr = y_ptr + src_w * src_h;

    dim3 blocks(32, 32);

    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_to_yuv420sp_kernel<<<grids, blocks>>>(
            src_ptr, y_ptr, vu_ptr, src_h, src_w, src_stride, dst_stride, is_nv_12, b_idx, r_idx, channel);
}

__global__ void convert_from_I420_kernel(const unsigned char* y_ptr,
                                         const unsigned char* u_ptr,
                                         const unsigned char* v_ptr,
                                         unsigned char* dst_ptr,
                                         int src_h,
                                         int src_w,
                                         int src_stride,
                                         int dst_stride,
                                         int b_idx,
                                         int r_idx,
                                         int channel) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < src_w && y < src_h) {
        int Y_index = x + y * src_stride;
        int VU_index = x / 2 + y / 2 * (src_stride / 2);
        int dst_index = x * channel + y * dst_stride;
        int Y_value = FCV_MAX(y_ptr[Y_index], 16);
        int V_value = v_ptr[VU_index];
        int U_value = u_ptr[VU_index];

        // R
        dst_ptr[dst_index + r_idx] = fcv_cast_u8_cuda(1.164 * (Y_value - 16) + 1.596 * (V_value - 128));
        // G
        dst_ptr[dst_index + 1] =
                fcv_cast_u8_cuda(1.164 * (Y_value - 16) - 0.813 * (V_value - 128) - 0.392 * (U_value - 128));
        // B
        dst_ptr[dst_index + b_idx] = fcv_cast_u8_cuda(1.164 * (Y_value - 16) + 2.017 * (U_value - 128));

        // A
        if (channel == 4) {
            dst_ptr[dst_index + 3] = 255;
        }
    }
}

static void convert_from_I420(const CudaMat& src, CudaMat& dst, int b_idx, int r_idx, int channel) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();
    const int src_c = src.channels();

    CHECK_CVT_SIZE(((src_w % 2) == 0) && ((src_h % 2) == 0));

    const unsigned char* y_ptr = (const unsigned char*)src.data();
    const unsigned char* vu_ptr = y_ptr + src_stride * src_h;
    const unsigned char* u_ptr = vu_ptr;
    const unsigned char* v_ptr = vu_ptr + src_stride * (src_h / 4);

    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);

    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_from_I420_kernel<<<grids, blocks>>>(
            y_ptr, u_ptr, v_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, b_idx, r_idx, channel);
}

static void convert_from_I420(
        const CudaMat& src_y, CudaMat& src_u, CudaMat& src_v, CudaMat& dst, int b_idx, int r_idx, int channel) {
    const int src_w = src_y.width();
    const int src_h = src_y.height();
    const int src_c = src_y.channels();

    const int src_stride = src_y.stride();
    const int dst_stride = dst.stride();

    CHECK_CVT_SIZE(((src_w % 2) == 0) && ((src_h % 2) == 0));

    const unsigned char* y_ptr = (const unsigned char*)src_y.data();
    const unsigned char* u_ptr = (const unsigned char*)src_u.data();
    const unsigned char* v_ptr = (const unsigned char*)src_v.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);

    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_from_I420_kernel<<<grids, blocks>>>(
            y_ptr, u_ptr, v_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, b_idx, r_idx, channel);
}

__global__ void convert_bgr_to_gray_kernel(const unsigned char* src_ptr,
                                           unsigned char* dst_ptr,
                                           int src_h,
                                           int src_w,
                                           int src_stride,
                                           int dst_stride,
                                           int b_idx,
                                           int r_idx,
                                           int channel) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < src_w && y < src_h) {
        int src_index = x * channel + y * src_stride;
        int dst_index = x + y * dst_stride;

        int R_value = src_ptr[src_index + r_idx];
        int G_value = src_ptr[src_index + 1];
        int B_value = src_ptr[src_index + b_idx];

        // gray
        dst_ptr[dst_index] = fcv_cast_u8_cuda(R_value * 0.299 + G_value * 0.587 + B_value * 0.114);
    }
}

static void convert_bgr_to_gray(const CudaMat& src, CudaMat& dst) {
    const int src_h = src.height();
    const int src_w = src.width();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);

    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_bgr_to_gray_kernel<<<grids, blocks>>>(src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, 0, 2, 3);
}

static void convert_rgb_to_gray(const CudaMat& src, CudaMat& dst) {
    const int src_h = src.height();
    const int src_w = src.width();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_bgr_to_gray_kernel<<<grids, blocks>>>(src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, 2, 0, 3);
}

__global__ void convert_bgr_and_bgra_kernel(const unsigned char* src_ptr,
                                            unsigned char* dst_ptr,
                                            int src_h,
                                            int src_w,
                                            int src_stride,
                                            int dst_stride,
                                            int src_b_idx,
                                            int src_r_idx,
                                            int dst_b_idx,
                                            int dst_r_idx,
                                            int src_c,
                                            int dst_c) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < src_w && y < src_h) {
        int src_index = x * src_c + y * src_stride;
        int dst_index = x * dst_c + y * dst_stride;

        // R
        dst_ptr[dst_index + dst_r_idx] = src_ptr[src_index + src_r_idx];
        // G
        dst_ptr[dst_index + 1] = src_ptr[src_index + 1];
        // B
        dst_ptr[dst_index + dst_b_idx] = src_ptr[src_index + src_b_idx];

        if (src_c == 4 && dst_c == 4) {
            dst_ptr[dst_index + 3] = src_ptr[src_index + 3];
        } else if (src_c != 4 && dst_c == 4) {
            dst_ptr[dst_index + 3] = 255;
        }
    }
}

static void convert_bgr_to_rgb(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_bgr_and_bgra_kernel<<<grids, blocks>>>(
            src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, 0, 2, 2, 0, 3, 3);
}

static void convert_bgr_to_bgra(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_bgr_and_bgra_kernel<<<grids, blocks>>>(
            src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, 0, 2, 0, 2, 3, 4);
}

static void convert_bgra_to_bgr(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();
    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_bgr_and_bgra_kernel<<<grids, blocks>>>(
            src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, 0, 2, 0, 2, 4, 3);
}

static void convert_bgra_to_rgb(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();
    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_bgr_and_bgra_kernel<<<grids, blocks>>>(
            src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, 0, 2, 2, 0, 4, 3);
}

void convert_bgra_to_rgba(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_bgr_and_bgra_kernel<<<grids, blocks>>>(
            src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, 0, 2, 2, 0, 4, 4);
}

void convert_bgr_to_rgba(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_bgr_and_bgra_kernel<<<grids, blocks>>>(
            src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, 0, 2, 2, 0, 3, 4);
}

__global__ void convert_gray_to_bgr_or_bgra_kernel(const unsigned char* src_ptr,
                                                   unsigned char* dst_ptr,
                                                   int src_h,
                                                   int src_w,
                                                   int src_stride,
                                                   int dst_stride,
                                                   int channel) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < src_w && y < src_h) {
        int src_index = x + y * src_stride;
        int dst_index = x * channel + y * dst_stride;

        // RGB
        dst_ptr[dst_index] = src_ptr[src_index];
        dst_ptr[dst_index + 1] = src_ptr[src_index];
        dst_ptr[dst_index + 2] = src_ptr[src_index];
        if (channel == 4) {
            dst_ptr[dst_index + 3] = 255;
        }
    }
}

static void convert_gray_to_bgr(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_gray_to_bgr_or_bgra_kernel<<<grids, blocks>>>(src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, 3);
}

void convert_gray_to_bgra(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();
    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_gray_to_bgr_or_bgra_kernel<<<grids, blocks>>>(src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, 4);
}

__global__ void convert_package_to_planer_kernel(const unsigned char* src_ptr,
                                                 unsigned char* dst_ptr,
                                                 int src_h,
                                                 int src_w,
                                                 int src_stride,
                                                 int dst_stride,
                                                 int channel) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < src_w && y < src_h) {
        int src_index = x * channel + y * src_stride;
        int dst_R_index = x + y * dst_stride;
        int dst_G_index = x + y * dst_stride + src_h * src_w;
        int dst_B_index = x + y * dst_stride + src_h * src_w * 2;
        int dst_A_index = x + y * dst_stride + src_h * src_w * 3;

        // R
        dst_ptr[dst_R_index] = src_ptr[src_index];
        // G
        dst_ptr[dst_G_index] = src_ptr[src_index + 1];
        // B
        dst_ptr[dst_B_index] = src_ptr[src_index + 2];

        if (channel == 4) {
            dst_ptr[dst_A_index] = src_ptr[src_index + 3];
        }
    }
}

// bgrbgr..... to bbbb..ggg...rrr...
void convert_package_to_planer(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    const int channel = src.channels();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_package_to_planer_kernel<<<grids, blocks>>>(
            src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, channel);
}

__global__ void convert_planer_to_package_kernel(const unsigned char* src_ptr,
                                                 unsigned char* dst_ptr,
                                                 int src_h,
                                                 int src_w,
                                                 int src_stride,
                                                 int dst_stride,
                                                 int channel) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < src_w && y < src_h) {
        int src_R_index = x + y * src_stride;
        int src_G_index = x + y * src_stride + src_h * src_w;
        int src_B_index = x + y * src_stride + src_h * src_w * 2;
        int src_A_index = x + y * src_stride + src_h * src_w * 3;

        int dst_index = x * channel + y * dst_stride;

        // R
        dst_ptr[dst_index] = src_ptr[src_R_index];
        // G
        dst_ptr[dst_index + 1] = src_ptr[src_G_index];
        // B
        dst_ptr[dst_index + 2] = src_ptr[src_B_index];

        if (channel == 4) {
            dst_ptr[dst_index + 3] = src_ptr[src_A_index];
        }
    }
}

// bbb...ggg...rrr... convert bgrbgr...
static void convert_planer_to_package(const CudaMat& src, CudaMat& dst) {
    const int src_h = src.height();
    const int src_w = src.width();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();
    const int channel = src.channels();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_planer_to_package_kernel<<<grids, blocks>>>(
            src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride, channel);
}

// converts R, G, B (B, G, R) pixels to  RGB(BGR)565 format respectively
// 0xF800 1111 1000 0000 0000 high
// 0x07E0 0000 0111 1110 0000
// 0X001F 0000 0000 0001 1111 low
__device__ void convertTo565_device(const unsigned short b,
                                    const unsigned short g,
                                    const unsigned short r,
                                    unsigned short* dst) {
    // rrrr rggg gggb bbbb
    *dst = (b >> 3) | ((g << 3) & (0x07E0)) | ((r << 8) & (0xF800));
    // printf("%x %x ", b, *dst);
}

__global__ void convert_gray_to_bgr565_kernel(
        const unsigned char* src_ptr, unsigned short* dst_ptr, int src_h, int src_w, int src_stride, int dst_stride) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < src_w && y < src_h) {
        int src_index = x + y * src_stride;
        int dst_index = x + y * dst_stride / sizeof(unsigned short);

        convertTo565_device(src_ptr[src_index], src_ptr[src_index], src_ptr[src_index], &(dst_ptr[dst_index]));
    }
}

static void convert_gray_to_bgr565(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_stride = src.stride();
    const int dst_stride = dst.stride();
    // printf("%d ", dst_stride);

    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned short* dst_ptr = (unsigned short*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_gray_to_bgr565_kernel<<<grids, blocks>>>(src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride);
}

__global__ void convert_bgr_or_bgra_to_bgr565_kernel(const unsigned char* src_ptr,
                                                     unsigned short* dst_ptr,
                                                     int src_h,
                                                     int src_w,
                                                     int src_c,
                                                     int src_stride,
                                                     int dst_stride,
                                                     int b_idx,
                                                     int r_idx) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < src_w && y < src_h) {
        int src_index = x * src_c + y * src_stride;
        int dst_index = x + y * dst_stride / sizeof(unsigned short);

        convertTo565_device(
                src_ptr[src_index + b_idx], src_ptr[src_index + 1], src_ptr[src_index + r_idx], &(dst_ptr[dst_index]));
    }
}

static void convert_bgr_to_bgr565(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_c = src.channels();

    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned short* dst_ptr = (unsigned short*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_bgr_or_bgra_to_bgr565_kernel<<<grids, blocks>>>(
            src_ptr, dst_ptr, src_h, src_w, src_c, src_stride, dst_stride, 0, 2);
}

static void convert_rgb_to_bgr565(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_c = src.channels();

    const int src_stride = src.stride();
    const int dst_stride = dst.stride();
    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned short* dst_ptr = (unsigned short*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_bgr_or_bgra_to_bgr565_kernel<<<grids, blocks>>>(
            src_ptr, dst_ptr, src_h, src_w, src_c, src_stride, dst_stride, 2, 0);
}

static void convert_bgra_to_bgr565(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_c = src.channels();

    const int src_stride = src.stride();
    const int dst_stride = dst.stride();
    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned short* dst_ptr = (unsigned short*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_bgr_or_bgra_to_bgr565_kernel<<<grids, blocks>>>(
            src_ptr, dst_ptr, src_h, src_w, src_c, src_stride, dst_stride, 0, 2);
}

static void convert_rgba_to_bgr565(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int src_c = src.channels();

    const int src_stride = src.stride();
    const int dst_stride = dst.stride();
    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned short* dst_ptr = (unsigned short*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_bgr_or_bgra_to_bgr565_kernel<<<grids, blocks>>>(
            src_ptr, dst_ptr, src_h, src_w, src_c, src_stride, dst_stride, 2, 0);
}

__global__ void convert_rgba_to_mrgba_kernel(
        const unsigned char* src_ptr, unsigned char* dst_ptr, int src_h, int src_w, int src_stride, int dst_stride) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    const unsigned char max_val = 255;
    const unsigned char half_val = 128;

    if (x < src_w && y < src_h) {
        int src_index = x * 4 + y * src_stride;
        int dst_index = x * 4 + y * dst_stride;

        unsigned char v0 = src_ptr[src_index];
        unsigned char v1 = src_ptr[src_index + 1];
        unsigned char v2 = src_ptr[src_index + 2];
        unsigned char v3 = src_ptr[src_index + 3];

        dst_ptr[dst_index] = (v0 * v3 + half_val) / max_val;
        dst_ptr[dst_index + 1] = (v1 * v3 + half_val) / max_val;
        dst_ptr[dst_index + 2] = (v2 * v3 + half_val) / max_val;
        dst_ptr[dst_index + 3] = v3;
    }
}

static void convert_rgba_to_mrgba(const CudaMat& src, CudaMat& dst) {
    const int src_w = src.width();
    const int src_h = src.height();

    const int src_stride = src.stride();
    const int dst_stride = dst.stride();

    // printf("src_stride: %d dst_stride: %d \n", src_stride, dst_stride);
    const unsigned char* src_ptr = (const unsigned char*)src.data();
    unsigned char* dst_ptr = (unsigned char*)dst.data();

    dim3 blocks(32, 32);
    int grid_x = fcv_ceil((src_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);
    convert_rgba_to_mrgba_kernel<<<grids, blocks>>>(src_ptr, dst_ptr, src_h, src_w, src_stride, dst_stride);
}

int cvt_color(const CudaMat& src, CudaMat& dst, ColorConvertType cvt_type, Stream& stream) {
    if (src.empty()) {
        LOG_ERR("Input CudaMat to cvtColor is empty!");
        return -1;
    }

    int type = get_cvt_color_dst_mat_type(cvt_type);

    if (dst.empty()) {
        dst = CudaMat(src.width(), src.height(), static_cast<FCVImageType>(type));
    }

    if (src.width() != dst.width() || src.height() != dst.height() || int(dst.type()) != int(type)) {
        LOG_ERR("illegal size [%d, %d] or format %d of the dst object while "
                "cvt_color",
                dst.width(),
                dst.height(),
                int(dst.type()));
        return -1;
    }

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag = CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(src.data(), src.total_byte_size(), device_id));
    }

    switch (cvt_type) {
    case ColorConvertType::CVT_NV212PA_BGR:
        convert_from_yuv420sp(src, dst, false, 0, 2, 3);
        break;
    case ColorConvertType::CVT_NV122PA_BGR:
        convert_from_yuv420sp(src, dst, true, 0, 2, 3);
        break;
    case ColorConvertType::CVT_NV212PA_RGB:
        convert_from_yuv420sp(src, dst, false, 2, 0, 3);
        break;
    case ColorConvertType::CVT_NV122PA_RGB:
        convert_from_yuv420sp(src, dst, true, 2, 0, 3);
        break;
    case ColorConvertType::CVT_NV212PA_BGRA:
        convert_from_yuv420sp(src, dst, false, 0, 2, 4);
        break;
    case ColorConvertType::CVT_NV122PA_BGRA:
        convert_from_yuv420sp(src, dst, true, 0, 2, 4);
        break;
    case ColorConvertType::CVT_NV212PA_RGBA:
        convert_from_yuv420sp(src, dst, false, 2, 0, 4);
        break;
    case ColorConvertType::CVT_NV122PA_RGBA:
        convert_from_yuv420sp(src, dst, true, 2, 0, 4);
        break;
    case ColorConvertType::CVT_I4202PA_BGR:
        convert_from_I420(src, dst, 0, 2, 3);
        break;
    case ColorConvertType::CVT_PA_BGR2NV12:
        convert_to_yuv420sp(src, dst, true, 0, 2, 3);
        break;
    case ColorConvertType::CVT_PA_BGR2NV21:
        convert_to_yuv420sp(src, dst, false, 0, 2, 3);
        break;
    case ColorConvertType::CVT_PA_RGB2NV12:
        convert_to_yuv420sp(src, dst, true, 2, 0, 3);
        break;
    case ColorConvertType::CVT_PA_RGB2NV21:
        convert_to_yuv420sp(src, dst, false, 2, 0, 3);
        break;
    case ColorConvertType::CVT_PA_BGRA2NV12:
        convert_to_yuv420sp(src, dst, true, 0, 2, 4);
        break;
    case ColorConvertType::CVT_PA_BGRA2NV21:
        convert_to_yuv420sp(src, dst, false, 0, 2, 4);
        break;
    case ColorConvertType::CVT_PA_RGBA2NV12:
        convert_to_yuv420sp(src, dst, true, 2, 0, 4);
        break;
    case ColorConvertType::CVT_PA_RGBA2NV21:
        convert_to_yuv420sp(src, dst, false, 2, 0, 4);
        break;
    case ColorConvertType::CVT_PA_BGR2GRAY:
        convert_bgr_to_gray(src, dst);
        break;
    case ColorConvertType::CVT_PA_RGB2GRAY:
        convert_rgb_to_gray(src, dst);
        break;
    // cvt from bgr/rgb to rgb/bgr/rgba/bgra
    case ColorConvertType::CVT_PA_BGR2PA_RGB:
    case ColorConvertType::CVT_PA_RGB2PA_BGR:
        convert_bgr_to_rgb(src, dst);
        break;
    case ColorConvertType::CVT_PA_BGR2PA_BGRA:
    case ColorConvertType::CVT_PA_RGB2PA_RGBA:
        convert_bgr_to_bgra(src, dst);
        break;
    case ColorConvertType::CVT_PA_BGR2PA_RGBA:
    case ColorConvertType::CVT_PA_RGB2PA_BGRA:
        convert_bgr_to_rgba(src, dst);
        break;
    // cvt from bgra/rgba to rgb/bgr/rgba/bgra
    case ColorConvertType::CVT_PA_BGRA2PA_BGR:
    case ColorConvertType::CVT_PA_RGBA2PA_RGB:
        convert_bgra_to_bgr(src, dst);
        break;
    case ColorConvertType::CVT_PA_BGRA2PA_RGB:
    case ColorConvertType::CVT_PA_RGBA2PA_BGR:
        convert_bgra_to_rgb(src, dst);
        break;
    case ColorConvertType::CVT_PA_RGBA2PA_BGRA:
    case ColorConvertType::CVT_PA_BGRA2PA_RGBA:
        convert_bgra_to_rgba(src, dst);
        break;
    // cvt from gray to bgr/bgra
    case ColorConvertType::CVT_GRAY2PA_RGB:
    case ColorConvertType::CVT_GRAY2PA_BGR:
        convert_gray_to_bgr(src, dst);
        break;

    case ColorConvertType::CVT_GRAY2PA_RGBA:
    case ColorConvertType::CVT_GRAY2PA_BGRA:
        convert_gray_to_bgra(src, dst);
        break;
    // cvt planer and package
    case ColorConvertType::CVT_PL_BGR2PA_BGR:
        convert_planer_to_package(src, dst);
        break;
    case ColorConvertType::CVT_PA_BGR2PL_BGR:
        convert_package_to_planer(src, dst);
        break;
    // cvt to bgr565
    case ColorConvertType::CVT_GRAY2PA_BGR565:
        convert_gray_to_bgr565(src, dst);
        break;
    case ColorConvertType::CVT_PA_BGR2PA_BGR565:
        convert_bgr_to_bgr565(src, dst);
        break;
    case ColorConvertType::CVT_PA_RGB2PA_BGR565:
        convert_rgb_to_bgr565(src, dst);
        break;
    case ColorConvertType::CVT_PA_BGRA2PA_BGR565:
        convert_bgra_to_bgr565(src, dst);
        break;
    case ColorConvertType::CVT_PA_RGBA2PA_BGR565:
        convert_rgba_to_bgr565(src, dst);
        break;
    case ColorConvertType::CVT_PA_RGBA2PA_mRGBA:
        convert_rgba_to_mrgba(src, dst);
        break;
    default:
        LOG_ERR("cvt type not support yet !");
        return -1;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

int cvt_color(
        const CudaMat& src_y, CudaMat& src_u, CudaMat& src_v, CudaMat& dst, ColorConvertType cvt_type, Stream& stream) {
    if (src_y.empty() || src_u.empty() || src_v.empty()) {
        LOG_ERR("Input CudaMat to cvtColor is empty !");
        return -1;
    }

    if (dst.empty()) {
        int type = get_cvt_color_dst_mat_type(cvt_type);
        dst = CudaMat(src_y.width(), src_y.height(), static_cast<FCVImageType>(type));
    }

    switch (cvt_type) {
    case ColorConvertType::CVT_I4202PA_BGR:
        convert_from_I420(src_y, src_u, src_v, dst, 0, 2, 3);
        break;
    default:
        LOG_ERR("cvt type not support yet !");
        return -1;
    };

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

G_FCV_NAMESPACE1_END()
