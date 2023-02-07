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

#include "modules/img_transform/flip/interface/flip_cuda.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

constexpr int chn_dim = 4;
__constant__ int gpu_chn_table[chn_dim][chn_dim];
static int cpu_chn_table[chn_dim][chn_dim] = {{-1}, {-2, 0}, {-3, -1, 1}, {-4, -2, 0, 2}};
/*
 1 2 3
 4 5 6
 7 8 9
---------- flip_x
 7 8 9
 4 5 6
 1 2 3
*/
// src_w = img_width * img_channel
template <typename T>
__global__ void flip_x_c(const T* src, int src_h, int src_w, int stride,
                         T* dst) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < src_w && y < src_h) {
        dst[x + y * stride] = src[x + (src_h - 1 - y) * stride];
    }
}

/*
 1 2 3  | 3 2 1
 4 5 6  | 6 5 4
 7 8 9  | 9 8 7
*****flip_y*****
*/
// src_w = img_width * img_channel
template <typename T>
__global__ void flip_y_c(const T* src, int src_h, int src_w, int src_c,
                         int stride, T* dst) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < src_w && y < src_h) {
        const int m = gpu_chn_table[src_c - 1][x % src_c];
        // printf("x: %d y: %d coef: %d ", x, y, m);
        dst[x + y * stride] = src[(src_w + m - x) + y * stride];
    }
}

template <typename T>
void flip_c(const T* src, int src_h, int src_w, int src_c, int stride, T* dst,
            FlipType type) {
    dim3 blocks(32, 32);

    int data_w = src_w * src_c;
    int grid_x = fcv_ceil((data_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((src_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);

    if (FlipType::X == type) {
        flip_x_c<<<grids, blocks>>>(src, src_h, data_w, stride, dst);
    } else if (FlipType::Y == type) {
        cudaMemcpyToSymbol(gpu_chn_table, cpu_chn_table, sizeof(int) * chn_dim * chn_dim);
        flip_y_c<<<grids, blocks>>>(src, src_h, data_w, src_c, stride, dst);
    } else {
        LOG_ERR("flip type not support yet !");
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int flip(const CudaMat& src, CudaMat& dst, FlipType type, Stream& stream) {
    if (src.empty()) {
        LOG_ERR("Input CudaMat of flip is empty!");
        return -1;
    }

    if (src.channels() != 1 && src.channels() != 3 && src.channels() != 4) {
        LOG_ERR("Unsupport src channel");
        return -1;
    }

    if (dst.empty()) {
        dst = CudaMat(src.width(), src.height(), src.type());
    }

    if (dst.width() != src.width() || dst.height() != src.height() ||
        dst.stride() != src.stride() || dst.type() != src.type()) {
        LOG_ERR("illegal size or type of dst CudaMat to flip");
    }

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag =
        CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(
            cudaMemPrefetchAsync(src.data(), src.total_byte_size(), device_id));
    }

    const int src_w = src.width();
    const int src_h = src.height();
    const void* src_ptr = (const void*)src.data();
    void* dst_ptr = (void*)dst.data();

    const int src_c = src.channels();
    const int stride = src.stride();

    // printf("channel: %d \n", src_c);

    switch (src.type()) {
        case FCVImageType::GRAY_U8:
        case FCVImageType::PKG_RGB_U8:
        case FCVImageType::PKG_BGR_U8:
        case FCVImageType::PKG_RGBA_U8:
        case FCVImageType::PKG_BGRA_U8:
            flip_c((unsigned char*)src_ptr, src_h, src_w, src_c, stride,
                   (unsigned char*)dst_ptr, type);
            break;
        case FCVImageType::GRAY_F32:
        case FCVImageType::PKG_RGB_F32:
        case FCVImageType::PKG_BGR_F32:
        case FCVImageType::PKG_RGBA_F32:
        case FCVImageType::PKG_BGRA_F32:
            flip_c((float*)src_ptr, src_h, src_w, src_c, stride / sizeof(float),
                   (float*)dst_ptr, type);
            break;

        case FCVImageType::NV12:
        case FCVImageType::NV21:
            flip_c((unsigned char*)src_ptr, src_h, src_w, 1, stride,
                   (unsigned char*)dst_ptr, type);

            flip_c(((unsigned char*)src_ptr + stride * src_h), (src_h >> 1),
                   (src_w >> 1), 2, stride, ((unsigned char*)dst_ptr + stride * src_h),
                   type);
            break;
        default:
            LOG_ERR("flip type not support yet!");
            break;
    }

    return 0;
}

G_FCV_NAMESPACE1_END()
