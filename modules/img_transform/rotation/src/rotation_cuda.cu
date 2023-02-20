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

#include "modules/img_transform/rotation/interface/rotation_cuda.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

// with coalesced write
template <typename T>
__global__ void transpose_kernel(
        const T* src, int src_s, T* dst, int dst_w, int dst_h, int dst_s, int channel) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x < dst_w && y < dst_h && z < channel) {
        const int src_index = y * channel + x * src_s + z;
        const int dst_index = x * channel + y * dst_s + z;
        dst[dst_index] = __ldg(&src[src_index]);
    }
}

int transpose(const CudaMat& src, CudaMat& dst, Stream& stream) {
    if (src.empty()) {
        LOG_ERR("Input CudaMat of transpose is empty!");
        return -1;
    }

    if (dst.empty()) {
        dst = CudaMat(src.height(), src.width(), src.type());
    }

    if (dst.width() != src.height() || dst.height() != src.width() || dst.type() != src.type()) {
        LOG_ERR("illegal size of dst mat");
        return -1;
    }

    const int src_w = src.width();
    const int src_h = src.height();
    const int dst_w = dst.width();
    const int dst_h = dst.height();

    if ((src_w != dst_h) || (src_h != dst_w)) {
        LOG_ERR("size of input or output is not match!");
        return -1;
    }

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag = CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(src.data(), src.total_byte_size(), device_id));
    }

    int channel = src.channels();
    const int src_s = src_w * channel;
    const int dst_s = dst_w * channel;

    dim3 blocks(32, 8, 4);
    int grid_x = fcv_ceil((dst_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((dst_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);

    switch (src.type()) {
    case FCVImageType::GRAY_U8:
    case FCVImageType::PKG_RGB_U8:
    case FCVImageType::PKG_BGR_U8:
    case FCVImageType::PKG_RGBA_U8:
    case FCVImageType::PKG_BGRA_U8: {
        const unsigned char* src_data = (const unsigned char*)src.data();
        unsigned char* dst_data = (unsigned char*)dst.data();
        transpose_kernel<<<grids, blocks>>>(src_data, src_s, dst_data, dst_w, dst_h, dst_s, channel);
    }

    break;
    case FCVImageType::GRAY_F32:
    case FCVImageType::PKG_RGB_F32:
    case FCVImageType::PKG_BGR_F32:
    case FCVImageType::PKG_RGBA_F32:
    case FCVImageType::PKG_BGRA_F32: {
        const float* src_data = (const float*)src.data();
        float* dst_data = (float*)dst.data();
        transpose_kernel<<<grids, blocks>>>(src_data, src_s, dst_data, dst_w, dst_h, dst_s, channel);
    } break;
    default:
        LOG_ERR("transpose type not support yet!");
        break;
    };

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

G_FCV_NAMESPACE1_END()
