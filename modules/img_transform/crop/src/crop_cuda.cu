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

#include "modules/core/base/include/type_info.h"
#include "modules/img_transform/crop/interface/crop_cuda.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

__global__ void crop_kernel(const uint8_t* src, const int32_t src_stride,
                            const int32_t dst_stride, const int32_t dst_height,
                            uint8_t* dst) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < dst_stride && y < dst_height) {
        dst[x + y * dst_stride] = src[x + y * src_stride];
    }
}

int crop(const CudaMat& src, CudaMat& dst, Rect& drect, Stream& stream) {
    if (src.empty()) {
        LOG_ERR("The src Mat to crop from is empty!");
        return -1;
    }

    if (drect.x() < 0 || drect.y() < 0) {
        return false;
    }
    if ((drect.x() + drect.width()) > src.width() ||
        (drect.y() + drect.height()) > src.height()) {
        return false;
    }

    TypeInfo type_info;
    int status = get_type_info(src.type(), type_info);
    if (status) {
        LOG_ERR("Unsupport image type!");
        return -1;
    }

    int step = type_info.pixel_size;
    if (step <= 0) {
        LOG_ERR("invalid Mat type for crop!");
        return -1;
    }

    // check image type
    switch (type_info.layout) {
        case LayoutType::SINGLE:
        case LayoutType::PACKAGE:
            break;
        default:
            LOG_ERR("crop not support yuv or planar image_type now!");
            return -1;
    }

    if (dst.empty() || dst.type() != src.type() ||
        dst.width() != drect.width() || dst.height() != drect.height()) {
        // LOG_ERR("create new CudaMat!");
        dst = CudaMat(drect.width(), drect.height(), src.type());
    }

    uint8_t* dst_addr = reinterpret_cast<uint8_t*>(dst.data());
    uint8_t* src_addr = reinterpret_cast<uint8_t*>(src.data()) +
                        step * src.width() * drect.y() + drect.x() * step;
    int dst_step = step * dst.width();

    dim3 blocks(32, 32);

    int grid_x = fcv_ceil((dst_step + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((drect.height() + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag =
        CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(
            cudaMemPrefetchAsync(src.data(), src.total_byte_size(), device_id));
    }

    crop_kernel<<<grids, blocks>>>(src_addr, step * src.width(), dst_step,
                                   dst.height(), dst_addr);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

G_FCV_NAMESPACE1_END()
