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
#include "modules/core/basic_math/interface/basic_math_cuda.h"
#include "modules/img_transform/add_weighted/interface/add_weighted_cuda.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

template <typename T>
__global__ void add_weighted_kernel(const T* src1_ptr,
                                    double alpha,
                                    const T* src2_ptr,
                                    double beta,
                                    T* dst_ptr,
                                    double gamma,
                                    const int width,
                                    const int height,
                                    const int channel,
                                    const int stride) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x < width && y < height && z < channel) {
        double value =
                src1_ptr[x * channel + y * stride + z] * alpha + src2_ptr[x * channel + y * stride + z] * beta + gamma;
        dst_ptr[x * channel + y * stride + z] = fcv_cast_u8_cuda(fcv_round_cuda(value));
    }
}

int add_weighted(CudaMat& src1, double alpha, CudaMat& src2, double beta, double gamma, CudaMat& dst, Stream& stream) {
    if (src1.empty()) {
        LOG_ERR("The first input mat is empty!");
        return -1;
    }

    if (src2.empty()) {
        LOG_ERR("The second input mat is empty!");
        return -1;
    }

    if (src1.type() != src2.type() ||
        (src1.type() != FCVImageType::GRAY_U8 && src1.type() != FCVImageType::PKG_BGR_U8 &&
         src1.type() != FCVImageType::PKG_RGB_U8 && src1.type() != FCVImageType::PKG_BGRA_U8 &&
         src1.type() != FCVImageType::PKG_RGBA_U8)) {
        LOG_ERR("The input type is not surpport, which is %d \n", int(src1.type()));
        return -1;
    }

    if (src1.width() != src2.width() || src1.height() != src2.height()) {
        LOG_ERR("src1 and src2 is not match, they width height type must be same.");
        return -1;
    }

    if (dst.empty()) {
        dst = CudaMat(src1.width(), src1.height(), src1.type());
    }

    TypeInfo cur_type_info;
    if (get_type_info(src1.type(), cur_type_info)) {
        LOG_ERR("failed to get type info from src mat while get_type_info");
        return -1;
    }

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag = CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(src1.data(), src1.total_byte_size(), device_id));
        CUDA_CHECK(cudaMemPrefetchAsync(src2.data(), src2.total_byte_size(), device_id));
    }

    const int width = src1.width();
    const int height = src1.height();
    const int stride = src1.stride() / src1.type_byte_size();
    const int channel = src1.channels();

    dim3 blocks(32, 8, 4);
    int grid_x = fcv_ceil((width + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((height + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);

    switch (cur_type_info.data_type) {
    case DataType::UINT8: {
        const unsigned char* src1_ptr = (const unsigned char*)src1.data();
        const unsigned char* src2_ptr = (const unsigned char*)src2.data();
        unsigned char* dst_ptr = (unsigned char*)dst.data();
        add_weighted_kernel<<<grids, blocks>>>(
                src1_ptr, alpha, src2_ptr, beta, dst_ptr, gamma, width, height, channel, stride);
    } break;
    case DataType::F32: {
        const float* src1_ptr = (const float*)src1.data();
        const float* src2_ptr = (const float*)src2.data();
        float* dst_ptr = (float*)dst.data();
        add_weighted_kernel<<<grids, blocks>>>(
                src1_ptr, alpha, src2_ptr, beta, dst_ptr, gamma, width, height, channel, stride);
    } break;
    default:
        LOG_ERR("add weighted is not support this type, the current src element data type is %d",
                int(cur_type_info.data_type));
        return -1;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

G_FCV_NAMESPACE1_END()
