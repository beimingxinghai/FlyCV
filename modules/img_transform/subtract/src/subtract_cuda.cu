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

#include <type_traits>

#include "modules/core/base/include/type_info.h"
#include "modules/core/basic_math/interface/basic_math_cuda.h"
#include "modules/img_transform/subtract/interface/subtract_cuda.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

__constant__ double gpu_scaler[4];

template <typename T>
__global__ void subtract_kernel(
        const T* src, const int src_w, const int src_h, const int src_s, const int src_c, T* dst) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x < src_w && y < src_h && z < src_c) {
        // printf("x: %d y: %d z: %d len: %d \n", x,y,z, x * src_c + y * src_s + z);
        if (std::is_same<T, float>::value) {
            dst[x * src_c + y * src_s + z] = src[x * src_c + y * src_s + z] - gpu_scaler[z];
        } else {
            dst[x * src_c + y * src_s + z] = fcv_cast_u8_cuda(src[x * src_c + y * src_s + z] - gpu_scaler[z]);
        }
    }
}

int subtract(const CudaMat& src, Scalar scalar, CudaMat& dst, Stream& stream) {
    if (src.empty()) {
        LOG_ERR("the src is empty!");
        return -1;
    }

    TypeInfo cur_type_info;
    if (get_type_info(src.type(), cur_type_info)) {
        LOG_ERR("failed to get type info from src mat while get_type_info");
        return -1;
    }

    if (dst.empty()) {
        dst = CudaMat(src.width(), src.height(), src.type());
    }

    if (dst.width() != src.width() || dst.height() != src.height() || dst.channels() != src.channels() ||
        dst.type() != src.type()) {
        LOG_ERR("illegal format of dst mat to subtract, which should be same size and type with src");
    }

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag = CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(src.data(), src.total_byte_size(), device_id));
    }

    const int width = src.width();
    const int height = src.height();
    const int stride = src.stride() / src.type_byte_size();
    const int channel = src.channels();

    const double* cpu_scalar = scalar.val();
    cudaMemcpyToSymbol(gpu_scaler, cpu_scalar, sizeof(double) * 4);

    dim3 blocks(32, 8, 4);
    int grid_x = fcv_ceil((width + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((height + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);

    switch (cur_type_info.data_type) {
    case DataType::UINT8: {
        const unsigned char* src_data = (const unsigned char*)src.data();
        unsigned char* dst_data = (unsigned char*)dst.data();
        subtract_kernel<<<grids, blocks>>>(src_data, width, height, stride, channel, dst_data);
    } break;
    case DataType::F32: {
        const float* src_data = (const float*)src.data();
        float* dst_data = (float*)dst.data();
        subtract_kernel<<<grids, blocks>>>(src_data, width, height, stride, channel, dst_data);
    } break;
    default:
        LOG_ERR("subtract is not support this type, the current src element data type is %d",
                int(cur_type_info.data_type));
        return -1;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

G_FCV_NAMESPACE1_END()
