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

#include <cstdint>

#include "modules/core/base/include/type_info.h"
#include "modules/core/base/include/utils.h"
#include "modules/img_transform/extract_channel/interface/extract_channel_cuda.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

template <typename T>
__global__ void extract_with_kernel(
        const T* src, int src_w, int src_h, int src_c, int src_s, int dst_s, int dst_i, T* dst) {
    const int nx = blockDim.x * blockIdx.x + threadIdx.x;
    const int ny = blockDim.y * blockIdx.y + threadIdx.y;
    if (nx < src_w && ny < src_h) {
        dst[nx + ny * dst_s] = src[nx * src_c + ny * src_s + dst_i];
    }
}

int extract_channel(CudaMat& _src, CudaMat& _dst, int _index, Stream& stream) {
    if (_src.type() != FCVImageType::PKG_BGR_U8) {
        LOG_ERR("src type is not support");
        return -1;
    }

    if (_dst.size().width() != _src.size().width() || _dst.size().height() != _src.size().height()
        || _dst.type() != FCVImageType::GRAY_U8) {
        _dst = CudaMat(_src.size(), FCVImageType::GRAY_U8);
    }

    if (_index >= _src.channels()) {
        LOG_ERR("Input index must less than mat channel count");
        return -1;
    }

    TypeInfo cur_type_info;
    if (get_type_info(_src.type(), cur_type_info)) {
        LOG_ERR("failed to get type info from src mat while get_type_info");
        return -1;
    }

    if (cur_type_info.data_type != DataType::UINT8) {
        LOG_ERR("extract_channel only support u8 data, the current src element "
                "data type is %d",
                int(cur_type_info.data_type));
        return -1;
    }

    if (_src.channels() != 3) {
        LOG_ERR("extract_channel only support 3 or 4 channels, current src "
                "channels is %d",
                _src.channels());
        return -1;
    }

    const size_t src_size = _src.total_byte_size();
    const size_t dst_size = _dst.total_byte_size();

    dim3 blocks(32, 8);
    int grid_x = fcv_ceil((_src.width() + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((_src.height() + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag = CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(_src.data(), src_size, device_id));
        CUDA_CHECK(cudaMemPrefetchAsync(_dst.data(), dst_size, device_id));
    }

    extract_with_kernel<<<grids, blocks>>>((const unsigned char*)_src.data(),
                                           _src.width(),
                                           _src.height(),
                                           _src.channels(),
                                           _src.stride() / _src.type_byte_size(),
                                           _dst.stride() / _dst.type_byte_size(),
                                           _index,
                                           (unsigned char*)_dst.data());

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

G_FCV_NAMESPACE1_END()
