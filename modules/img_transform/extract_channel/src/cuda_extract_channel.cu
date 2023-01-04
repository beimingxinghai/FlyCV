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
#include "modules/img_transform/extract_channel/interface/cuda_extract_channel.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

const int TILE_DIM = 32;

template <typename T>
__global__ void extract_with_kernel(const T* src, int width, int height, int channel, int index, T* dst) {
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    const int nn = ny * width + nx;
    if (nx < width && ny < height) {
        dst[nn] = src[nn * channel + index];
    }
}

int extract_channel(CudaMat& _src, CudaMat& _dst, int _index, Stream& stream) {
    if (_src.type() != FCVImageType::PKG_BGR_U8) {
        LOG_ERR("src type is not support");
        return -1;
    }

    if (_dst.size().width() != _src.size().width() || _dst.size().height() != _src.size().height() ||
        _dst.type() != FCVImageType::GRAY_U8) {
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
    const int grid_size_x = (_src.width() + TILE_DIM - 1) / TILE_DIM;
    const int grid_size_y = (_src.height() + TILE_DIM - 1) / TILE_DIM;
    const dim3 block_size(TILE_DIM, TILE_DIM);
    const dim3 grid_size(grid_size_x, grid_size_y);

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag = CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(_src.data(), src_size, device_id));
        CUDA_CHECK(cudaMemPrefetchAsync(_dst.data(), dst_size, device_id));
    }

    extract_with_kernel<<<grid_size, block_size>>>((const unsigned char*)_src.data(),
                                                   _src.width(),
                                                   _src.height(),
                                                   _src.channels(),
                                                   _index,
                                                   (unsigned char*)_dst.data());

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

G_FCV_NAMESPACE1_END()
