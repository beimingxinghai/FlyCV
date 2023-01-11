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
#include "modules/fusion_api/split_to_memcpy/interface/cuda_split_to_memcpy.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

__global__ void split_3Channel_kernel(const float *src, const int count, float *dst) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < count) {
        *(dst + n) = *(src + n * 3);
        *(dst + count + n) = *(src + n * 3 + 1);
        *(dst + 2 * count + n) = *(src + n * 3 + 2);
    }
}

__global__ void split_4Channel_kernel(const float *src, const int count, float *dst) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < count) {
        *(dst + n) = *(src + n * 4);
        *(dst + count + n) = *(src + n * 4 + 1);
        *(dst + 2 * count + n) = *(src + n * 4 + 2);
        *(dst + 3 * count + n) = *(src + n * 4 + 3);
    }
}

int split_to_memcpy(const CudaMat &src, CudaMat *dst, Stream &stream) {
    if (src.empty()) {
        LOG_ERR("The input src is empty!");
        return -1;
    }

    if (dst->empty() || dst->width() != src.stride() / src.type_byte_size() || dst->height() != src.height() ||
        dst->type() != FCVImageType::GRAY_F32) {
        *dst = CudaMat(src.stride() / src.type_byte_size(), src.height(), FCVImageType::GRAY_F32);
    }

    TypeInfo cur_type_info;

    if (get_type_info(src.type(), cur_type_info)) {
        LOG_ERR("The src type is not supported!");
        return -1;
    }

    if (cur_type_info.data_type != DataType::F32) {
        LOG_ERR("Only support f32, the current src element data type is %d", int(cur_type_info.data_type));
        return -1;
    }

    if (src.channels() != 3 && src.channels() != 4) {
        LOG_ERR("Only support 3 or 4 channels, current src channels is %d", src.channels());
        return -1;
    }

    const int width = src.width();
    const int height = src.height();
    const int channel = src.channels();
    const float *src_ptr = (const float *)src.data();
    float *dst_ptr = (float *)dst->data();
    int N = width * height;

    size_t threadsPerBlock = 1024;
    size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag = CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(src.data(), src.total_byte_size(), device_id));
    }

    if (channel == 3) {
        split_3Channel_kernel<<<numberOfBlocks, threadsPerBlock>>>(src_ptr, N, dst_ptr);
    } else if (channel == 4) {
        split_4Channel_kernel<<<numberOfBlocks, threadsPerBlock>>>(src_ptr, N, dst_ptr);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

G_FCV_NAMESPACE1_END()
