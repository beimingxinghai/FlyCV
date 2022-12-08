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

#include <iostream>

#include "modules/core/base/include/type_info.h"
#include "modules/core/base/include/utils.h"
#include "modules/core/mat/include/mat_dot_common.h"
#include "modules/core/mat/interface/cuda_mat.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

template <typename T>
__global__ void dot_product(T* src1, T* src2, double* result, int len) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < len) {
        result[n] = double(src1[n]) * double(src2[n]);
    }
}

__global__ void reduce_shared(double* d_x, double* d_y, int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ double s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_y[bid] = s_y[0];
    }
}

double CudaMat::dot(CudaMat& m) const {
    if (m.empty()) {
        m = CudaMat(_width, _height, _type);
    }

    const int m_w = m.width();
    const int m_h = m.height();
    const int m_c = m.channels();

    if ((m_w != _width) || (m_h != _height) || (m_c != _channels)) {
        LOG_ERR("The size of dot-product operands should be the same!");
        return 0;
    }

    TypeInfo type_info;
    int status = get_type_info(m.type(), type_info);
    double result = 0.;

    if (status != 0) {
        LOG_ERR("The mat type is not supported!");
        return result;
    }

    int N = _width * _height * _channels;
    size_t src_size = N * type_info.type_byte_size;
    void* ma_data = _data;
    void* mb_data = m.data();

    double* mc_data;
    size_t dst_size = N * sizeof(double);
    CUDA_CHECK(cudaMallocManaged((void**)&mc_data, dst_size));

    size_t threadsPerBlock = 1024;
    size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag = CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(ma_data, src_size, device_id));
        CUDA_CHECK(cudaMemPrefetchAsync(mb_data, src_size, device_id));
        CUDA_CHECK(cudaMemPrefetchAsync(mc_data, dst_size, device_id));
    }

    switch (type_info.data_type) {
    case DataType::UINT8:
        dot_product<unsigned char><<<numberOfBlocks, threadsPerBlock>>>(
                reinterpret_cast<unsigned char*>(ma_data), reinterpret_cast<unsigned char*>(mb_data), mc_data, N);
        break;
    case DataType::UINT16:
        dot_product<unsigned short><<<numberOfBlocks, threadsPerBlock>>>(
                reinterpret_cast<unsigned short*>(ma_data), reinterpret_cast<unsigned short*>(mb_data), mc_data, N);
        break;
    case DataType::SINT32:
        dot_product<int><<<numberOfBlocks, threadsPerBlock>>>(
                reinterpret_cast<int*>(ma_data), reinterpret_cast<int*>(mb_data), mc_data, N);
        break;
    case DataType::F32:
        dot_product<float><<<numberOfBlocks, threadsPerBlock>>>(
                reinterpret_cast<float*>(ma_data), reinterpret_cast<float*>(mb_data), mc_data, N);
        break;
    default:
        LOG_ERR("The src type is not supported!");
        break;
    };

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double* md_data;
    const int md_size = numberOfBlocks * sizeof(double);
    CUDA_CHECK(cudaMallocManaged((void**)&md_data, md_size));

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(md_data, md_size, device_id));
    }

    reduce_shared<<<numberOfBlocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(mc_data, md_data, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(md_data, md_size, cudaCpuDeviceId));
    }

    for (size_t n = 0; n < numberOfBlocks; ++n) {
        result += md_data[n];
    }

    return result;
}

G_FCV_NAMESPACE1_END()
