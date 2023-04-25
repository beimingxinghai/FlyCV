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

#include "modules/core/base/include/type_info.h"
#include "modules/img_calculation/matrix_mul/interface/matrix_mul_cuda.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

#define BLOCK_SIZE 32

template <typename T>
__global__ void matrix_mul_c1(T* dst, T* src0, T* src1, int w, int h, int k, int b) {
    __shared__ T tile_0[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_1[BLOCK_SIZE][BLOCK_SIZE];

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = blockIdx.z * blockDim.z + threadIdx.z;

    int N = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    T data = 0;
    for (int i = 0; i < N; i++) {
        int row0 = tz * h * k + ty * k + i * BLOCK_SIZE + threadIdx.x;
        int row1 = tz * k * w + w * (i * BLOCK_SIZE + threadIdx.y) + tx;

        if (tx < w && ty < h && tz < b) {
            tile_0[threadIdx.y][threadIdx.x] = src0[row0];
            tile_1[threadIdx.y][threadIdx.x] = src1[row1];
        } else {
            tile_0[threadIdx.y][threadIdx.x] = 0;
            tile_1[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            data += tile_0[threadIdx.y][j] * tile_1[j][threadIdx.x];
        }
        __syncthreads();
    }

    if (tx < w && ty < h && tz < b) {
        dst[tz * w * h + ty * w + tx] = data;
    }
}

int matrix_mul(const CudaMat& src0, const CudaMat& src1, CudaMat& dst, Stream& stream) {
    if (src0.empty() || src1.empty()) {
        LOG_ERR("The input data is empty!");
        return -1;
    }

    if (src0.width() != src1.height()) {
        LOG_ERR("The width of src0 should be the same with the height of src1!");
        return -1;
    }

    if (src0.channels() != src1.channels() || src0.type() != src1.type()) {
        LOG_ERR("The channel and data type of matrix_mul should be the same!");
        return -1;
    }

    if (src0.batch() != src1.batch()) {
        LOG_ERR("The batch of src0 should be the same with the height of src1!");
        return -1;
    }

    if (dst.empty()) {
        dst = CudaMat(src1.width(), src0.height(), src0.type(), src0.batch());
    }

    TypeInfo type_info;
    int status = get_type_info(src0.type(), type_info);
    size_t src0_size = src0.batch() * src0.height() * src0.width() * type_info.type_byte_size;
    size_t src1_size = src1.batch() * src1.height() * src1.width() * type_info.type_byte_size;
    size_t dst_size = dst.batch() * src1.width() * src0.height() * type_info.type_byte_size;

    const int dst_b = dst.batch();
    const int dst_c = dst.channels();
    const int dst_w = src1.width();
    const int dst_h = src0.height();
    const int mat_k = src0.width();

    const int grid_x_size = (dst_w + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int grid_y_size = (dst_h + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // printf("grid: %d %d %d \n", grid_x_size, grid_y_size, dst_b);

    const dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 gridsize(grid_x_size, grid_y_size, dst_b);

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag = CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;
    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(dst.data(), dst_size, device_id));
        CUDA_CHECK(cudaMemPrefetchAsync(src0.data(), src0_size, device_id));
        CUDA_CHECK(cudaMemPrefetchAsync(src1.data(), src1_size, device_id));
    }

    if (dst_c == 1) {
        switch (type_info.data_type) {
        case DataType::UINT16:
            matrix_mul_c1<unsigned short><<<gridsize, blocksize>>>((unsigned short*)dst.data(),
                                                                   (unsigned short*)src0.data(),
                                                                   (unsigned short*)src1.data(),
                                                                   dst_w,
                                                                   dst_h,
                                                                   mat_k,
                                                                   dst_b);
            break;
        case DataType::SINT32:
            matrix_mul_c1<int><<<gridsize, blocksize>>>(
                    (int*)dst.data(), (int*)src0.data(), (int*)src1.data(), dst_w, dst_h, mat_k, dst_b);
            break;
        case DataType::F32:
            matrix_mul_c1<float><<<gridsize, blocksize>>>(
                    (float*)dst.data(), (float*)src0.data(), (float*)src1.data(), dst_w, dst_h, mat_k, dst_b);
            break;
        case DataType::F64:
            matrix_mul_c1<double><<<gridsize, blocksize>>>(
                    (double*)dst.data(), (double*)src0.data(), (double*)src1.data(), dst_w, dst_h, mat_k, dst_b);
            break;
        default:
            LOG_ERR("The src type is not supported!");
            break;
        }
    } else {
        LOG_ERR("The src channel is not supported!");
        return -1;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

G_FCV_NAMESPACE1_END()
