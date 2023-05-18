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

#include <float.h>

#include <cmath>

#include "modules/core/base/include/type_info.h"
#include "modules/core/basic_math/interface/basic_math_cuda.h"
#include "modules/img_transform/warp_affine/interface/warp_affine_cuda.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

static bool __host__ __device__ inverse_matrix_2x3(const double* src_mat, double* dst_mat) {
    if ((nullptr == src_mat) || (nullptr == dst_mat)) {
        LOG_ERR("Mat is empty!");
        return false;
    }

    double d = src_mat[0] * src_mat[4] - src_mat[1] * src_mat[3];
    d = (fabs(d) < FCV_EPSILON) ? 0.0f : 1.0f / d;
    double a11 = src_mat[4] * d, a22 = src_mat[0] * d;
    dst_mat[0] = a11;
    dst_mat[1] = src_mat[1] * (-d);
    dst_mat[3] = src_mat[3] * (-d);
    dst_mat[4] = a22;
    double b1 = -dst_mat[0] * src_mat[2] - dst_mat[1] * src_mat[5];
    double b2 = -dst_mat[3] * src_mat[2] - dst_mat[4] * src_mat[5];
    dst_mat[2] = b1;
    dst_mat[5] = b2;

    return true;
}

template <typename T>
static T __device__ getPixel(const T* src,
                             const int x,
                             const int y,
                             const int k,
                             const int src_w,
                             const int src_h,
                             const int src_s,
                             const int src_c,
                             const double* brd_v) {
    const int src_index = x * src_c + y * src_s + k;
    if (x >= 0 && x < src_w && y >= 0 && y < src_h) {
        return src[src_index];
    }

    return brd_v[k];
}

template <typename T>
__global__ void warp_affine_nearest_kernel(const T* src_ptr,
                                           const int src_w,
                                           const int src_h,
                                           const int src_s,
                                           const double* xform,
                                           const int channel,
                                           const int dst_w,
                                           const int dst_h,
                                           const int dst_s,
                                           T* dst_ptr,
                                           const double* brd_v) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < dst_w && dst_y < dst_h) {
        const int src_x = __float2int_rz(dst_x * xform[0] + dst_y * xform[1] + xform[2]);
        const int src_y = __float2int_rz(dst_x * xform[3] + dst_y * xform[4] + xform[5]);
        for (int z = 0; z < channel; z++) {
            const int dst_index = dst_x * channel + dst_y * dst_s + z;

            dst_ptr[dst_index] = getPixel<T>(src_ptr, src_x, src_y, z, src_w, src_h, src_s, channel, brd_v);
        }
    }
}

template <typename T>
__global__ void warp_affine_linear_cn_kernel(const T* src_ptr,
                                             const int src_w,
                                             const int src_h,
                                             const int src_s,
                                             const double* xform,
                                             const int channel,
                                             const int dst_w,
                                             const int dst_h,
                                             const int dst_s,
                                             T* dst_ptr,
                                             const double* brd_v) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < dst_w && dst_y < dst_h) {
        float delta_x = dst_x * xform[0] + dst_y * xform[1] + xform[2];
        float delta_y = dst_x * xform[3] + dst_y * xform[4] + xform[5];

        int src_x = __float2int_rd(delta_x);
        int src_y = __float2int_rd(delta_y);

        delta_x -= src_x;
        delta_y -= src_y;

        for (int z = 0; z < channel; z++) {
            float out = 0;
            T src_reg = getPixel<T>(src_ptr, src_x, src_y, z, src_w, src_h, src_s, channel, brd_v);
            out += src_reg * ((1.0 - delta_x) * (1.0 - delta_y));

            src_reg = getPixel<T>(src_ptr, src_x + 1, src_y, z, src_w, src_h, src_s, channel, brd_v);
            out += src_reg * ((delta_x) * (1.0 - delta_y));

            src_reg = getPixel<T>(src_ptr, src_x, src_y + 1, z, src_w, src_h, src_s, channel, brd_v);
            out += src_reg * ((1.0 - delta_x) * (delta_y));

            src_reg = getPixel<T>(src_ptr, src_x + 1, src_y + 1, z, src_w, src_h, src_s, channel, brd_v);
            out += src_reg * ((delta_x) * (delta_y));

            int dst_index = dst_x * channel + dst_y * dst_s + z;

            dst_ptr[dst_index] = out < 0 ? 0 : (out > 255 ? 255 : out);
        }
    }
}

static bool check_warp_affine_support(FCVImageType type) {
    if (!(type == FCVImageType::GRAY_U8 || type == FCVImageType::PKG_RGB_U8 || type == FCVImageType::PKG_BGR_U8
          || type == FCVImageType::PKG_RGBA_U8 || type == FCVImageType::PKG_BGRA_U8 || type == FCVImageType::GRAY_F32
          || type == FCVImageType::PKG_RGB_F32 || type == FCVImageType::PKG_BGR_F32
          || type == FCVImageType::PKG_RGBA_F32 || type == FCVImageType::PKG_BGRA_F32)) {
        LOG_ERR("Invalid DataType: %d ", int(type));
        return false;
    }
    return true;
}

int warp_affine(const CudaMat& src,
                CudaMat& dst,
                const double* m_data,
                InterpolationType interpolation,
                BorderType border_method,
                const Scalar border_value,
                Stream& stream) {
    if (src.empty()) {
        LOG_ERR("Input CudaMat of warp_affine is empty!");
        return -1;
    }

    if (!check_warp_affine_support(src.type())) {
        return -1;
    }

    if (border_method != BorderType::BORDER_CONSTANT) {
        LOG_ERR("warp_affine interpolation type not support yet!");
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

    if (src.type() != dst.type()) {
        LOG_ERR("The src and dst Mat type should be the same for warp_affine!");
        return -1;
    }

    double m_ivt[6];

    // 求逆矩阵，用于根据目标坐标求愿坐标点
    inverse_matrix_2x3(m_data, m_ivt);
    double* brd_v;
    double* xform;

    cudaMallocManaged(&brd_v, sizeof(double) * 4);
    cudaMallocManaged(&xform, sizeof(double) * 6);

    cudaMemcpy(brd_v, border_value.val(), sizeof(double) * 4, cudaMemcpyKind::cudaMemcpyDefault);
    cudaMemcpy(xform, m_ivt, sizeof(double) * 6, cudaMemcpyKind::cudaMemcpyDefault);

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag = CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(src.data(), src.total_byte_size(), device_id));
    }

    const int src_w = src.width();
    const int src_h = src.height();
    const int src_s = src.stride() / src.type_byte_size();
    int channel = src.channels();
    const int dst_w = dst.width();
    const int dst_h = dst.height();
    const int dst_s = dst.stride() / dst.type_byte_size();

    dim3 blocks(32, 8);
    int grid_x = fcv_ceil((dst_w + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((dst_h + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);

    switch (interpolation) {
    case InterpolationType::INTER_NEAREST:
        switch (cur_type_info.data_type) {
        case DataType::UINT8: {
            const unsigned char* src_data = (const unsigned char*)src.data();
            unsigned char* dst_data = (unsigned char*)dst.data();
            warp_affine_nearest_kernel<<<grids, blocks>>>(
                    src_data, src_w, src_h, src_s, xform, channel, dst_w, dst_h, dst_s, dst_data, brd_v);
        } break;
        case DataType::F32: {
            const float* src_data = (const float*)src.data();
            float* dst_data = (float*)dst.data();
            warp_affine_nearest_kernel<<<grids, blocks>>>(
                    src_data, src_w, src_h, src_s, xform, channel, dst_w, dst_h, dst_s, dst_data, brd_v);
        } break;
        default:
            LOG_ERR("subtract is not support this type, the current src element data type is %d",
                    int(cur_type_info.data_type));
            return -1;
        }
        break;
    case InterpolationType::INTER_LINEAR:
        switch (cur_type_info.data_type) {
        case DataType::UINT8: {
            const unsigned char* src_data = (const unsigned char*)src.data();
            unsigned char* dst_data = (unsigned char*)dst.data();

            warp_affine_linear_cn_kernel<<<grids, blocks>>>(
                    src_data, src_w, src_h, src_s, xform, channel, dst_w, dst_h, dst_s, dst_data, brd_v);

        } break;
        case DataType::F32: {
            const float* src_data = (const float*)src.data();
            float* dst_data = (float*)dst.data();

            warp_affine_linear_cn_kernel<<<grids, blocks>>>(
                    src_data, src_w, src_h, src_s, xform, channel, dst_w, dst_h, dst_s, dst_data, brd_v);

        } break;
        default:
            LOG_ERR("subtract is not support this type, the current src element data type is %d",
                    int(cur_type_info.data_type));
            return -1;
        }
        break;
    default:
        LOG_ERR("The warp_affine interpolation %d is unsupported now", int(interpolation));
        return -1;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(xform);
    cudaFree(brd_v);

    return 0;
}

G_FCV_NAMESPACE1_END()
