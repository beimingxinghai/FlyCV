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
#include "modules/img_transform/copy_make_border/interface/copy_make_border_cuda.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

static __constant__ double gpu_scaler[4];

template <typename T>
__global__ void copy_make_border_constant_kernel(const T* src,
                                                 const int src_s,
                                                 T* dst,
                                                 const int dst_w,
                                                 const int dst_h,
                                                 const int dst_s,
                                                 const int channel,
                                                 const int left,
                                                 const int top,
                                                 const int right,
                                                 const int bottom) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x < dst_w && y < dst_h && z < channel) {
        if (x >= left && x < (dst_w - right) && y >= top && y < (dst_h - bottom)) {
            const int x_shift = x - left;
            const int y_shift = y - top;
            dst[x * channel + y * dst_s + z] = src[x_shift * channel + y_shift * src_s + z];
        } else {
            dst[x * channel + y * dst_s + z] = gpu_scaler[z];
        }
    }
}

int copy_make_border(CudaMat& src,
                     CudaMat& dst,
                     int top,
                     int bottom,
                     int left,
                     int right,
                     BorderType border_type,
                     const Scalar& scaler,
                     Stream& stream) {
    // Check whether the parameter is legal
    if (src.empty()) {
        LOG_ERR("The src image is empty for copy_make_border!");
        return -1;
    }

    switch (src.type()) {
    case FCVImageType::PLA_BGR_U8:
    case FCVImageType::PLA_RGB_U8:
    case FCVImageType::PLA_BGRA_U8:
    case FCVImageType::PLA_RGBA_U8:
    case FCVImageType::PLA_BGR_F32:
    case FCVImageType::PLA_RGB_F32:
    case FCVImageType::PLA_BGRA_F32:
    case FCVImageType::PLA_RGBA_F32:
    case FCVImageType::NV21:
    case FCVImageType::NV12:
    case FCVImageType::I420:
        LOG_ERR("Unsupported src image type for copy_make_border : %d", int(src.type()));
        return -1;
    default:
        break;
    };

    if (top < 0 || bottom < 0 || left < 0 || right < 0) {
        LOG_ERR("The top : %d, bottom : %d, left : %d, right : %d has "
                "negative value for copy_make_border!",
                top,
                bottom,
                left,
                right);
        return -1;
    }

    TypeInfo type_info;
    int status = get_type_info(src.type(), type_info);
    if (status != 0) {
        LOG_ERR("The src type is not supported!");
        return -1;
    }

    // TODO(chenlong22) : add other BorderType later
    if (border_type != BorderType::BORDER_CONSTANT) {
        LOG_ERR("Only support copy_make_border type BORDER_CONSTANT now!");
        return -1;
    }

    int expected_width = src.width() + left + right;
    int expected_height = src.height() + top + bottom;
    // Check whether dst image mat is legal
    if (dst.empty() || dst.width() != expected_width || dst.height() != expected_height || dst.type() != src.type()) {
        dst = CudaMat(expected_width, expected_height, src.type());
    }

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coherent_flag = CUDADeviceInfo::get_instance()->device_attrs[device_id].coherent_flag;

    if (coherent_flag) {
        CUDA_CHECK(cudaMemPrefetchAsync(src.data(), src.total_byte_size(), device_id));
    }

    const int channel = src.channels();
    const int src_stride = src.stride() / src.type_byte_size();
    const int dst_stride = dst.stride() / dst.type_byte_size();
    const int dst_width = dst.width();
    const int dst_height = dst.height();

    const double* cpu_scaler = scaler.val();
    cudaMemcpyToSymbol(gpu_scaler, cpu_scaler, sizeof(double) * 4);

    dim3 blocks(32, 8, 4);
    int grid_x = fcv_ceil((dst_width + blocks.x - 1) / (float)blocks.x);
    int grid_y = fcv_ceil((dst_height + blocks.y - 1) / (float)blocks.y);
    dim3 grids(grid_x, grid_y);

    if (border_type == BorderType::BORDER_CONSTANT) {
        switch (type_info.data_type) {
        case DataType::UINT8: {
            const unsigned char* src_data = (const unsigned char*)src.data();
            unsigned char* dst_data = (unsigned char*)dst.data();
            copy_make_border_constant_kernel<<<grids, blocks>>>(src_data,
                                                                src_stride,
                                                                dst_data,
                                                                dst_width,
                                                                dst_height,
                                                                dst_stride,
                                                                channel,
                                                                left,
                                                                top,
                                                                right,
                                                                bottom);
        } break;
        case DataType::UINT16: {
            const unsigned short* src_data = (const unsigned short*)src.data();
            unsigned short* dst_data = (unsigned short*)dst.data();
            copy_make_border_constant_kernel<<<grids, blocks>>>(src_data,
                                                                src_stride,
                                                                dst_data,
                                                                dst_width,
                                                                dst_height,
                                                                dst_stride,
                                                                channel,
                                                                left,
                                                                top,
                                                                right,
                                                                bottom);
        } break;
        case DataType::SINT32: {
            const signed int* src_data = (const signed int*)src.data();
            signed int* dst_data = (signed int*)dst.data();
            copy_make_border_constant_kernel<<<grids, blocks>>>(src_data,
                                                                src_stride,
                                                                dst_data,
                                                                dst_width,
                                                                dst_height,
                                                                dst_stride,
                                                                channel,
                                                                left,
                                                                top,
                                                                right,
                                                                bottom);
        } break;
        case DataType::F32: {
            const float* src_data = (const float*)src.data();
            float* dst_data = (float*)dst.data();
            copy_make_border_constant_kernel<<<grids, blocks>>>(src_data,
                                                                src_stride,
                                                                dst_data,
                                                                dst_width,
                                                                dst_height,
                                                                dst_stride,
                                                                channel,
                                                                left,
                                                                top,
                                                                right,
                                                                bottom);
        } break;
        case DataType::F64: {
            const double* src_data = (const double*)src.data();
            double* dst_data = (double*)dst.data();
            copy_make_border_constant_kernel<<<grids, blocks>>>(src_data,
                                                                src_stride,
                                                                dst_data,
                                                                dst_width,
                                                                dst_height,
                                                                dst_stride,
                                                                channel,
                                                                left,
                                                                top,
                                                                right,
                                                                bottom);
        } break;
        default:
            LOG_ERR("copy_make_border is not support this type, the current src element data type is %d",
                    int(type_info.data_type));
            return -1;
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

G_FCV_NAMESPACE1_END()
