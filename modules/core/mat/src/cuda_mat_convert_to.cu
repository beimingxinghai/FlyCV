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
#include <iostream>

#include "modules/core/base/include/type_info.h"
#include "modules/core/basic_math/interface/basic_math.h"
#include "modules/core/mat/interface/cuda_mat.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

template <typename ST, typename DT>
__global__ void convert_type(ST* src, DT* dst, int count, double scale, double shift) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < count) {
        dst[n] = static_cast<DT>(src[n] * scale + shift);
    }
}

int CudaMat::convert_to(CudaMat& dst, FCVImageType dst_type, double scale, double shift, Stream& stream) const {
    if (dst.empty()) {
        dst = CudaMat(_width, _height, dst_type);
    }

    int dst_w = dst.width();
    int dst_h = dst.height();
    int dst_c = dst.channels();

    if ((dst_w != _width) || (dst_h != _height) || (dst_c != _channels)) {
        LOG_ERR("The size of dst and src should be the same, "
                "width: %d -> %d, height: %d -> %d, channels: "
                "%d -> %d",
                dst_w,
                _width,
                dst_h,
                _height,
                dst_c,
                _channels);
        return -1;
    }

    TypeInfo type_info;
    int status = get_type_info(dst_type, type_info);

    if (status != 0 || (type_info.data_type != DataType::F32 && type_info.data_type != DataType::F64)) {
        LOG_ERR("The dst_type is not supported!");
        return -1;
    }

    int N = _width * _height * _channels;
    // std::cout << "N: " << N << std::endl;
    size_t threadsPerBlock = 1024;
    size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    switch (_type) {
    case FCVImageType::GRAY_U8:
    case FCVImageType::PLA_BGR_U8:
    case FCVImageType::PLA_RGB_U8:
    case FCVImageType::PKG_BGR_U8:
    case FCVImageType::PKG_RGB_U8:
    case FCVImageType::PLA_BGRA_U8:
    case FCVImageType::PLA_RGBA_U8:
    case FCVImageType::PKG_BGRA_U8:
    case FCVImageType::PKG_RGBA_U8:
        dst_w = FCV_MAX(dst_w, (int)_stride);
        // std::cout << "count: " << dst_w * dst_h << std::endl;
        convert_type<unsigned char, float><<<numberOfBlocks, threadsPerBlock>>>(
                static_cast<unsigned char*>(_data), static_cast<float*>(dst.data()), dst_w * dst_h, scale, shift);
        break;
    case FCVImageType::GRAY_U16:
        dst_w = FCV_MAX(dst_w, (int)(_stride / sizeof(unsigned short)));
        // std::cout << "count: " << dst_w * dst_h << std::endl;
        convert_type<unsigned short, float><<<numberOfBlocks, threadsPerBlock>>>(
                static_cast<unsigned short*>(_data), static_cast<float*>(dst.data()), dst_w * dst_h, scale, shift);
        break;
    case FCVImageType::GRAY_S32:
        dst_w = FCV_MAX(dst_w, (int)(_stride / sizeof(int)));
        // std::cout << "count: " << dst_w * dst_h << std::endl;
        convert_type<int, float><<<numberOfBlocks, threadsPerBlock>>>(
                static_cast<int*>(_data), static_cast<float*>(dst.data()), dst_w * dst_h, scale, shift);
        break;
    case FCVImageType::GRAY_F64:
        dst_w = FCV_MAX(dst_w, (int)(_stride / sizeof(double)));
        // std::cout << "count: " << dst_w * dst_h << std::endl;
        convert_type<double, double><<<numberOfBlocks, threadsPerBlock>>>(
                static_cast<double*>(_data), static_cast<double*>(dst.data()), dst_w * dst_h, scale, shift);
        break;
    default:
        LOG_ERR("The src type is not supported!");
        return -1;
    };

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

G_FCV_NAMESPACE1_END()
