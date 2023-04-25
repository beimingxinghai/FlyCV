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

#include "modules/core/mat/interface/cuda_mat.h"
#include "modules/core/base/include/type_info.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

CudaMat::CudaMat() :
        _width(0),
        _height(0),
        _batch(0),
        _stride(0),
        _flag(0),
        _type(FCVImageType::GRAY_U8),
        _platform(PlatformType::CUDA),
        _data(nullptr),
        _allocator(nullptr) {
    parse_type_info();
}

CudaMat::CudaMat(
        int width,
        int height,
        FCVImageType type,
        void* data,
        int batch,
        int stride) :
        _width(width),
        _height(height),
        _batch(batch),
        _stride(stride),
        _flag(0),
        _type(type),
        _platform(PlatformType::CUDA),
        _data(data),
        _allocator(nullptr) {
    parse_type_info();
}

CudaMat::CudaMat(
        Size size,
        FCVImageType type,
        void* data,
        int batch,
        int stride) :
        _width(size.width()),
        _height(size.height()),
        _batch(batch),
        _stride(stride),
        _flag(0),
        _type(type),
        _platform(PlatformType::CUDA),
        _data(data),
        _allocator(nullptr) {
    parse_type_info();
}

CudaMat::CudaMat(
        int width,
        int height,
        FCVImageType type,
        int batch,
        int stride,
        int flag,
        PlatformType platform) :
        _width(width),
        _height(height),
        _batch(batch),
        _stride(stride),
        _flag(flag),
        _type(type),
        _platform(platform),
        _data(nullptr),
        _allocator(nullptr) {
    parse_type_info();
    _allocator = get_allocator_from_platform(_total_byte_size, _platform, _flag);
    if (!_allocator) {
        LOG_ERR("Failed to init CudaMat! size: %d \n", _total_byte_size);
        return;
    }

    bool res = _allocator->get_data(&_data);
    if (!res) {
        LOG_ERR("Failed to get CudaMat data address!");
        return;
    }
}

CudaMat::CudaMat(
        Size size,
        FCVImageType type,
        int batch,
        int stride,
        int flag,
        PlatformType platform) :
        _width(size.width()),
        _height(size.height()),
        _batch(batch),
        _flag(flag),
        _stride(stride),
        _type(type),
        _platform(platform),
        _data(nullptr),
        _allocator(nullptr) {
    parse_type_info();
    _allocator = get_allocator_from_platform(_total_byte_size, _platform, _flag);
    if (!_allocator) {
        LOG_ERR("Failed to init CudaMat!");
        return;
    }

    bool res = _allocator->get_data(&_data);

    if (!res) {
        LOG_ERR("Failed to get CudaMat data address!");
        return;
    }
}

CudaMat::CudaMat(const CudaMat& m)
    : _width(m._width),
      _height(m._height),
      _batch(m._batch),
      _stride(m._stride),
      _flag(m._flag),
      _type(m._type),
      _platform(PlatformType::CUDA),
      _data(m._data),
      _allocator(m._allocator) {
    parse_type_info();
}

CudaMat& CudaMat::operator=(const CudaMat& m) {
    if (this == &m) {
        return *this;
    }

    _width = m._width;
    _height = m._height;
    _batch = m._batch;
    _stride = m._stride;
    _flag = m._flag;
    _type = m._type;
    _platform = PlatformType::CUDA;
    _data = m._data;
    _allocator = m._allocator;

    parse_type_info();

    return *this;
}

CudaMat::~CudaMat() { _allocator = nullptr; }

int CudaMat::width() const { return _width; }

int CudaMat::height() const { return _height; }

Size2i CudaMat::size() const {
    return Size2i(_width, _height);
}

int CudaMat::channels() const { return _channels; }

int CudaMat::batch() const { return _batch; }

int CudaMat::stride() const { return _stride; }

int CudaMat::flag() const { return _flag; }

FCVImageType CudaMat::type() const { return _type; }

int CudaMat::type_byte_size() const {
    return _type_byte_size;
}

uint64_t CudaMat::total_byte_size() const {
    return _total_byte_size;
}

bool CudaMat::empty() const {
    return (!_data) || (_width == 0) || (_height == 0);
}

void* CudaMat::data() const {
    return _data;
}

CudaMat CudaMat::clone() const {
    CudaMat tmp(_width, _height, _type, _batch, _stride, _flag, _platform);
    CUDA_CHECK(cudaMemcpy(tmp.data(), _data, _total_byte_size, cudaMemcpyDefault));
    return tmp;
}

int CudaMat::parse_type_info() {
    TypeInfo type_info;
    int status = get_type_info(_type, type_info);

    if (status != 0) {
        LOG_ERR("Unsupport image type: %d", _type);
        return -1;
    }

    _type_byte_size = type_info.type_byte_size;
    _channels = type_info.channels;

    int min_stride = _width * type_info.pixel_offset;

    _stride = (_stride > min_stride) ? _stride : min_stride;
    _pixel_offset = type_info.pixel_offset;

    if (type_info.layout == LayoutType::SINGLE) {
        _channel_offset = 0;
        _batch_offset = _stride * _height;
        _total_byte_size = _batch * _batch_offset;
    } else if (type_info.layout == LayoutType::PACKAGE) {
        _channel_offset = _type_byte_size;
        _batch_offset = _stride * _height;
        _total_byte_size = _batch * _batch_offset;
    } else if (type_info.layout == LayoutType::PLANAR) {
        _channel_offset = _stride * _height;
        _batch_offset = _stride * _height * _channels;
        _total_byte_size = _batch * _batch_offset;
    } else if (type_info.layout == LayoutType::YUV) {
        _channel_offset = -1;
        _batch_offset = _stride * _height * 3 / 2;
        _total_byte_size = _batch * _batch_offset;
    } else {
        LOG_ERR("Unsupported image format, can not get extra info!");
        return -1;
    }
    return 0;
}

void* CudaMat::get_pixel_address(int x, int y, int c, int b) const {
    if (x < 0 || y < 0 || c < 0 || b < 0 || x >= _width
            || y >= _height || c >= _channels || b >= _batch) {
        LOG_ERR("The pixel coordinate (%d, %d, %d, %d) is out of range", x, y, c, b);
        return nullptr;
    }

    char* ptr = nullptr;
    char* data = reinterpret_cast<char*>(_data);
    if (_channel_offset >= 0) { // RGB
        ptr = data + (b * _batch_offset) + (y * _stride) + (x * _pixel_offset) + (c * _channel_offset);
    } else { // YUV
        if (c == 0) { // Y : the same calculation formula
            ptr = data + (b * _batch_offset) + y * _stride + x;
        } else if (c == 1) { // UV planar
            switch (_type) {
            case FCVImageType::NV12:
            case FCVImageType::NV21:
                ptr = data + (b * _batch_offset) + _height * _stride + (y >> 1) * _stride + ((x >> 1) << 1);
                break;
            case FCVImageType::I420:
                ptr = data + (b * _batch_offset) + _height * _stride + ((y * _stride) >> 2) + (x >> 1);
                break;
            default:
                break;
            }
        } else if (c == 2) {
            switch (_type) {
            case FCVImageType::NV12:
            case FCVImageType::NV21:
                ptr = data + (b * _batch_offset) + _height * _stride + (y >> 1) * _stride + (x | int(1));
                break;
            case FCVImageType::I420:
                ptr = data + (b * _batch_offset) + _height * _stride + (_height >> 1) * (_stride >> 1) + (y >> 1) * (_stride >> 1) + (x >> 1);
                break;
            default:
                break;
            }
        }
    }

    return ptr;
}

bool CudaMat::is_continuous() const {
    return _flag & 0b10000;
}

CUDAMemoryType CudaMat::memory_type() const {
    switch (_flag & 0b0111) {
    case 0x0:
         return CUDAMemoryType::UNIFIED;
        break;
    case 0x1:
         return CUDAMemoryType::GENERAL;
        break;
    case 0x2:
         return CUDAMemoryType::CONSTANT;
        break;
    default:
        break;
    }
    return CUDAMemoryType::UNDEFINED;
}

template<typename T>
CudaMat allocate_cudamat(int width, int height, int channels) {
    int type_size = sizeof(T);
    FCVImageType type = FCVImageType::I420;
    if (channels == 1) {
        switch (type_size) {
        case 1:
            type = FCVImageType::GRAY_U8;
            break;
        case 2:
            type = FCVImageType::GRAY_U16;
            break;
        case 4:
            type = FCVImageType::GRAY_F32;
            break;
        case 8:
            type = FCVImageType::GRAY_F64;
            break;
        default:
            LOG_ERR("Wrong type size!");
            break;
        }
    } else if (channels == 3) {
        switch (type_size) {
        case 1:
            type = FCVImageType::PKG_BGR_U8;
            break;
        case 2:
            type = FCVImageType::PKG_BGR565_U8;
            break;
        case 4:
            type = FCVImageType::PKG_BGR_F32;
            break;
        // case 8:
        //     type = FCVImageType::GRAY_F64;
        //     break;
        default:
            LOG_ERR("Wrong type size!");
            break;
        }
    } else {
        LOG_ERR("Unsupport channel num : %d\n", channels);
    }

    if (type == FCVImageType::I420) {
        return CudaMat();
    }

    return CudaMat(width, height, type);
}

template<>
CudaMat allocate_cudamat<int>(int width, int height, int channels) {
    if (channels == 1) {
        return CudaMat(width, height, FCVImageType::GRAY_S32);
    } else {
        return CudaMat();
    }
}

template CudaMat allocate_cudamat<char>(int width, int height, int channels);
template CudaMat allocate_cudamat<unsigned char>(int width, int height, int channels);
template CudaMat allocate_cudamat<short>(int width, int height, int channels);
template CudaMat allocate_cudamat<unsigned short>(int width, int height, int channels);
template CudaMat allocate_cudamat<float>(int width, int height, int channels);
template CudaMat allocate_cudamat<double>(int width, int height, int channels);

G_FCV_NAMESPACE1_END()
