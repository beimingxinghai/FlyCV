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

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include <vector>

#include "flycv_namespace.h"
#include "modules/core/allocator/interface/base_allocator.h"
#include "modules/core/base/interface/basic_types.h"
#include "modules/core/base/interface/log.h"
#include "modules/core/base/interface/macro_ns.h"
#include "modules/core/basic_math/interface/basic_math.h"
#include "modules/core/base/interface/cuda_types.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

class FCV_CLASS CudaMat {
public:
    CudaMat();

    //! constructor for GpuMat headers pointing to user-allocated data
    CudaMat(int width, int height, FCVImageType type, void* data, int stride = 0);
    CudaMat(Size size, FCVImageType type, void* data, int stride = 0);

    //! constructs CudaMat of the specified size and type
    CudaMat(
            int width,
            int height,
            FCVImageType type,
            int stride = 0,
            int flag = 0,
            PlatformType platform = PlatformType::CUDA);
    CudaMat(
            Size size,
            FCVImageType type,
            int stride = 0,
            int flag = 0,
            PlatformType platform = PlatformType::CUDA);


    //! copy constructor
    CudaMat(const CudaMat& m);

    //! destructor - calls release()
    ~CudaMat();

    /** @brief Performs data upload to CudaMat (Blocking call)

    This function copies data from host memory to device memory. As being a
    blocking call, it is guaranteed that the copy operation is finished when
    this function returns.
    */
    void upload(CudaMat host);

    /** @brief Performs data upload to CudaMat (Non-Blocking call)

    This function copies data from host memory to device memory. As being a
    non-blocking call, this function may return even if the copy operation is
    not finished.

    The copy operation may be overlapped with operations in other non-default
    streams if \p stream is not the default stream and \p dst is HostMem
    allocated with HostMem::PAGE_LOCKED option.
    */
    void upload(CudaMat host, Stream& stream);

    /** @brief Performs data download from CudaMat (Blocking call)

    This function copies data from device memory to host memory. As being a
    blocking call, it is guaranteed that the copy operation is finished when
    this function returns.
    */
    void download(CudaMat device) const;

    /** @brief Performs data download from CudaMat (Non-Blocking call)

    This function copies data from device memory to host memory. As being a
    non-blocking call, this function may return even if the copy operation is
    not finished.

    The copy operation may be overlapped with operations in other non-default
    streams if \p stream is not the default stream and \p dst is HostMem
    allocated with HostMem::PAGE_LOCKED option.
    */
    void download(CudaMat device, Stream& stream) const;

    //! return number of columns
    int width() const;

    //! return number of rows
    int height() const;

    //! returns CudaMat size : width == number of columns, height == number of
    //! rows
    Size2i size() const;

    //! returns number of channels
    int channels() const;

    //! returns the element size in bytes of step
    int stride() const;

    // ! returns the memory flag
    int flag() const;

    //! returns element type
    FCVImageType type() const;

    //! returns the type size in bytes
    int type_byte_size() const;

    //! returns the total size in bytes
    uint64_t total_byte_size() const;

    //! returns true if CudaMat data is NULL
    bool empty() const;

    // returns pointer to cuda memory
    void* data() const;

    //! returns deep copy of the CudaMat, i.e. the data is copied
    CudaMat clone() const;

    //! returns reference to pixel location template version
    template<typename T>
    T& at(int x, int y, int c = 0) {
        return *reinterpret_cast<T*>(get_pixel_address(x, y, c));
    }

    template<typename T>
    const T& at(int x, int y, int c = 0) const {
        return *reinterpret_cast<T*>(get_pixel_address(x, y, c));
    }

    //! returns pointer to pixel location template version
    template <typename T>
    T* ptr(int x, int y, int c = 0) {
        return *reinterpret_cast<T*>(get_pixel_address(x, y, c));
    }

    template <typename T>
    const T* ptr(int x, int y, int c = 0) const {
        return *reinterpret_cast<T*>(get_pixel_address(x, y, c));
    }

    //! returns true i the CudaMat data is continuous
    //! (i.e. when there are no gaps between successive rows)
    bool is_continuous() const;

    //!  returns GPU memory info
    CUDAMemoryType memory_type() const;

    /** @brief Converts an CudaMat array to another data type with optional scaling.(Blocking call)
    The method converts source pixel values to the target data type. saturate_cast\<\> is applied at
    the end to avoid possible overflows:
    @param dst output CudaMat; if it does not have a proper size or type before the operation, it is
    reallocated.
    @param rtype desired output matrix type
    @param scale optional scale factor.
    @param shift optional delta added to the scaled values.
     */
    int convert_to(CudaMat& dst, FCVImageType rtype, double scale = 1.0, double shift = 0.0) const;

    /** @brief Converts an CudaMat array to another data type with optional scaling.(Non-Blocking call)
    The method converts source pixel values to the target data type. saturate_cast\<\> is applied at
    the end to avoid possible overflows:
    @param dst output CudaMat; if it does not have a proper size or type before the operation, it is
    reallocated.
    @param rtype desired output matrix type
    @param stream cuda stream for bound
    @param scale optional scale factor.
    @param shift optional delta added to the scaled values.
     */
    int convert_to(CudaMat& dst, FCVImageType rtype, Stream& stream, double scale = 1.0, double shift = 0.0) const;

    /** @brief Copies the CudaMat to another memory. (Blocking call)
    @param dst Destination matrix. If it does not have a proper size or type before the operation, it is
    reallocated.
     */
    void copy_to(CudaMat& dst) const;

    /** @brief Copies the CudaMat to another memory. (Non-Blocking call)
    @param dst Destination matrix. If it does not have a proper size or type before the operation, it is
    reallocated.
    @param stream cuda stream for bound
     */
    void copy_to(CudaMat& dst, Stream& stream) const;

    /** @overload
    @param dst Destination matrix. If it does not have a proper size or type before the operation, it is
    reallocated.
    @param mask Operation mask of the same size as \*this. Its non-zero elements indicate which matrix
    elements need to be copied. The mask has to be of type unsigned char and can have 1 or multiple channels.
    */
    int copy_to(CudaMat& dst, CudaMat& mask) const;

    /** @overload
    @param dst Destination matrix. If it does not have a proper size or type before the operation, it is
    reallocated.
    @param mask Operation mask of the same size as \*this. Its non-zero elements indicate which matrix
    elements need to be copied. The mask has to be of type unsigned char and can have 1 or multiple channels.
    @param stream cuda stream for bound
    */
    int copy_to(CudaMat& dst, CudaMat& mask, Stream& stream) const;

    /** @overload
      * copy src to the area oriented of dst, so the size of dst cannot samller than src's.
    @param dst Destination matrix. If it does not have a proper size or type before the operation, it is
    return.
    @param rect dst rect Rect_(T x, T y, T width, T height).
    */
    int copy_to(CudaMat& dst, Rect& rect) const;

    /** @overload
      * copy src to the area oriented of dst, so the size of dst cannot samller than src's.
    @param dst Destination matrix. If it does not have a proper size or type before the operation, it is
    return.
    @param rect dst rect Rect_(T x, T y, T width, T height).
    @param stream cuda stream for bound
    */
    int copy_to(CudaMat& dst, Rect& rect, Stream& stream) const;

    /** @brief Computes a dot-product of two vectors.
    The method computes a dot-product of two matrices. The vectors must have the same size and type. If the matrices have more than one channel,
    the dot products from all the channels are summed together.
    @param m another dot-product operand.
     */
    double dot(CudaMat& m) const;

    /** @brief Computes a dot-product of two vectors.
    The method computes a dot-product of two matrices. The vectors must have the same size and type. If the matrices have more than one channel,
    the dot products from all the channels are summed together.
    @param m another dot-product operand.
    @param stream cuda stream for bound
     */
    double dot(CudaMat& m, Stream& stream) const;

    /** @brief Compute the inverse of a matrix.
    The method performs a matrix inversion by means of matrix expressions. This means that a temporary
    matrix inversion object is returned by the method and can be used further as a part of more complex
    matrix expressions or can be assigned to a matrix.
    @param dst Destination matrix. If it does not have a proper size or type before the operation, it is
    return
     */
    bool invert(CudaMat& dst) const;

    /** @brief Compute the inverse of a matrix.
    The method performs a matrix inversion by means of matrix expressions. This means that a temporary
    matrix inversion object is returned by the method and can be used further as a part of more complex
    matrix expressions or can be assigned to a matrix.
    @param dst Destination matrix. If it does not have a proper size or type before the operation, it is
    return
    @param stream cuda stream for bound
     */
    bool invert(CudaMat& dst, Stream& stream) const;

private:
    //! the number of width and height
    int _width;
    int _height;

    //! the number of channel
    int _channels;

    //! a distance between successive rows in bytes; includes the gap if any
    int _stride;

    /*! includes several bit-fields:
    - use general/unified memory 3bit
      - 0: use unified memory
      - 1: use general memory
      - 2: use constant memory
    - use memory pool 1bit
      - 4: use memory pool 0: not use 1: use
    - continuity flag 1bit
      - 5: whether data address continue 0: continue 1: not continue
    */
    int _flag;

    //! total size in bytes
    uint64_t _total_byte_size;

    //! image type
    FCVImageType _type;

    //! image type size in bytes
    int _type_byte_size;

    //! use in which platform
    PlatformType _platform;

    //! pointer to the data
    void* _data;

    //! pixel size in bytes of all channel
    int _pixel_offset;

    //! the same pixel interval of different channels
    int _channel_offset;

    //! parse FCVImageType info
    int parse_type_info();

    //! get data address point from pixel addrees
    void* get_pixel_address(int x, int y, int c) const;

    //! data allocator, manage different alloc method of image data memory
    std::shared_ptr<BaseAllocator> _allocator;
};

template<typename T>
CudaMat allocate_cudamat(int width, int height, int channels);

G_FCV_NAMESPACE1_END()
