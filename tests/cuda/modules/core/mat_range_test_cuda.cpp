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

#include "flycv.h"
#include "gtest/gtest.h"
#include "test_util.h"

using namespace g_fcv_ns;

class CudaMatRangeTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(prepare_gray_u8_720p_cuda(gray_u8_src), 0);
        ASSERT_EQ(prepare_pkg_bgr_u8_720p_cuda(pkg_bgr_u8_src), 0);
        ASSERT_EQ(prepare_pkg_bgr_u8_720p_cuda_batch(pkg_bgr_u8_batch_src, 3), 0);
    }

public:
    int height_start = 100;
    int height_end = 300;
    int width_start = 100;
    int width_end = 300;
    CudaMat gray_u8_src;
    CudaMat pkg_bgr_u8_src;
    CudaMat pkg_bgr_u8_batch_src;
};

TEST_F(CudaMatRangeTest, RangeSingleBatch_GrayU8) {
    CudaMat dst(gray_u8_src, Range(height_start, height_end), Range(width_start, width_end));

    ASSERT_EQ(dst.width(), width_end - width_start);
    ASSERT_EQ(dst.height(), height_end - height_start);
    ASSERT_EQ(dst.channels(), gray_u8_src.channels());
    ASSERT_EQ(dst.batch(), gray_u8_src.batch());
    ASSERT_EQ(dst.stride(), gray_u8_src.stride());
    ASSERT_EQ(dst.total_byte_size(), gray_u8_src.total_byte_size());

    unsigned char* src_data = static_cast<unsigned char*>(gray_u8_src.data());
    unsigned char* dst_data = static_cast<unsigned char*>(dst.data());

    for (int i = 0; i < dst.height(); ++i) {
        for (int j = 0; j < dst.width() * dst.channels(); j++) {
            ASSERT_EQ(src_data[height_start * gray_u8_src.stride() + width_start * gray_u8_src.pixel_byte_size()
                               + i * gray_u8_src.stride() + j],
                      dst_data[i * dst.stride() + j]);
        }
    }

    CudaMat dst2 = dst.clone();
    unsigned char* dst2_data = static_cast<unsigned char*>(dst2.data());
    for (int i = 0; i < dst2.height(); ++i) {
        for (int j = 0; j < dst2.width() * dst2.channels(); j++) {
            ASSERT_EQ(dst_data[i * dst.stride() + j], dst2_data[i * dst2.stride() + j]);
        }
    }
}

TEST_F(CudaMatRangeTest, RangeSingleBatch_PkgBgrU8) {
    CudaMat dst(pkg_bgr_u8_src, Range(height_start, height_end), Range(width_start, width_end));

    ASSERT_EQ(dst.width(), width_end - width_start);
    ASSERT_EQ(dst.height(), height_end - height_start);
    ASSERT_EQ(dst.channels(), pkg_bgr_u8_src.channels());
    ASSERT_EQ(dst.batch(), pkg_bgr_u8_src.batch());
    ASSERT_EQ(dst.stride(), pkg_bgr_u8_src.stride());
    ASSERT_EQ(dst.total_byte_size(), pkg_bgr_u8_src.total_byte_size());

    unsigned char* src_data = static_cast<unsigned char*>(pkg_bgr_u8_src.data());
    unsigned char* dst_data = static_cast<unsigned char*>(dst.data());

    for (int i = 0; i < dst.height(); ++i) {
        for (int j = 0; j < dst.width() * dst.channels(); j++) {
            ASSERT_EQ(src_data[height_start * pkg_bgr_u8_src.stride() + width_start * pkg_bgr_u8_src.pixel_byte_size()
                               + i * pkg_bgr_u8_src.stride() + j],
                      dst_data[i * dst.stride() + j]);
        }
    }

    CudaMat dst2 = dst.clone();
    unsigned char* dst2_data = static_cast<unsigned char*>(dst2.data());
    for (int i = 0; i < dst2.height(); ++i) {
        for (int j = 0; j < dst2.width() * dst2.channels(); j++) {
            ASSERT_EQ(dst_data[i * dst.stride() + j], dst2_data[i * dst2.stride() + j]);
        }
    }
}

TEST_F(CudaMatRangeTest, RangeMultiBatch_PkgBgrU8) {
    CudaMat dst(pkg_bgr_u8_batch_src, Range(height_start, height_end), Range(width_start, width_end));

    ASSERT_EQ(dst.width(), width_end - width_start);
    ASSERT_EQ(dst.height(), height_end - height_start);
    ASSERT_EQ(dst.channels(), pkg_bgr_u8_batch_src.channels());
    ASSERT_EQ(dst.batch(), pkg_bgr_u8_batch_src.batch());
    ASSERT_EQ(dst.stride(), pkg_bgr_u8_batch_src.stride());
    ASSERT_EQ(dst.total_byte_size(), pkg_bgr_u8_batch_src.total_byte_size());

    unsigned char* src_data = static_cast<unsigned char*>(pkg_bgr_u8_batch_src.data());
    unsigned char* dst_data = static_cast<unsigned char*>(dst.data());

    for (int k = 0; k < dst.batch(); k++) {
        for (int i = 0; i < dst.height(); ++i) {
            for (int j = 0; j < dst.width() * dst.channels(); j++) {
                ASSERT_EQ(src_data[k * pkg_bgr_u8_batch_src.batch_byte_size()
                                   + height_start * pkg_bgr_u8_batch_src.stride()
                                   + width_start * pkg_bgr_u8_batch_src.pixel_byte_size()
                                   + i * pkg_bgr_u8_batch_src.stride() + j],
                          dst_data[k * dst.batch_byte_size() + i * dst.stride() + j]);
                // printf("%d ", dst_data[k * dst.batch_byte_size() + i * dst.stride() + j]);
            }
        }
    }

    CudaMat dst2 = dst.clone();
    unsigned char* dst2_data = static_cast<unsigned char*>(dst2.data());
    for (int k = 0; k < dst.batch(); k++) {
        for (int i = 0; i < dst2.height(); ++i) {
            for (int j = 0; j < dst2.width() * dst2.channels(); j++) {
                ASSERT_EQ(dst_data[k * dst.batch_byte_size() + i * dst.stride() + j],
                          dst2_data[k * dst2.batch_byte_size() + i * dst2.stride() + j]) << " " << i << " " << j << " " << k;
            }
        }
    }
}
