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

#include "gtest/gtest.h"
#include "flycv.h"
#include "test_util.h"

using namespace g_fcv_ns;

class CudaFlipTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(prepare_gray_u8_720p_cuda(gray_u8_src), 0);
        ASSERT_EQ(prepare_pkg_bgr_u8_720p_cuda(pkg_bgr_u8_src), 0);
    }

    CudaMat gray_u8_src;
    CudaMat pkg_bgr_u8_src;
};

TEST_F(CudaFlipTest, FlipXPositiveInput) {
    CudaMat gray_u8_dst;
    int status = flip(gray_u8_src, gray_u8_dst, FlipType::X);
    ASSERT_EQ(status, 0);

    unsigned char* src_data = (unsigned char*)gray_u8_src.data();
    unsigned char* dst_data = (unsigned char*)gray_u8_dst.data();

    const unsigned char* ptr_src = src_data;
    unsigned char* ptr_dst = dst_data + (IMG_720P_HEIGHT - 1) * gray_u8_dst.stride();

    int i = 0, j = 0;
    for (; i < IMG_720P_HEIGHT; i++) {
        const unsigned char* src_row = ptr_src;
        unsigned char* dst_row = ptr_dst;

        for (j = 0; j < IMG_720P_WIDTH; j++) {
            ASSERT_EQ(dst_row[j], src_row[j]);
        }

        ptr_dst -= gray_u8_dst.stride();
        ptr_src += gray_u8_src.stride();
    }

    CudaMat bgr_u8_dst;
    status = flip(pkg_bgr_u8_src, bgr_u8_dst, FlipType::X);
    ASSERT_EQ(status, 0);

    src_data = (unsigned char*)pkg_bgr_u8_src.data();
    dst_data = (unsigned char*)bgr_u8_dst.data();

    ptr_src = src_data;
    ptr_dst = dst_data + (IMG_720P_HEIGHT - 1) * bgr_u8_dst.stride();

    for (; i < IMG_720P_HEIGHT; i++) {
        const unsigned char* src_row = ptr_src;
        unsigned char* dst_row = ptr_dst;

        for (j = 0; j < IMG_720P_WIDTH; j++) {
            ASSERT_EQ(dst_row[j], src_row[j]);
        }

        ptr_dst -= bgr_u8_dst.stride();
        ptr_src += pkg_bgr_u8_src.stride();
    }
}

TEST_F(CudaFlipTest, FlipYPositiveInput) {
    CudaMat gray_u8_dst;
    int status = flip(gray_u8_src, gray_u8_dst, FlipType::Y);
    ASSERT_EQ(status, 0);

    unsigned char* src_data = (unsigned char*)gray_u8_src.data();
    unsigned char* dst_data = (unsigned char*)gray_u8_dst.data();

    int i = 0, j = 0;
    for (; i < IMG_720P_HEIGHT; i++) {
        const unsigned char* src_row = src_data;
        unsigned char* dst_row = dst_data + gray_u8_dst.stride() - 1;

        for (j = 0; j < IMG_720P_WIDTH; j++) {
            ASSERT_EQ(*(dst_row--), *(src_row++));
        }

        dst_data += gray_u8_dst.stride();
        src_data += gray_u8_src.stride();
    }

    CudaMat bgr_u8_dst;
    status = flip(pkg_bgr_u8_src, bgr_u8_dst, FlipType::Y);
    ASSERT_EQ(status, 0);

    src_data = (unsigned char*)pkg_bgr_u8_src.data();
    dst_data = (unsigned char*)bgr_u8_dst.data();


    for (; i < IMG_720P_HEIGHT; i++) {
        const unsigned char* src_row = src_data;
        unsigned char* dst_row = dst_data + bgr_u8_dst.stride() - 3;

        for (j = 0; j < IMG_720P_WIDTH; j++) {
            ASSERT_EQ(*(dst_row++), *(src_row++));
            ASSERT_EQ(*(dst_row++), *(src_row++));
            ASSERT_EQ(*(dst_row++), *(src_row++));
            dst_row -= 6;
        }

        dst_data += bgr_u8_dst.stride();
        src_data += pkg_bgr_u8_src.stride();
    }
}
