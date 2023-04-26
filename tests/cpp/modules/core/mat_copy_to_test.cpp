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

class MatCopyToTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(prepare_pkg_bgr_u8_720p(pkg_bgr_u8_src), 0);
    }

    Mat pkg_bgr_u8_src;
};

TEST_F(MatCopyToTest, CopyToPositiveInput) {
    Mat dst;

    pkg_bgr_u8_src.copy_to(dst);
    unsigned char* src_data = static_cast<unsigned char*>(pkg_bgr_u8_src.data());
    unsigned char* dst_data = static_cast<unsigned char*>(dst.data());

   for (int i = 0; i < dst.width() * dst.height() * dst.channels(); ++i) {
        ASSERT_EQ(src_data[i], dst_data[i]);
    }
}

TEST_F(MatCopyToTest, CopyToWithMaskPositiveInput) {
    Mat mask(IMG_720P_WIDTH, IMG_720P_HEIGHT, FCVImageType::GRAY_U8);
    for (int y = 0; y < mask.height(); y++) {
        for (int x = 0; x < mask.width(); x++) {
            mask.at<char>(x, y) = 1;
        }
    }

    Mat src_gray_u8(IMG_720P_WIDTH, IMG_720P_HEIGHT, FCVImageType::GRAY_U8);
    unsigned char min = 0;
    unsigned char max = 255;
    // init src Mat
    int num = IMG_720P_WIDTH * IMG_720P_HEIGHT;
    init_random((unsigned char *)src_gray_u8.data(), num * src_gray_u8.channels(), min, max);

    Mat gray_u8_dst;
    Mat pkg_bgr_u8_dst;

    src_gray_u8.copy_to(gray_u8_dst, mask);
    pkg_bgr_u8_src.copy_to(pkg_bgr_u8_dst, mask);

    unsigned char* src_data = static_cast<unsigned char*>(pkg_bgr_u8_src.data());
    unsigned char* dst_data = static_cast<unsigned char*>(pkg_bgr_u8_dst.data());

    for (int i = 0; i < pkg_bgr_u8_dst.width() * pkg_bgr_u8_dst.height() *
            pkg_bgr_u8_dst.channels(); ++i) {
        ASSERT_EQ(src_data[i], dst_data[i]);
    }

    src_data = static_cast<unsigned char*>(gray_u8_dst.data());
    dst_data = static_cast<unsigned char*>(gray_u8_dst.data());

    for (int i = 0; i < gray_u8_dst.width() * gray_u8_dst.height(); ++i) {
        ASSERT_EQ(src_data[i], dst_data[i]);
    }
}