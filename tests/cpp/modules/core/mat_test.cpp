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

class MatTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(prepare_gray_u8_720p(gray_u8_src), 0);
        ASSERT_EQ(prepare_gray_u16_720p(gray_u16_src), 0);
        ASSERT_EQ(prepare_pkg_bgr_u8_720p(pkg_bgr_u8_src), 0);
        ASSERT_EQ(prepare_pkg_bgr_f32_720p(pkg_bgr_f32_src), 0);
        ASSERT_EQ(prepare_nv21_720p(nv21_src), 0);
        ASSERT_EQ(prepare_i420_720p(i420_src), 0);
    }

    Mat gray_u8_src;
    Mat gray_u16_src;
    Mat pkg_bgr_u8_src;
    Mat pkg_bgr_f32_src;
    Mat nv21_src;
    Mat i420_src;
};

TEST_F(MatTest, ConstructorPositiveInput) {
    Mat gray_u8 = Mat(1280, 720, FCVImageType::GRAY_U8);
    EXPECT_EQ(1280, gray_u8.stride());

    Mat gray_u16 = Mat(1280, 720, FCVImageType::GRAY_U16);
    EXPECT_EQ(1280 * 2, gray_u16.stride());

    Mat gray_s16 = Mat(1280, 720, FCVImageType::GRAY_S16);
    EXPECT_EQ(1280 * 2, gray_s16.stride());

    Mat gray_s32 = Mat(1280, 720, FCVImageType::GRAY_S32);
    EXPECT_EQ(1280 * 4, gray_s32.stride());

    Mat gray_f64 = Mat(1280, 720, FCVImageType::GRAY_F64);
    EXPECT_EQ(1280 * 8, gray_f64.stride());

    Mat pkg_bgr_u8 = Mat(1280, 720, FCVImageType::PKG_BGR_U8);
    EXPECT_EQ(1280 * 3, pkg_bgr_u8.stride());

    Mat pkg_bgr_f32 = Mat(1280, 720, FCVImageType::PKG_BGR_F32);
    EXPECT_EQ(1280 * 3 * 4, pkg_bgr_f32.stride());

    Mat nv12 = Mat(1280, 720, FCVImageType::NV12);
    EXPECT_EQ(1280, nv12.stride());
    EXPECT_EQ(3, nv12.channels());

    Mat nv21 = Mat(1280, 720, FCVImageType::NV21);
    EXPECT_EQ(1280, nv21.stride());
    EXPECT_EQ(3, nv21.channels());

    Mat i420 = Mat(1280, 720, FCVImageType::I420);
    EXPECT_EQ(1280, i420.stride());
    EXPECT_EQ(3, i420.channels());

    Size size(1280, 720);
    Mat pla_rgb_u8 = Mat(size, FCVImageType::PLA_RGB_U8);
    EXPECT_EQ(1280, pla_rgb_u8.width());
    EXPECT_EQ(720, pla_rgb_u8.height());
}