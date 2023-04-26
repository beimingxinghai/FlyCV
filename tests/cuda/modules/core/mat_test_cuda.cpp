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

class CudaMatTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(prepare_gray_u8_720p_cuda(gray_u8_src), 0);
        ASSERT_EQ(prepare_gray_u16_720p_cuda(gray_u16_src), 0);
        ASSERT_EQ(prepare_pkg_bgr_u8_720p_cuda(pkg_bgr_u8_src), 0);
        ASSERT_EQ(prepare_pkg_bgr_f32_720p_cuda(pkg_bgr_f32_src), 0);
        ASSERT_EQ(prepare_nv21_720p_cuda(nv21_src), 0);
        ASSERT_EQ(prepare_i420_720p_cuda(i420_src), 0);
    }

public:
    CudaMat gray_u8_src;
    CudaMat gray_u16_src;
    CudaMat pkg_bgr_u8_src;
    CudaMat pkg_bgr_f32_src;
    CudaMat nv21_src;
    CudaMat i420_src;
};

TEST_F(CudaMatTest, ConstructorPositiveInput) {
    CudaMat gray_u8 = CudaMat(1280, 720, FCVImageType::GRAY_U8);
    EXPECT_EQ(1280, gray_u8.stride());

    CudaMat gray_u16 = CudaMat(1280, 720, FCVImageType::GRAY_U16);
    EXPECT_EQ(1280 * 2, gray_u16.stride());

    CudaMat gray_s16 = CudaMat(1280, 720, FCVImageType::GRAY_S16);
    EXPECT_EQ(1280 * 2, gray_s16.stride());

    CudaMat gray_s32 = CudaMat(1280, 720, FCVImageType::GRAY_S32);
    EXPECT_EQ(1280 * 4, gray_s32.stride());

    CudaMat gray_f64 = CudaMat(1280, 720, FCVImageType::GRAY_F64);
    EXPECT_EQ(1280 * 8, gray_f64.stride());

    CudaMat pkg_bgr_u8 = CudaMat(1280, 720, FCVImageType::PKG_BGR_U8);
    EXPECT_EQ(1280 * 3, pkg_bgr_u8.stride());

    CudaMat pkg_bgr_f32 = CudaMat(1280, 720, FCVImageType::PKG_BGR_F32);
    EXPECT_EQ(1280 * 3 * 4, pkg_bgr_f32.stride());

    CudaMat nv12 = CudaMat(1280, 720, FCVImageType::NV12);
    EXPECT_EQ(1280, nv12.stride());
    EXPECT_EQ(3, nv12.channels());

    CudaMat nv21 = CudaMat(1280, 720, FCVImageType::NV21);
    EXPECT_EQ(1280, nv21.stride());
    EXPECT_EQ(3, nv21.channels());

    CudaMat i420 = CudaMat(1280, 720, FCVImageType::I420);
    EXPECT_EQ(1280, i420.stride());
    EXPECT_EQ(3, i420.channels());

    Size size(1280, 720);
    CudaMat pla_rgb_u8 = CudaMat(size, FCVImageType::PLA_RGB_U8);
    EXPECT_EQ(1280, pla_rgb_u8.width());
    EXPECT_EQ(720, pla_rgb_u8.height());
}