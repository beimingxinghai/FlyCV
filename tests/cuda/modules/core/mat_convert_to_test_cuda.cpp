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

class CudaMatConvertToTest : public ::testing::Test {
protected:
    void SetUp() override {
        int status = 0;

        pkg_bgr_u8_src = CudaMat(IMG_720P_WIDTH, IMG_720P_HEIGHT, FCVImageType::PKG_BGR_U8);
        status = read_binary_file(BGR_1280X720_U8_BIN, pkg_bgr_u8_src.data(),
                pkg_bgr_u8_src.total_byte_size());
        EXPECT_EQ(status, 0);
    }

public:
    CudaMat pkg_bgr_u8_src;
};

TEST_F(CudaMatConvertToTest, ConvertToPositiveInput) {
    CudaMat dst;
    pkg_bgr_u8_src.convert_to(dst, FCVImageType::PKG_BGR_F32, 0.5, 10);

    float* dst_data = reinterpret_cast<float*>(dst.data());
    std::vector<float> groundtruth = {10.0f, 51.0f, 33.5f,
            52.0f, 55.0f, 62.0f, 87.5f, 89.5f, 137.5f};

    for (size_t i = 0; i < C3_1280X720_IDX.size(); ++i) {
        ASSERT_FLOAT_EQ(groundtruth[i], dst_data[C3_1280X720_IDX[i]]);
    }
}

TEST_F(CudaMatConvertToTest, ConvertToNegativeInput) {
    CudaMat dst_gray_u8;
    int status = pkg_bgr_u8_src.convert_to(dst_gray_u8, FCVImageType::GRAY_U8, 0.5, 10);
    EXPECT_NE(status, 0);

    CudaMat dst_gray_u16;
    status = pkg_bgr_u8_src.convert_to(dst_gray_u16, FCVImageType::GRAY_U16, 0.5, 10);
    EXPECT_NE(status, 0);

    CudaMat dst_gray_s32;
    status = pkg_bgr_u8_src.convert_to(dst_gray_s32, FCVImageType::GRAY_S32, 0.5, 10);
    EXPECT_NE(status, 0);

    CudaMat dst_pkg_rgb;
    status = pkg_bgr_u8_src.convert_to(dst_pkg_rgb, FCVImageType::PKG_RGB_U8, 0.5, 10);
    EXPECT_NE(status, 0);

    CudaMat dst_pla_bgra;
    status = pkg_bgr_u8_src.convert_to(dst_pla_bgra, FCVImageType::PLA_BGRA_U8, 0.5, 10);
    EXPECT_NE(status, 0);
}