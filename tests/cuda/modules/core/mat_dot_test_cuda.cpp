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

class CudaMatDotTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(prepare_gray_u8_720p_cuda(gray_u8_src), 0);
        ASSERT_EQ(prepare_gray_u16_720p_cuda(gray_u16_src), 0);
        ASSERT_EQ(prepare_pkg_bgr_u8_720p_cuda(pkg_bgr_u8_src), 0);
        ASSERT_EQ(prepare_pkg_bgr_f32_720p_cuda(pkg_bgr_f32_src), 0);
    }

public:
    CudaMat gray_u8_src;
    CudaMat gray_u16_src;
    CudaMat pkg_bgr_u8_src;
    CudaMat pkg_bgr_f32_src;
};

TEST_F(CudaMatDotTest, DotPositiveInput) {
    double result0 = gray_u8_src.dot(gray_u8_src);
    EXPECT_DOUBLE_EQ(result0, 23713082282);

    double result1 = gray_u16_src.dot(gray_u16_src);
    EXPECT_DOUBLE_EQ(result1, 23713082282);

    double result2 = pkg_bgr_u8_src.dot(pkg_bgr_u8_src);
    EXPECT_DOUBLE_EQ(result2, 70768231298);

    double result3 = pkg_bgr_f32_src.dot(pkg_bgr_f32_src);
    EXPECT_DOUBLE_EQ(result3, 70768231298);
}
