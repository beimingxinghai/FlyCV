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

#include "gtest/gtest.h"
#include "flycv.h"
#include "test_util.h"

using namespace g_fcv_ns;

class CudaWarpAffine : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(prepare_pkg_bgr_u8_720p_cuda(pkg_bgr_u8_src), 0);
        ASSERT_EQ(prepare_pkg_bgr_f32_720p_cuda(pkg_bgr_f32_src), 0);
        ASSERT_EQ(prepare_gray_u8_720p_cuda(gray_u8_src), 0);
        ASSERT_EQ(prepare_gray_f32_720p_cuda(gray_f32_src), 0);
    }

    CudaMat pkg_bgr_u8_src;
    CudaMat pkg_bgr_f32_src;
    CudaMat gray_u8_src;
    CudaMat gray_f32_src;
};

TEST_F(CudaWarpAffine, GrayU8PositiveInput) {
    double m[6] = {0.996, -0.08, 0, 0.08, 0.996, 0};
    int status;
    CudaMat gray_u8_dst;
    std::vector<int> groundtruth_gray = {62, 58, 55, 0, 84, 148, 0, 0, 0};

    status = warp_affine(gray_u8_src, gray_u8_dst, m);
    ASSERT_EQ(status, 0);

    unsigned char* data = reinterpret_cast<unsigned char*>(gray_u8_dst.data());

    for (size_t i = 0; i < C1_1280X720_IDX.size(); ++i) {
        ASSERT_NEAR((int)data[C1_1280X720_IDX[i]], groundtruth_gray[i], 1) << "===: " << i;
    }
}

TEST_F(CudaWarpAffine, PkgBGRU8PositiveInput) {
    double m[6] = {0.996, -0.08, 0, 0.08, 0.996, 0};

    int status;
    CudaMat pkg_bgr_u8_dst;
    status = warp_affine(pkg_bgr_u8_src, pkg_bgr_u8_dst, m);
    ASSERT_EQ(status, 0);

    std::vector<int> groundtruth_bgr = {0, 82, 47, 0, 77, 96, 0, 0, 0};
    unsigned char* data = reinterpret_cast<unsigned char*>(pkg_bgr_u8_dst.data());

    for (size_t i = 0; i < C3_1280X720_IDX.size(); ++i) {
        ASSERT_NEAR((int)data[C3_1280X720_IDX[i]], groundtruth_bgr[i], 1) << "===: " << i;
    }
}

TEST_F(CudaWarpAffine, GrayF32PositiveInput) {
    double m[6] = {0.996, -0.08, 0, 0.08, 0.996, 0};

    CudaMat gray_f32_dst;

    int status = warp_affine(gray_f32_src, gray_f32_dst, m);
    ASSERT_EQ(status, 0);

    std::vector<float> groundtruth_gray = {62.0f, 58.0f, 54.84375f,
            0.0f, 84.472656f, 147.957031f, 0.0f, 0.0f, 0.0f};

    float* gray_data = reinterpret_cast<float*>(gray_f32_dst.data());

    for (size_t i = 0; i < C1_1280X720_IDX.size(); ++i) {
        ASSERT_NEAR(gray_data[C1_1280X720_IDX[i]], groundtruth_gray[i], 1) << "===: " << i;
    }
}

TEST_F(CudaWarpAffine, PkgBGRF32PositiveInput) {
    double m[6] = {0.996, -0.08, 0, 0.08, 0.996, 0};

    CudaMat pkg_bgr_f32_dst;

    int status = warp_affine(pkg_bgr_f32_src, pkg_bgr_f32_dst, m);
    ASSERT_EQ(status, 0);

    std::vector<int> index = {0, 3, 6, 100, 1000, 1382400, 1843200};
    std::vector<float> groundtruth_bgr = {0.0f, 82.0f, 47.0f, 0.0f, 76.675781f,
            96.390625f, 0.0f, 0.0f, 0.0f};

    float* bgr_data = reinterpret_cast<float*>(pkg_bgr_f32_dst.data());

    for (size_t i = 0; i < C3_1280X720_IDX.size(); ++i) {
        ASSERT_NEAR(bgr_data[C3_1280X720_IDX[i]], groundtruth_bgr[i], 1) << "===: " << i;
    }
}
