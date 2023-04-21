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

class Y420SPToResizeToBGRTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(prepare_nv12_720p(nv12_src), 0);
        ASSERT_EQ(prepare_nv21_720p(nv21_src), 0);
    }

    Mat nv12_src;
    Mat nv21_src;
};

TEST_F(Y420SPToResizeToBGRTest, NV12ToResizeToBGRLinearPositiveInput) {
    Mat dst(640, 360, FCVImageType::PKG_BGR_U8);
    int status = nv12_to_resize_to_bgr(nv12_src, dst);
    EXPECT_EQ(status, 0);

    unsigned char* dst_data = reinterpret_cast<unsigned char*>(dst.data());
    std::vector<int> groundtruth = {3, 86, 51, 85, 114, 125, 184, 158, 254};

    for (size_t i = 0; i < C3_640X360_IDX.size(); ++i) {
        EXPECT_NEAR(groundtruth[i], (int)dst_data[C3_640X360_IDX[i]], 1);
    }
}

TEST_F(Y420SPToResizeToBGRTest, NV21ToResizeToBGRLinearPositiveInput) {
    Mat dst;
    int status = nv21_to_resize_to_bgr(nv21_src, dst, Size(640, 360));
    EXPECT_EQ(status, 0);

    unsigned char* dst_data = reinterpret_cast<unsigned char*>(dst.data());
    std::vector<int> groundtruth = {3, 86, 51, 85, 114, 125, 184, 158, 254};

    for (size_t i = 0; i < C3_640X360_IDX.size(); ++i) {
        EXPECT_NEAR(groundtruth[i], (int)dst_data[C3_640X360_IDX[i]], 1);
    }
}

TEST_F(Y420SPToResizeToBGRTest, NV12ToResizeToBGRNearestDn2xPositiveInput) {
    Mat dst(640, 360, FCVImageType::PKG_BGR_U8);
    int status = nv12_to_resize_to_bgr(nv12_src, dst, Size(640, 360), InterpolationType::INTER_NEAREST);
    EXPECT_EQ(status, 0);

    unsigned char* dst_data = reinterpret_cast<unsigned char*>(dst.data());
    std::vector<int> groundtruth = {0, 92, 53, 85, 156, 149, 184, 160, 255};

    for (size_t i = 0; i < C3_640X360_IDX.size(); ++i) {
        EXPECT_NEAR(groundtruth[i], (int)dst_data[C3_640X360_IDX[i]], 1);
    }
}

TEST_F(Y420SPToResizeToBGRTest, NV21ToResizeToBGRNearestDn2xPositiveInput) {
    Mat dst;
    int status = nv21_to_resize_to_bgr(nv21_src, dst, Size(640, 360), InterpolationType::INTER_NEAREST);
    EXPECT_EQ(status, 0);

    unsigned char* dst_data = reinterpret_cast<unsigned char*>(dst.data());
    std::vector<int> groundtruth = {0, 92, 53, 85, 156, 149, 184, 160, 255};

    for (size_t i = 0; i < C3_640X360_IDX.size(); ++i) {
        // std::cout << " " << (int)dst_data[C3_640X360_IDX[i]] << std::endl;
        EXPECT_NEAR(groundtruth[i], (int)dst_data[C3_640X360_IDX[i]], 1);
    }
}

TEST_F(Y420SPToResizeToBGRTest, NV12ToResizeToBGRNearestGenericPositiveInput) {
    Mat dst(320, 180, FCVImageType::PKG_BGR_U8);
    int status = nv12_to_resize_to_bgr(nv12_src, dst, Size(320, 180), InterpolationType::INTER_NEAREST);
    EXPECT_EQ(status, 0);

    unsigned char* dst_data = reinterpret_cast<unsigned char*>(dst.data());
    std::vector<int> groundtruth = {0, 107, 58, 84, 126, 148, 183, 158, 255};

    for (size_t i = 0; i < C3_640X360_IDX.size(); ++i) {
        EXPECT_NEAR(groundtruth[i], (int)dst_data[C3_320X180_IDX[i]], 1);
    }
}

TEST_F(Y420SPToResizeToBGRTest, NV21ToResizeToBGRNearestGenericPositiveInput) {
    Mat dst;
    int status = nv21_to_resize_to_bgr(nv21_src, dst, Size(320, 180), InterpolationType::INTER_NEAREST);
    EXPECT_EQ(status, 0);

    unsigned char* dst_data = reinterpret_cast<unsigned char*>(dst.data());
    std::vector<int> groundtruth = {0, 107, 58, 84, 126, 148, 183, 158, 255};

    for (size_t i = 0; i < C3_640X360_IDX.size(); ++i) {
        // std::cout << " " << (int)dst_data[C3_320X180_IDX[i]] << std::endl;
        EXPECT_NEAR(groundtruth[i], (int)dst_data[C3_320X180_IDX[i]], 1);
    }
}
