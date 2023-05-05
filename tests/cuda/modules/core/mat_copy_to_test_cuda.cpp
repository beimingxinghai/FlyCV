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

class CudaMatCopyToTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(prepare_gray_u8_720p_cuda(gray_u8_src), 0);
        ASSERT_EQ(prepare_pkg_bgr_u8_720p_cuda(pkg_bgr_u8_src), 0);
        ASSERT_EQ(prepare_pkg_bgr_u8_720p_cuda_batch(pkg_bgr_u8_batch_src, 3), 0);
    }

public:
    CudaMat gray_u8_src;
    CudaMat pkg_bgr_u8_src;
    CudaMat pkg_bgr_u8_batch_src;
};

TEST_F(CudaMatCopyToTest, CopyToSingleBatch_PkgBgrU8) {
    CudaMat dst;
    int status = pkg_bgr_u8_src.copy_to(dst);
    ASSERT_EQ(status, 0);

    unsigned char* src_data = static_cast<unsigned char*>(pkg_bgr_u8_src.data());
    unsigned char* dst_data = static_cast<unsigned char*>(dst.data());

    for (int i = 0; i < dst.width() * dst.height() * dst.channels(); ++i) {
        ASSERT_EQ(src_data[i], dst_data[i]);
    }
}

TEST_F(CudaMatCopyToTest, CopyToMultiBatch_PkgBgrU8) {
    unsigned char* src_data = reinterpret_cast<unsigned char*>(pkg_bgr_u8_batch_src.data());

    CudaMat dst;
    int status = pkg_bgr_u8_batch_src.copy_to(dst);
    ASSERT_EQ(status, 0);

    unsigned char* dst_data = static_cast<unsigned char*>(dst.data());
    ASSERT_EQ(pkg_bgr_u8_batch_src.batch(), dst.batch());

    for (int i = 0; i < dst.width() * dst.height()
            * dst.channels() * dst.batch(); ++i) {
        ASSERT_EQ(src_data[i], dst_data[i]);
    }
}

TEST_F(CudaMatCopyToTest, CopyToWithRectSingleBatch_PkgBgrU8) {
    Rect rect(10, 20, 50, 100);
    CudaMat dst(640, 360, FCVImageType::PKG_BGR_U8);

    unsigned char* dst_data = reinterpret_cast<unsigned char*>(dst.data());

    for (int i = 0; i < dst.total_byte_size(); ++i) {
        dst_data[i] = 1;
    }

    int status = pkg_bgr_u8_src.copy_to(dst, rect);
    ASSERT_EQ(status, 0);

    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 50; ++x) {
            for (int c = 0; c < 3; ++c) {
                ASSERT_EQ(pkg_bgr_u8_src.at<unsigned char>(x, y, c),
                        dst.at<unsigned char>(x + 10, y + 20, c));
            }
        }
    }
}

TEST_F(CudaMatCopyToTest, CopyToWithRectMultiBatch_PkgBgrU8) {
    Rect rect(10, 20, 50, 100);
    CudaMat dst(640, 360, FCVImageType::PKG_BGR_U8, pkg_bgr_u8_batch_src.batch());

    unsigned char* dst_data = reinterpret_cast<unsigned char*>(dst.data());

    for (int i = 0; i < dst.total_byte_size(); ++i) {
        dst_data[i] = 1;
    }

    int status = pkg_bgr_u8_batch_src.copy_to(dst, rect);
    ASSERT_EQ(status, 0);

    for (int i = 0; i < pkg_bgr_u8_batch_src.batch(); ++i) {
        for (int y = 0; y < 100; ++y) {
            for (int x = 0; x < 50; ++x) {
                for (int c = 0; c < 3; ++c) {
                    ASSERT_EQ(pkg_bgr_u8_batch_src.at<unsigned char>(x, y, c, i),
                            dst.at<unsigned char>(x + 10, y + 20, c, i));
                }
            }
        }
    }
}

TEST_F(CudaMatCopyToTest, CopyToWithMaskSingleBatch_GrayU8) {
    CudaMat mask(IMG_720P_WIDTH, IMG_720P_HEIGHT, FCVImageType::GRAY_U8);
    unsigned char* mask_data = reinterpret_cast<unsigned char*>(mask.data());

    for (int i = 0; i < mask.total_byte_size(); ++i) {
        mask_data[i] = i % 2;
    }

    CudaMat dst;
    int status = gray_u8_src.copy_to(dst, mask);
    ASSERT_EQ(status, 0);

    for (int y = 0; y < mask.height(); y++) {
        for (int x = 0; x < mask.width(); x++) {
            if (mask.at<unsigned char>(x, y)) {
                ASSERT_EQ(gray_u8_src.at<unsigned char>(x, y), dst.at<unsigned char>(x, y));
            }
        }
    }
}

TEST_F(CudaMatCopyToTest, CopyToWithMask_PkgBgrU8) {
    CudaMat mask(IMG_720P_WIDTH, IMG_720P_HEIGHT, FCVImageType::GRAY_U8);
    unsigned char* mask_data = reinterpret_cast<unsigned char*>(mask.data());

    for (int i = 0; i < mask.total_byte_size(); ++i) {
        mask_data[i] = i % 2;
    }

    CudaMat dst;
    int status = pkg_bgr_u8_src.copy_to(dst, mask);
    ASSERT_EQ(status, 0);

    for (int y = 0; y < mask.height(); y++) {
        for (int x = 0; x < mask.width(); x++) {
            if (mask.at<unsigned char>(x, y)) {
                for (int c = 0; c < pkg_bgr_u8_src.channels(); ++c) {
                    ASSERT_EQ(pkg_bgr_u8_src.at<unsigned char>(x, y, c), dst.at<unsigned char>(x, y, c));
                }
            }
        }
    }
}
