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
        int status = 0;

        gray_u8_src = CudaMat(IMG_720P_WIDTH, IMG_720P_HEIGHT, FCVImageType::GRAY_U8);
        status = read_binary_file(GRAY_1280X720_U8_BIN, gray_u8_src.data(),
                gray_u8_src.total_byte_size());
        EXPECT_EQ(status, 0);

        gray_u16_src = CudaMat(IMG_720P_WIDTH, IMG_720P_HEIGHT, FCVImageType::GRAY_U16);
        status = read_binary_file(GRAY_1280X720_U16_BIN, gray_u16_src.data(),
                gray_u16_src.total_byte_size());
        EXPECT_EQ(status, 0);

        pkg_bgr_u8_src = CudaMat(IMG_720P_WIDTH, IMG_720P_HEIGHT, FCVImageType::PKG_BGR_U8);
        status = read_binary_file(BGR_1280X720_U8_BIN, pkg_bgr_u8_src.data(),
                pkg_bgr_u8_src.total_byte_size());
        EXPECT_EQ(status, 0);

        pkg_bgr_f32_src = CudaMat(IMG_720P_WIDTH, IMG_720P_HEIGHT, FCVImageType::PKG_BGR_F32);
        status = read_binary_file(BGR_1280X720_F32_BIN, pkg_bgr_f32_src.data(),
                pkg_bgr_f32_src.total_byte_size());
        EXPECT_EQ(status, 0);

        nv21_src = CudaMat(IMG_720P_WIDTH, IMG_720P_HEIGHT, FCVImageType::NV21);
        status = read_binary_file(NV21_1280X720_U8_BIN, nv21_src.data(),
                nv21_src.total_byte_size());
        EXPECT_EQ(status, 0);

        i420_src= CudaMat(IMG_720P_WIDTH, IMG_720P_HEIGHT, FCVImageType::I420);
        status = read_binary_file(I420_1280X720_U8_BIN, (char*)i420_src.data(),
                i420_src.total_byte_size());
        EXPECT_EQ(status, 0);
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

TEST_F(CudaMatTest, AtPositiveInput) {
    CudaMat gray_u8(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::GRAY_U8);
    for (int y = 0; y < gray_u8.height(); ++y) {
        for (int x = 0; x < gray_u8.width(); ++x) {
            gray_u8.at<unsigned char>(x, y) = y % 256;
            ASSERT_EQ((int)gray_u8.at<unsigned char>(x, y), y % 256);
        }
    }

    CudaMat gray_u16(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::GRAY_U16);
    for (int y = 0; y < gray_u16.height(); ++y) {
        for (int x = 0; x < gray_u16.width(); ++x) {
            gray_u16.at<unsigned short>(x, y) = y % 256;
            ASSERT_EQ((int)gray_u16.at<unsigned short>(x, y), y % 256);
        }
    }

    CudaMat gray_s16(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::GRAY_S16);
    for (int y = 0; y < gray_s16.height(); ++y) {
        for (int x = 0; x < gray_s16.width(); ++x) {
            gray_s16.at<signed short>(x, y) = y % 256;
            ASSERT_EQ((int)gray_s16.at<signed short>(x, y), y % 256);
        }
    }

    CudaMat gray_s32(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::GRAY_S32);
    for (int y = 0; y < gray_s32.height(); ++y) {
        for (int x = 0; x < gray_s32.width(); ++x) {
            gray_s32.at<int>(x, y) = y % 256;
            ASSERT_EQ((int)gray_s32.at<int>(x, y), y % 256);
        }
    }

    CudaMat gray_f32(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::GRAY_F32);
    for (int y = 0; y < gray_f32.height(); ++y) {
        for (int x = 0; x < gray_f32.width(); ++x) {
            gray_f32.at<float>(x, y) = y % 256;
            ASSERT_FLOAT_EQ(gray_f32.at<float>(x, y), y % 256);
        }
    }

    CudaMat gray_f64(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::GRAY_F64);
    for (int y = 0; y < gray_f64.height(); ++y) {
        for (int x = 0; x < gray_f64.width(); ++x) {
            gray_f64.at<double>(x, y) = y % 256;
            ASSERT_DOUBLE_EQ((double)gray_f64.at<double>(x, y), y % 256);
        }
    }

    CudaMat pla_bgr_u8(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PLA_BGR_U8);
    for (int y = 0; y < pla_bgr_u8.height(); ++y) {
        for (int x = 0; x < pla_bgr_u8.width(); ++x) {
            for (int c = 0; c < pla_bgr_u8.channels(); ++c) {
                pla_bgr_u8.at<unsigned char>(x, y, c) = y % 256;
                ASSERT_EQ((int)pla_bgr_u8.at<unsigned char>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pla_rgb_u8(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PLA_RGB_U8);
    for (int y = 0; y < pla_rgb_u8.height(); ++y) {
        for (int x = 0; x < pla_rgb_u8.width(); ++x) {
            for (int c = 0; c < pla_rgb_u8.channels(); ++c) {
                pla_rgb_u8.at<unsigned char>(x, y, c) = y % 256;
                ASSERT_EQ((int)pla_rgb_u8.at<unsigned char>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_bgr_u8(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_BGR_U8);
    for (int y = 0; y < pkg_bgr_u8.height(); ++y) {
        for (int x = 0; x < pkg_bgr_u8.width(); ++x) {
            for (int c = 0; c < pkg_bgr_u8.channels(); ++c) {
                pkg_bgr_u8.at<unsigned char>(x, y, c) = y % 256;
                ASSERT_EQ((int)pkg_bgr_u8.at<unsigned char>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_rgb_u8(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_RGB_U8);
    for (int y = 0; y < pkg_rgb_u8.height(); ++y) {
        for (int x = 0; x < pkg_rgb_u8.width(); ++x) {
            for (int c = 0; c < pkg_rgb_u8.channels(); ++c) {
                pkg_rgb_u8.at<unsigned char>(x, y, c) = y % 256;
                ASSERT_EQ((int)pkg_rgb_u8.at<unsigned char>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pla_bgra_u8(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PLA_BGRA_U8);
    for (int y = 0; y < pla_bgra_u8.height(); ++y) {
        for (int x = 0; x < pla_bgra_u8.width(); ++x) {
            for (int c = 0; c < pla_bgra_u8.channels(); ++c) {
                pla_bgra_u8.at<unsigned char>(x, y, c) = y % 256;
                ASSERT_EQ((int)pla_bgra_u8.at<unsigned char>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pla_rgba_u8(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PLA_RGBA_U8);
    for (int y = 0; y < pla_rgba_u8.height(); ++y) {
        for (int x = 0; x < pla_rgba_u8.width(); ++x) {
            for (int c = 0; c < pla_rgba_u8.channels(); ++c) {
                pla_rgba_u8.at<unsigned char>(x, y, c) = y % 256;
                ASSERT_EQ((int)pla_rgba_u8.at<unsigned char>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_bgra_u8(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_BGRA_U8);
    for (int y = 0; y < pkg_bgra_u8.height(); ++y) {
        for (int x = 0; x < pkg_bgra_u8.width(); ++x) {
            for (int c = 0; c < pkg_bgra_u8.channels(); ++c) {
                pkg_bgra_u8.at<unsigned char>(x, y, c) = y % 256;
                ASSERT_EQ((int)pkg_bgra_u8.at<unsigned char>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_rgba_u8(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_RGBA_U8);
    for (int y = 0; y < pkg_rgba_u8.height(); ++y) {
        for (int x = 0; x < pkg_rgba_u8.width(); ++x) {
            for (int c = 0; c < pkg_rgba_u8.channels(); ++c) {
                pkg_rgba_u8.at<unsigned char>(x, y, c) = y % 256;
                ASSERT_EQ((int)pkg_rgba_u8.at<unsigned char>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pla_bgr_f32(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PLA_BGR_F32);
    for (int y = 0; y < pla_bgr_f32.height(); ++y) {
        for (int x = 0; x < pla_bgr_f32.width(); ++x) {
            for (int c = 0; c < pla_bgr_f32.channels(); ++c) {
                pla_bgr_f32.at<float>(x, y, c) = y % 256;
                ASSERT_EQ((int)pla_bgr_f32.at<float>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pla_rgb_f32(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PLA_RGB_F32);
    for (int y = 0; y < pla_rgb_f32.height(); ++y) {
        for (int x = 0; x < pla_rgb_f32.width(); ++x) {
            for (int c = 0; c < pla_rgb_f32.channels(); ++c) {
                pla_rgb_f32.at<float>(x, y, c) = y % 256;
                ASSERT_FLOAT_EQ(pla_rgb_f32.at<float>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_bgr_f32(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_BGR_F32);
    for (int y = 0; y < pkg_bgr_f32.height(); ++y) {
        for (int x = 0; x < pkg_bgr_f32.width(); ++x) {
            for (int c = 0; c < pkg_bgr_f32.channels(); ++c) {
                pkg_bgr_f32.at<float>(x, y, c) = y % 256;
                ASSERT_FLOAT_EQ(pkg_bgr_f32.at<float>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_rgb_f32(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_RGB_F32);
    for (int y = 0; y < pkg_rgb_f32.height(); ++y) {
        for (int x = 0; x < pkg_rgb_f32.width(); ++x) {
            for (int c = 0; c < pkg_rgb_f32.channels(); ++c) {
                pkg_rgb_f32.at<float>(x, y, c) = y % 256;
                ASSERT_FLOAT_EQ(pkg_rgb_f32.at<float>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pla_bgra_f32(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PLA_BGRA_F32);
    for (int y = 0; y < pla_bgra_f32.height(); ++y) {
        for (int x = 0; x < pla_bgra_f32.width(); ++x) {
            for (int c = 0; c < pla_bgra_f32.channels(); ++c) {
                pla_bgra_f32.at<float>(x, y, c) = y % 256;
                ASSERT_FLOAT_EQ(pla_bgra_f32.at<float>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pla_rgba_f32(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PLA_RGBA_F32);
    for (int y = 0; y < pla_rgba_f32.height(); ++y) {
        for (int x = 0; x < pla_rgba_f32.width(); ++x) {
            for (int c = 0; c < pla_rgba_f32.channels(); ++c) {
                pla_rgba_f32.at<float>(x, y, c) = y % 256;
                ASSERT_EQ((int)pla_rgba_f32.at<float>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_bgra_f32(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_BGRA_F32);
    for (int y = 0; y < pkg_bgra_f32.height(); ++y) {
        for (int x = 0; x < pkg_bgra_f32.width(); ++x) {
            for (int c = 0; c < pkg_bgra_f32.channels(); ++c) {
                pkg_bgra_f32.at<float>(x, y, c) = y % 256;
                ASSERT_FLOAT_EQ(pkg_bgra_f32.at<float>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_rgba_f32(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_RGBA_F32);
    for (int y = 0; y < pkg_rgba_f32.height(); ++y) {
        for (int x = 0; x < pkg_rgba_f32.width(); ++x) {
            for (int c = 0; c < pkg_rgba_f32.channels(); ++c) {
                pkg_rgba_f32.at<float>(x, y, c) = y % 256;
                ASSERT_FLOAT_EQ(pkg_rgba_f32.at<float>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_bgr_f64(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_BGR_F64);
    for (int y = 0; y < pkg_bgr_f64.height(); ++y) {
        for (int x = 0; x < pkg_bgr_f64.width(); ++x) {
            for (int c = 0; c < pkg_bgr_f64.channels(); ++c) {
                pkg_bgr_f64.at<double>(x, y, c) = y % 256;
                ASSERT_DOUBLE_EQ(pkg_bgr_f64.at<double>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_rgb_f64(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_RGB_F64);
    for (int y = 0; y < pkg_rgb_f64.height(); ++y) {
        for (int x = 0; x < pkg_rgb_f64.width(); ++x) {
            for (int c = 0; c < pkg_rgb_f64.channels(); ++c) {
                pkg_rgb_f64.at<double>(x, y, c) = y % 256;
                ASSERT_DOUBLE_EQ(pkg_rgb_f64.at<double>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_bgra_f64(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_BGRA_F64);
    for (int y = 0; y < pkg_bgra_f64.height(); ++y) {
        for (int x = 0; x < pkg_bgra_f64.width(); ++x) {
            for (int c = 0; c < pkg_bgra_f64.channels(); ++c) {
                pkg_bgra_f64.at<double>(x, y, c) = y % 256;
                ASSERT_DOUBLE_EQ(pkg_bgra_f64.at<double>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_rgba_f64(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_RGBA_F64);
    for (int y = 0; y < pkg_rgba_f64.height(); ++y) {
        for (int x = 0; x < pkg_rgba_f64.width(); ++x) {
            for (int c = 0; c < pkg_rgba_f64.channels(); ++c) {
                pkg_rgba_f64.at<double>(x, y, c) = y % 256;
                ASSERT_DOUBLE_EQ(pkg_rgba_f64.at<double>(x, y, c), y % 256);
            }
        }
    }

    CudaMat pkg_bgr565_u8(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_BGR565_U8);
    for (int y = 0; y < pkg_bgr565_u8.height(); ++y) {
        for (int x = 0; x < pkg_bgr565_u8.width(); ++x) {
            for (int c = 0; c < pkg_bgr565_u8.channels(); ++c) {
                pkg_bgr565_u8.at<unsigned char>(x, y) = y % 256;
                ASSERT_EQ((int)pkg_bgr565_u8.at<unsigned char>(x, y), y % 256);
            }
        }
    }

    CudaMat pkg_rgb565_u8(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::PKG_RGB565_U8);
    for (int y = 0; y < pkg_rgb565_u8.height(); ++y) {
        for (int x = 0; x < pkg_rgb565_u8.width(); ++x) {
            for (int c = 0; c < pkg_rgb565_u8.channels(); ++c) {
                pkg_rgb565_u8.at<unsigned char>(x, y) = y % 256;
                ASSERT_EQ((int)pkg_rgb565_u8.at<unsigned char>(x, y), y % 256);
            }
        }
    }

    CudaMat nv12(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::NV12);
    for (int y = 0; y < nv12.height(); ++y) {
        for (int x = 0; x < nv12.width(); ++x) {
            for (int c = 0; c < nv12.channels(); ++c) {
                nv12.at<unsigned char>(x, y) = y % 256;
                ASSERT_EQ((int)nv12.at<unsigned char>(x, y), y % 256);
            }
        }
    }

    CudaMat nv21(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::NV21);
    for (int y = 0; y < nv21.height(); ++y) {
        for (int x = 0; x < nv21.width(); ++x) {
            for (int c = 0; c < nv21.channels(); ++c) {
                nv21.at<unsigned char>(x, y) = y % 256;
                ASSERT_EQ((int)nv21.at<unsigned char>(x, y), y % 256);
            }
        }
    }

    CudaMat i420(IMG_480P_WIDTH, IMG_480P_HEIGHT, FCVImageType::I420);
    for (int y = 0; y < i420.height(); ++y) {
        for (int x = 0; x < i420.width(); ++x) {
            for (int c = 0; c < i420.channels(); ++c) {
                i420.at<unsigned char>(x, y) = y % 256;
                ASSERT_EQ((int)i420.at<unsigned char>(x, y), y % 256);
            }
        }
    }
}

TEST_F(CudaMatTest, AtNV21PositiveInput) {
    CudaMat dst(nv21_src.width(), nv21_src.height(), FCVImageType::NV21);

    for (int y = 0; y < dst.height(); y++) {
        for (int x = 0; x < dst.width(); x++) {
            for (int c = 0; c < dst.channels(); c++) {
                dst.at<char>(x, y, c) = nv21_src.at<char>(x, y, c);
            }
        }
    }

    unsigned char* src_data = static_cast<unsigned char*>(nv21_src.data());
    unsigned char* dst_data = static_cast<unsigned char*>(dst.data());

    for (int i = 0; i < dst.width() * dst.height() * dst.channels() / 2; ++i) {
        ASSERT_EQ(src_data[i], dst_data[i]);
    }
}

TEST_F(CudaMatTest, AtI420PositiveInput) {
    CudaMat dst(i420_src.width(), i420_src.height(), FCVImageType::I420);

    for (int y = 0; y < dst.height(); y++) {
        for (int x = 0; x < dst.width(); x++) {
            for (int c = 0; c < dst.channels(); c++) {
                dst.at<char>(x, y, c) = i420_src.at<char>(x, y, c);
            }
        }
    }

    unsigned char* src_data = static_cast<unsigned char*>(i420_src.data());
    unsigned char* dst_data = static_cast<unsigned char*>(dst.data());

    for (int i = 0; i < dst.width() * dst.height() * dst.channels() / 2; ++i) {
        ASSERT_EQ(src_data[i], dst_data[i]);
    }
}

TEST_F(CudaMatTest, DotPositiveInput) {
    double result0 = gray_u8_src.dot(gray_u8_src);
    EXPECT_DOUBLE_EQ(result0, 23713082282);

    double result1 = gray_u16_src.dot(gray_u16_src);
    EXPECT_DOUBLE_EQ(result1, 23713082282);

    double result2 = pkg_bgr_u8_src.dot(pkg_bgr_u8_src);
    EXPECT_DOUBLE_EQ(result2, 70768231298);

    double result3 = pkg_bgr_f32_src.dot(pkg_bgr_f32_src);
    EXPECT_DOUBLE_EQ(result3, 70768231298);
}
