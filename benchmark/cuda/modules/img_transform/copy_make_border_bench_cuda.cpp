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

#include "benchmark/benchmark.h"
#include "utils/utils.h"
#include "flycv.h"

using namespace g_fcv_ns;

class CudaCopyMakeBorder : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {
        feed_num = state.range(0);
        set_thread_num(G_THREAD_NUM);

        gray_u8_720 = CudaMat(1280, 720, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_720.total_byte_size(), feed_num, gray_u8_720.data());
        gray_u8_720_dst = CudaMat(1280 + left + right, 720 + top + bottom, FCVImageType::GRAY_U8);

        pkg_bgr_u8_720 = CudaMat(1280, 720, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_720.total_byte_size(), feed_num, pkg_bgr_u8_720.data());
        pkg_bgr_u8_720_dst = CudaMat(1280 + left + right, 720 + top + bottom, FCVImageType::PKG_BGR_U8);

        pkg_bgra_u8_720 = CudaMat(1280, 720, FCVImageType::PKG_BGRA_U8);
        construct_data<unsigned char>(pkg_bgra_u8_720.total_byte_size(), feed_num, pkg_bgra_u8_720.data());
        pkg_bgra_u8_720_dst = CudaMat(1280 + left + right, 720 + top + bottom, FCVImageType::PKG_BGRA_U8);

        // 1080
        gray_u8_1080 = CudaMat(1920, 1080, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_1080.total_byte_size(), feed_num, gray_u8_1080.data());
        gray_u8_1080_dst = CudaMat(1920 + left + right, 1080 + top + bottom, FCVImageType::GRAY_U8);

        pkg_bgr_u8_1080 = CudaMat(1920, 1080, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_1080.total_byte_size(), feed_num, pkg_bgr_u8_1080.data());
        pkg_bgr_u8_1080_dst = CudaMat(1920 + left + right, 1080 + top + bottom, FCVImageType::PKG_BGR_U8);

        pkg_bgra_u8_1080 = CudaMat(1920, 1080, FCVImageType::PKG_BGRA_U8);
        construct_data<unsigned char>(pkg_bgra_u8_1080.total_byte_size(), feed_num, pkg_bgra_u8_1080.data());
        pkg_bgra_u8_1080_dst = CudaMat(1920 + left + right, 1080 + top + bottom, FCVImageType::PKG_BGRA_U8);

        // 4K
        gray_u8_4k = CudaMat(4032, 3024, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_4k.total_byte_size(), feed_num, gray_u8_4k.data());
        gray_u8_4k_dst = CudaMat(4032 + left + right, 3024 + top + bottom, FCVImageType::GRAY_U8);

        pkg_bgr_u8_4k = CudaMat(4032, 3024, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_4k.total_byte_size(), feed_num, pkg_bgr_u8_4k.data());
        pkg_bgr_u8_4k_dst = CudaMat(4032 + left + right, 3024 + top + bottom, FCVImageType::PKG_BGR_U8);

        pkg_bgra_u8_4k = CudaMat(4032, 3024, FCVImageType::PKG_BGRA_U8);
        construct_data<unsigned char>(pkg_bgra_u8_4k.total_byte_size(), feed_num, pkg_bgra_u8_4k.data());
        pkg_bgra_u8_4k_dst = CudaMat(4032 + left + right, 3024 + top + bottom, FCVImageType::PKG_BGRA_U8);
    }

    void TearDown(const ::benchmark::State& state) {
        feed_num = state.range(0);

        gray_u8_720.~CudaMat();
        gray_u8_720_dst.~CudaMat();
        pkg_bgra_u8_720.~CudaMat();
        pkg_bgra_u8_720_dst.~CudaMat();
        pkg_bgr_u8_720.~CudaMat();
        pkg_bgr_u8_720_dst.~CudaMat();

        gray_u8_1080.~CudaMat();
        gray_u8_1080_dst.~CudaMat();
        pkg_bgra_u8_1080.~CudaMat();
        pkg_bgra_u8_1080_dst.~CudaMat();
        pkg_bgr_u8_1080.~CudaMat();
        pkg_bgr_u8_1080_dst.~CudaMat();

        gray_u8_4k.~CudaMat();
        gray_u8_4k_dst.~CudaMat();
        pkg_bgra_u8_4k.~CudaMat();
        pkg_bgra_u8_4k_dst.~CudaMat();
        pkg_bgr_u8_4k.~CudaMat();
        pkg_bgr_u8_4k_dst.~CudaMat();
    }

public:
    int feed_num;
    int top = 50;
    int bottom = 50;
    int left = 64;
    int right = 64;

    CudaMat gray_u8_720;
    CudaMat gray_u8_720_dst;
    CudaMat pkg_bgr_u8_720;
    CudaMat pkg_bgr_u8_720_dst;
    CudaMat pkg_bgra_u8_720;
    CudaMat pkg_bgra_u8_720_dst;

    CudaMat gray_u8_1080;
    CudaMat gray_u8_1080_dst;
    CudaMat pkg_bgr_u8_1080;
    CudaMat pkg_bgr_u8_1080_dst;
    CudaMat pkg_bgra_u8_1080;
    CudaMat pkg_bgra_u8_1080_dst;

    CudaMat gray_u8_4k;
    CudaMat gray_u8_4k_dst;
    CudaMat pkg_bgr_u8_4k;
    CudaMat pkg_bgr_u8_4k_dst;
    CudaMat pkg_bgra_u8_4k;
    CudaMat pkg_bgra_u8_4k_dst;
};

BENCHMARK_DEFINE_F(CudaCopyMakeBorder, GRAYU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        copy_make_border(gray_u8_720, gray_u8_720_dst, top, bottom, left, right, BorderType::BORDER_CONSTANT);
    }
}

BENCHMARK_DEFINE_F(CudaCopyMakeBorder, RGBU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        copy_make_border(pkg_bgr_u8_720, pkg_bgr_u8_720_dst, top, bottom, left, right, BorderType::BORDER_CONSTANT);
    }
}

BENCHMARK_DEFINE_F(CudaCopyMakeBorder, RGBAU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        copy_make_border(pkg_bgra_u8_720, pkg_bgra_u8_720_dst, top, bottom, left, right, BorderType::BORDER_CONSTANT);
    }
}

BENCHMARK_REGISTER_F(CudaCopyMakeBorder, GRAYU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaCopyMakeBorder, RGBU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaCopyMakeBorder, RGBAU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

// 1080P
BENCHMARK_DEFINE_F(CudaCopyMakeBorder, GRAYU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        copy_make_border(gray_u8_1080, gray_u8_1080_dst, top, bottom, left, right, BorderType::BORDER_CONSTANT);
    }
}

BENCHMARK_DEFINE_F(CudaCopyMakeBorder, RGBU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        copy_make_border(pkg_bgr_u8_1080, pkg_bgr_u8_1080_dst, top, bottom, left, right, BorderType::BORDER_CONSTANT);
    }
}

BENCHMARK_DEFINE_F(CudaCopyMakeBorder, RGBAU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        copy_make_border(pkg_bgra_u8_1080, pkg_bgra_u8_1080_dst, top, bottom, left, right, BorderType::BORDER_CONSTANT);
    }
}

BENCHMARK_REGISTER_F(CudaCopyMakeBorder, GRAYU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaCopyMakeBorder, RGBU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaCopyMakeBorder, RGBAU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

// 4K
BENCHMARK_DEFINE_F(CudaCopyMakeBorder, GRAYU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        copy_make_border(gray_u8_4k, gray_u8_4k_dst, top, bottom, left, right, BorderType::BORDER_CONSTANT);
    }
}

BENCHMARK_DEFINE_F(CudaCopyMakeBorder, RGBU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        copy_make_border(pkg_bgr_u8_4k, pkg_bgr_u8_4k_dst, top, bottom, left, right, BorderType::BORDER_CONSTANT);
    }
}

BENCHMARK_DEFINE_F(CudaCopyMakeBorder, RGBAU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        copy_make_border(pkg_bgra_u8_4k, pkg_bgra_u8_4k_dst, top, bottom, left, right, BorderType::BORDER_CONSTANT);
    }
}

BENCHMARK_REGISTER_F(CudaCopyMakeBorder, GRAYU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaCopyMakeBorder, RGBU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaCopyMakeBorder, RGBAU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);