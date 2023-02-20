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
#include "common/utils.h"
#include "flycv.h"

using namespace g_fcv_ns;
class CudaAddWeightedBench : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {
        feed_num = state.range(0);
        set_thread_num(G_THREAD_NUM);

        // 720P
        gray_u8_720_src0 = CudaMat(1280, 720, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_720_src0.total_byte_size(), feed_num, gray_u8_720_src0.data());
        gray_u8_720_src1 = CudaMat(1280, 720, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_720_src1.total_byte_size(), feed_num, gray_u8_720_src1.data());
        gray_u8_720_dst = CudaMat(1280, 720, FCVImageType::GRAY_U8);

        pkg_bgr_u8_720_src0 = CudaMat(1280, 720, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_720_src0.total_byte_size(), feed_num, pkg_bgr_u8_720_src0.data());
        pkg_bgr_u8_720_src1 = CudaMat(1280, 720, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_720_src1.total_byte_size(), feed_num, pkg_bgr_u8_720_src1.data());
        pkg_bgr_u8_720_dst = CudaMat(1280, 720, FCVImageType::PKG_BGR_U8);

        pkg_bgra_u8_720_src0 = CudaMat(1280, 720, FCVImageType::PKG_BGRA_U8);
        construct_data<unsigned char>(pkg_bgra_u8_720_src0.total_byte_size(), feed_num, pkg_bgra_u8_720_src0.data());
        pkg_bgra_u8_720_src1 = CudaMat(1280, 720, FCVImageType::PKG_BGRA_U8);
        construct_data<unsigned char>(pkg_bgra_u8_720_src1.total_byte_size(), feed_num, pkg_bgra_u8_720_src1.data());
        pkg_bgra_u8_720_dst = CudaMat(1280, 720, FCVImageType::PKG_BGRA_U8);

        // 1080P
        gray_u8_1080_src0 = CudaMat(1920, 1080, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_1080_src0.total_byte_size(), feed_num, gray_u8_1080_src0.data());
        gray_u8_1080_src1 = CudaMat(1920, 1080, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_1080_src1.total_byte_size(), feed_num, gray_u8_1080_src1.data());
        gray_u8_1080_dst = CudaMat(1920, 1080, FCVImageType::GRAY_U8);

        pkg_bgr_u8_1080_src0 = CudaMat(1920, 1080, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_1080_src0.total_byte_size(), feed_num, pkg_bgr_u8_1080_src0.data());
        pkg_bgr_u8_1080_src1 = CudaMat(1920, 1080, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_1080_src1.total_byte_size(), feed_num, pkg_bgr_u8_1080_src1.data());
        pkg_bgr_u8_1080_dst = CudaMat(1920, 1080, FCVImageType::PKG_BGR_U8);

        pkg_bgra_u8_1080_src0 = CudaMat(1920, 1080, FCVImageType::PKG_BGRA_U8);
        construct_data<unsigned char>(pkg_bgra_u8_1080_src0.total_byte_size(), feed_num, pkg_bgra_u8_1080_src0.data());
        pkg_bgra_u8_1080_src1 = CudaMat(1920, 1080, FCVImageType::PKG_BGRA_U8);
        construct_data<unsigned char>(pkg_bgra_u8_1080_src1.total_byte_size(), feed_num, pkg_bgra_u8_1080_src1.data());
        pkg_bgra_u8_1080_dst = CudaMat(1920, 1080, FCVImageType::PKG_BGRA_U8);

        // 4k
        gray_u8_4k_src0 = CudaMat(4032, 3024, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_4k_src0.total_byte_size(), feed_num, gray_u8_4k_src0.data());
        gray_u8_4k_src1 = CudaMat(4032, 3024, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_4k_src1.total_byte_size(), feed_num, gray_u8_4k_src1.data());
        gray_u8_4k_dst = CudaMat(4032, 3024, FCVImageType::GRAY_U8);

        pkg_bgr_u8_4k_src0 = CudaMat(4032, 3024, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_4k_src0.total_byte_size(), feed_num, pkg_bgr_u8_4k_src0.data());
        pkg_bgr_u8_4k_src1 = CudaMat(4032, 3024, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_4k_src1.total_byte_size(), feed_num, pkg_bgr_u8_4k_src1.data());
        pkg_bgr_u8_4k_dst = CudaMat(4032, 3024, FCVImageType::PKG_BGR_U8);

        pkg_bgra_u8_4k_src0 = CudaMat(4032, 3024, FCVImageType::PKG_BGRA_U8);
        construct_data<unsigned char>(pkg_bgra_u8_4k_src0.total_byte_size(), feed_num, pkg_bgra_u8_4k_src0.data());
        pkg_bgra_u8_4k_src1 = CudaMat(4032, 3024, FCVImageType::PKG_BGRA_U8);
        construct_data<unsigned char>(pkg_bgra_u8_4k_src1.total_byte_size(), feed_num, pkg_bgra_u8_4k_src1.data());
        pkg_bgra_u8_4k_dst = CudaMat(4032, 3024, FCVImageType::PKG_BGRA_U8);
    }

    void TearDown(const ::benchmark::State& state) {
        feed_num = state.range(0);
        gray_u8_720_src0.~CudaMat();
        gray_u8_720_src1.~CudaMat();
        gray_u8_720_dst.~CudaMat();
        pkg_bgr_u8_720_src0.~CudaMat();
        pkg_bgr_u8_720_src1.~CudaMat();
        pkg_bgr_u8_720_dst.~CudaMat();
        pkg_bgra_u8_720_src0.~CudaMat();
        pkg_bgra_u8_720_src1.~CudaMat();
        pkg_bgra_u8_720_dst.~CudaMat();

        gray_u8_1080_src0.~CudaMat();
        gray_u8_1080_src1.~CudaMat();
        gray_u8_1080_dst.~CudaMat();
        pkg_bgr_u8_1080_src0.~CudaMat();
        pkg_bgr_u8_1080_src1.~CudaMat();
        pkg_bgr_u8_1080_dst.~CudaMat();
        pkg_bgra_u8_1080_src0.~CudaMat();
        pkg_bgra_u8_1080_src1.~CudaMat();
        pkg_bgra_u8_1080_dst.~CudaMat();

        gray_u8_4k_src0.~CudaMat();
        gray_u8_4k_src1.~CudaMat();
        gray_u8_4k_dst.~CudaMat();
        pkg_bgr_u8_4k_src0.~CudaMat();
        pkg_bgr_u8_4k_src1.~CudaMat();
        pkg_bgr_u8_4k_dst.~CudaMat();
        pkg_bgra_u8_4k_src0.~CudaMat();
        pkg_bgra_u8_4k_src1.~CudaMat();
        pkg_bgra_u8_4k_dst.~CudaMat();
    }

public:
    int feed_num;
    double alpha = 0.333;
    double beta = 0.555;
    double gamma = 0.666;
    CudaMat gray_u8_720_src0;
    CudaMat gray_u8_720_src1;
    CudaMat gray_u8_720_dst;
    CudaMat pkg_bgr_u8_720_src0;
    CudaMat pkg_bgr_u8_720_src1;
    CudaMat pkg_bgr_u8_720_dst;
    CudaMat pkg_bgra_u8_720_src0;
    CudaMat pkg_bgra_u8_720_src1;
    CudaMat pkg_bgra_u8_720_dst;

    CudaMat gray_u8_1080_src0;
    CudaMat gray_u8_1080_src1;
    CudaMat gray_u8_1080_dst;
    CudaMat pkg_bgr_u8_1080_src0;
    CudaMat pkg_bgr_u8_1080_src1;
    CudaMat pkg_bgr_u8_1080_dst;
    CudaMat pkg_bgra_u8_1080_src0;
    CudaMat pkg_bgra_u8_1080_src1;
    CudaMat pkg_bgra_u8_1080_dst;

    CudaMat gray_u8_4k_src0;
    CudaMat gray_u8_4k_src1;
    CudaMat gray_u8_4k_dst;
    CudaMat pkg_bgr_u8_4k_src0;
    CudaMat pkg_bgr_u8_4k_src1;
    CudaMat pkg_bgr_u8_4k_dst;
    CudaMat pkg_bgra_u8_4k_src0;
    CudaMat pkg_bgra_u8_4k_src1;
    CudaMat pkg_bgra_u8_4k_dst;
};

BENCHMARK_DEFINE_F(CudaAddWeightedBench, GRAYU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        add_weighted(gray_u8_720_src0, alpha, gray_u8_720_src1, beta, gamma, gray_u8_720_dst);
    }
}

BENCHMARK_DEFINE_F(CudaAddWeightedBench, RGBU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        add_weighted(pkg_bgr_u8_720_src0, alpha, pkg_bgr_u8_720_src1, beta, gamma, pkg_bgr_u8_720_dst);
    }
}

BENCHMARK_DEFINE_F(CudaAddWeightedBench, RGBAU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        add_weighted(pkg_bgra_u8_720_src0, alpha, pkg_bgra_u8_720_src1, beta, gamma, pkg_bgra_u8_720_dst);
    }
}

// 1080
BENCHMARK_DEFINE_F(CudaAddWeightedBench, GRAYU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        add_weighted(gray_u8_1080_src0, alpha, gray_u8_1080_src1, beta, gamma, gray_u8_1080_dst);
    }
}

BENCHMARK_DEFINE_F(CudaAddWeightedBench, RGBU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        add_weighted(pkg_bgr_u8_1080_src0, alpha, pkg_bgr_u8_1080_src1, beta, gamma, pkg_bgr_u8_1080_dst);
    }
}

BENCHMARK_DEFINE_F(CudaAddWeightedBench, RGBAU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        add_weighted(pkg_bgra_u8_1080_src0, alpha, pkg_bgra_u8_1080_src1, beta, gamma, pkg_bgra_u8_1080_dst);
    }
}

// 4k
BENCHMARK_DEFINE_F(CudaAddWeightedBench, GRAYU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        add_weighted(gray_u8_4k_src0, alpha, gray_u8_4k_src1, beta, gamma, gray_u8_4k_dst);
    }
}

BENCHMARK_DEFINE_F(CudaAddWeightedBench, RGBU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        add_weighted(pkg_bgr_u8_4k_src0, alpha, pkg_bgr_u8_4k_src1, beta, gamma, pkg_bgr_u8_4k_dst);
    }
}

BENCHMARK_DEFINE_F(CudaAddWeightedBench, RGBAU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        add_weighted(pkg_bgra_u8_4k_src0, alpha, pkg_bgra_u8_4k_src1, beta, gamma, pkg_bgra_u8_4k_dst);
    }
}

BENCHMARK_REGISTER_F(CudaAddWeightedBench, GRAYU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaAddWeightedBench, RGBU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaAddWeightedBench, RGBAU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaAddWeightedBench, GRAYU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaAddWeightedBench, RGBU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaAddWeightedBench, RGBAU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaAddWeightedBench, GRAYU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaAddWeightedBench, RGBU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaAddWeightedBench, RGBAU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);
