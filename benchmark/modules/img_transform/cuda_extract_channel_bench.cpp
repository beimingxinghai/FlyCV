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

class CudaExtractChannelBench : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {
        feed_num = state.range(0);
        set_thread_num(G_THREAD_NUM);

        pkg_bgr_u8_720 = CudaMat(1280, 720, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_720.total_byte_size(), feed_num, pkg_bgr_u8_720.data());
        pkg_bgra_u8_720 = CudaMat(1280, 720, FCVImageType::PKG_BGRA_U8);
        construct_data<unsigned char>(pkg_bgra_u8_720.total_byte_size(), feed_num, pkg_bgra_u8_720.data());
        gray_u8_720 = CudaMat(pkg_bgr_u8_720.size(), FCVImageType::GRAY_U8);

        pkg_bgr_u8_1080 = CudaMat(1920, 1080, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_1080.total_byte_size(), feed_num, pkg_bgr_u8_1080.data());
        pkg_bgra_u8_1080 = CudaMat(1920, 1080, FCVImageType::PKG_BGRA_U8);
        construct_data<unsigned char>(pkg_bgra_u8_1080.total_byte_size(), feed_num, pkg_bgra_u8_1080.data());
        gray_u8_1080 = CudaMat(pkg_bgr_u8_1080.size(), FCVImageType::GRAY_U8);

        pkg_bgr_u8_4K = CudaMat(4032, 3024, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_4K.total_byte_size(), feed_num, pkg_bgr_u8_4K.data());
        pkg_bgra_u8_4K = CudaMat(4032, 3024, FCVImageType::PKG_BGRA_U8);
        construct_data<unsigned char>(pkg_bgra_u8_4K.total_byte_size(), feed_num, pkg_bgra_u8_4K.data());
        gray_u8_4K = CudaMat(pkg_bgr_u8_4K.size(), FCVImageType::GRAY_U8);
    }

    void TearDown(const ::benchmark::State& state) {
        feed_num = state.range(0);
        gray_u8_4K.~CudaMat();
        pkg_bgra_u8_4K.~CudaMat();
        pkg_bgr_u8_4K.~CudaMat();

        gray_u8_1080.~CudaMat();
        pkg_bgra_u8_1080.~CudaMat();
        pkg_bgr_u8_1080.~CudaMat();

        gray_u8_720.~CudaMat();
        pkg_bgra_u8_720.~CudaMat();
        pkg_bgr_u8_720.~CudaMat();
    }

public:
    int feed_num;
    CudaMat pkg_bgr_u8_720;
    CudaMat pkg_bgra_u8_720;
    CudaMat gray_u8_720;
    CudaMat pkg_bgr_u8_1080;
    CudaMat pkg_bgra_u8_1080;
    CudaMat gray_u8_1080;
    CudaMat pkg_bgr_u8_4K;
    CudaMat pkg_bgra_u8_4K;
    CudaMat gray_u8_4K;
};

BENCHMARK_DEFINE_F(CudaExtractChannelBench, BGRU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        extract_channel(pkg_bgr_u8_720, gray_u8_720, state.range(1));
    }
}

BENCHMARK_DEFINE_F(CudaExtractChannelBench, BGRAU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        extract_channel(pkg_bgra_u8_720, gray_u8_720, state.range(1));
    }
}

BENCHMARK_REGISTER_F(CudaExtractChannelBench, BGRU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->ArgsProduct({benchmark::CreateDenseRange(55, 255, 200), benchmark::CreateDenseRange(0, 2, 1)});

// BENCHMARK_REGISTER_F(CudaExtractChannelBench, BGRAU8_720P)
//         ->Unit(benchmark::kMicrosecond)
//         ->Iterations(100)
//         ->ArgsProduct({benchmark::CreateDenseRange(55, 255, 200), benchmark::CreateDenseRange(0, 3, 1)});

// 1080
BENCHMARK_DEFINE_F(CudaExtractChannelBench, BGRU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        extract_channel(pkg_bgr_u8_1080, gray_u8_1080, state.range(1));
    }
}

BENCHMARK_DEFINE_F(CudaExtractChannelBench, BGRAU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        extract_channel(pkg_bgra_u8_1080, gray_u8_1080, state.range(1));
    }
}

BENCHMARK_REGISTER_F(CudaExtractChannelBench, BGRU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->ArgsProduct({benchmark::CreateDenseRange(55, 255, 200), benchmark::CreateDenseRange(0, 2, 1)});

// BENCHMARK_REGISTER_F(CudaExtractChannelBench, BGRAU8_1080P)
//         ->Unit(benchmark::kMicrosecond)
//         ->Iterations(100)
//         ->ArgsProduct({benchmark::CreateDenseRange(55, 255, 200), benchmark::CreateDenseRange(0, 3, 1)});

// 4K
BENCHMARK_DEFINE_F(CudaExtractChannelBench, BGRU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        extract_channel(pkg_bgr_u8_4K, gray_u8_4K, state.range(1));
    }
}

BENCHMARK_DEFINE_F(CudaExtractChannelBench, BGRAU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        extract_channel(pkg_bgra_u8_4K, gray_u8_4K, state.range(1));
    }
}

BENCHMARK_REGISTER_F(CudaExtractChannelBench, BGRU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->ArgsProduct({benchmark::CreateDenseRange(55, 255, 200), benchmark::CreateDenseRange(0, 2, 1)});

// BENCHMARK_REGISTER_F(CudaExtractChannelBench, BGRAU8_4K)
//         ->Unit(benchmark::kMicrosecond)
//         ->Iterations(100)
//         ->ArgsProduct({benchmark::CreateDenseRange(55, 255, 200), benchmark::CreateDenseRange(0, 3, 1)});
