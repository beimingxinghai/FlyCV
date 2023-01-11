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
// #include "modules/core/opencl/interface/opencl.h"

using namespace g_fcv_ns;

class CudaSplitToMemBench : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {
        feed_num = state.range(0);
        set_thread_num(G_THREAD_NUM);

        pkg_bgr_f32_720 = CudaMat(1280, 720, FCVImageType::PKG_BGR_F32);
        construct_data<float>(pkg_bgr_f32_720.total_byte_size() / pkg_bgr_f32_720.type_byte_size(), feed_num, pkg_bgr_f32_720.data());
        bgr_dst_f32_720 = CudaMat(1280 * 3, 720, FCVImageType::GRAY_F32);

        pkg_bgra_f32_720 = CudaMat(1280, 720, FCVImageType::PKG_BGRA_F32);
        construct_data<float>(pkg_bgra_f32_720.total_byte_size() / pkg_bgra_f32_720.type_byte_size(), feed_num, pkg_bgra_f32_720.data());
        bgra_dst_f32_720 = CudaMat(1280 * 4, 720, FCVImageType::GRAY_F32);

        pkg_bgr_f32_1080 = CudaMat(1920, 1080, FCVImageType::PKG_BGR_F32);
        construct_data<float>(pkg_bgr_f32_1080.total_byte_size() / pkg_bgr_f32_1080.type_byte_size(), feed_num, pkg_bgr_f32_1080.data());
        bgr_dst_f32_1080 = CudaMat(1920 * 3, 1080, FCVImageType::GRAY_F32);

        pkg_bgra_f32_1080 = CudaMat(1280, 720, FCVImageType::PKG_BGRA_F32);
        construct_data<float>(pkg_bgra_f32_1080.total_byte_size() / pkg_bgra_f32_1080.type_byte_size(), feed_num, pkg_bgra_f32_1080.data());
        bgra_dst_f32_1080 = CudaMat(1920 * 4, 1080, FCVImageType::GRAY_F32);

        pkg_bgr_f32_4K = CudaMat(4032, 3024, FCVImageType::PKG_BGR_F32);
        construct_data<float>(pkg_bgr_f32_4K.total_byte_size() / pkg_bgr_f32_4K.type_byte_size(), feed_num, pkg_bgr_f32_4K.data());
        bgr_dst_f32_4K = CudaMat(4032 * 3, 3024, FCVImageType::GRAY_F32);

        pkg_bgra_f32_4K = CudaMat(4032, 3024, FCVImageType::PKG_BGRA_F32);
        construct_data<float>(pkg_bgra_f32_4K.total_byte_size() / pkg_bgra_f32_4K.type_byte_size(), feed_num, pkg_bgra_f32_4K.data());
        bgra_dst_f32_4K = CudaMat(4032 * 4, 3024, FCVImageType::GRAY_F32);
    }

    void TearDown(const ::benchmark::State& state) {
        feed_num = state.range(0);
        bgra_dst_f32_4K.~CudaMat();
        pkg_bgra_f32_4K.~CudaMat();

        bgr_dst_f32_4K.~CudaMat();
        pkg_bgr_f32_4K.~CudaMat();

        bgra_dst_f32_1080.~CudaMat();
        pkg_bgra_f32_1080.~CudaMat();

        bgr_dst_f32_1080.~CudaMat();
        pkg_bgr_f32_1080.~CudaMat();

        bgra_dst_f32_720.~CudaMat();
        pkg_bgra_f32_720.~CudaMat();

        bgr_dst_f32_720.~CudaMat();
        pkg_bgr_f32_720.~CudaMat();
    }

public:
    int feed_num;
    CudaMat pkg_bgr_f32_720;
    CudaMat bgr_dst_f32_720;
    CudaMat pkg_bgra_f32_720;
    CudaMat bgra_dst_f32_720;
    CudaMat pkg_bgr_f32_1080;
    CudaMat bgr_dst_f32_1080;
    CudaMat pkg_bgra_f32_1080;
    CudaMat bgra_dst_f32_1080;
    CudaMat pkg_bgr_f32_4K;
    CudaMat bgr_dst_f32_4K;
    CudaMat pkg_bgra_f32_4K;
    CudaMat bgra_dst_f32_4K;
};

BENCHMARK_DEFINE_F(CudaSplitToMemBench, SplitToMemC3_720P)(benchmark::State& state) {
    for (auto _state : state) {
        split_to_memcpy(pkg_bgr_f32_720, &bgr_dst_f32_720);
    }
}

BENCHMARK_DEFINE_F(CudaSplitToMemBench, SplitToMemC4_720P)(benchmark::State& state) {
    for (auto _state : state) {
        split_to_memcpy(pkg_bgra_f32_720, &bgra_dst_f32_720);
    }
}

BENCHMARK_DEFINE_F(CudaSplitToMemBench, SplitToMemC3_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        split_to_memcpy(pkg_bgr_f32_1080, &bgr_dst_f32_1080);
    }
}

BENCHMARK_DEFINE_F(CudaSplitToMemBench, SplitToMemC4_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        split_to_memcpy(pkg_bgra_f32_1080, &bgra_dst_f32_1080);
    }
}

BENCHMARK_DEFINE_F(CudaSplitToMemBench, SplitToMemC3_4K)(benchmark::State& state) {
    for (auto _state : state) {
        split_to_memcpy(pkg_bgr_f32_4K, &bgr_dst_f32_4K);
    }
}

BENCHMARK_DEFINE_F(CudaSplitToMemBench, SplitToMemC4_4K)(benchmark::State& state) {
    for (auto _state : state) {
        split_to_memcpy(pkg_bgra_f32_4K, &bgra_dst_f32_4K);
    }
}

BENCHMARK_REGISTER_F(CudaSplitToMemBench, SplitToMemC3_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaSplitToMemBench, SplitToMemC4_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaSplitToMemBench, SplitToMemC3_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaSplitToMemBench, SplitToMemC4_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaSplitToMemBench, SplitToMemC3_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaSplitToMemBench, SplitToMemC4_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);
