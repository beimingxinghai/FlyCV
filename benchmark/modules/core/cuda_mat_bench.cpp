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
#include "flycv.h"
#include "common/utils.h"

using namespace g_fcv_ns;

class CudaMatBench : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {
        set_thread_num(G_THREAD_NUM);

        pkg_bgr_u8_720 = CudaMat(1280, 720, FCVImageType::PKG_BGR_U8);
        unsigned char* pkg_bgr_u8_720_data = (unsigned char*)pkg_bgr_u8_720.data();

        for (size_t i = 0; i < pkg_bgr_u8_720.total_byte_size(); ++i) {
            pkg_bgr_u8_720_data[i] = i * state.range(0) % 256;
        }

        pkg_bgr_u8_1080 = CudaMat(1920, 1080, FCVImageType::PKG_BGR_U8);
        unsigned char* pkg_bgr_u8_1080_data = (unsigned char*)pkg_bgr_u8_1080.data();

        for (size_t i = 0; i < pkg_bgr_u8_1080.total_byte_size(); ++i) {
            pkg_bgr_u8_1080_data[i] = i * state.range(0) % 256;
        }

        pkg_bgr_u8_4K = CudaMat(4032, 3024, FCVImageType::PKG_BGR_U8);
        unsigned char* pkg_bgr_u8_4K_data = (unsigned char*)pkg_bgr_u8_4K.data();

        for (size_t i = 0; i < pkg_bgr_u8_4K.total_byte_size(); ++i) {
            pkg_bgr_u8_4K_data[i] = i * state.range(0) % 256;
        }
    }

    void TearDown(const ::benchmark::State& state) {
        pkg_bgr_u8_4K.~CudaMat();
        pkg_bgr_u8_1080.~CudaMat();
        pkg_bgr_u8_720.~CudaMat();
    }

public:
    CudaMat pkg_bgr_u8_720;
    CudaMat pkg_bgr_u8_1080;
    CudaMat pkg_bgr_u8_4K;
};

BENCHMARK_DEFINE_F(CudaMatBench, ConstructorPkgBGRU8_720P)(
        benchmark::State& state) {
    for (auto _state : state) {
        CudaMat dst(1280, 720, FCVImageType::PKG_BGR_U8);
    }
}

BENCHMARK_DEFINE_F(CudaMatBench, ConstructorPkgBGRU8_1080P)(
        benchmark::State& state) {
    for (auto _state : state) {
        CudaMat dst(1920, 1080, FCVImageType::PKG_BGR_U8);
    }
}

BENCHMARK_DEFINE_F(CudaMatBench, ConstructorPkgBGRU8_4K)(
        benchmark::State& state) {
    for (auto _state : state) {
        CudaMat dst(4032, 3024, FCVImageType::PKG_BGR_U8);
    }
}

BENCHMARK_DEFINE_F(CudaMatBench, ConvertTo_PkgBGRU8ToPkgBGRF32_720P)(
        benchmark::State& state) {
    for (auto _state : state) {
        CudaMat dst;
        pkg_bgr_u8_720.convert_to(dst, FCVImageType::PKG_BGR_F32, 0.5f, 10);
    }
}

BENCHMARK_DEFINE_F(CudaMatBench, ConvertTo_PkgBGRU8ToPkgBGRF32_1080P)(
        benchmark::State& state) {
    for (auto _state : state) {
        CudaMat dst;
        pkg_bgr_u8_1080.convert_to(dst, FCVImageType::PKG_BGR_F32, 0.5f, 10);
    }
}

BENCHMARK_DEFINE_F(CudaMatBench, ConvertTo_PkgBGRU8ToPkgBGRF32_4K)(
        benchmark::State& state) {
    for (auto _state : state) {
        CudaMat dst;
        pkg_bgr_u8_4K.convert_to(dst, FCVImageType::PKG_BGR_F32, 0.5f, 10);
    }
}

BENCHMARK_REGISTER_F(CudaMatBench, ConstructorPkgBGRU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaMatBench, ConstructorPkgBGRU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaMatBench, ConstructorPkgBGRU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaMatBench, ConvertTo_PkgBGRU8ToPkgBGRF32_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaMatBench, ConvertTo_PkgBGRU8ToPkgBGRF32_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaMatBench, ConvertTo_PkgBGRU8ToPkgBGRF32_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);
