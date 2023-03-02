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

class CudaResizeBench : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {
        width = state.range(0);
        height = state.range(1);
        set_thread_num(G_THREAD_NUM);

        pkg_bgr_u8_1x = CudaMat(width, height, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_1x.total_byte_size(), 55, pkg_bgr_u8_1x.data());

        pkg_bgr_u8_2x = CudaMat(width / 2, height / 2, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_2x.total_byte_size(), 55, pkg_bgr_u8_2x.data());

        pkg_bgr_u8_300 = CudaMat(600, 300, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_300.total_byte_size(), 55, pkg_bgr_u8_300.data());

        pkg_bgr_u8_4x = CudaMat(width / 4, height / 4, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_4x.total_byte_size(), 55, pkg_bgr_u8_4x.data());

        gray_u8_1x = CudaMat(width, height, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_1x.total_byte_size(), 55, gray_u8_1x.data());

        gray_u8_2x = CudaMat(width / 2, height / 2, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_2x.total_byte_size(), 55, gray_u8_2x.data());

        gray_u8_4x = CudaMat(width / 4, height / 4, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_4x.total_byte_size(), 55, gray_u8_4x.data());

        pkg_rgba_u8_1x = CudaMat(width, height, FCVImageType::PKG_RGBA_U8);
        construct_data<unsigned char>(pkg_rgba_u8_1x.total_byte_size(), 55, pkg_rgba_u8_1x.data());

        pkg_rgba_u8_2x = CudaMat(width / 2, height / 2, FCVImageType::PKG_RGBA_U8);
        construct_data<unsigned char>(pkg_rgba_u8_2x.total_byte_size(), 55, pkg_rgba_u8_2x.data());

        pkg_rgba_u8_4x = CudaMat(width / 4, height / 4, FCVImageType::PKG_RGBA_U8);
        construct_data<unsigned char>(pkg_rgba_u8_4x.total_byte_size(), 55, pkg_rgba_u8_4x.data());
    }

    void TearDown(const ::benchmark::State& state) {
        pkg_bgr_u8_1x.~CudaMat();
        pkg_bgr_u8_2x.~CudaMat();
        pkg_bgr_u8_300.~CudaMat();
        pkg_bgr_u8_4x.~CudaMat();
        gray_u8_1x.~CudaMat();
        gray_u8_2x.~CudaMat();
        gray_u8_4x.~CudaMat();
        pkg_rgba_u8_1x.~CudaMat();
        pkg_rgba_u8_2x.~CudaMat();
        pkg_rgba_u8_4x.~CudaMat();
    }

public:
    int width;
    int height;

    CudaMat pkg_bgr_u8_1x;
    CudaMat pkg_bgr_u8_2x;
    CudaMat pkg_bgr_u8_300;
    CudaMat pkg_bgr_u8_4x;
    CudaMat gray_u8_1x;
    CudaMat gray_u8_2x;
    CudaMat gray_u8_4x;
    CudaMat pkg_rgba_u8_1x;
    CudaMat pkg_rgba_u8_2x;
    CudaMat pkg_rgba_u8_4x;
};

BENCHMARK_DEFINE_F(CudaResizeBench, Resize_INTER_LINEAR_IrregularScaleA)(benchmark::State& state) {
    for (auto _state : state) {
        resize(pkg_bgr_u8_1x, pkg_bgr_u8_300, Size(600, 300), 0, 0, InterpolationType::INTER_LINEAR);
    }
}

BENCHMARK_DEFINE_F(CudaResizeBench, Resize_INTER_LINEAR_IrregularScaleB)(benchmark::State& state) {
    for (auto _state : state) {
        resize(pkg_bgr_u8_300, pkg_bgr_u8_1x, Size(width, height), 0, 0, InterpolationType::INTER_LINEAR);
    }
}

BENCHMARK_DEFINE_F(CudaResizeBench, Resize_INTER_LINEAR_Half_C1_720P)(benchmark::State& state) {
    for(auto _state: state){
        resize(gray_u8_1x, gray_u8_2x, Size(width / 2, height / 2), 0, 0, InterpolationType::INTER_LINEAR);
    }
}

BENCHMARK_DEFINE_F(CudaResizeBench, Resize_INTER_LINEAR_Half_C3_720P)(benchmark::State& state) {
    for (auto _state : state) {
        resize(pkg_bgr_u8_1x, pkg_bgr_u8_2x, Size(width / 2, height / 2), 0, 0, InterpolationType::INTER_LINEAR);
    }
}

BENCHMARK_DEFINE_F(CudaResizeBench, Resize_INTER_LINEAR_Half_C4_720P)(benchmark::State& state) {
    for(auto _state: state){
        resize(pkg_rgba_u8_1x, pkg_rgba_u8_2x, Size(width / 2, height / 2), 0, 0, InterpolationType::INTER_LINEAR);
    }
}

BENCHMARK_DEFINE_F(CudaResizeBench, Resize_INTER_LINEAR_Quarter_C1_720P)(benchmark::State& state) {
    for (auto _state : state) {
        resize(gray_u8_1x, gray_u8_4x, Size(width / 4, height / 4), 0, 0, InterpolationType::INTER_LINEAR);
    }
}

BENCHMARK_DEFINE_F(CudaResizeBench, Resize_INTER_LINEAR_Quarter_C3_720P)(benchmark::State& state) {
    for (auto _state : state) {
        resize(pkg_bgr_u8_1x, pkg_bgr_u8_4x, Size(width / 4, height / 4), 0, 0, InterpolationType::INTER_LINEAR);
    }
}

BENCHMARK_DEFINE_F(CudaResizeBench, Resize_INTER_LINEAR_Quarter_C4_720P)(benchmark::State& state) {
    for (auto _state : state) {
        resize(pkg_rgba_u8_1x, pkg_rgba_u8_4x, Size(width / 4, height / 4), 0, 0, InterpolationType::INTER_LINEAR);
    }
}

BENCHMARK_DEFINE_F(CudaResizeBench, Resize_INTER_LINEAR_2X_720P)(benchmark::State& state) {
    for (auto _state : state) {
        resize(pkg_bgr_u8_2x, pkg_bgr_u8_1x, Size(width, height), 0, 0, InterpolationType::INTER_LINEAR);
    }
}

BENCHMARK_DEFINE_F(CudaResizeBench, Resize_INTER_LINEAR_4X_720P)(benchmark::State& state) {
    for (auto _state : state) {
        resize(pkg_bgr_u8_4x, pkg_bgr_u8_1x, Size(width, height), 0, 0, InterpolationType::INTER_LINEAR);
    }
}

BENCHMARK_REGISTER_F(CudaResizeBench, Resize_INTER_LINEAR_IrregularScaleA)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->Args({1280, 720})
        ->Args({1920, 1080})
        ->Args({4032, 3024});

BENCHMARK_REGISTER_F(CudaResizeBench, Resize_INTER_LINEAR_IrregularScaleB)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->Args({1280, 720})
        ->Args({1920, 1080})
        ->Args({4032, 3024});

BENCHMARK_REGISTER_F(CudaResizeBench, Resize_INTER_LINEAR_Half_C1_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->Args({1280, 720})
        ->Args({1920, 1080})
        ->Args({4032, 3024});

BENCHMARK_REGISTER_F(CudaResizeBench, Resize_INTER_LINEAR_Half_C3_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->Args({1280, 720})
        ->Args({1920, 1080})
        ->Args({4032, 3024});

BENCHMARK_REGISTER_F(CudaResizeBench, Resize_INTER_LINEAR_Half_C4_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->Args({1280, 720})
        ->Args({1920, 1080})
        ->Args({4032, 3024});

BENCHMARK_REGISTER_F(CudaResizeBench, Resize_INTER_LINEAR_Quarter_C1_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->Args({1280, 720})
        ->Args({1920, 1080})
        ->Args({4032, 3024});

BENCHMARK_REGISTER_F(CudaResizeBench, Resize_INTER_LINEAR_Quarter_C3_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->Args({1280, 720})
        ->Args({1920, 1080})
        ->Args({4032, 3024});

BENCHMARK_REGISTER_F(CudaResizeBench, Resize_INTER_LINEAR_Quarter_C4_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->Args({1280, 720})
        ->Args({1920, 1080})
        ->Args({4032, 3024});

BENCHMARK_REGISTER_F(CudaResizeBench, Resize_INTER_LINEAR_2X_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->Args({1280, 720})
        ->Args({1920, 1080})
        ->Args({4032, 3024});

BENCHMARK_REGISTER_F(CudaResizeBench, Resize_INTER_LINEAR_4X_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->Args({1280, 720})
        ->Args({1920, 1080})
        ->Args({4032, 3024});
