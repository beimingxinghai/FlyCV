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

class CudaColorConvertBench : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {
        feed_num = state.range(0);
        set_thread_num(G_THREAD_NUM);

        pkg_bgr_u8_720 = CudaMat(1280, 720, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_720.total_byte_size(), feed_num, pkg_bgr_u8_720.data());

        pkg_rgb_u8_720 = CudaMat(1280, 720, FCVImageType::PKG_RGB_U8);
        construct_data<unsigned char>(pkg_rgb_u8_720.total_byte_size(), feed_num, pkg_rgb_u8_720.data());

        gray_u8_720 = CudaMat(1280, 720, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_720.total_byte_size(), feed_num, gray_u8_720.data());

        I420_720 = CudaMat(1280, 720, FCVImageType::I420);
        construct_data<unsigned char>(I420_720.total_byte_size(), feed_num, I420_720.data());

        nv21_720 = CudaMat(1280, 720, FCVImageType::NV21);
        construct_data<unsigned char>(nv21_720.total_byte_size(), feed_num, nv21_720.data());

        nv12_720 = CudaMat(1280, 720, FCVImageType::NV12);
        construct_data<unsigned char>(nv12_720.total_byte_size(), feed_num, nv12_720.data());

        pkg_bgr_u8_1080 = CudaMat(1920, 1080, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_1080.total_byte_size(), feed_num, pkg_bgr_u8_1080.data());

        pkg_rgb_u8_1080 = CudaMat(1920, 1080, FCVImageType::PKG_RGB_U8);
        construct_data<unsigned char>(pkg_rgb_u8_1080.total_byte_size(), feed_num, pkg_rgb_u8_1080.data());

        gray_u8_1080 = CudaMat(1920, 1080, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_1080.total_byte_size(), feed_num, gray_u8_1080.data());

        I420_1080 = CudaMat(1920, 1080, FCVImageType::I420);
        construct_data<unsigned char>(I420_1080.total_byte_size(), feed_num, I420_1080.data());

        nv21_1080 = CudaMat(1920, 1080, FCVImageType::NV21);
        construct_data<unsigned char>(nv21_1080.total_byte_size(), feed_num, nv21_1080.data());

        nv12_1080 = CudaMat(1920, 1080, FCVImageType::NV12);
        construct_data<unsigned char>(nv12_1080.total_byte_size(), feed_num, nv12_1080.data());

        pkg_bgr_u8_4k = CudaMat(4032, 3024, FCVImageType::PKG_BGR_U8);
        construct_data<unsigned char>(pkg_bgr_u8_4k.total_byte_size(), feed_num, pkg_bgr_u8_4k.data());

        pkg_rgb_u8_4k = CudaMat(4032, 3024, FCVImageType::PKG_RGB_U8);
        construct_data<unsigned char>(pkg_rgb_u8_4k.total_byte_size(), feed_num, pkg_rgb_u8_4k.data());

        gray_u8_4k = CudaMat(4032, 3024, FCVImageType::GRAY_U8);
        construct_data<unsigned char>(gray_u8_4k.total_byte_size(), feed_num, gray_u8_4k.data());

        I420_4k = CudaMat(4032, 3024, FCVImageType::I420);
        construct_data<unsigned char>(I420_4k.total_byte_size(), feed_num, I420_4k.data());

        nv21_4k = CudaMat(4032, 3024, FCVImageType::NV21);
        construct_data<unsigned char>(nv21_4k.total_byte_size(), feed_num, nv21_4k.data());

        nv12_4k = CudaMat(4032, 3024, FCVImageType::NV12);
        construct_data<unsigned char>(nv12_4k.total_byte_size(), feed_num, nv12_4k.data());
    }

    void TearDown(const ::benchmark::State& state) {
        feed_num = state.range(0);
        nv12_4k.~CudaMat();
        nv21_4k.~CudaMat();
        I420_4k.~CudaMat();
        gray_u8_4k.~CudaMat();
        pkg_rgb_u8_4k.~CudaMat();
        pkg_bgr_u8_4k.~CudaMat();

        nv12_1080.~CudaMat();
        nv21_1080.~CudaMat();
        I420_1080.~CudaMat();
        gray_u8_1080.~CudaMat();
        pkg_rgb_u8_1080.~CudaMat();
        pkg_bgr_u8_1080.~CudaMat();

        nv12_720.~CudaMat();
        nv21_720.~CudaMat();
        I420_720.~CudaMat();
        gray_u8_720.~CudaMat();
        pkg_rgb_u8_720.~CudaMat();
        pkg_bgr_u8_720.~CudaMat();
    }

public:
    int feed_num;
    CudaMat pkg_bgr_u8_720;
    CudaMat pkg_rgb_u8_720;
    CudaMat gray_u8_720;
    CudaMat I420_720;
    CudaMat nv21_720;
    CudaMat nv12_720;

    CudaMat pkg_bgr_u8_1080;
    CudaMat pkg_rgb_u8_1080;
    CudaMat gray_u8_1080;
    CudaMat I420_1080;
    CudaMat nv21_1080;
    CudaMat nv12_1080;

    CudaMat pkg_bgr_u8_4k;
    CudaMat pkg_rgb_u8_4k;
    CudaMat gray_u8_4k;
    CudaMat I420_4k;
    CudaMat nv21_4k;
    CudaMat nv12_4k;

};

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgBGRU8ToGrayU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(pkg_bgr_u8_720, gray_u8_720, ColorConvertType::CVT_PA_BGR2GRAY);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgRGBU8ToGrayU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(pkg_rgb_u8_720, gray_u8_720, ColorConvertType::CVT_PA_RGB2GRAY);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgBGRU8ToPkgRGBU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(pkg_bgr_u8_720, pkg_rgb_u8_720, ColorConvertType::CVT_PA_BGR2PA_RGB);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgRGBU8ToPkgBGRU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(pkg_rgb_u8_720, pkg_bgr_u8_720, ColorConvertType::CVT_PA_RGB2PA_BGR);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgI420ToBGRU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(I420_720, pkg_bgr_u8_720, ColorConvertType::CVT_I4202PA_BGR);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgNV21ToBGRU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(nv21_720, pkg_bgr_u8_720, ColorConvertType::CVT_NV212PA_BGR);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgNV12ToBGRU8_720P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(nv12_720, pkg_bgr_u8_720, ColorConvertType::CVT_NV122PA_BGR);
    }
}

//1080
BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgBGRU8ToGrayU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(pkg_bgr_u8_1080, gray_u8_1080, ColorConvertType::CVT_PA_BGR2GRAY);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgRGBU8ToGrayU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(pkg_rgb_u8_1080, gray_u8_1080, ColorConvertType::CVT_PA_RGB2GRAY);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgBGRU8ToPkgRGBU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(pkg_bgr_u8_1080, pkg_rgb_u8_1080, ColorConvertType::CVT_PA_BGR2PA_RGB);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgRGBU8ToPkgBGRU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(pkg_rgb_u8_1080, pkg_bgr_u8_1080, ColorConvertType::CVT_PA_RGB2PA_BGR);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgI420ToBGRU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(I420_1080, pkg_bgr_u8_1080, ColorConvertType::CVT_I4202PA_BGR);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgNV21ToBGRU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(nv21_1080, pkg_bgr_u8_1080, ColorConvertType::CVT_NV212PA_BGR);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgNV12ToBGRU8_1080P)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(nv12_1080, pkg_bgr_u8_1080, ColorConvertType::CVT_NV122PA_BGR);
    }
}

//4K
BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgBGRU8ToGrayU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(pkg_bgr_u8_4k, gray_u8_4k, ColorConvertType::CVT_PA_BGR2GRAY);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgRGBU8ToGrayU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(pkg_rgb_u8_4k, gray_u8_4k, ColorConvertType::CVT_PA_RGB2GRAY);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgBGRU8ToPkgRGBU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(pkg_bgr_u8_4k, pkg_rgb_u8_4k, ColorConvertType::CVT_PA_BGR2PA_RGB);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgRGBU8ToPkgBGRU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(pkg_rgb_u8_4k, pkg_bgr_u8_4k, ColorConvertType::CVT_PA_RGB2PA_BGR);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgI420ToBGRU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(I420_4k, pkg_bgr_u8_4k, ColorConvertType::CVT_I4202PA_BGR);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgNV21ToBGRU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(nv21_4k, pkg_bgr_u8_4k, ColorConvertType::CVT_NV212PA_BGR);
    }
}

BENCHMARK_DEFINE_F(CudaColorConvertBench, PkgNV12ToBGRU8_4K)(benchmark::State& state) {
    for (auto _state : state) {
        cvt_color(nv12_4k, pkg_bgr_u8_4k, ColorConvertType::CVT_NV122PA_BGR);
    }
}

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgBGRU8ToGrayU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgRGBU8ToGrayU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgBGRU8ToPkgRGBU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgRGBU8ToPkgBGRU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgI420ToBGRU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgNV21ToBGRU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgNV12ToBGRU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

//1080
BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgBGRU8ToGrayU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgRGBU8ToGrayU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgBGRU8ToPkgRGBU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgRGBU8ToPkgBGRU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgI420ToBGRU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgNV21ToBGRU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgNV12ToBGRU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgBGRU8ToGrayU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgRGBU8ToGrayU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgBGRU8ToPkgRGBU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgRGBU8ToPkgBGRU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgI420ToBGRU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgNV21ToBGRU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CudaColorConvertBench, PkgNV12ToBGRU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);
