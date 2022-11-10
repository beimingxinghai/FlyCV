/***************************************************************************
 *
 * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
 *
 **************************************************************************/

/**
 * @contributor     taotianran
 * @created         2022-05-23 13:28
 * @brief
 */

#include "benchmark/benchmark.h"
#include "common/utils.h"
#include "flycv.h"

using namespace g_fcv_ns;

class CopyMakeBorder : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {
        feed_num = state.range(0);
        set_thread_num(1);
    }

public:
    int feed_num;
};

BENCHMARK_DEFINE_F(CopyMakeBorder, GRAYU8_720P)(benchmark::State& state) {
    Mat src = Mat(1280, 720, FCVImageType::GRAY_U8);
    construct_data<unsigned char>(src.total_byte_size(), feed_num, src.data());
    Mat dst;
    int top = 50;
    int bottom = 50;
    int left = 64;
    int right = 64;


    set_thread_num(1);
    for (auto _state : state) {
        copy_make_border(src,dst,top,bottom,left,right,BorderTypes::BORDER_CONSTANT);
    }
}

BENCHMARK_DEFINE_F(CopyMakeBorder, RGBU8_720P)(benchmark::State& state) {
    Mat src = Mat(1280, 720, FCVImageType::PKG_RGB_U8);
    construct_data<unsigned char>(src.total_byte_size(), feed_num, src.data());
    Mat dst;
    int top = 50;
    int bottom = 50;
    int left = 64;
    int right = 64;


    set_thread_num(1);
    for (auto _state : state) {
        copy_make_border(src,dst,top,bottom,left,right,BorderTypes::BORDER_CONSTANT);
    }
}

BENCHMARK_DEFINE_F(CopyMakeBorder, RGBAU8_720P)(benchmark::State& state) {
    Mat src = Mat(1280, 720, FCVImageType::PKG_RGBA_U8);
    construct_data<unsigned char>(src.total_byte_size(), feed_num, src.data());
    Mat dst;
    int top = 50;
    int bottom = 50;
    int left = 64;
    int right = 64;


    set_thread_num(1);
    for (auto _state : state) {
        copy_make_border(src,dst,top,bottom,left,right,BorderTypes::BORDER_CONSTANT);
    }
}


BENCHMARK_REGISTER_F(CopyMakeBorder, GRAYU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CopyMakeBorder, RGBU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CopyMakeBorder, RGBAU8_720P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

//1080P
BENCHMARK_DEFINE_F(CopyMakeBorder, GRAYU8_1080P)(benchmark::State& state) {
    Mat src = Mat(1920, 1080, FCVImageType::GRAY_U8);
    construct_data<unsigned char>(src.total_byte_size(), feed_num, src.data());
    Mat dst;
    int top = 50;
    int bottom = 50;
    int left = 64;
    int right = 64;


    set_thread_num(1);
    for (auto _state : state) {
        copy_make_border(src,dst,top,bottom,left,right,BorderTypes::BORDER_CONSTANT);
    }
}

BENCHMARK_DEFINE_F(CopyMakeBorder, RGBU8_1080P)(benchmark::State& state) {
    Mat src = Mat(1920, 1080, FCVImageType::PKG_RGB_U8);
    construct_data<unsigned char>(src.total_byte_size(), feed_num, src.data());
    Mat dst;
    int top = 50;
    int bottom = 50;
    int left = 64;
    int right = 64;


    set_thread_num(1);
    for (auto _state : state) {
        copy_make_border(src,dst,top,bottom,left,right,BorderTypes::BORDER_CONSTANT);
    }
}

BENCHMARK_DEFINE_F(CopyMakeBorder, RGBAU8_1080P)(benchmark::State& state) {
    Mat src = Mat(1920, 1080, FCVImageType::PKG_RGBA_U8);
    construct_data<unsigned char>(src.total_byte_size(), feed_num, src.data());
    Mat dst;
    int top = 50;
    int bottom = 50;
    int left = 64;
    int right = 64;


    set_thread_num(1);
    for (auto _state : state) {
        copy_make_border(src,dst,top,bottom,left,right,BorderTypes::BORDER_CONSTANT);
    }
}


BENCHMARK_REGISTER_F(CopyMakeBorder, GRAYU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CopyMakeBorder, RGBU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CopyMakeBorder, RGBAU8_1080P)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);  

//4K
BENCHMARK_DEFINE_F(CopyMakeBorder, GRAYU8_4K)(benchmark::State& state) {
    Mat src = Mat(4032, 3024, FCVImageType::GRAY_U8);
    construct_data<unsigned char>(src.total_byte_size(), feed_num, src.data());
    Mat dst;
    int top = 50;
    int bottom = 50;
    int left = 64;
    int right = 64;


    set_thread_num(1);
    for (auto _state : state) {
        copy_make_border(src,dst,top,bottom,left,right,BorderTypes::BORDER_CONSTANT);
    }
}

BENCHMARK_DEFINE_F(CopyMakeBorder, RGBU8_4K)(benchmark::State& state) {
    Mat src = Mat(4032, 3024, FCVImageType::PKG_RGB_U8);
    construct_data<unsigned char>(src.total_byte_size(), feed_num, src.data());
    Mat dst;
    int top = 50;
    int bottom = 50;
    int left = 64;
    int right = 64;


    set_thread_num(1);
    for (auto _state : state) {
        copy_make_border(src,dst,top,bottom,left,right,BorderTypes::BORDER_CONSTANT);
    }
}

BENCHMARK_DEFINE_F(CopyMakeBorder, RGBAU8_4K)(benchmark::State& state) {
    Mat src = Mat(4032, 3024, FCVImageType::PKG_RGBA_U8);
    construct_data<unsigned char>(src.total_byte_size(), feed_num, src.data());
    Mat dst;
    int top = 50;
    int bottom = 50;
    int left = 64;
    int right = 64;


    set_thread_num(1);
    for (auto _state : state) {
        copy_make_border(src,dst,top,bottom,left,right,BorderTypes::BORDER_CONSTANT);
    }
}


BENCHMARK_REGISTER_F(CopyMakeBorder, GRAYU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CopyMakeBorder, RGBU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);

BENCHMARK_REGISTER_F(CopyMakeBorder, RGBAU8_4K)
        ->Unit(benchmark::kMicrosecond)
        ->Iterations(100)
        ->DenseRange(55, 255, 200);  