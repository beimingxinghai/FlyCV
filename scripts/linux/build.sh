#!/bin/bash

current_dir=$(cd `dirname $0`; pwd)
repository_dir=$(cd ${current_dir}/../..; pwd)
build_dir=${repository_dir}/build

core_num=$(grep -c processor /proc/cpuinfo)
arch=$(uname -m)
CUDA_SW=OFF

if [ $# -lt 1 ];then
    echo "Select architecture by serial number:"
    echo "    0 CPU only"
    echo "    1 CPU with CUDA"
    read -r index
else
    index=$1
fi

case ${index} in
    0|cpu|CPU)
        CUDA_SW=OFF
        ;;
    1|cuda|CUDA)
        CUDA_SW=ON
        ;;
    *)
        echo "Unsupported select"
        exit 1
    ;;
esac

if [ $# -lt 2 ];then
    echo "Do you need clean previous project files? [Y/N]"
    read -n 1 index
    echo ""
else
    index=${2}
fi

case ${index} in
    Y|y)
        rebuild=1
        ;;
    N|n)
        rebuild=0
        ;;
    *)
        echo "Unsupported arguments"
        exit 1
    ;;
esac

mkdir -p ${build_dir}
if [ ${rebuild} -eq 1 ];then
    echo "clean previous project files ..."
    rm -rf ${build_dir}/*
fi

cd ${build_dir} || exit 0

cmake \
    -DWITH_CUDA_SUPPORT="${CUDA_SW}" \
    -DBUILD_TEST=ON \
    -DBUILD_BENCHMARK=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${build_dir}/install \
    -DBUILD_FCV_MEDIA_IO=ON \
    -DWITH_LIB_PNG=ON \
    -DWITH_LIB_JPEG_TURBO=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCUDA_MULTI_ARCH=ON \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    ..

make -j"${core_num}"
make install
