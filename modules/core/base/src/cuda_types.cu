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
#include <mutex>

#include "modules/core/base/interface/cuda_types.h"

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

static std::shared_ptr<CUDADeviceInfo> singleton = nullptr;
static std::once_flag singleton_flag;

std::shared_ptr<CUDADeviceInfo> CUDADeviceInfo::get_instance() {
    std::call_once(singleton_flag, [&] {
        singleton = std::shared_ptr<CUDADeviceInfo>(new CUDADeviceInfo());
    });
    return singleton;
}

CUDADeviceInfo::CUDADeviceInfo() {
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

    for (int dev = 0; dev < device_count; ++dev) {
        CUDADeviceAttr attr;
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaGetDeviceProperties(&attr.device_prop, dev));

        CUDA_CHECK(cudaDeviceGetAttribute(&attr.integrated_flag, cudaDevAttrIntegrated, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&attr.coherent_flag, cudaDevAttrConcurrentManagedAccess, dev));
        device_attrs.push_back(attr);
    }
}

G_FCV_NAMESPACE1_END()
