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

#include "modules/img_transform/resize/interface/resize_cuda.h"

#include <cmath>

G_FCV_NAMESPACE1_BEGIN(g_fcv_ns)

int resize(
        CudaMat& src,
        CudaMat& dst,
        const Size& dsize,
        double fx,
        double fy,
        InterpolationType interpolation,
        Stream& stream) {
    if (src.empty()) {
        LOG_ERR("Input CudaMat of resize is empty!");
        return -1;
    }

    if (dst.empty()) {
        if (dsize.width() > 0 && dsize.height() > 0) {
            dst = CudaMat(dsize.width(), dsize.height(), src.type());
        } else if (fx > 0 && fy > 0) {
            dst = CudaMat(std::round(src.width() * fx), std::round(src.height() * fy), src.type());
        } else {
            LOG_ERR("Dst CudaMat width or height is zero which is illegal!");
            return -1;
        }
    }

    switch (interpolation) {
    case InterpolationType::INTER_LINEAR:
        break;
    case InterpolationType::INTER_CUBIC:
        break;
    case InterpolationType::INTER_AREA:
        break;
    default:
        LOG_ERR("The resize interpolation %d is unsupported now", int(interpolation));
        return -1;
    }



    return 0;
}

G_FCV_NAMESPACE1_END()
