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

#include "gtest/gtest.h"
#include "flycv.h"
#include "test_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace g_fcv_ns;

class CudaAddWeightedTest : public ::testing::Test {
public:
    void SetUp() override {
        ASSERT_EQ(prepare_pkg_bgr_u8_720p_cuda(src1), 0);
        ASSERT_EQ(prepare_pkg_bgr_u8_720p_2_cuda(src2), 0);
    }

public:
    CudaMat src1;
    CudaMat src2;
};

TEST_F(CudaAddWeightedTest, AddWeightedCommonPositiveInput) {
    double alpha = 0.5;
    double beta = 0.5;
    double gama = 20;

    CudaMat add_weighted_dst;
    add_weighted(src1, alpha, src2, beta, gama, add_weighted_dst);
}
