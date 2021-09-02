// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include <fp16/fp16.h>

#include "models/models.h"

namespace models {

ExecutionPlan FP16MobileNetV1(pthreadpool_t threadpool) {
  alignas(16) static uint16_t v0[150528];
  alignas(16) static uint16_t v1[401408];
  alignas(16) static uint16_t v2[401408];
  alignas(16) static uint16_t v3[802816];
  alignas(16) static uint16_t v4[200704];
  alignas(16) static uint16_t v5[401408];
  alignas(16) static uint16_t v6[401408];
  alignas(16) static uint16_t v7[401408];
  alignas(16) static uint16_t v8[100352];
  alignas(16) static uint16_t v9[200704];
  alignas(16) static uint16_t v10[200704];
  alignas(16) static uint16_t v11[200704];
  alignas(16) static uint16_t v12[50176];
  alignas(16) static uint16_t v13[100352];
  alignas(16) static uint16_t v14[100352];
  alignas(16) static uint16_t v15[100352];
  alignas(16) static uint16_t v16[100352];
  alignas(16) static uint16_t v17[100352];
  alignas(16) static uint16_t v18[100352];
  alignas(16) static uint16_t v19[100352];
  alignas(16) static uint16_t v20[100352];
  alignas(16) static uint16_t v21[100352];
  alignas(16) static uint16_t v22[100352];
  alignas(16) static uint16_t v23[100352];
  alignas(16) static uint16_t v24[25088];
  alignas(16) static uint16_t v25[50176];
  alignas(16) static uint16_t v26[50176];
  alignas(16) static uint16_t v27[50176];
  alignas(16) static uint16_t v28[1024];
  alignas(16) static uint16_t v29[1001];
  alignas(16) static uint16_t w30[864];
  alignas(16) static uint16_t w31[32];
  alignas(16) static uint16_t w32[288];
  alignas(16) static uint16_t w33[32];
  alignas(16) static uint16_t w34[2048];
  alignas(16) static uint16_t w35[64];
  alignas(16) static uint16_t w36[576];
  alignas(16) static uint16_t w37[64];
  alignas(16) static uint16_t w38[8192];
  alignas(16) static uint16_t w39[128];
  alignas(16) static uint16_t w40[1152];
  alignas(16) static uint16_t w41[128];
  alignas(16) static uint16_t w42[16384];
  alignas(16) static uint16_t w43[128];
  alignas(16) static uint16_t w44[1152];
  alignas(16) static uint16_t w45[128];
  alignas(16) static uint16_t w46[32768];
  alignas(16) static uint16_t w47[256];
  alignas(16) static uint16_t w48[2304];
  alignas(16) static uint16_t w49[256];
  alignas(16) static uint16_t w50[65536];
  alignas(16) static uint16_t w51[256];
  alignas(16) static uint16_t w52[2304];
  alignas(16) static uint16_t w53[256];
  alignas(16) static uint16_t w54[131072];
  alignas(16) static uint16_t w55[512];
  alignas(16) static uint16_t w56[4608];
  alignas(16) static uint16_t w57[512];
  alignas(16) static uint16_t w58[262144];
  alignas(16) static uint16_t w59[512];
  alignas(16) static uint16_t w60[4608];
  alignas(16) static uint16_t w61[512];
  alignas(16) static uint16_t w62[262144];
  alignas(16) static uint16_t w63[512];
  alignas(16) static uint16_t w64[4608];
  alignas(16) static uint16_t w65[512];
  alignas(16) static uint16_t w66[262144];
  alignas(16) static uint16_t w67[512];
  alignas(16) static uint16_t w68[4608];
  alignas(16) static uint16_t w69[512];
  alignas(16) static uint16_t w70[262144];
  alignas(16) static uint16_t w71[512];
  alignas(16) static uint16_t w72[4608];
  alignas(16) static uint16_t w73[512];
  alignas(16) static uint16_t w74[262144];
  alignas(16) static uint16_t w75[512];
  alignas(16) static uint16_t w76[4608];
  alignas(16) static uint16_t w77[512];
  alignas(16) static uint16_t w78[524288];
  alignas(16) static uint16_t w79[1024];
  alignas(16) static uint16_t w80[9216];
  alignas(16) static uint16_t w81[1024];
  alignas(16) static uint16_t w82[1048576];
  alignas(16) static uint16_t w83[1024];
  alignas(16) static uint16_t w84[1025024];
  alignas(16) static uint16_t w85[1001];

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);
  std::generate(v0, v0 + 150528, std::ref(f16rng));
  std::generate(v1, v1 + 401408, std::ref(f16rng));
  std::generate(v2, v2 + 401408, std::ref(f16rng));
  std::generate(v3, v3 + 802816, std::ref(f16rng));
  std::generate(v4, v4 + 200704, std::ref(f16rng));
  std::generate(v5, v5 + 401408, std::ref(f16rng));
  std::generate(v6, v6 + 401408, std::ref(f16rng));
  std::generate(v7, v7 + 401408, std::ref(f16rng));
  std::generate(v8, v8 + 100352, std::ref(f16rng));
  std::generate(v9, v9 + 200704, std::ref(f16rng));
  std::generate(v10, v10 + 200704, std::ref(f16rng));
  std::generate(v11, v11 + 200704, std::ref(f16rng));
  std::generate(v12, v12 + 50176, std::ref(f16rng));
  std::generate(v13, v13 + 100352, std::ref(f16rng));
  std::generate(v14, v14 + 100352, std::ref(f16rng));
  std::generate(v15, v15 + 100352, std::ref(f16rng));
  std::generate(v16, v16 + 100352, std::ref(f16rng));
  std::generate(v17, v17 + 100352, std::ref(f16rng));
  std::generate(v18, v18 + 100352, std::ref(f16rng));
  std::generate(v19, v19 + 100352, std::ref(f16rng));
  std::generate(v20, v20 + 100352, std::ref(f16rng));
  std::generate(v21, v21 + 100352, std::ref(f16rng));
  std::generate(v22, v22 + 100352, std::ref(f16rng));
  std::generate(v23, v23 + 100352, std::ref(f16rng));
  std::generate(v24, v24 + 25088, std::ref(f16rng));
  std::generate(v25, v25 + 50176, std::ref(f16rng));
  std::generate(v26, v26 + 50176, std::ref(f16rng));
  std::generate(v27, v27 + 50176, std::ref(f16rng));
  std::generate(v28, v28 + 1024, std::ref(f16rng));
  std::generate(v29, v29 + 1001, std::ref(f16rng));
  std::generate(w30, w30 + 864, std::ref(f16rng));
  std::generate(w31, w31 + 32, std::ref(f16rng));
  std::generate(w32, w32 + 288, std::ref(f16rng));
  std::generate(w33, w33 + 32, std::ref(f16rng));
  std::generate(w34, w34 + 2048, std::ref(f16rng));
  std::generate(w35, w35 + 64, std::ref(f16rng));
  std::generate(w36, w36 + 576, std::ref(f16rng));
  std::generate(w37, w37 + 64, std::ref(f16rng));
  std::generate(w38, w38 + 8192, std::ref(f16rng));
  std::generate(w39, w39 + 128, std::ref(f16rng));
  std::generate(w40, w40 + 1152, std::ref(f16rng));
  std::generate(w41, w41 + 128, std::ref(f16rng));
  std::generate(w42, w42 + 16384, std::ref(f16rng));
  std::generate(w43, w43 + 128, std::ref(f16rng));
  std::generate(w44, w44 + 1152, std::ref(f16rng));
  std::generate(w45, w45 + 128, std::ref(f16rng));
  std::generate(w46, w46 + 32768, std::ref(f16rng));
  std::generate(w47, w47 + 256, std::ref(f16rng));
  std::generate(w48, w48 + 2304, std::ref(f16rng));
  std::generate(w49, w49 + 256, std::ref(f16rng));
  std::generate(w50, w50 + 65536, std::ref(f16rng));
  std::generate(w51, w51 + 256, std::ref(f16rng));
  std::generate(w52, w52 + 2304, std::ref(f16rng));
  std::generate(w53, w53 + 256, std::ref(f16rng));
  std::generate(w54, w54 + 131072, std::ref(f16rng));
  std::generate(w55, w55 + 512, std::ref(f16rng));
  std::generate(w56, w56 + 4608, std::ref(f16rng));
  std::generate(w57, w57 + 512, std::ref(f16rng));
  std::generate(w58, w58 + 262144, std::ref(f16rng));
  std::generate(w59, w59 + 512, std::ref(f16rng));
  std::generate(w60, w60 + 4608, std::ref(f16rng));
  std::generate(w61, w61 + 512, std::ref(f16rng));
  std::generate(w62, w62 + 262144, std::ref(f16rng));
  std::generate(w63, w63 + 512, std::ref(f16rng));
  std::generate(w64, w64 + 4608, std::ref(f16rng));
  std::generate(w65, w65 + 512, std::ref(f16rng));
  std::generate(w66, w66 + 262144, std::ref(f16rng));
  std::generate(w67, w67 + 512, std::ref(f16rng));
  std::generate(w68, w68 + 4608, std::ref(f16rng));
  std::generate(w69, w69 + 512, std::ref(f16rng));
  std::generate(w70, w70 + 262144, std::ref(f16rng));
  std::generate(w71, w71 + 512, std::ref(f16rng));
  std::generate(w72, w72 + 4608, std::ref(f16rng));
  std::generate(w73, w73 + 512, std::ref(f16rng));
  std::generate(w74, w74 + 262144, std::ref(f16rng));
  std::generate(w75, w75 + 512, std::ref(f16rng));
  std::generate(w76, w76 + 4608, std::ref(f16rng));
  std::generate(w77, w77 + 512, std::ref(f16rng));
  std::generate(w78, w78 + 524288, std::ref(f16rng));
  std::generate(w79, w79 + 1024, std::ref(f16rng));
  std::generate(w80, w80 + 9216, std::ref(f16rng));
  std::generate(w81, w81 + 1024, std::ref(f16rng));
  std::generate(w82, w82 + 1048576, std::ref(f16rng));
  std::generate(w83, w83 + 1024, std::ref(f16rng));
  std::generate(w84, w84 + 1025024, std::ref(f16rng));
  std::generate(w85, w85 + 1001, std::ref(f16rng));

  ExecutionPlan operators;
  xnn_status status;

  xnn_operator_t op0 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    3 /* input channels per group */,
    32 /* output_channels_per_group */,
    3 /* input pixel stride */,
    32 /* output pixel stride */,
    w30, w31,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op1 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    32 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    32 /* input pixel stride */,
    32 /* output pixel stride */,
    w32, w33,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #1" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op1, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    32 /* input channels per group */,
    64 /* output_channels_per_group */,
    32 /* input pixel stride */,
    64 /* output pixel stride */,
    w34, w35,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    64 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    64 /* input pixel stride */,
    64 /* output pixel stride */,
    w36, w37,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    128 /* output_channels_per_group */,
    64 /* input pixel stride */,
    128 /* output pixel stride */,
    w38, w39,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #4" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op4, xnn_delete_operator);

  xnn_operator_t op5 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    128 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    128 /* input pixel stride */,
    128 /* output pixel stride */,
    w40, w41,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    128 /* input channels per group */,
    128 /* output_channels_per_group */,
    128 /* input pixel stride */,
    128 /* output pixel stride */,
    w42, w43,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #6" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op6, xnn_delete_operator);

  xnn_operator_t op7 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    128 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    128 /* input pixel stride */,
    128 /* output pixel stride */,
    w44, w45,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #7" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op7, xnn_delete_operator);

  xnn_operator_t op8 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    128 /* input channels per group */,
    256 /* output_channels_per_group */,
    128 /* input pixel stride */,
    256 /* output pixel stride */,
    w46, w47,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    256 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    256 /* input pixel stride */,
    256 /* output pixel stride */,
    w48, w49,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  xnn_operator_t op10 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    256 /* input channels per group */,
    256 /* output_channels_per_group */,
    256 /* input pixel stride */,
    256 /* output pixel stride */,
    w50, w51,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    256 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    256 /* input pixel stride */,
    256 /* output pixel stride */,
    w52, w53,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    256 /* input channels per group */,
    512 /* output_channels_per_group */,
    256 /* input pixel stride */,
    512 /* output pixel stride */,
    w54, w55,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #12" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op12, xnn_delete_operator);

  xnn_operator_t op13 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    w56, w57,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    512 /* input channels per group */,
    512 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    w58, w59,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #14" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op14, xnn_delete_operator);

  xnn_operator_t op15 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    w60, w61,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    512 /* input channels per group */,
    512 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    w62, w63,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    w64, w65,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    512 /* input channels per group */,
    512 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    w66, w67,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    w68, w69,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    512 /* input channels per group */,
    512 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    w70, w71,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    w72, w73,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    512 /* input channels per group */,
    512 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    w74, w75,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #22" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op22, xnn_delete_operator);

  xnn_operator_t op23 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    w76, w77,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #23" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op23, xnn_delete_operator);

  xnn_operator_t op24 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    512 /* input channels per group */,
    1024 /* output_channels_per_group */,
    512 /* input pixel stride */,
    1024 /* output pixel stride */,
    w78, w79,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1024 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    1024 /* input pixel stride */,
    1024 /* output pixel stride */,
    w80, w81,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    1024 /* input channels per group */,
    1024 /* output_channels_per_group */,
    1024 /* input pixel stride */,
    1024 /* output pixel stride */,
    w82, w83,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f16(
    1024 /* channels */, 1024 /* input stride */, 1024 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    1024 /* input channels per group */,
    1001 /* output_channels_per_group */,
    1024 /* input pixel stride */,
    1001 /* output pixel stride */,
    w84, w85,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);



  status = xnn_setup_convolution2d_nhwc_f16(
    op0,
    1 /* batch size */, 224 /* input height */, 224 /* input width */,
    v0 /* input */, v1 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op1,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v1 /* input */, v2 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op2,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v2 /* input */, v3 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op3,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v3 /* input */, v4 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op4,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v4 /* input */, v5 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op5,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v5 /* input */, v6 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op6,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v6 /* input */, v7 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op7,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v7 /* input */, v8 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op8,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v8 /* input */, v9 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op9,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v9 /* input */, v10 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op10,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v10 /* input */, v11 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op11,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v11 /* input */, v12 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op12,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v12 /* input */, v13 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op13,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v13 /* input */, v14 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op14,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v14 /* input */, v15 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op15,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v15 /* input */, v16 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op16,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v16 /* input */, v17 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op17,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v17 /* input */, v18 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op18,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v18 /* input */, v19 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op19,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v19 /* input */, v20 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op20,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v20 /* input */, v21 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op21,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v21 /* input */, v22 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op22,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v22 /* input */, v23 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op23,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v23 /* input */, v24 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op24,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v24 /* input */, v25 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op25,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v25 /* input */, v26 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op26,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v26 /* input */, v27 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f16(
    op27,
    1 /* batch size */, 49 /* width */,
    v27 /* input */, v28 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op28,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v28 /* input */, v29 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wpessimizing-move"
  return operators;
  #pragma clang diagnostic pop
}

}  // namespace models
