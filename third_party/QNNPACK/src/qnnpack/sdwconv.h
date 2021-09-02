/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/params.h>
#include <qnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_SUPDWCONV_UKERNEL_FUNCTION(fn_name)           \
  QNNP_INTERNAL void fn_name(                                 \
    size_t channels,                                          \
    size_t output_width,                                      \
    const float** input,                                      \
    const float* weights,                                     \
    float* output,                                            \
    size_t input_stride,                                      \
    size_t output_increment,                                  \
    const struct qnnp_fp32_clamping_params* clamping_params);

DECLARE_SUPDWCONV_UKERNEL_FUNCTION(sdwconv_ukernel_up4x9__psimd)

#define DECLARE_SMPDWCONV_UKERNEL_FUNCTION(fn_name)           \
  QNNP_INTERNAL void fn_name(                                 \
    size_t channels,                                          \
    size_t output_width,                                      \
    const uint8_t** input,                                    \
    const void* weights,                                      \
    int32_t* buffer,                                          \
    uint8_t* output,                                          \
    size_t input_stride,                                      \
    size_t output_increment,                                  \
    const struct qnnp_fp32_clamping_params* clamping_params);

#ifdef __cplusplus
} /* extern "C" */
#endif
