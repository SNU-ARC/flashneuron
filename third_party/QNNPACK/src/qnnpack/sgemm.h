/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>

#include <qnnpack/params.h>
#include <qnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_SGEMM_UKERNEL_FUNCTION(fn_name)                 \
  QNNP_INTERNAL void fn_name(                                   \
      size_t mr,                                                \
      size_t nr,                                                \
      size_t k,                                                 \
      const float* a,                                           \
      size_t a_stride,                                          \
      const float* w,                                           \
      float* c,                                                 \
      size_t c_stride,                                          \
      const struct qnnp_fp32_clamping_params* clamping_params);

DECLARE_SGEMM_UKERNEL_FUNCTION(sgemm_ukernel_5x8__neon)
DECLARE_SGEMM_UKERNEL_FUNCTION(sgemm_ukernel_6x8__neon)

DECLARE_SGEMM_UKERNEL_FUNCTION(sgemm_ukernel_6x8__psimd)

#ifdef __cplusplus
} /* extern "C" */
#endif
