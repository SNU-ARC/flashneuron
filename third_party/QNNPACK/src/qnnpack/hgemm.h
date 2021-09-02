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

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_HGEMM_UKERNEL_FUNCTION(fn_name)                 \
  void fn_name(                                                 \
      size_t mr,                                                \
      size_t nr,                                                \
      size_t k,                                                 \
      const void* a,                                            \
      size_t a_stride,                                          \
      const void* w,                                            \
      void* c,                                                  \
      size_t c_stride,                                          \
      const struct qnnp_fp16_clamping_params* clamping_params);

DECLARE_HGEMM_UKERNEL_FUNCTION(hgemm_ukernel_8x8__neonfp16arith)
DECLARE_HGEMM_UKERNEL_FUNCTION(hgemm_ukernel_8x8__aarch32_neonfp16arith)

#ifdef __cplusplus
} /* extern "C" */
#endif
