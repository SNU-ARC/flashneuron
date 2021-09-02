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

#define DECLARE_Q8VADD_UKERNEL_FUNCTION(fn_name)                      \
  QNNP_INTERNAL void fn_name(                                         \
      size_t n,                                                       \
      const uint8_t* a,                                               \
      const uint8_t* b,                                               \
      uint8_t* y,                                                     \
      const union qnnp_add_quantization_params* quantization_params);


DECLARE_Q8VADD_UKERNEL_FUNCTION(q8vadd_ukernel__neon)
DECLARE_Q8VADD_UKERNEL_FUNCTION(q8vadd_ukernel__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
