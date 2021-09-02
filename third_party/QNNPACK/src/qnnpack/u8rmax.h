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

#define DECLARE_U8RMAX_UKERNEL_FUNCTION(fn_name) \
  QNNP_INTERNAL uint8_t fn_name(                 \
      size_t n,                                  \
      const uint8_t* x);

DECLARE_U8RMAX_UKERNEL_FUNCTION(u8rmax_ukernel__neon)
DECLARE_U8RMAX_UKERNEL_FUNCTION(u8rmax_ukernel__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
