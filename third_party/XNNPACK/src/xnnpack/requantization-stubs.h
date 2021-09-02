// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdint.h>
#include <stddef.h>

#include <xnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef void (*xnn_qu8_requantization_function)(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output);

#define DECLARE_QU8_REQUANTIZATION_FUNCTION(fn_name) \
    void fn_name(                                    \
        size_t n,                                    \
        const int32_t* input,                        \
        float scale,                                 \
        uint8_t zero_point,                          \
        uint8_t qmin,                                \
        uint8_t qmax,                                \
        uint8_t* output);

DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_precise__scalar_unsigned32)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_precise__scalar_unsigned64)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_precise__scalar_signed64)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_precise__sse2)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_precise__ssse3)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_precise__sse4)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_precise__neon)

DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_fp32__scalar_lrintf)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_fp32__scalar_magic)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_fp32__sse2)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_fp32__neon)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_fp32__wasmsimd)

DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_q31__scalar)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_q31__sse2)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_q31__ssse3)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_q31__sse4)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_q31__neon)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_q31__wasmsimd)


typedef void (*xnn_qs8_requantization_function)(
    size_t n,
    const int32_t* input,
    float scale,
    int8_t zero_point,
    int8_t qmin,
    int8_t qmax,
    int8_t* output);

#define DECLARE_QS8_REQUANTIZATION_FUNCTION(fn_name) \
    void fn_name(                                    \
        size_t n,                                    \
        const int32_t* input,                        \
        float scale,                                 \
        int8_t zero_point,                           \
        int8_t qmin,                                 \
        int8_t qmax,                                 \
        int8_t* output);

DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_precise__scalar_unsigned32)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_precise__scalar_unsigned64)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_precise__scalar_signed64)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_precise__sse2)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_precise__ssse3)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_precise__sse4)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_precise__neon)

DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_fp32__scalar_lrintf)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_fp32__scalar_magic)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_fp32__sse2)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_fp32__sse4)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_fp32__neon)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_fp32__wasmsimd)

DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_q31__scalar)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_q31__sse2)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_q31__ssse3)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_q31__sse4)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_q31__neon)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_q31__wasmsimd)


#ifdef __cplusplus
}  // extern "C"
#endif
