// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-dwconv-minmax.yaml
//   Generator: tools/generate-dwconv-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/dwconv.h>
#include "dwconv-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__neon_mul16);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__neon_mul16);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, c_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(24)
      .kr(9)
      .channels(24)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, c_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, c_div_24_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, c_div_24_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, c_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 24; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, c_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, c_gt_24_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, c_gt_24_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(24)
        .width(5)
        .output_stride(127)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .input_offset(464)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 48; channels < 384; channels += 72) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .input_offset(464)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__neon_mul16);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, c_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(9)
      .channels(32)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, c_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, c_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, c_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__neon_mul16);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, c_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, c_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, c_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, c_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, multipixel) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE2_MUL16, zero) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse2_mul16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, c_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, c_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, c_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, c_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, multipixel) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE2_MUL16, zero) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse2_mul16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, c_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    DWConvMicrokernelTester()
      .cr(24)
      .kr(9)
      .channels(24)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, c_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, c_div_24_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, c_div_24_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, c_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 1; channels < 24; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, c_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, c_gt_24_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, c_gt_24_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, multipixel) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(24)
        .width(5)
        .output_stride(127)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .input_offset(464)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE2_MUL16, zero) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 48; channels < 384; channels += 72) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .input_offset(464)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse2_mul16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, c_eq_8) {
    TEST_REQUIRES_X86_SSSE3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, c_div_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, c_lt_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, c_gt_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, multipixel) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, multipixel_with_step) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, input_offset) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSSE3_MUL16, zero) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__ssse3_mul16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, c_eq_16) {
    TEST_REQUIRES_X86_SSSE3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, c_div_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, c_lt_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, c_gt_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, multipixel) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, multipixel_with_step) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, input_offset) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSSE3_MUL16, zero) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__ssse3_mul16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, c_eq_24) {
    TEST_REQUIRES_X86_SSSE3;
    DWConvMicrokernelTester()
      .cr(24)
      .kr(9)
      .channels(24)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, c_div_24) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, c_div_24_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, c_div_24_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, c_lt_24) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 1; channels < 24; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, c_gt_24) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, c_gt_24_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, c_gt_24_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, multipixel) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, multipixel_with_step) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(24)
        .width(5)
        .output_stride(127)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, input_offset) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .input_offset(464)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSSE3_MUL16, zero) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 48; channels < 384; channels += 72) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .input_offset(464)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__ssse3_mul16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, c_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, c_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, c_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, c_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, multipixel) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, input_offset) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__SSE41_MUL16, zero) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__sse41_mul16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, c_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, c_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, c_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, c_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, multipixel) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, input_offset) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__SSE41_MUL16, zero) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__sse41_mul16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, c_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    DWConvMicrokernelTester()
      .cr(24)
      .kr(9)
      .channels(24)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, c_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, c_div_24_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, c_div_24_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, c_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 1; channels < 24; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, c_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, c_gt_24_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, c_gt_24_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, multipixel) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(24)
        .width(5)
        .output_stride(127)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, input_offset) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .input_offset(464)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__SSE41_MUL16, zero) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 48; channels < 384; channels += 72) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .input_offset(464)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__sse41_mul16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, c_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, c_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, c_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, c_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, multipixel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, input_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL16, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, c_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(9)
      .channels(32)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, c_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, c_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, c_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, multipixel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, input_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL16, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, c_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, c_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, c_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, c_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, multipixel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, input_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__AVX2_MUL32, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__avx2_mul32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, c_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, c_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, c_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, c_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, multipixel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, input_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX2_MUL32, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx2_mul32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, c_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    DWConvMicrokernelTester()
      .cr(24)
      .kr(9)
      .channels(24)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, c_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, c_div_24_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, c_div_24_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, c_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 1; channels < 24; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, c_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, c_gt_24_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, c_gt_24_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, multipixel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(24)
        .width(5)
        .output_stride(127)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, input_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .input_offset(464)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__AVX2_MUL32, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 48; channels < 384; channels += 72) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .input_offset(464)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__avx2_mul32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, c_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(9)
      .channels(32)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, c_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, c_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, c_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, multipixel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, input_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX2_MUL32, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx2_mul32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, c_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, c_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, c_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, c_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, multipixel) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, input_offset) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__AVX512SKX_MUL32, zero) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__avx512skx_mul32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, c_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(9)
      .channels(32)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, c_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, c_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, c_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, multipixel) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, input_offset) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP32X9__AVX512SKX_MUL32, zero) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up32x9__avx512skx_mul32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP8X9__WASMSIMD_MUL16, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up8x9__wasmsimd_mul16);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, c_eq_16) {
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, c_div_16) {
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, c_div_16_with_qmin) {
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, c_div_16_with_qmax) {
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, c_lt_16) {
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, c_gt_16) {
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, c_gt_16_with_qmin) {
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, c_gt_16_with_qmax) {
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, multipixel_with_step) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, input_offset) {
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP16X9__WASMSIMD_MUL16, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up16x9__wasmsimd_mul16);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, c_eq_24) {
    DWConvMicrokernelTester()
      .cr(24)
      .kr(9)
      .channels(24)
      .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, c_div_24) {
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, c_div_24_with_qmin) {
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, c_div_24_with_qmax) {
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, c_lt_24) {
    for (uint32_t channels = 1; channels < 24; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, c_gt_24) {
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, c_gt_24_with_qmin) {
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, c_gt_24_with_qmax) {
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, multipixel) {
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, multipixel_with_step) {
    for (size_t channels = 1; channels <= 120; channels += 23) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(24)
        .width(5)
        .output_stride(127)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, input_offset) {
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .input_offset(464)
        .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
    }
  }

  TEST(QS8_DWCONV_MINMAX_UP24X9__WASMSIMD_MUL16, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 48; channels < 384; channels += 72) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .input_offset(464)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_ukernel_up24x9__wasmsimd_mul16);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD
