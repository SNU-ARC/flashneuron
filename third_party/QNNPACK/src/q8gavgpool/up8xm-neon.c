/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <arm_neon.h>

#include <qnnpack/q8gavgpool.h>


void q8gavgpool_ukernel_up8xm__neon(
    size_t m,
    size_t n,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union qnnp_avgpool_quantization_params quantization_params[restrict static 1])
{
  assert(m >= 1);
  assert(n < 8);

  const int32x4_t vbias = vld1q_dup_s32(&quantization_params->neon.bias);
  int32x4_t vacc_lo = vbias;
  int32x4_t vacc_hi = vbias;
  while (m >= 8) {
    const uint8x8_t vinput = vld1_u8(input); input += input_stride;
    const int16x8_t vxinput = vreinterpretq_s16_u16(vmovl_u8(vinput));
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vxinput));
    vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vxinput));

    m--;
  }
  while (m-- != 0) {
    input += n;
    uint8x8_t vinput = vmov_n_u8(0);
    if (n & 1) {
      input -= 1;
      vinput = vld1_lane_u8(input, vinput, 0);
    }
    if (n & 2) {
      vinput = vext_u8(vinput, vinput, 6);
      input -= 2;
      vinput = vreinterpret_u8_u16(vld1_lane_u16(__builtin_assume_aligned(input, 1), vreinterpret_u16_u8(vinput), 0));
    }
    if (n & 4) {
      vinput = vext_u8(vinput, vinput, 4);
      input -= 4;
      vinput = vreinterpret_u8_u32(vld1_lane_u32(__builtin_assume_aligned(input, 1), vreinterpret_u32_u8(vinput), 0));
    }
    input += input_stride;

    const int16x8_t vxinput = vreinterpretq_s16_u16(vmovl_u8(vinput));
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vxinput));
    vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vxinput));
  }

#ifdef __aarch64__
  const int32x4_t vmultiplier = vld1q_dup_s32(&quantization_params->neon.multiplier);
#else
  const int32x2_t vmultiplier = vld1_dup_s32(&quantization_params->neon.multiplier);
#endif
  const int64x2_t vleft_shift = vld1q_dup_s64(&quantization_params->neon.left_shift);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&quantization_params->neon.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&quantization_params->neon.output_max);

  const int32x4_t vneg_mask_lo = vreinterpretq_s32_u32(vcltq_s32(vacc_lo, vmovq_n_s32(0)));
  const int32x4_t vneg_mask_hi = vreinterpretq_s32_u32(vcltq_s32(vacc_hi, vmovq_n_s32(0)));

#if defined(__aarch64__)
  const int64x2_t vproduct01 = vmull_s32(vget_low_s32(vacc_lo), vget_low_s32(vmultiplier));
  const int64x2_t vproduct23 = vmull_high_s32(vacc_lo, vmultiplier);
  const int64x2_t vproduct45 = vmull_s32(vget_low_s32(vacc_hi), vget_low_s32(vmultiplier));
  const int64x2_t vproduct67 = vmull_high_s32(vacc_hi, vmultiplier);

  const int64x2_t vadjusted_product01 = vaddw_s32(vproduct01, vget_low_s32(vneg_mask_lo));
  const int64x2_t vadjusted_product23 = vaddw_high_s32(vproduct23, vneg_mask_lo);
  const int64x2_t vadjusted_product45 = vaddw_s32(vproduct45, vget_low_s32(vneg_mask_hi));
  const int64x2_t vadjusted_product67 = vaddw_high_s32(vproduct67, vneg_mask_hi);
#else
  const int64x2_t vproduct01 = vmull_s32(vget_low_s32(vacc_lo), vmultiplier);
  const int64x2_t vproduct23 = vmull_s32(vget_high_s32(vacc_lo), vmultiplier);
  const int64x2_t vproduct45 = vmull_s32(vget_low_s32(vacc_hi), vmultiplier);
  const int64x2_t vproduct67 = vmull_s32(vget_high_s32(vacc_hi), vmultiplier);

  const int64x2_t vadjusted_product01 = vaddw_s32(vproduct01, vget_low_s32(vneg_mask_lo));
  const int64x2_t vadjusted_product23 = vaddw_s32(vproduct23, vget_high_s32(vneg_mask_lo));
  const int64x2_t vadjusted_product45 = vaddw_s32(vproduct45, vget_low_s32(vneg_mask_hi));
  const int64x2_t vadjusted_product67 = vaddw_s32(vproduct67, vget_high_s32(vneg_mask_hi));
#endif

  const int64x2_t vscaled_acc01 = vrshlq_s64(vadjusted_product01, vleft_shift);
  const int64x2_t vscaled_acc23 = vrshlq_s64(vadjusted_product23, vleft_shift);
  const int64x2_t vscaled_acc45 = vrshlq_s64(vadjusted_product45, vleft_shift);
  const int64x2_t vscaled_acc67 = vrshlq_s64(vadjusted_product67, vleft_shift);

#ifdef __aarch64__
  vacc_lo = vuzp1q_s32(vreinterpretq_s32_s64(vscaled_acc01), vreinterpretq_s32_s64(vscaled_acc23));
  vacc_hi = vuzp1q_s32(vreinterpretq_s32_s64(vscaled_acc45), vreinterpretq_s32_s64(vscaled_acc67));

  const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
#else
  vacc_lo = vcombine_s32(vmovn_s64(vscaled_acc01), vmovn_s64(vscaled_acc23));
  vacc_hi = vcombine_s32(vmovn_s64(vscaled_acc45), vmovn_s64(vscaled_acc67));

  const int16x8_t vacc = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)), voutput_zero_point);
#endif

  uint8x8_t vout = vqmovun_s16(vacc);
  vout = vmax_u8(vout, voutput_min);
  vout = vmin_u8(vout, voutput_max);

  if (n & 4) {
    vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0); output += 4;
    vout = vext_u8(vout, vout, 4);
  }
  if (n & 2) {
    vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0); output += 2;
    vout = vext_u8(vout, vout, 2);
  }
  if (n & 1) {
    vst1_lane_u8(output, vout, 0);
  }
}
