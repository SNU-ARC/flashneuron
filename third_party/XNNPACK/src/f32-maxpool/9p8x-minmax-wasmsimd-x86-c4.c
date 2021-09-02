// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/maxpool.h>


void xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const v128_t voutput_max = wasm_v32x4_load_splat(&params->scalar.max);
  const v128_t voutput_min = wasm_v32x4_load_splat(&params->scalar.min);
  do {
    float* o = output;
    {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      const float* i8 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
      if (kernel_elements < 2) {
        i1 = i0;
      }
      if (kernel_elements <= 2) {
        i2 = i0;
      }
      if (kernel_elements < 4) {
        i3 = i0;
      }
      if (kernel_elements <= 4) {
        i4 = i0;
      }
      if (kernel_elements < 6) {
        i5 = i0;
      }
      if (kernel_elements <= 6) {
        i6 = i0;
      }
      if (kernel_elements < 8) {
        i7 = i0;
      }
      if (kernel_elements <= 8) {
        i8 = i0;
      }

      size_t c = channels;
      for (; c >= 4; c -= 4) {
        const v128_t vi0 = wasm_v128_load(i0);
        i0 += 4;
        const v128_t vi1 = wasm_v128_load(i1);
        i1 += 4;
        const v128_t vi2 = wasm_v128_load(i2);
        i2 += 4;
        const v128_t vi3 = wasm_v128_load(i3);
        i3 += 4;
        const v128_t vi4 = wasm_v128_load(i4);
        i4 += 4;
        const v128_t vi5 = wasm_v128_load(i5);
        i5 += 4;
        const v128_t vi6 = wasm_v128_load(i6);
        i6 += 4;
        const v128_t vi7 = wasm_v128_load(i7);
        i7 += 4;
        const v128_t vi8 = wasm_v128_load(i8);
        i8 += 4;

        const v128_t vmax01 = wasm_v128_bitselect(vi1, vi0, wasm_f32x4_lt(vi0, vi1));
        const v128_t vmax23 = wasm_v128_bitselect(vi3, vi2, wasm_f32x4_lt(vi2, vi3));
        const v128_t vmax45 = wasm_v128_bitselect(vi5, vi4, wasm_f32x4_lt(vi4, vi5));
        const v128_t vmax018 = wasm_v128_bitselect(vi8, vmax01, wasm_f32x4_lt(vmax01, vi8));
        const v128_t vmax67 = wasm_v128_bitselect(vi7, vi6, wasm_f32x4_lt(vi6, vi7));

        const v128_t vmax2345 = wasm_v128_bitselect(vmax45, vmax23, wasm_f32x4_lt(vmax23, vmax45));
        const v128_t vmax01678 = wasm_v128_bitselect(vmax67, vmax018, wasm_f32x4_lt(vmax018, vmax67));
        const v128_t vmax = wasm_v128_bitselect(vmax2345, vmax01678, wasm_f32x4_lt(vmax01678, vmax2345));

        const v128_t vmaskmin = wasm_f32x4_lt(vmax, voutput_min);
        const v128_t vmaskmax = wasm_f32x4_le(vmax, voutput_max);

        v128_t vout = wasm_v128_bitselect(voutput_min, vmax, vmaskmin);
        vout = wasm_v128_bitselect(vout, voutput_max, vmaskmax);

        wasm_v128_store(o, vout);
        o += 4;
      }
      if (c != 0) {
        const v128_t vi0 = wasm_v128_load(i0);
        i0 += 4;
        const v128_t vi1 = wasm_v128_load(i1);
        i1 += 4;
        const v128_t vi2 = wasm_v128_load(i2);
        i2 += 4;
        const v128_t vi3 = wasm_v128_load(i3);
        i3 += 4;
        const v128_t vi4 = wasm_v128_load(i4);
        i4 += 4;
        const v128_t vi5 = wasm_v128_load(i5);
        i5 += 4;
        const v128_t vi6 = wasm_v128_load(i6);
        i6 += 4;
        const v128_t vi7 = wasm_v128_load(i7);
        i7 += 4;
        const v128_t vi8 = wasm_v128_load(i8);
        i8 += 4;

        const v128_t vmax01 = wasm_v128_bitselect(vi1, vi0, wasm_f32x4_lt(vi0, vi1));
        const v128_t vmax23 = wasm_v128_bitselect(vi3, vi2, wasm_f32x4_lt(vi2, vi3));
        const v128_t vmax45 = wasm_v128_bitselect(vi5, vi4, wasm_f32x4_lt(vi4, vi5));
        const v128_t vmax018 = wasm_v128_bitselect(vi8, vmax01, wasm_f32x4_lt(vmax01, vi8));
        const v128_t vmax67 = wasm_v128_bitselect(vi7, vi6, wasm_f32x4_lt(vi6, vi7));

        const v128_t vmax2345 = wasm_v128_bitselect(vmax45, vmax23, wasm_f32x4_lt(vmax23, vmax45));
        const v128_t vmax01678 = wasm_v128_bitselect(vmax67, vmax018, wasm_f32x4_lt(vmax018, vmax67));
        const v128_t vmax = wasm_v128_bitselect(vmax2345, vmax01678, wasm_f32x4_lt(vmax01678, vmax2345));

        const v128_t vmaskmin = wasm_f32x4_lt(vmax, voutput_min);
        const v128_t vmaskmax = wasm_f32x4_le(vmax, voutput_max);

        v128_t vout = wasm_v128_bitselect(voutput_min, vmax, vmaskmin);
        vout = wasm_v128_bitselect(vout, voutput_max, vmaskmax);

        if (c & 2) {
          *((double*) o) = wasm_f64x2_extract_lane(vout, 0);
          vout = wasm_v32x4_shuffle(vout, vout, 2, 3, 2, 3);
          o += 2;
        }
        if (c & 1) {
          *o++ = wasm_f32x4_extract_lane(vout, 0);
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k < 8) {
        i7 = i0;
      }

      o = output;
      size_t c = channels;
      for (; c >= 4; c -= 4) {
        const v128_t vi0 = wasm_v128_load(i0);
        i0 += 4;
        const v128_t vi1 = wasm_v128_load(i1);
        i1 += 4;
        const v128_t vi2 = wasm_v128_load(i2);
        i2 += 4;
        const v128_t vi3 = wasm_v128_load(i3);
        i3 += 4;
        const v128_t vi4 = wasm_v128_load(i4);
        i4 += 4;
        const v128_t vi5 = wasm_v128_load(i5);
        i5 += 4;
        const v128_t vi6 = wasm_v128_load(i6);
        i6 += 4;
        const v128_t vi7 = wasm_v128_load(i7);
        i7 += 4;
        const v128_t vo = wasm_v128_load(o);

        const v128_t vmax01 = wasm_v128_bitselect(vi1, vi0, wasm_f32x4_lt(vi0, vi1));
        const v128_t vmax23 = wasm_v128_bitselect(vi3, vi2, wasm_f32x4_lt(vi2, vi3));
        const v128_t vmax45 = wasm_v128_bitselect(vi5, vi4, wasm_f32x4_lt(vi4, vi5));
        const v128_t vmax01o = wasm_v128_bitselect(vo, vmax01, wasm_f32x4_lt(vmax01, vo));
        const v128_t vmax67 = wasm_v128_bitselect(vi7, vi6, wasm_f32x4_lt(vi6, vi7));

        const v128_t vmax2345 = wasm_v128_bitselect(vmax45, vmax23, wasm_f32x4_lt(vmax23, vmax45));
        const v128_t vmax0167 = wasm_v128_bitselect(vmax67, vmax01o, wasm_f32x4_lt(vmax01o, vmax67));
        const v128_t vmax = wasm_v128_bitselect(vmax2345, vmax0167, wasm_f32x4_lt(vmax0167, vmax2345));

        const v128_t vmaskmin = wasm_f32x4_lt(vmax, voutput_min);
        const v128_t vmaskmax = wasm_f32x4_le(vmax, voutput_max);

        v128_t vout = wasm_v128_bitselect(voutput_min, vmax, vmaskmin);
        vout = wasm_v128_bitselect(vout, voutput_max, vmaskmax);

        wasm_v128_store(o, vout);
        o += 4;
      }
      if (c != 0) {
        const v128_t vi0 = wasm_v128_load(i0);
        const v128_t vi1 = wasm_v128_load(i1);
        const v128_t vi2 = wasm_v128_load(i2);
        const v128_t vi3 = wasm_v128_load(i3);
        const v128_t vi4 = wasm_v128_load(i4);
        const v128_t vi5 = wasm_v128_load(i5);
        const v128_t vi6 = wasm_v128_load(i6);
        const v128_t vi7 = wasm_v128_load(i7);
        const v128_t vo = wasm_v128_load(o);

        const v128_t vmax01 = wasm_v128_bitselect(vi1, vi0, wasm_f32x4_lt(vi0, vi1));
        const v128_t vmax23 = wasm_v128_bitselect(vi3, vi2, wasm_f32x4_lt(vi2, vi3));
        const v128_t vmax45 = wasm_v128_bitselect(vi5, vi4, wasm_f32x4_lt(vi4, vi5));
        const v128_t vmax01o = wasm_v128_bitselect(vo, vmax01, wasm_f32x4_lt(vmax01, vo));
        const v128_t vmax67 = wasm_v128_bitselect(vi7, vi6, wasm_f32x4_lt(vi6, vi7));

        const v128_t vmax2345 = wasm_v128_bitselect(vmax45, vmax23, wasm_f32x4_lt(vmax23, vmax45));
        const v128_t vmax0167 = wasm_v128_bitselect(vmax67, vmax01o, wasm_f32x4_lt(vmax01o, vmax67));
        const v128_t vmax = wasm_v128_bitselect(vmax2345, vmax0167, wasm_f32x4_lt(vmax0167, vmax2345));

        const v128_t vmaskmin = wasm_f32x4_lt(vmax, voutput_min);
        const v128_t vmaskmax = wasm_f32x4_le(vmax, voutput_max);

        v128_t vout = wasm_v128_bitselect(voutput_min, vmax, vmaskmin);
        vout = wasm_v128_bitselect(vout, voutput_max, vmaskmax);

        if (c & 2) {
          *((double*) o) = wasm_f64x2_extract_lane(vout, 0);
          vout = wasm_v32x4_shuffle(vout, vout, 2, 3, 2, 3);
          o += 2;
        }
        if (c & 1) {
          *o++ = wasm_f32x4_extract_lane(vout, 0);
        }
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}
