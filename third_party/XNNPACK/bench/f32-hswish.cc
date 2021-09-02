// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/hswish.h>
#include <xnnpack/params.h>
#include <xnnpack/params-init.h>


static void f32_hswish(
  benchmark::State& state,
  xnn_f32_hswish_ukernel_function hswish,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);
  std::vector<float, AlignedAllocator<float, 64>> input(elements);
  std::vector<float, AlignedAllocator<float, 64>> output(elements);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::fill(output.begin(), output.end(), std::nanf(""));

  const union xnn_f32_hswish_params params = xnn_init_f32_hswish_params();
  for (auto _ : state) {
    hswish(elements * sizeof(float), input.data(), output.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_hswish, neon_x4, xnn_f32_hswish_ukernel__neon_x4, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_hswish, neon_x8, xnn_f32_hswish_ukernel__neon_x8, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_hswish, neon_x16, xnn_f32_hswish_ukernel__neon_x16, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_hswish, sse_x4, xnn_f32_hswish_ukernel__sse_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_hswish, sse_x8, xnn_f32_hswish_ukernel__sse_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_hswish, avx_x8, xnn_f32_hswish_ukernel__avx_x8, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_hswish, avx_x16, xnn_f32_hswish_ukernel__avx_x16, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_hswish, fma3_x8, xnn_f32_hswish_ukernel__fma3_x8, benchmark::utils::CheckFMA3)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_hswish, fma3_x16, xnn_f32_hswish_ukernel__fma3_x16, benchmark::utils::CheckFMA3)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_hswish, avx512f_x16, xnn_f32_hswish_ukernel__avx512f_x16, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_hswish, avx512f_x32, xnn_f32_hswish_ukernel__avx512f_x32, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD
  BENCHMARK_CAPTURE(f32_hswish, wasmsimd_x4, xnn_f32_hswish_ukernel__wasmsimd_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_hswish, wasmsimd_x8, xnn_f32_hswish_ukernel__wasmsimd_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_hswish, wasmsimd_x16, xnn_f32_hswish_ukernel__wasmsimd_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  BENCHMARK_CAPTURE(f32_hswish, wasm_x1, xnn_f32_hswish_ukernel__wasm_x1)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_hswish, wasm_x2, xnn_f32_hswish_ukernel__wasm_x2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_hswish, wasm_x4, xnn_f32_hswish_ukernel__wasm_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD

BENCHMARK_CAPTURE(f32_hswish, scalar_x1, xnn_f32_hswish_ukernel__scalar_x1)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_hswish, scalar_x2, xnn_f32_hswish_ukernel__scalar_x2)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_hswish, scalar_x4, xnn_f32_hswish_ukernel__scalar_x4)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
