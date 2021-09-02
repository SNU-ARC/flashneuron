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
#include <xnnpack/vunary.h>
#include <xnnpack/params.h>
#include <xnnpack/params-init.h>


static void f32_elu(
  benchmark::State& state,
  xnn_f32_velu_ukernel_function elu,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-20.0f, 10.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> x(elements);
  std::vector<float, AlignedAllocator<float, 64>> y(elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::fill(y.begin(), y.end(), std::nanf(""));

  const union xnn_f32_elu_params params =
    xnn_init_f32_elu_params(1.0f /* prescale */, 1.0f /* alpha */, 1.0f /* beta */);
  for (auto _ : state) {
    elu(elements * sizeof(float), x.data(), y.data(), &params);
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
  BENCHMARK_CAPTURE(f32_elu, neonfma_lut16_p3_x4, xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x4, benchmark::utils::CheckNEONFMA)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neonfma_lut16_p3_x8, xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x8, benchmark::utils::CheckNEONFMA)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neonfma_lut16_p3_x12, xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x12, benchmark::utils::CheckNEONFMA)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neonfma_lut16_p3_x16, xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x16, benchmark::utils::CheckNEONFMA)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neonfma_lut16_p3_x20, xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x20, benchmark::utils::CheckNEONFMA)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neonfma_lut16_p3_x24, xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x24, benchmark::utils::CheckNEONFMA)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, neonfma_p6_x4, xnn_f32_velu_ukernel__neonfma_rr1_p6_x4, benchmark::utils::CheckNEONFMA)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neonfma_p6_x8, xnn_f32_velu_ukernel__neonfma_rr1_p6_x8, benchmark::utils::CheckNEONFMA)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neonfma_p6_x12, xnn_f32_velu_ukernel__neonfma_rr1_p6_x12, benchmark::utils::CheckNEONFMA)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neonfma_p6_x16, xnn_f32_velu_ukernel__neonfma_rr1_p6_x16, benchmark::utils::CheckNEONFMA)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neonfma_p6_x20, xnn_f32_velu_ukernel__neonfma_rr1_p6_x20, benchmark::utils::CheckNEONFMA)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neonfma_p6_x24, xnn_f32_velu_ukernel__neonfma_rr1_p6_x24, benchmark::utils::CheckNEONFMA)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, neon_lut16_p3_x4, xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x4, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neon_lut16_p3_x8, xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x8, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neon_lut16_p3_x12, xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x12, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neon_lut16_p3_x16, xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x16, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neon_lut16_p3_x20, xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x20, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neon_lut16_p3_x24, xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x24, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, neon_p6_x4, xnn_f32_velu_ukernel__neon_rr2_p6_x4, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neon_p6_x8, xnn_f32_velu_ukernel__neon_rr2_p6_x8, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neon_p6_x12, xnn_f32_velu_ukernel__neon_rr2_p6_x12, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neon_p6_x16, xnn_f32_velu_ukernel__neon_rr2_p6_x16, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neon_p6_x20, xnn_f32_velu_ukernel__neon_rr2_p6_x20, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, neon_p6_x24, xnn_f32_velu_ukernel__neon_rr2_p6_x24, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_elu, avx512f_lut16_p3_x16, xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x16, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_lut16_p3_x32, xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x32, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_lut16_p3_x48, xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x48, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_lut16_p3_x64, xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_lut16_p3_x80, xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x80, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_lut16_p3_x96, xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x96, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_lut16_p3_x112, xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x112, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_lut16_p3_x128, xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x128, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, avx512f_p6_x16, xnn_f32_velu_ukernel__avx512f_rr1_p6_x16, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_p6_x32, xnn_f32_velu_ukernel__avx512f_rr1_p6_x32, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_p6_x48, xnn_f32_velu_ukernel__avx512f_rr1_p6_x48, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_p6_x64, xnn_f32_velu_ukernel__avx512f_rr1_p6_x64, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_p6_x80, xnn_f32_velu_ukernel__avx512f_rr1_p6_x80, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_p6_x96, xnn_f32_velu_ukernel__avx512f_rr1_p6_x96, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_p6_x112, xnn_f32_velu_ukernel__avx512f_rr1_p6_x112, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx512f_p6_x128, xnn_f32_velu_ukernel__avx512f_rr1_p6_x128, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, avx2_lut4_p4_x8, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x8, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut4_p4_x16, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x16, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut4_p4_x24, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x24, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut4_p4_x32, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x32, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut4_p4_x40, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x40, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut4_p4_x48, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x48, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut4_p4_x56, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x56, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut4_p4_x64, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x64, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut4_p4_x72, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x72, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut4_p4_x80, xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x80, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, avx2_lut8_p4_x8, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x8, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut8_p4_x16, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x16, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut8_p4_x24, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x24, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut8_p4_x32, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x32, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut8_p4_x40, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x40, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut8_p4_x48, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x48, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut8_p4_x56, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x56, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut8_p4_x64, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x64, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut8_p4_x72, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x72, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut8_p4_x80, xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x80, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, avx2_lut16_p3_x8, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x8, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut16_p3_x16, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x16, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut16_p3_x24, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x24, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut16_p3_x32, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x32, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut16_p3_x40, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x40, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut16_p3_x48, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x48, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut16_p3_x56, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x56, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut16_p3_x64, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x64, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut16_p3_x72, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x72, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_lut16_p3_x80, xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x80, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, avx2_p6_x8, xnn_f32_velu_ukernel__avx2_rr1_p6_x8, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_p6_x16, xnn_f32_velu_ukernel__avx2_rr1_p6_x16, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_p6_x24, xnn_f32_velu_ukernel__avx2_rr1_p6_x24, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_p6_x32, xnn_f32_velu_ukernel__avx2_rr1_p6_x32, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_p6_x40, xnn_f32_velu_ukernel__avx2_rr1_p6_x40, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_p6_x48, xnn_f32_velu_ukernel__avx2_rr1_p6_x48, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_p6_x56, xnn_f32_velu_ukernel__avx2_rr1_p6_x56, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_p6_x64, xnn_f32_velu_ukernel__avx2_rr1_p6_x64, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_p6_x72, xnn_f32_velu_ukernel__avx2_rr1_p6_x72, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx2_p6_x80, xnn_f32_velu_ukernel__avx2_rr1_p6_x80, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, avx_lut4_p4_x8, xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x8, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_lut4_p4_x16, xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x16, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_lut4_p4_x24, xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x24, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_lut4_p4_x32, xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x32, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_lut4_p4_x40, xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x40, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_lut4_p4_x48, xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x48, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, avx_lut16_p3_x8, xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x8, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_lut16_p3_x16, xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x16, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_lut16_p3_x24, xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x24, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_lut16_p3_x32, xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x32, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_lut16_p3_x40, xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x40, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_lut16_p3_x48, xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x48, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, avx_p6_x8, xnn_f32_velu_ukernel__avx_rr2_p6_x8, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_p6_x16, xnn_f32_velu_ukernel__avx_rr2_p6_x16, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_p6_x24, xnn_f32_velu_ukernel__avx_rr2_p6_x24, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_p6_x32, xnn_f32_velu_ukernel__avx_rr2_p6_x32, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_p6_x40, xnn_f32_velu_ukernel__avx_rr2_p6_x40, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, avx_p6_x48, xnn_f32_velu_ukernel__avx_rr2_p6_x48, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, sse41_lut16_p3_x4, xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x4, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse41_lut16_p3_x8, xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x8, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse41_lut16_p3_x12, xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x12, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse41_lut16_p3_x16, xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x16, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse41_lut16_p3_x20, xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x20, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse41_lut16_p3_x24, xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x24, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, sse41_p6_x4, xnn_f32_velu_ukernel__sse41_rr2_p6_x4, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse41_p6_x8, xnn_f32_velu_ukernel__sse41_rr2_p6_x8, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse41_p6_x12, xnn_f32_velu_ukernel__sse41_rr2_p6_x12, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse41_p6_x16, xnn_f32_velu_ukernel__sse41_rr2_p6_x16, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse41_p6_x20, xnn_f32_velu_ukernel__sse41_rr2_p6_x20, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse41_p6_x24, xnn_f32_velu_ukernel__sse41_rr2_p6_x24, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, sse2_lut16_p3_x4, xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse2_lut16_p3_x8, xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse2_lut16_p3_x12, xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse2_lut16_p3_x16, xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse2_lut16_p3_x20, xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse2_lut16_p3_x24, xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, sse2_p6_x4, xnn_f32_velu_ukernel__sse2_rr2_p6_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse2_p6_x8, xnn_f32_velu_ukernel__sse2_rr2_p6_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse2_p6_x12, xnn_f32_velu_ukernel__sse2_rr2_p6_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse2_p6_x16, xnn_f32_velu_ukernel__sse2_rr2_p6_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse2_p6_x20, xnn_f32_velu_ukernel__sse2_rr2_p6_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, sse2_p6_x24, xnn_f32_velu_ukernel__sse2_rr2_p6_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_arm_lut16_p3_x4, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_arm_lut16_p3_x8, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_arm_lut16_p3_x12, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_arm_lut16_p3_x16, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_arm_lut16_p3_x20, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_arm_lut16_p3_x24, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, wasmsimd_x86_lut16_p3_x4, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_x86_lut16_p3_x8, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_x86_lut16_p3_x12, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_x86_lut16_p3_x16, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_x86_lut16_p3_x20, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_x86_lut16_p3_x24, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, wasmsimd_arm_p6_x4, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_arm_p6_x8, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_arm_p6_x12, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_arm_p6_x16, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_arm_p6_x20, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_arm_p6_x24, xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, wasmsimd_x86_p6_x4, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_x86_p6_x8, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_x86_p6_x12, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_x86_p6_x16, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_x86_p6_x20, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasmsimd_x86_p6_x24, xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  BENCHMARK_CAPTURE(f32_elu, wasm_lut16_p3_x1, xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x1)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasm_lut16_p3_x2, xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasm_lut16_p3_x3, xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x3)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasm_lut16_p3_x4, xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasm_lut16_p3_x5, xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x5)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasm_lut16_p3_x6, xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x6)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_elu, wasm_p6_x1, xnn_f32_velu_ukernel__wasm_rr2_p6_x1)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasm_p6_x2, xnn_f32_velu_ukernel__wasm_rr2_p6_x2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasm_p6_x3, xnn_f32_velu_ukernel__wasm_rr2_p6_x3)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasm_p6_x4, xnn_f32_velu_ukernel__wasm_rr2_p6_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasm_p6_x5, xnn_f32_velu_ukernel__wasm_rr2_p6_x5)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_elu, wasm_p6_x6, xnn_f32_velu_ukernel__wasm_rr2_p6_x6)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD

BENCHMARK_CAPTURE(f32_elu, scalar_lut16_p3_x1, xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x1)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_elu, scalar_lut16_p3_x2, xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x2)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_elu, scalar_lut16_p3_x3, xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x3)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_elu, scalar_lut16_p3_x4, xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_elu, scalar_lut16_p3_x5, xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x5)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_elu, scalar_lut16_p3_x6, xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x6)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_elu, scalar_p6_x1, xnn_f32_velu_ukernel__scalar_rr2_p6_x1)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_elu, scalar_p6_x2, xnn_f32_velu_ukernel__scalar_rr2_p6_x2)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_elu, scalar_p6_x3, xnn_f32_velu_ukernel__scalar_rr2_p6_x3)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_elu, scalar_p6_x4, xnn_f32_velu_ukernel__scalar_rr2_p6_x4)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_elu, scalar_p6_x5, xnn_f32_velu_ukernel__scalar_rr2_p6_x5)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_elu, scalar_p6_x6, xnn_f32_velu_ukernel__scalar_rr2_p6_x6)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
