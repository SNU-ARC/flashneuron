// Copyright 2019 Google LLC
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
#include <xnnpack/params.h>
#include <xnnpack/vunary.h>


static void f32_sigmoid(
  benchmark::State& state,
  xnn_f32_vunary_ukernel_function sigmoid,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> x(elements);
  std::vector<float, AlignedAllocator<float, 64>> y(elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::fill(y.begin(), y.end(), std::nanf(""));

  for (auto _ : state) {
    sigmoid(elements * sizeof(float), x.data(), y.data(), nullptr /* params */);
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

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_div_x4, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_div_x8, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_div_x12, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_div_x16, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_div_x20, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_div_x24, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_div_x4, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_div_x8, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_div_x12, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_div_x16, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_div_x20, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_div_x24, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_div_x4, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_div_x8, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_div_x12, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_div_x16, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_div_x20, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_div_x24, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr2fma_x4, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr2fma_x8, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr2fma_x12, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr2fma_x16, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr2fma_x20, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr2fma_x24, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr1recps1fma_x4, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr1recps1fma_x8, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr1recps1fma_x12, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr1recps1fma_x16, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr1recps1fma_x20, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr1recps1fma_x24, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr2recps_x4, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr2recps_x8, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr2recps_x12, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr2recps_x16, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr2recps_x20, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_p5_nr2recps_x24, xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_p5_nr2recps_x4, xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_p5_nr2recps_x8, xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_p5_nr2recps_x12, xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_p5_nr2recps_x16, xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_p5_nr2recps_x20, xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_p5_nr2recps_x24, xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr2fma_x4, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr2fma_x8, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr2fma_x12, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr2fma_x16, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr2fma_x20, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr2fma_x24, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr1recps1fma_x4, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr1recps1fma_x8, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr1recps1fma_x12, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr1recps1fma_x16, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr1recps1fma_x20, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr1recps1fma_x24, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr2recps_x4, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr2recps_x8, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr2recps_x12, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr2recps_x16, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr2recps_x20, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut64_p2_nr2recps_x24, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_lut64_p2_nr2recps_x4, xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_lut64_p2_nr2recps_x8, xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_lut64_p2_nr2recps_x12, xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_lut64_p2_nr2recps_x16, xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_lut64_p2_nr2recps_x20, xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_lut64_p2_nr2recps_x24, xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr2fma_x4, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr2fma_x8, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr2fma_x12, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr2fma_x16, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr2fma_x20, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr2fma_x24, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr1recps1fma_x4, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr1recps1fma_x8, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr1recps1fma_x12, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr1recps1fma_x16, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr1recps1fma_x20, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr1recps1fma_x24, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr2recps_x4, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr2recps_x8, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr2recps_x12, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr2recps_x16, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr2recps_x20, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neonfma_rr1_lut2048_p1_nr2recps_x24, xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_lut2048_p1_nr2recps_x4, xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_lut2048_p1_nr2recps_x8, xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_lut2048_p1_nr2recps_x12, xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_lut2048_p1_nr2recps_x16, xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_lut2048_p1_nr2recps_x20, xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, neon_rr2_lut2048_p1_nr2recps_x24, xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_div_x16, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_div_x16, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_div_x32, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_div_x32, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_div_x48, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_div_x48, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_div_x64, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_div_x64, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_div_x80, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_div_x80, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_div_x96, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_div_x96, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_div_x112, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_div_x112, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_div_x128, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_div_x128, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_nr1fma_x16, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x16, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_nr1fma_x32, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x32, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_nr1fma_x48, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x48, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_nr1fma_x64, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x64, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_nr1fma_x80, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x80, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_nr1fma_x96, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x96, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_nr1fma_x112, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x112, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_p5_scalef_nr1fma_x128, xnn_f32_sigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x128, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_div_x16, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x16, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_div_x32, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x32, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_div_x48, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x48, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_div_x64, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x64, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_div_x80, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x80, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_div_x96, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x96, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_div_x112, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x112, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_div_x128, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x128, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_nr1fma_x16, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x16, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_nr1fma_x32, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x32, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_nr1fma_x48, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x48, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_nr1fma_x64, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x64, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_nr1fma_x80, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x80, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_nr1fma_x96, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x96, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_nr1fma_x112, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x112, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut16_p3_perm_scalef_nr1fma_x128, xnn_f32_sigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x128, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_div_x16, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x16, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_div_x32, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x32, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_div_x48, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x48, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_div_x64, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x64, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_div_x80, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x80, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_div_x96, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x96, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_div_x112, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x112, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_div_x128, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x128, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_nr1fma_x16, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x16, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_nr1fma_x32, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x32, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_nr1fma_x48, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x48, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_nr1fma_x64, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x64, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_nr1fma_x80, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x80, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_nr1fma_x96, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x96, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_nr1fma_x112, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x112, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx512f_lut32_p2_perm2_scalef_nr1fma_x128, xnn_f32_sigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x128, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_div_x8, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x8, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_div_x16, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x16, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_div_x24, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x24, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_div_x32, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x32, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_div_x40, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x40, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_div_x48, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x48, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_div_x56, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x56, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_div_x64, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x64, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_div_x72, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x72, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_div_x80, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x80, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr1fma_x8, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x8, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr1fma_x16, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x16, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr1fma_x24, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x24, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr1fma_x32, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x32, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr1fma_x40, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x40, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr1fma_x48, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x48, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr1fma_x56, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x56, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr1fma_x64, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x64, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr1fma_x72, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x72, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr1fma_x80, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x80, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr2fma_x8, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x8, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr2fma_x16, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x16, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr2fma_x24, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x24, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr2fma_x32, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x32, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr2fma_x40, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x40, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr2fma_x48, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x48, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr2fma_x56, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x56, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr2fma_x64, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x64, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr2fma_x72, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x72, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx2_p5_nr2fma_x80, xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x80, benchmark::utils::CheckAVX2)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_div_x8, xnn_f32_sigmoid_ukernel__avx_rr2_p5_div_x8, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_div_x16, xnn_f32_sigmoid_ukernel__avx_rr2_p5_div_x16, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_div_x24, xnn_f32_sigmoid_ukernel__avx_rr2_p5_div_x24, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_div_x32, xnn_f32_sigmoid_ukernel__avx_rr2_p5_div_x32, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_div_x40, xnn_f32_sigmoid_ukernel__avx_rr2_p5_div_x40, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_div_x48, xnn_f32_sigmoid_ukernel__avx_rr2_p5_div_x48, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_div_x56, xnn_f32_sigmoid_ukernel__avx_rr2_p5_div_x56, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_div_x64, xnn_f32_sigmoid_ukernel__avx_rr2_p5_div_x64, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_div_x72, xnn_f32_sigmoid_ukernel__avx_rr2_p5_div_x72, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_div_x80, xnn_f32_sigmoid_ukernel__avx_rr2_p5_div_x80, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_nr2_x8, xnn_f32_sigmoid_ukernel__avx_rr2_p5_nr2_x8, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_nr2_x16, xnn_f32_sigmoid_ukernel__avx_rr2_p5_nr2_x16, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_nr2_x24, xnn_f32_sigmoid_ukernel__avx_rr2_p5_nr2_x24, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_nr2_x32, xnn_f32_sigmoid_ukernel__avx_rr2_p5_nr2_x32, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_nr2_x40, xnn_f32_sigmoid_ukernel__avx_rr2_p5_nr2_x40, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_nr2_x48, xnn_f32_sigmoid_ukernel__avx_rr2_p5_nr2_x48, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_nr2_x56, xnn_f32_sigmoid_ukernel__avx_rr2_p5_nr2_x56, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_nr2_x64, xnn_f32_sigmoid_ukernel__avx_rr2_p5_nr2_x64, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_nr2_x72, xnn_f32_sigmoid_ukernel__avx_rr2_p5_nr2_x72, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, avx_p5_nr2_x80, xnn_f32_sigmoid_ukernel__avx_rr2_p5_nr2_x80, benchmark::utils::CheckAVX)
      ->RangeMultiplier(10)
      ->Range(1000, 1000000)
      ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, sse41_p5_div_x4, xnn_f32_sigmoid_ukernel__sse41_p5_div_x4, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse41_p5_div_x8, xnn_f32_sigmoid_ukernel__sse41_p5_div_x8, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse41_p5_div_x12, xnn_f32_sigmoid_ukernel__sse41_p5_div_x12, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse41_p5_div_x16, xnn_f32_sigmoid_ukernel__sse41_p5_div_x16, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse41_p5_div_x20, xnn_f32_sigmoid_ukernel__sse41_p5_div_x20, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse41_p5_div_x24, xnn_f32_sigmoid_ukernel__sse41_p5_div_x24, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, sse41_lut64_p2_div_x4, xnn_f32_sigmoid_ukernel__sse41_lut64_p2_div_x4, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse41_lut64_p2_div_x8, xnn_f32_sigmoid_ukernel__sse41_lut64_p2_div_x8, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse41_lut64_p2_div_x12, xnn_f32_sigmoid_ukernel__sse41_lut64_p2_div_x12, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse41_lut64_p2_div_x16, xnn_f32_sigmoid_ukernel__sse41_lut64_p2_div_x16, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse41_lut64_p2_div_x20, xnn_f32_sigmoid_ukernel__sse41_lut64_p2_div_x20, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse41_lut64_p2_div_x24, xnn_f32_sigmoid_ukernel__sse41_lut64_p2_div_x24, benchmark::utils::CheckSSE41)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, sse2_p5_div_x4, xnn_f32_sigmoid_ukernel__sse2_p5_div_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse2_p5_div_x8, xnn_f32_sigmoid_ukernel__sse2_p5_div_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse2_p5_div_x12, xnn_f32_sigmoid_ukernel__sse2_p5_div_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse2_p5_div_x16, xnn_f32_sigmoid_ukernel__sse2_p5_div_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse2_p5_div_x20, xnn_f32_sigmoid_ukernel__sse2_p5_div_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse2_p5_div_x24, xnn_f32_sigmoid_ukernel__sse2_p5_div_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, sse2_lut64_p2_div_x4, xnn_f32_sigmoid_ukernel__sse2_lut64_p2_div_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse2_lut64_p2_div_x8, xnn_f32_sigmoid_ukernel__sse2_lut64_p2_div_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse2_lut64_p2_div_x12, xnn_f32_sigmoid_ukernel__sse2_lut64_p2_div_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse2_lut64_p2_div_x16, xnn_f32_sigmoid_ukernel__sse2_lut64_p2_div_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse2_lut64_p2_div_x20, xnn_f32_sigmoid_ukernel__sse2_lut64_p2_div_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, sse2_lut64_p2_div_x24, xnn_f32_sigmoid_ukernel__sse2_lut64_p2_div_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD
  BENCHMARK_CAPTURE(f32_sigmoid, wasmsimd_lut64_p2_div_x4, xnn_f32_sigmoid_ukernel__wasmsimd_lut64_p2_div_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, wasmsimd_lut64_p2_div_x8, xnn_f32_sigmoid_ukernel__wasmsimd_lut64_p2_div_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, wasmsimd_lut64_p2_div_x12, xnn_f32_sigmoid_ukernel__wasmsimd_lut64_p2_div_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, wasmsimd_lut64_p2_div_x16, xnn_f32_sigmoid_ukernel__wasmsimd_lut64_p2_div_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, wasmsimd_lut64_p2_div_x20, xnn_f32_sigmoid_ukernel__wasmsimd_lut64_p2_div_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, wasmsimd_lut64_p2_div_x24, xnn_f32_sigmoid_ukernel__wasmsimd_lut64_p2_div_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_sigmoid, wasmsimd_p5_div_x4, xnn_f32_sigmoid_ukernel__wasmsimd_p5_div_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, wasmsimd_p5_div_x8, xnn_f32_sigmoid_ukernel__wasmsimd_p5_div_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, wasmsimd_p5_div_x12, xnn_f32_sigmoid_ukernel__wasmsimd_p5_div_x12)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, wasmsimd_p5_div_x16, xnn_f32_sigmoid_ukernel__wasmsimd_p5_div_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, wasmsimd_p5_div_x20, xnn_f32_sigmoid_ukernel__wasmsimd_p5_div_x20)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_sigmoid, wasmsimd_p5_div_x24, xnn_f32_sigmoid_ukernel__wasmsimd_p5_div_x24)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD

BENCHMARK_CAPTURE(f32_sigmoid, scalar_lut2048_p1_div_x1, xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x1)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_sigmoid, scalar_lut2048_p1_div_x2, xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x2)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_sigmoid, scalar_lut2048_p1_div_x4, xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x4)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_sigmoid, scalar_lut64_p2_div_x1, xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x1)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_sigmoid, scalar_lut64_p2_div_x2, xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x2)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_sigmoid, scalar_lut64_p2_div_x4, xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x4)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_sigmoid, scalar_p5_div_x1, xnn_f32_sigmoid_ukernel__scalar_p5_div_x1)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_sigmoid, scalar_p5_div_x2, xnn_f32_sigmoid_ukernel__scalar_p5_div_x2)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_sigmoid, scalar_p5_div_x4, xnn_f32_sigmoid_ukernel__scalar_p5_div_x4)
  ->RangeMultiplier(10)
  ->Range(1000, 1000000)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
