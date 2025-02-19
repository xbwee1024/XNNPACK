// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/gemm.h"
#include "bench/utils.h"

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/math.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/pack.h>
#include <xnnpack/packx.h>
#include <xnnpack/ppmm.h>


static void GEMMBenchmark(benchmark::State& state,
  xnn_f32_qc8w_gemm_minmax_ukernel_fn gemm,
  xnn_init_f32_minmax_params_fn init_params,
  size_t mr, size_t nr, size_t kr, size_t sr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  const size_t nc_stride = benchmark::utils::RoundUp(nc, nr);
  const size_t kc_stride = benchmark::utils::RoundUp(kc, kr * sr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));
  auto s8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<float> a(mc * kc + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<int8_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(s8rng));
  std::vector<float> b(nc);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  const size_t w_size = nc_stride * 2 * sizeof(float) + kc_stride * nc_stride * sizeof(int8_t);
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * c_elements + w_size);

  std::vector<int8_t, AlignedAllocator<int8_t, 64>> w(w_size * num_buffers);
  std::fill(w.begin(), w.end(), 0);
  xnn_pack_f32_qs8w_gemm_goi_w(1 /* groups */, nc, kc, nr, kr, sr, k.data(), b.data(), w.data(), nr * sizeof(float), nullptr);
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params params;
  init_params(&params,
    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(
        mb, nc, kc * sizeof(float),
        a.data() + m * kc, kc * sizeof(float),
        w.data() + buffer_index * w_size,
        c.data() + (buffer_index * mc + m) * nc, nc * sizeof(float), nr * sizeof(float),
        &params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_qc8w_gemm_1x8__asm_aarch64_neon_ld128_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neon_ld128_acc2_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld64_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld64_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld64_acc2_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld64_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld64_acc4_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld128_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld128_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld128_acc2_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld128_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld128_acc4_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_4x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_6x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_4x2__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }

  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neon_ld128_acc2)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neon_ld128_acc2_prfm)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld64_prfm)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld64_acc2)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld64_acc2_prfm)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld64_acc4)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld64_acc4_prfm)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld128_prfm)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld128_acc2)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld128_acc2_prfm)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld128_acc4)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__asm_aarch64_neonfma_ld128_acc4_prfm)
  BENCHMARK_GEMM(f32_qc8w_gemm_4x2__asm_aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_4x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_6x8__asm_aarch64_neonfma_ld64)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM64
  static void f32_qc8w_gemm_4x2__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_6x2__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_4x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_6x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_4x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_6x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_1x16__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  static void f32_qc8w_gemm_4x16__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }

  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_4x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_6x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_4x2__aarch64_neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_6x2__aarch64_neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__aarch64_neonfma_lane_ld128)
  BENCHMARK_GEMM(f32_qc8w_gemm_4x8__aarch64_neonfma_lane_ld128)
  BENCHMARK_GEMM(f32_qc8w_gemm_6x8__aarch64_neonfma_lane_ld128)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x16__aarch64_neonfma_lane_ld128)
  BENCHMARK_GEMM(f32_qc8w_gemm_4x16__aarch64_neonfma_lane_ld128)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_qc8w_gemm_4x2__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_6x2__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_1x8__neon_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_1x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_1x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_4x8__neon_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_4x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_4x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_5x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_6x8__neon_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_6x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_6x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_1x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_4x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_qc8w_gemm_6x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_qc8w_gemm_4x2__neon_lane_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_6x2__neon_lane_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__neon_dup_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__neon_lane_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8__neonfma_dup_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_4x8__neon_dup_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_4x8__neon_lane_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_4x8__neonfma_dup_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_5x8__neon_lane_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_6x8__neon_dup_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_6x8__neon_lane_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_6x8__neonfma_dup_ld64)
  BENCHMARK_GEMM(f32_qc8w_gemm_1x8s4__neonfma)
  BENCHMARK_GEMM(f32_qc8w_gemm_4x8s4__neonfma)
  BENCHMARK_GEMM(f32_qc8w_gemm_6x8s4__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

static void f32_qc8w_gemm_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void f32_qc8w_gemm_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void f32_qc8w_gemm_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_GEMM(f32_qc8w_gemm_1x4__scalar)
BENCHMARK_GEMM(f32_qc8w_gemm_2x4__scalar)
BENCHMARK_GEMM(f32_qc8w_gemm_4x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
