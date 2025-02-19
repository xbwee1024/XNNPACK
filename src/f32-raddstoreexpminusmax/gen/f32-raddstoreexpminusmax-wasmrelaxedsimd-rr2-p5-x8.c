// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/raddstoreexpminusmax.h>


void xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x8(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const v128_t vi_max = wasm_v128_load32_splat(max);
  const v128_t vlog2e = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.log2e);
  const v128_t vmagic_bias = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.magic_bias);
  const v128_t vminus_ln2_hi = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.minus_ln2_hi);
  const v128_t vminus_ln2_lo = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.minus_ln2_lo);
  const v128_t vc5 = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.c5);
  const v128_t vc4 = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.c4);
  const v128_t vc3 = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.c3);
  const v128_t vc2 = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.c2);
  const v128_t vc1 = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.c1);
  const v128_t vdenorm_cutoff = wasm_v128_load64_splat(params->wasmsimd_rr2_p5.denorm_cutoff);

  v128_t vacc0 = wasm_f32x4_const_splat(0.0f);
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    // Load 8 (2x4) inputs at a time.
    const v128_t vi0123 = wasm_v128_load(input);
    const v128_t vi4567 = wasm_v128_load(input + 4);
    input += 8;

    const v128_t vx0123 = wasm_f32x4_sub(vi0123, vi_max);
    const v128_t vx4567 = wasm_f32x4_sub(vi4567, vi_max);

    v128_t vn0123 = __builtin_wasm_relaxed_madd_f32x4(vx0123, vlog2e, vmagic_bias);
    v128_t vn4567 = __builtin_wasm_relaxed_madd_f32x4(vx4567, vlog2e, vmagic_bias);

    const v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);
    const v128_t vs4567 = wasm_i32x4_shl(vn4567, 23);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);

    v128_t vt0123 = __builtin_wasm_relaxed_madd_f32x4(vn0123, vminus_ln2_hi, vx0123);
    v128_t vt4567 = __builtin_wasm_relaxed_madd_f32x4(vn4567, vminus_ln2_hi, vx4567);

    vt0123 = __builtin_wasm_relaxed_madd_f32x4(vn0123, vminus_ln2_lo, vt0123);
    vt4567 = __builtin_wasm_relaxed_madd_f32x4(vn4567, vminus_ln2_lo, vt4567);

    v128_t vp0123 = __builtin_wasm_relaxed_madd_f32x4(vc5, vt0123, vc4);
    v128_t vp4567 = __builtin_wasm_relaxed_madd_f32x4(vc5, vt4567, vc4);

    vp0123 = __builtin_wasm_relaxed_madd_f32x4(vp0123, vt0123, vc3);
    vp4567 = __builtin_wasm_relaxed_madd_f32x4(vp4567, vt4567, vc3);

    vp0123 = __builtin_wasm_relaxed_madd_f32x4(vp0123, vt0123, vc2);
    vp4567 = __builtin_wasm_relaxed_madd_f32x4(vp4567, vt4567, vc2);

    vp0123 = __builtin_wasm_relaxed_madd_f32x4(vp0123, vt0123, vc1);
    vp4567 = __builtin_wasm_relaxed_madd_f32x4(vp4567, vt4567, vc1);

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);

    v128_t vf0123 = __builtin_wasm_relaxed_madd_f32x4(vt0123, vp0123, vs0123);
    v128_t vf4567 = __builtin_wasm_relaxed_madd_f32x4(vt4567, vp4567, vs4567);

    vf0123 = wasm_v128_andnot(vf0123, wasm_f32x4_lt(vx0123, vdenorm_cutoff));
    vf4567 = wasm_v128_andnot(vf4567, wasm_f32x4_lt(vx4567, vdenorm_cutoff));

    wasm_v128_store(output, vf0123);
    wasm_v128_store(output + 4, vf4567);
    output += 8;

    vacc0 = wasm_f32x4_add(vacc0, vf0123);
    vacc0 = wasm_f32x4_add(vacc0, vf4567);
  }

  v128_t vacc = vacc0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vi = wasm_v128_load(input);
    input += 4;

    const v128_t vx = wasm_f32x4_sub(vi, vi_max);

    v128_t vn = __builtin_wasm_relaxed_madd_f32x4(vx, vlog2e, vmagic_bias);

    const v128_t vs = wasm_i32x4_shl(vn, 23);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = __builtin_wasm_relaxed_madd_f32x4(vn, vminus_ln2_hi, vx);
    vt = __builtin_wasm_relaxed_madd_f32x4(vn, vminus_ln2_lo, vt);

    v128_t vp = __builtin_wasm_relaxed_madd_f32x4(vc5, vt, vc4);
    vp = __builtin_wasm_relaxed_madd_f32x4(vp, vt, vc3);
    vp = __builtin_wasm_relaxed_madd_f32x4(vp, vt, vc2);
    vp = __builtin_wasm_relaxed_madd_f32x4(vp, vt, vc1);

    vt = wasm_f32x4_mul(vt, vs);
    v128_t vf = __builtin_wasm_relaxed_madd_f32x4(vt, vp, vs);

    vf = wasm_v128_andnot(vf, wasm_f32x4_lt(vx, vdenorm_cutoff));

    wasm_v128_store(output, vf);
    output += 4;

    vacc = wasm_f32x4_add(vacc, vf);
  }
  vacc = wasm_f32x4_add(vacc, wasm_v64x2_shuffle(vacc, vacc, 1, 1));
  float vsum = wasm_f32x4_extract_lane(vacc, 0) + wasm_f32x4_extract_lane(vacc, 1);
  if (batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));

    const v128_t vi = wasm_v128_load(input);

    const v128_t vx = wasm_f32x4_sub(vi, vi_max);

    v128_t vn = __builtin_wasm_relaxed_madd_f32x4(vx, vlog2e, vmagic_bias);

    const v128_t vs = wasm_i32x4_shl(vn, 23);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = __builtin_wasm_relaxed_madd_f32x4(vn, vminus_ln2_hi, vx);
    vt = __builtin_wasm_relaxed_madd_f32x4(vn, vminus_ln2_lo, vt);

    v128_t vp = __builtin_wasm_relaxed_madd_f32x4(vc5, vt, vc4);
    vp = __builtin_wasm_relaxed_madd_f32x4(vp, vt, vc3);
    vp = __builtin_wasm_relaxed_madd_f32x4(vp, vt, vc2);
    vp = __builtin_wasm_relaxed_madd_f32x4(vp, vt, vc1);

    vt = wasm_f32x4_mul(vt, vs);
    v128_t vf = __builtin_wasm_relaxed_madd_f32x4(vt, vp, vs);

    vf = wasm_v128_andnot(vf, wasm_f32x4_lt(vx, vdenorm_cutoff));

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vf, 0);
      output += 2;

      vsum += wasm_f32x4_extract_lane(vf, 0) + wasm_f32x4_extract_lane(vf, 1);
      vf = wasm_v64x2_shuffle(vf, vf, 1, 1);
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vf, 0);
      vsum += wasm_f32x4_extract_lane(vf, 0);
    }
  }
  *sum = vsum;
}
