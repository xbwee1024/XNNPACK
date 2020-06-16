// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/wasmsimd-s4.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/gemm.h>


void xnn_f32_gemminc_minmax_ukernel_3x8s4__wasmsimd_arm(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const float*restrict acc,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);
  assert(acc != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(acc + 0);
    v128_t vacc0x4567 = wasm_v128_load(acc + 4);
    v128_t vacc1x0123 = wasm_v128_load(acc + 8);
    v128_t vacc1x4567 = wasm_v128_load(acc + 12);
    v128_t vacc2x0123 = wasm_v128_load(acc + 16);
    v128_t vacc2x4567 = wasm_v128_load(acc + 20);
    acc += 24;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      v128_t va2 = wasm_v128_load(a2);
      a2 += 4;


      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = wasm_f32x4_add(vacc0x0123, wasm_f32x4_mul(va0, vb0123c0));
      vacc1x0123 = wasm_f32x4_add(vacc1x0123, wasm_f32x4_mul(va1, vb0123c0));
      vacc2x0123 = wasm_f32x4_add(vacc2x0123, wasm_f32x4_mul(va2, vb0123c0));
      vacc0x4567 = wasm_f32x4_add(vacc0x4567, wasm_f32x4_mul(va0, vb4567c0));
      vacc1x4567 = wasm_f32x4_add(vacc1x4567, wasm_f32x4_mul(va1, vb4567c0));
      vacc2x4567 = wasm_f32x4_add(vacc2x4567, wasm_f32x4_mul(va2, vb4567c0));

      va0 = wasm_v32x4_shuffle(va0, va0, 1, 2, 3, 0);
      va1 = wasm_v32x4_shuffle(va1, va1, 1, 2, 3, 0);
      va2 = wasm_v32x4_shuffle(va2, va2, 1, 2, 3, 0);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = wasm_f32x4_add(vacc0x0123, wasm_f32x4_mul(va0, vb0123c1));
      vacc1x0123 = wasm_f32x4_add(vacc1x0123, wasm_f32x4_mul(va1, vb0123c1));
      vacc2x0123 = wasm_f32x4_add(vacc2x0123, wasm_f32x4_mul(va2, vb0123c1));
      vacc0x4567 = wasm_f32x4_add(vacc0x4567, wasm_f32x4_mul(va0, vb4567c1));
      vacc1x4567 = wasm_f32x4_add(vacc1x4567, wasm_f32x4_mul(va1, vb4567c1));
      vacc2x4567 = wasm_f32x4_add(vacc2x4567, wasm_f32x4_mul(va2, vb4567c1));

      va0 = wasm_v32x4_shuffle(va0, va0, 1, 2, 3, 0);
      va1 = wasm_v32x4_shuffle(va1, va1, 1, 2, 3, 0);
      va2 = wasm_v32x4_shuffle(va2, va2, 1, 2, 3, 0);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = wasm_f32x4_add(vacc0x0123, wasm_f32x4_mul(va0, vb0123c2));
      vacc1x0123 = wasm_f32x4_add(vacc1x0123, wasm_f32x4_mul(va1, vb0123c2));
      vacc2x0123 = wasm_f32x4_add(vacc2x0123, wasm_f32x4_mul(va2, vb0123c2));
      vacc0x4567 = wasm_f32x4_add(vacc0x4567, wasm_f32x4_mul(va0, vb4567c2));
      vacc1x4567 = wasm_f32x4_add(vacc1x4567, wasm_f32x4_mul(va1, vb4567c2));
      vacc2x4567 = wasm_f32x4_add(vacc2x4567, wasm_f32x4_mul(va2, vb4567c2));

      va0 = wasm_v32x4_shuffle(va0, va0, 1, 2, 3, 0);
      va1 = wasm_v32x4_shuffle(va1, va1, 1, 2, 3, 0);
      va2 = wasm_v32x4_shuffle(va2, va2, 1, 2, 3, 0);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = wasm_f32x4_add(vacc0x0123, wasm_f32x4_mul(va0, vb0123c3));
      vacc1x0123 = wasm_f32x4_add(vacc1x0123, wasm_f32x4_mul(va1, vb0123c3));
      vacc2x0123 = wasm_f32x4_add(vacc2x0123, wasm_f32x4_mul(va2, vb0123c3));
      vacc0x4567 = wasm_f32x4_add(vacc0x4567, wasm_f32x4_mul(va0, vb4567c3));
      vacc1x4567 = wasm_f32x4_add(vacc1x4567, wasm_f32x4_mul(va1, vb4567c3));
      vacc2x4567 = wasm_f32x4_add(vacc2x4567, wasm_f32x4_mul(va2, vb4567c3));


      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v32x4_load_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v32x4_load_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v32x4_load_splat(a2);
        a2 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = wasm_f32x4_add(vacc0x0123, wasm_f32x4_mul(va0, vb0123));
        vacc1x0123 = wasm_f32x4_add(vacc1x0123, wasm_f32x4_mul(va1, vb0123));
        vacc2x0123 = wasm_f32x4_add(vacc2x0123, wasm_f32x4_mul(va2, vb0123));
        vacc0x4567 = wasm_f32x4_add(vacc0x4567, wasm_f32x4_mul(va0, vb4567));
        vacc1x4567 = wasm_f32x4_add(vacc1x4567, wasm_f32x4_mul(va1, vb4567));
        vacc2x4567 = wasm_f32x4_add(vacc2x4567, wasm_f32x4_mul(va2, vb4567));

        k -= sizeof(float);
      } while (k != 0);
    }

    const v128_t vmin = wasm_v32x4_load_splat(&params->scalar.min);
    vacc0x0123 = wasm_f32x4_max(vacc0x0123, vmin);
    vacc1x0123 = wasm_f32x4_max(vacc1x0123, vmin);
    vacc2x0123 = wasm_f32x4_max(vacc2x0123, vmin);
    vacc0x4567 = wasm_f32x4_max(vacc0x4567, vmin);
    vacc1x4567 = wasm_f32x4_max(vacc1x4567, vmin);
    vacc2x4567 = wasm_f32x4_max(vacc2x4567, vmin);

    const v128_t vmax = wasm_v32x4_load_splat(&params->scalar.max);
    vacc0x0123 = wasm_f32x4_min(vacc0x0123, vmax);
    vacc1x0123 = wasm_f32x4_min(vacc1x0123, vmax);
    vacc2x0123 = wasm_f32x4_min(vacc2x0123, vmax);
    vacc0x4567 = wasm_f32x4_min(vacc0x4567, vmax);
    vacc1x4567 = wasm_f32x4_min(vacc1x4567, vmax);
    vacc2x4567 = wasm_f32x4_min(vacc2x4567, vmax);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        *((double*) c2) = wasm_f64x2_extract_lane(vacc2x0123, 0);
        *((double*) c1) = wasm_f64x2_extract_lane(vacc1x0123, 0);
        *((double*) c0) = wasm_f64x2_extract_lane(vacc0x0123, 0);

        vacc2x0123 = wasm_v32x4_shuffle(vacc2x0123, vacc2x0123, 2, 3, 2, 3);
        vacc1x0123 = wasm_v32x4_shuffle(vacc1x0123, vacc1x0123, 2, 3, 2, 3);
        vacc0x0123 = wasm_v32x4_shuffle(vacc0x0123, vacc0x0123, 2, 3, 2, 3);

        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        *c2 = wasm_f32x4_extract_lane(vacc2x0123, 0);
        *c1 = wasm_f32x4_extract_lane(vacc1x0123, 0);
        *c0 = wasm_f32x4_extract_lane(vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
