// Auto-generated file. Do not edit!
//   Template: src/f32-rminmax/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/reduce.h>


void xnn_f32_rmin_ukernel__wasm_x4_acc2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  float vacc0 = *input;
  float vacc1 = vacc0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vt0 = input[0];
    const float vt1 = input[1];
    const float vt2 = input[2];
    const float vt3 = input[3];
    input += 4;

    vacc0 = __builtin_wasm_min_f32(vacc0, vt0);
    vacc1 = __builtin_wasm_min_f32(vacc1, vt1);
    vacc0 = __builtin_wasm_min_f32(vacc0, vt2);
    vacc1 = __builtin_wasm_min_f32(vacc1, vt3);
  }
  vacc0 = __builtin_wasm_min_f32(vacc0, vacc1);

  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vt = *input++;
      vacc0 = __builtin_wasm_min_f32(vacc0, vt);
      batch -= sizeof(float);
    } while (batch != 0);
  }
  *output = vacc0;
}
