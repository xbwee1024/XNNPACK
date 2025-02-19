// Auto-generated file. Do not edit!
//   Template: src/f32-rminmax/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/reduce.h>


void xnn_f32_rmin_ukernel__neon_x16_acc4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  float32x4_t vacc0 = vld1q_dup_f32(input);
  float32x4_t vacc1 = vacc0;
  float32x4_t vacc2 = vacc0;
  float32x4_t vacc3 = vacc0;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const float32x4_t vt0 = vld1q_f32(input); input += 4;
    const float32x4_t vt1 = vld1q_f32(input); input += 4;
    const float32x4_t vt2 = vld1q_f32(input); input += 4;
    const float32x4_t vt3 = vld1q_f32(input); input += 4;

    vacc0 = vminq_f32(vacc0, vt0);
    vacc1 = vminq_f32(vacc1, vt1);
    vacc2 = vminq_f32(vacc2, vt2);
    vacc3 = vminq_f32(vacc3, vt3);
  }
  vacc0 = vminq_f32(vacc0, vacc1);
  vacc2 = vminq_f32(vacc2, vacc3);
  vacc0 = vminq_f32(vacc0, vacc2);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vt = vld1q_f32(input); input += 4;
    vacc0 = vminq_f32(vacc0, vt);
  }
  float32x2_t vacc = vmin_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(float))) {
    const float32x2_t vt = vld1_f32(input); input += 2;
    vacc = vmin_f32(vacc, vt);
  }
  vacc = vpmin_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(float))) {
    const float32x2_t vt = vld1_dup_f32(input);
    vacc = vmin_f32(vacc, vt);
  }
  vst1_lane_f32(output, vacc, 0);
}
