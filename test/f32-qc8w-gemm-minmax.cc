// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-qc8w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, k_eq_1) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, strided_cn) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(1)
      .cn_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, k_eq_1_strided_a) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, k_eq_1_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, k_eq_1_subtile_m) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(4)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, k_eq_1_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, k_gt_1_subtile) {
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, strided_cm_subtile) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X4__WASM, strided_cm) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(1)
      .cm_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, k_eq_1) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, strided_cn) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(1)
      .cn_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, k_eq_1_strided_a) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, k_eq_1_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, k_eq_1_subtile_m) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(4)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, k_eq_1_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, k_gt_1_subtile) {
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, strided_cm_subtile) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_2X4__WASM, strided_cm) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(1)
      .cm_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, k_eq_1) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, strided_cn) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(1)
      .cn_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, k_eq_1_strided_a) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, k_eq_1_subtile) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, k_eq_1_subtile_m) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(2)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, k_eq_1_subtile_n) {
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, k_gt_1_subtile) {
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, n_gt_2) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, n_gt_2_strided_cn) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, n_gt_2_strided_a) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, n_gt_2_subtile) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, n_div_2) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, n_div_2_strided_cn) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, n_div_2_strided_a) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, n_div_2_subtile) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, strided_cm_subtile) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(5)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__WASM, strided_cm) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(1)
      .cm_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, k_eq_1) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, strided_cn) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(1)
      .cn_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, k_eq_1_strided_a) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, k_eq_1_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, k_eq_1_subtile_m) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(4)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, k_eq_1_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, k_gt_1_subtile) {
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, strided_cm_subtile) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X4__WASM, strided_cm) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(1)
      .cm_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(1)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 1; m++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(k)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
}


TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 2; m++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(k)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
}


TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .cn_stride(5)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 2; n++) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 4; m++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(m)
      .n(2)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 2; n++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(k)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, n_gt_2) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, n_gt_2_strided_cn) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, n_gt_2_strided_a) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, n_gt_2_subtile) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, n_div_2) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, n_div_2_strided_cn) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, n_div_2_strided_a) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, n_div_2_subtile) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .cm_stride(5)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params);
}


TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 4; m++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
}


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(2)
      .cn_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(2)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, n_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, n_gt_2_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, n_gt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, n_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, n_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, n_div_2_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, n_div_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, n_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(5)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(2)
      .cm_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(2)
      .cn_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(2)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, n_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, n_gt_2_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, n_gt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, n_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, n_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, n_div_2_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, n_div_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, n_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(5)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(2)
      .cm_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(2)
      .cn_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(2)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, n_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, n_gt_2_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, n_gt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, n_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, n_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, n_div_2_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, n_div_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, n_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(5)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(2)
      .cm_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 5; m++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 5; m++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 5; m++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 5; m++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(6)
      .n(2)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(6)
      .n(2)
      .k(2)
      .cn_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(6)
      .n(2)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(2)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(6)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(6)
        .n(2)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(6)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(6)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, n_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, n_gt_2_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, n_gt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, n_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, n_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, n_div_2_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, n_div_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, n_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(5)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(6)
      .n(2)
      .k(2)
      .cm_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(6)
      .n(2)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(6)
      .n(2)
      .k(2)
      .cn_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(6)
      .n(2)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(2)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(6)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(6)
        .n(2)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(6)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(6)
        .n(2)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, n_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, n_gt_2_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, n_gt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, n_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, n_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, n_div_2_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(5)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, n_div_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, n_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(5)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(6)
      .n(2)
      .k(2)
      .cm_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(4)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(4)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
