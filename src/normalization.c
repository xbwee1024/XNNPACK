// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <string.h>
#include <stddef.h>

void xnn_normalize_tensor_dimensions(const size_t num_dims, const size_t element_size, const size_t* perm,
                                                   const size_t* shape, size_t* normalized_num_dims,
                                                   size_t* normalized_element_size, size_t* normalized_perm,
                                                   size_t* normalized_shape)
{
  size_t output_dims = num_dims;
  memcpy(normalized_perm, perm, num_dims * sizeof(size_t));
  memcpy(normalized_shape, shape, num_dims * sizeof(size_t));
  for (size_t i = num_dims; i-- > 1;) {
    if (normalized_perm[i] == normalized_perm[i-1] + 1 || normalized_shape[normalized_perm[i]] == 1) {
      output_dims--;
      normalized_shape[normalized_perm[i-1]] *= normalized_shape[normalized_perm[i]];
      for (size_t j = normalized_perm[i]; j + 1 < num_dims; ++j) {
        normalized_shape[j] = normalized_shape[j+1];
      }
      for (size_t j = 0; j < num_dims; ++j) {
        if (normalized_perm[j] > normalized_perm[i]) --normalized_perm[j];
      }
      for (size_t j = i; j + 1 < num_dims; ++j) {
        normalized_perm[j] = normalized_perm[j+1];
      }
    }
  }
  *normalized_element_size = element_size;
  if (normalized_perm[output_dims - 1] == output_dims - 1) {
    *normalized_element_size = element_size * normalized_shape[output_dims - 1];
    normalized_shape[output_dims - 1] = 1;
    if (output_dims > 1) {
      --output_dims;
    }
  }
  *normalized_num_dims = output_dims;
}
