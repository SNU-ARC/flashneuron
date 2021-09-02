/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cassert>

#include <pack_block_sparse.h>

namespace qnnpack {
std::unique_ptr<BCSRMatrix> generateBlockCSRMatrix(
    const uint8_t* a,
    const size_t N,
    const size_t K,
    const uint32_t col_block_size,
    const uint8_t* zero_points) {
  assert(K > 0);
  std::unique_ptr<BCSRMatrix> bcsr_mat_ptr = std::make_unique<BCSRMatrix>();
  auto& bcsr_mat = *bcsr_mat_ptr;
  // K must be > 0
  const uint32_t num_blocks = (K + col_block_size - 1) / col_block_size;
  const uint64_t num_blocks_full =
    (K % col_block_size) == 0 ? num_blocks : num_blocks - 1;
  const uint32_t remainder_elements = K % col_block_size;

  bcsr_mat.row_values.reserve(N);
  uint32_t num_nnz_blocks{0};
  bcsr_mat.row_values.push_back(num_nnz_blocks);
  // Note that values and col_indices vectors will be
  // dynamically resized and copied. This has some overhead
  // but chosing not to optimize prematurely.
  // Plus loops can be vectorized as well
  for (uint32_t n = 0; n < N ; ++n) {
    // Process all blocks but last
    uint32_t k_block = 0;
    for (; k_block < num_blocks_full; ++k_block) {
      bool block_zero{true};
      for (uint32_t m = 0; m < col_block_size; ++m) {
        if (*(a + n * K + k_block * col_block_size + m) != zero_points[n]) {
          block_zero = false;
          break;
        }
      }
      if (!block_zero) {
        bcsr_mat.col_indices.push_back(k_block);
        num_nnz_blocks++;
        for (uint32_t m = 0; m < col_block_size; ++m) {
          uint8_t val = *(a + n * K + k_block * col_block_size + m);
          bcsr_mat.values.push_back(val);
        }
      }
    }
    for (; k_block < num_blocks; ++k_block) {
      bool block_zero{true};
      uint32_t m = 0;
      for (; m < remainder_elements; ++m) {
        if (*(a + n * K + k_block * col_block_size + m) != zero_points[n]) {
          block_zero = false;
          break;
        }
      }
      if (!block_zero) {
        bcsr_mat.col_indices.push_back(k_block);
        num_nnz_blocks++;
        for (m = 0; m < remainder_elements; ++m) {
          uint8_t val = *(a + n * K + k_block * col_block_size + m);
          bcsr_mat.values.push_back(val);
        }
        for (; m < col_block_size; ++m) {
          bcsr_mat.values.push_back(zero_points[n]);
        }
      }
    }
    bcsr_mat.row_values.push_back(num_nnz_blocks);
  }
  return bcsr_mat_ptr;
}
} // namsepace qnnpack
