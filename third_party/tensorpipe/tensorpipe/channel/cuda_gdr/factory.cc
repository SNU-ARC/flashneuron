/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/channel/cuda_gdr/factory.h>

#include <tensorpipe/channel/cuda_gdr/context.h>

namespace tensorpipe {
namespace channel {
namespace cuda_gdr {

std::shared_ptr<CudaContext> create(
    optional<std::vector<std::string>> gpuIdxToNicName) {
  return std::make_shared<cuda_gdr::Context>(std::move(gpuIdxToNicName));
}

} // namespace cuda_gdr
} // namespace channel
} // namespace tensorpipe
