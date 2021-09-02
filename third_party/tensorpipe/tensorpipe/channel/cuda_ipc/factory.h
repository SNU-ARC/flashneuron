/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <tensorpipe/channel/cuda_context.h>

namespace tensorpipe {
namespace channel {
namespace cuda_ipc {

std::shared_ptr<CudaContext> create();

} // namespace cuda_ipc
} // namespace channel
} // namespace tensorpipe
