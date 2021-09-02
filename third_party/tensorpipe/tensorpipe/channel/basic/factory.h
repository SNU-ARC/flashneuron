/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <tensorpipe/channel/cpu_context.h>

namespace tensorpipe {
namespace channel {
namespace basic {

std::shared_ptr<CpuContext> create();

} // namespace basic
} // namespace channel
} // namespace tensorpipe
