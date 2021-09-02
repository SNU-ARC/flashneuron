/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/channel/error.h>

#include <cstring>
#include <sstream>

namespace tensorpipe {
namespace channel {

std::string ContextClosedError::what() const {
  return "context closed";
}

std::string ChannelClosedError::what() const {
  return "channel closed";
}

} // namespace channel
} // namespace tensorpipe
