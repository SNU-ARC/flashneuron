/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#include <tensorpipe/common/error.h>

namespace tensorpipe {
namespace transport {

class ListenerClosedError final : public BaseError {
 public:
  ListenerClosedError() {}

  std::string what() const override;
};

class ConnectionClosedError final : public BaseError {
 public:
  ConnectionClosedError() {}

  std::string what() const override;
};

} // namespace transport
} // namespace tensorpipe
