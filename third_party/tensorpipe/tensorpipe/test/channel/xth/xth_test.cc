/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/channel/xth/factory.h>
#include <tensorpipe/test/channel/channel_test.h>

namespace {

class XthChannelTestHelper : public ChannelTestHelper<tensorpipe::CpuBuffer> {
 protected:
  std::shared_ptr<tensorpipe::channel::CpuContext> makeContextInternal(
      std::string id) override {
    auto context = tensorpipe::channel::xth::create();
    context->setId(std::move(id));
    return context;
  }
};

XthChannelTestHelper helper;

} // namespace

INSTANTIATE_TEST_CASE_P(Xth, CpuChannelTestSuite, ::testing::Values(&helper));
