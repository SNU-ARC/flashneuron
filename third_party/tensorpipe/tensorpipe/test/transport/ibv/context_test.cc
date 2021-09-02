/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/test/transport/ibv/ibv_test.h>

#include <gtest/gtest.h>

namespace {

class IbvTransportContextTest : public TransportTest {};

IbvTransportTestHelper helper;

} // namespace

using namespace tensorpipe;

// Linux-only because OSX machines on CircleCI cannot resolve their hostname
#ifdef __linux__
TEST_P(IbvTransportContextTest, LookupHostnameAddress) {
  auto context = std::dynamic_pointer_cast<transport::ibv::Context>(
      GetParam()->getContext());
  ASSERT_TRUE(context);

  Error error;
  std::string addr;
  std::tie(error, addr) = context->lookupAddrForHostname();
  EXPECT_FALSE(error) << error.what();
  EXPECT_NE(addr, "");
}
#endif

// Interface name conventions change based on platform. Linux uses "lo", OSX
// uses lo0, Windows uses integers.
#ifdef __linux__
#define LOOPBACK_INTERFACE "lo"
#elif __APPLE__
#define LOOPBACK_INTERFACE "lo0"
#endif

#ifdef LOOPBACK_INTERFACE
TEST_P(IbvTransportContextTest, LookupInterfaceAddress) {
  auto context = std::dynamic_pointer_cast<transport::ibv::Context>(
      GetParam()->getContext());
  ASSERT_TRUE(context);

  Error error;
  std::string addr;
  std::tie(error, addr) = context->lookupAddrForIface(LOOPBACK_INTERFACE);
  EXPECT_FALSE(error) << error.what();
  EXPECT_NE(addr, "");
}
#endif

INSTANTIATE_TEST_CASE_P(
    Ibv,
    IbvTransportContextTest,
    ::testing::Values(&helper));
