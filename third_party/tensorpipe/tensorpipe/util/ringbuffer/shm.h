/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <tensorpipe/common/fd.h>
#include <tensorpipe/util/ringbuffer/ringbuffer.h>
#include <tensorpipe/util/shm/segment.h>

namespace tensorpipe {
namespace util {
namespace ringbuffer {
namespace shm {

/// Creates ringbuffer on shared memory.
///
/// RingBuffer's data can have any <util::shm::PageType>
/// (e.g. 4KB or a HugeTLB Page of 2MB or 1GB). If  <data_page_type> is not
/// provided, then choose the largest page that would result in
/// close to full occupancy.
///
/// If <persistent>, the shared memory will not be unlinked
/// when RingBuffer is destroyed.
///
/// <min_rb_byte_size> is the minimum size of the data section
/// of a RingBuffer (or each CPU's RingBuffer).
///
std::tuple<Error, util::shm::Segment, util::shm::Segment, RingBuffer> create(
    size_t minRbByteSize,
    optional<util::shm::PageType> dataPageType = nullopt,
    bool permWrite = true);

std::tuple<Error, util::shm::Segment, util::shm::Segment, RingBuffer> load(
    Fd headerFd,
    Fd dataFd,
    optional<util::shm::PageType> dataPageType = nullopt,
    bool permWrite = true);

} // namespace shm
} // namespace ringbuffer
} // namespace util
} // namespace tensorpipe
