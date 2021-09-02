#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

struct TensorIterator;

namespace native {

using copy_fn = void (*)(TensorIterator&, bool non_blocking);
using FN_copy_fn = void (*)(TensorIterator&, bool non_blocking, int tid, bool is_csr);

DECLARE_DISPATCH(copy_fn, copy_stub);
DECLARE_DISPATCH(FN_copy_fn, FN_copy_stub);

} // namespace native
} // namespace at
