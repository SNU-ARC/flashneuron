/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <cuda.h>

#include <tensorpipe/common/defs.h>
#include <tensorpipe/common/dl.h>

#define TP_CUDA_DRIVER_CHECK(cuda_lib, a)                                 \
  do {                                                                    \
    CUresult error = (a);                                                 \
    if (error != CUDA_SUCCESS) {                                          \
      CUresult res;                                                       \
      const char* errorName;                                              \
      const char* errorStr;                                               \
      res = cuda_lib.getErrorName(error, &errorName);                     \
      TP_THROW_ASSERT_IF(res != CUDA_SUCCESS);                            \
      res = cuda_lib.getErrorString(error, &errorStr);                    \
      TP_THROW_ASSERT_IF(res != CUDA_SUCCESS);                            \
      TP_THROW_ASSERT() << __TP_EXPAND_OPD(a) << " " << errorName << " (" \
                        << errorStr << ")";                               \
    }                                                                     \
  } while (false)

namespace tensorpipe {

// Master list of all symbols we care about from libcuda.

#define TP_FORALL_CUDA_SYMBOLS(_)                               \
  _(ctxGetCurrent, cuCtxGetCurrent, (CUcontext*))               \
  _(ctxSetCurrent, cuCtxSetCurrent, (CUcontext))                \
  _(deviceGet, cuDeviceGet, (CUdevice*, int))                   \
  _(deviceGetCount, cuDeviceGetCount, (int*))                   \
  _(deviceGetUuid, cuDeviceGetUuid, (CUuuid*, CUdevice))        \
  _(getErrorName, cuGetErrorName, (CUresult, const char**))     \
  _(getErrorString, cuGetErrorString, (CUresult, const char**)) \
  _(init, cuInit, (unsigned int))                               \
  _(memGetAddressRange_v2,                                      \
    cuMemGetAddressRange_v2,                                    \
    (CUdeviceptr*, size_t*, CUdeviceptr))                       \
  _(pointerGetAttribute,                                        \
    cuPointerGetAttribute,                                      \
    (void*, CUpointer_attribute, CUdeviceptr))

// Wrapper for libcuda.

class CudaLib {
 private:
  explicit CudaLib(DynamicLibraryHandle dlhandle)
      : dlhandle_(std::move(dlhandle)) {}

  DynamicLibraryHandle dlhandle_;

#define TP_DECLARE_FIELD(method_name, function_name, args_types) \
  CUresult(*function_name##_ptr_) args_types = nullptr;
  TP_FORALL_CUDA_SYMBOLS(TP_DECLARE_FIELD)
#undef TP_DECLARE_FIELD

 public:
  CudaLib() = default;

#define TP_FORWARD_CALL(method_name, function_name, args_types)  \
  template <typename... Args>                                    \
  auto method_name(Args&&... args) const {                       \
    return (*function_name##_ptr_)(std::forward<Args>(args)...); \
  }
  TP_FORALL_CUDA_SYMBOLS(TP_FORWARD_CALL)
#undef TP_FORWARD_CALL

  static std::tuple<Error, CudaLib> create() {
    Error error;
    DynamicLibraryHandle dlhandle;
    // To keep things "neat" and contained, we open in "local" mode (as
    // opposed to global) so that the cuda symbols can only be resolved
    // through this handle and are not exposed (a.k.a., "leaked") to other
    // shared objects.
    std::tie(error, dlhandle) =
        createDynamicLibraryHandle("libcuda.so.1", RTLD_LOCAL | RTLD_LAZY);
    if (error) {
      return std::make_tuple(std::move(error), CudaLib());
    }
    CudaLib lib(std::move(dlhandle));
#define TP_LOAD_SYMBOL(method_name, function_name, args_types)        \
  {                                                                   \
    void* ptr;                                                        \
    std::tie(error, ptr) = loadSymbol(lib.dlhandle_, #function_name); \
    if (error) {                                                      \
      return std::make_tuple(std::move(error), CudaLib());            \
    }                                                                 \
    TP_THROW_ASSERT_IF(ptr == nullptr);                               \
    lib.function_name##_ptr_ =                                        \
        reinterpret_cast<decltype(function_name##_ptr_)>(ptr);        \
  }
    TP_FORALL_CUDA_SYMBOLS(TP_LOAD_SYMBOL)
#undef TP_LOAD_SYMBOL
    TP_CUDA_DRIVER_CHECK(lib, lib.init(0));
    return std::make_tuple(Error::kSuccess, std::move(lib));
  }

  CUresult memGetAddressRange(
      CUdeviceptr* pbase,
      size_t* psize,
      CUdeviceptr dptr) const {
    // NOTE: We are forwarding to cuMemGetAddressRange_v2() directly, because
    // the name cuMemGetAddressRange is #defined to its _v2 variant in cuda.h.
    // Calling the actual cuMemGetAddressRange() function here would lead to a
    // CUDA_ERROR_INVALID_CONTEXT.
    return memGetAddressRange_v2(pbase, psize, dptr);
  }
};

#undef TP_FORALL_CUDA_SYMBOLS

} // namespace tensorpipe
