// This file registers special JIT operators used to implement the PyTorch CUDA
// API in TorchScript.
#ifndef __HIP_PLATFORM_HCC__
#include <torch/csrc/api/include/torch/utils.h>
#include <torch/csrc/jit/cuda/cuda.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace jit {

namespace {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

RegisterOperators const reg({
    Operator(
        "cuda::current_stream(int64_t val) -> __torch__.torch.classes.cuda.Stream",
        [](Stack* stack) {
          auto idx = uint16_t(pop(stack).toInt());
          auto s = c10::cuda::getCurrentCUDAStream(idx);
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::default_stream(int64_t val) -> __torch__.torch.classes.cuda.Stream",
        [](Stack* stack) {
          auto idx = uint16_t(pop(stack).toInt());
          auto s = c10::cuda::getDefaultCUDAStream(idx);
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::_current_device() -> int",
        [](Stack* stack) {
          auto v = c10::cuda::current_device();
          push(stack, static_cast<int>(v));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::_set_device(int64_t val) -> ()",
        [](Stack* stack) {
          int64_t idx = -1;
          pop(stack, idx);
          c10::cuda::set_device(static_cast<c10::DeviceIndex>(idx));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::device_index(Device device) -> int",
        [](Stack* stack) {
          auto device = pop(stack);
          auto idx = device.toDevice().index();
          push(stack, idx);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::device_count() -> int",
        [](Stack* stack) { push(stack, at::cuda::device_count()); },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::set_stream(__torch__.torch.classes.cuda.Stream stream) -> ()",
        [](Stack* stack) {
          auto v = pop(stack);
          auto s = v.toCustomClass<torch::jit::CUDAStream>();
          // To set the current CUDA stream using
          // c10::cuda::setCurrentCUDAStream, the jit::CUDAStream object needs
          // to be converted to c10::cuda::CUDAStream. Since the latter cannot
          // be returned from a class registered via TorchBind, this can only be
          // achieved by packing the c10::cuda::CUDAStream instance contained
          // inside the jit::CUDAStream object to a uint64_t representation, and
          // unpacking it inside this operator. The unpacked stream is then used
          // to set the current CUDA stream.
          auto packed = s->pack();
          auto unpacked = c10::cuda::CUDAStream::unpack(packed);
          c10::cuda::setCurrentCUDAStream(unpacked);
        },
        aliasAnalysisFromSchema()),
});
} // namespace
} // namespace jit
} // namespace torch
#endif
