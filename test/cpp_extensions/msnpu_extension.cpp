#include <torch/extension.h>
#include <torch/library.h>

using namespace at;

static int test_int;

Tensor get_tensor(caffe2::TypeMeta dtype, IntArrayRef size) {
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          Storage::use_byte_size_t(),
          0,
          at::DataPtr(nullptr, Device(DeviceType::MSNPU, 0)),
          nullptr,
          false),
      DispatchKey::MSNPU,
      dtype);
  // This is a hack to workaround the shape checks in _convolution.
  tensor_impl->set_sizes_contiguous(size);
  return Tensor(std::move(tensor_impl));
}

Tensor empty_override(IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device,
                      c10::optional<bool> pin_memory, c10::optional<c10::MemoryFormat> optional_memory_format) {
  test_int = 0;
  return get_tensor(scalarTypeToTypeMeta(dtype_or_default(dtype)), size);
}

Tensor add_override(const Tensor & a, const Tensor & b , Scalar c) {
  test_int = 1;
  return get_tensor(a.dtype(), a.sizes());
}

Tensor fake_convolution(
    const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups) {
  test_int = 2;
  // Only the first 2 dimension of output shape is correct.
  return get_tensor(input.dtype(), {input.size(0), weight.size(0), input.size(2), input.size(3)});
}

std::tuple<Tensor,Tensor,Tensor> fake_convolution_backward(
        const Tensor & grad_output, const Tensor & input, const Tensor & weight,
        IntArrayRef stride, IntArrayRef padding,
        IntArrayRef dilation, bool transposed, IntArrayRef output_padding,
        int64_t groups, std::array<bool,3> output_mask) {
    test_int = 3;
    return std::tuple<Tensor, Tensor, Tensor>(
            get_tensor(input.dtype(), input.sizes()),
            get_tensor(weight.dtype(), weight.sizes()),
            get_tensor(input.dtype(), {}));
}

TORCH_LIBRARY_IMPL(aten, MSNPU, m) {
  m.impl("empty.memory_format",                empty_override);
  m.impl("add.Tensor",                         add_override);
  m.impl("convolution_overrideable",           fake_convolution);
  m.impl("convolution_backward_overrideable",  fake_convolution_backward);
}

// TODO: Extend this to exercise multi-device setting.  In that case,
// we need to add a thread local variable to track the current device.
struct MSNPUGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::MSNPU;
  MSNPUGuardImpl() {}
  MSNPUGuardImpl(DeviceType t) {
    AT_ASSERT(t == DeviceType::MSNPU);
  }
  DeviceType type() const override {
    return DeviceType::MSNPU;
  }
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::MSNPU);
    AT_ASSERT(d.index() == 0);
    return d;
  }
  Device getDevice() const override {
    return Device(DeviceType::MSNPU, 0);
  }
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::MSNPU);
    AT_ASSERT(d.index() == 0);
  }
  void uncheckedSetDevice(Device d) const noexcept override {
  }
  Stream getStream(Device d) const noexcept override {
    return Stream(Stream::DEFAULT, Device(DeviceType::MSNPU, 0));
  }
  Stream exchangeStream(Stream s) const noexcept override {
    return Stream(Stream::DEFAULT, Device(DeviceType::MSNPU, 0));
  }
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }

  // Event-related functions
  void record(void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const EventFlag flag) const override {
    TORCH_CHECK(false, "MSNPU backend doesn't support events.");
  }
  void block(
    void* event,
    const Stream& stream) const override {
    TORCH_CHECK(false, "MSNPU backend doesn't support events.");
  }
  bool queryEvent(void* event) const override {
    TORCH_CHECK(false, "MSNPU backend doesn't support events.");
  }
  void destroyEvent(
    void* event,
    const DeviceIndex device_index) const noexcept override { }
};

constexpr DeviceType MSNPUGuardImpl::static_type;
C10_REGISTER_GUARD_IMPL(MSNPU, MSNPUGuardImpl);

int get_test_int() {
  return test_int;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_test_int", &get_test_int);
}
