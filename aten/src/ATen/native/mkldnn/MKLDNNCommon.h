#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>

namespace at { namespace native {

// Mapping ScalarType to ideep tensor data_type
ideep::tensor::data_type get_mkldnn_dtype(ScalarType type);

// Construct aten MKL-DNN tensor given an ideep tensor
Tensor new_with_itensor_mkldnn(ideep::tensor&& it, c10::optional<ScalarType> dtype, c10::optional<Device> device);

// Retrieve `ideep::tensor` from MKL-DNN tensor
ideep::tensor& itensor_from_mkldnn(const Tensor& mkldnn_tensor);

// Construct an `ideep::tensor` "view" from dense tensor, note the
// ideep::tensor will share the underlying buffer
ideep::tensor itensor_view_from_dense(const Tensor& tensor);

// Helper function for getting an ideep tensor out of an aten Tensor or MKL-DNN tensor.
ideep::tensor itensor_from_tensor(const Tensor& tensor);

}}

#endif // AT_MKLDNN_ENABLED
