// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack.h>

#include <memory>
#include <vector>

namespace models {

typedef std::vector<std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>> ExecutionPlan;
typedef ExecutionPlan (*ExecutionPlanFactory)(pthreadpool_t threadpool);

ExecutionPlan FP32MobileNetV1(pthreadpool_t threadpool);
ExecutionPlan FP32MobileNetV2(pthreadpool_t threadpool);
ExecutionPlan FP32MobileNetV3Large(pthreadpool_t threadpool);
ExecutionPlan FP32MobileNetV3Small(pthreadpool_t threadpool);

ExecutionPlan FP16MobileNetV1(pthreadpool_t threadpool);
ExecutionPlan FP16MobileNetV2(pthreadpool_t threadpool);
ExecutionPlan FP16MobileNetV3Large(pthreadpool_t threadpool);
ExecutionPlan FP16MobileNetV3Small(pthreadpool_t threadpool);

ExecutionPlan QS8MobileNetV1(pthreadpool_t threadpool);
ExecutionPlan QS8MobileNetV2(pthreadpool_t threadpool);

}  // namespace models
