/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <qnnpack.h>


class FullyConnectedOperatorTester {
 public:
  inline FullyConnectedOperatorTester& inputChannels(size_t inputChannels) {
    assert(inputChannels >= 1);
    this->inputChannels_ = inputChannels;
    return *this;
  }

  inline size_t inputChannels() const {
    return this->inputChannels_;
  }

  inline FullyConnectedOperatorTester& outputChannels(size_t outputChannels) {
    assert(outputChannels >= 1);
    this->outputChannels_ = outputChannels;
    return *this;
  }

  inline size_t outputChannels() const {
    return this->outputChannels_;
  }

  inline FullyConnectedOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline FullyConnectedOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride >= 1);
    this->inputStride_ = inputStride;
    return *this;
  }

  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return inputChannels();
    } else {
      assert(this->inputStride_ >= inputChannels());
      return this->inputStride_;
    }
  }

  inline FullyConnectedOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride >= 1);
    this->outputStride_ = outputStride;
    return *this;
  }

  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return outputChannels();
    } else {
      assert(this->outputStride_ >= outputChannels());
      return this->outputStride_;
    }
  }

  inline FullyConnectedOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline FullyConnectedOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline FullyConnectedOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void testQ8() const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> input((batchSize() - 1) * inputStride() + inputChannels() + 8);
    std::vector<uint8_t> kernel(outputChannels() * inputChannels());
    std::vector<int32_t> bias(outputChannels());
    std::vector<uint8_t> output((batchSize() - 1) * outputStride() + outputChannels());
    std::vector<int32_t> accumulators(batchSize() * outputChannels());

    const uint8_t* inputPtr = input.data() + 8;
    const uint8_t inputZeroPoint = 127;
    const uint8_t kernelZeroPoint = 127;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(output.begin(), output.end(), 0xA5);
      std::fill(accumulators.begin(), accumulators.end(), 0);

      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t oc = 0; oc < outputChannels(); oc++) {
          accumulators[i * outputChannels() + oc] = bias[oc];
        }
      }
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t oc = 0; oc < outputChannels(); oc++) {
          for (size_t ic = 0; ic < inputChannels(); ic++) {
            accumulators[i * outputChannels() + oc] +=
              (int32_t(inputPtr[i * inputStride() + ic]) - int32_t(inputZeroPoint)) *
              (int32_t(kernel[oc * inputChannels() + ic]) - int32_t(kernelZeroPoint));
          }
        }
      }
      // Create dummy min/max for empty inputs.
      // These are only used to compute scale and zero point,
      // and real callers will just pull those values from the model.
      const int32_t accumulatorsMin = accumulators.empty() ?   0 : *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulatorsMax = accumulators.empty() ? 900 : *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double outputScale = double(uint32_t(accumulatorsMax - accumulatorsMin)) / 255.0;
      const uint8_t outputZeroPoint = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accumulatorsMin + accumulatorsMax) / outputScale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      ASSERT_EQ(qnnp_status_success, qnnp_initialize());
      qnnp_operator_t convolution = nullptr;


      ASSERT_EQ(qnnp_status_success,
        qnnp_create_fully_connected_nc_q8(
          inputChannels(), outputChannels(),
          inputZeroPoint, 1.0f /* input scale */,
          kernelZeroPoint, 1.0f /* kernel scale */,
          kernel.data(), bias.data(),
          outputZeroPoint, outputScale, qmin(), qmax(),
          0, &convolution));

      ASSERT_EQ(qnnp_status_success,
        qnnp_setup_fully_connected_nc_q8(
          convolution,
          batchSize(),
          inputPtr,
          inputStride(),
          output.data(),
          outputStride()));

      ASSERT_EQ(qnnp_status_success,
        qnnp_run_operator(convolution, nullptr /* thread pool */));

      ASSERT_EQ(qnnp_status_success,
        qnnp_delete_operator(convolution));
      convolution = nullptr;

      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < outputChannels(); c++) {
          const double scaledAccumulator = accumulators[i * outputChannels() + c] / outputScale;
          const double clampedAccumulator = std::max(std::min(scaledAccumulator,
            double(qmax()) - double(outputZeroPoint)),
            double(qmin()) - double(outputZeroPoint));
          ASSERT_NEAR(
            clampedAccumulator,
            (int32_t(output[i * outputStride() + c]) - outputZeroPoint),
            0.9) << "batch index = " << i << ", channel = " << c;
        }
      }
    }
  }

 private:
  size_t inputChannels_{1};
  size_t inputStride_{0};
  size_t outputChannels_{1};
  size_t outputStride_{0};
  size_t batchSize_{1};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
