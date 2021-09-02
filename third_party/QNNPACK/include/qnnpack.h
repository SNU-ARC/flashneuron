/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <pthreadpool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status code for any QNNPACK function call.
 */
enum qnnp_status {
  /** The call succeeded, and all output arguments now contain valid data. */
  qnnp_status_success = 0,
  qnnp_status_uninitialized = 1,
  qnnp_status_invalid_parameter = 2,
  qnnp_status_unsupported_parameter = 3,
  qnnp_status_unsupported_hardware = 4,
  qnnp_status_out_of_memory = 5,
};

enum qnnp_status qnnp_initialize(void);

enum qnnp_status qnnp_deinitialize(void);

typedef struct qnnp_operator* qnnp_operator_t;

enum qnnp_status qnnp_create_convolution2d_nhwc_q8(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t subsampling_height,
    uint32_t subsampling_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t kernel_zero_point,
    float kernel_scale,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    qnnp_operator_t* convolution);

enum qnnp_status qnnp_setup_convolution2d_nhwc_q8(
    qnnp_operator_t convolution,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride,
    pthreadpool_t threadpool);

enum qnnp_status qnnp_create_deconvolution2d_nhwc_q8(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t adjustment_height,
    uint32_t adjustment_width,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t kernel_zero_point,
    float kernel_scale,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    qnnp_operator_t* deconvolution);

enum qnnp_status qnnp_setup_deconvolution2d_nhwc_q8(
    qnnp_operator_t deconvolution,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride,
    pthreadpool_t threadpool);

enum qnnp_status qnnp_create_fully_connected_nc_q8(
    size_t input_channels,
    size_t output_channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t kernel_zero_point,
    float kernel_scale,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    qnnp_operator_t* fully_connected);

enum qnnp_status qnnp_setup_fully_connected_nc_q8(
    qnnp_operator_t fully_connected,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum qnnp_status qnnp_create_global_average_pooling_nwc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    qnnp_operator_t* global_average_pooling);

enum qnnp_status qnnp_setup_global_average_pooling_nwc_q8(
    qnnp_operator_t global_average_pooling,
    size_t batch_size,
    size_t width,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum qnnp_status qnnp_create_average_pooling2d_nhwc_q8(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t pooling_height,
    uint32_t pooling_width,
    uint32_t stride_height,
    uint32_t stride_width,
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    qnnp_operator_t* average_pooling);

enum qnnp_status qnnp_setup_average_pooling2d_nhwc_q8(
    qnnp_operator_t average_pooling,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride,
    pthreadpool_t threadpool);

enum qnnp_status qnnp_create_max_pooling2d_nhwc_u8(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t pooling_height,
    uint32_t pooling_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    size_t channels,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    qnnp_operator_t* max_pooling);

enum qnnp_status qnnp_setup_max_pooling2d_nhwc_u8(
    qnnp_operator_t max_pooling,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride,
    pthreadpool_t threadpool);

enum qnnp_status qnnp_create_channel_shuffle_nc_x8(
    size_t groups,
    size_t group_channels,
    uint32_t flags,
    qnnp_operator_t* channel_shuffle);

enum qnnp_status qnnp_setup_channel_shuffle_nc_x8(
    qnnp_operator_t channel_shuffle,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum qnnp_status qnnp_create_add_nc_q8(
    size_t channels,
    uint8_t a_zero_point,
    float a_scale,
    uint8_t b_zero_point,
    float b_scale,
    uint8_t sum_zero_point,
    float sum_scale,
    uint8_t sum_min,
    uint8_t sum_max,
    uint32_t flags,
    qnnp_operator_t* add);

enum qnnp_status qnnp_setup_add_nc_q8(
    qnnp_operator_t add,
    size_t batch_size,
    const uint8_t* a,
    size_t a_stride,
    const uint8_t* b,
    size_t b_stride,
    uint8_t* sum,
    size_t sum_stride);

enum qnnp_status qnnp_create_clamp_nc_u8(
    size_t channels,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    qnnp_operator_t* clamp);

enum qnnp_status qnnp_setup_clamp_nc_u8(
    qnnp_operator_t clamp,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum qnnp_status qnnp_create_sigmoid_nc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    qnnp_operator_t* sigmoid);

enum qnnp_status qnnp_setup_sigmoid_nc_q8(
    qnnp_operator_t sigmoid,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum qnnp_status qnnp_create_leaky_relu_nc_q8(
    size_t channels,
    float negative_slope,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    qnnp_operator_t* leaky_relu);

enum qnnp_status qnnp_setup_leaky_relu_nc_q8(
    qnnp_operator_t leaky_relu,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum qnnp_status qnnp_create_softargmax_nc_q8(
    size_t channels,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint32_t flags,
    qnnp_operator_t* softargmax);

enum qnnp_status qnnp_setup_softargmax_nc_q8(
    qnnp_operator_t softargmax,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum qnnp_status qnnp_run_operator(
    qnnp_operator_t op,
    pthreadpool_t threadpool);

enum qnnp_status qnnp_delete_operator(
    qnnp_operator_t op);

#ifdef __cplusplus
} /* extern "C" */
#endif
