#ifndef MPSCNNTests_h
#define MPSCNNTests_h

bool test_synchronization();
bool test_nchw_to_nc4_cpu();
bool test_copy_nchw_to_metal();
bool test_conv2d();
bool test_depthwiseConv();
bool test_max_pool2d();
bool test_relu();
bool test_addmm();
bool test_add();
bool test_add_broadcast();
bool test_sub();
bool test_sub_broadcast();
bool test_sub_broadcast2();
bool test_mul();
bool test_mul_broadcast();
bool test_mul_broadcast2();
bool test_t();
bool test_view();
bool test_softmax();
bool test_sigmoid();
bool test_hardsigmoid();
bool test_hardswish();
bool test_upsampling_nearest2d_vec();
bool test_adaptive_avg_pool2d();
bool test_hardtanh_();
bool test_reshape();

#endif
