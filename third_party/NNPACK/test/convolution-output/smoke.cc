#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/convolution.h>

/*
 * Test that implementation works for a single tile of transformation
 */

TEST(FT8x8, single_tile) {
	ConvolutionTester()
		.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, single_tile_with_relu) {
	ConvolutionTester()
		.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, single_tile) {
	ConvolutionTester()
		.inputSize(16, 16)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, single_tile_with_relu) {
	ConvolutionTester()
		.inputSize(16, 16)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, single_tile) {
	ConvolutionTester()
		.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-3)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, single_tile_with_relu) {
	ConvolutionTester()
		.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-3)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

/*
 * Test that the implementation handles extraction of input subtile
 */

TEST(FT8x8, input_subtile) {
	ConvolutionTester()
		.inputSize(4, 4)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, input_subtile_with_relu) {
	ConvolutionTester()
		.inputSize(4, 4)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, input_subtile) {
	ConvolutionTester()
		.inputSize(4, 4)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, input_subtile_with_relu) {
	ConvolutionTester()
		.inputSize(4, 4)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, input_subtile) {
	ConvolutionTester()
		.inputSize(4, 4)
		.iterations(100)
		.errorLimit(1.0e-4)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, input_subtile_with_relu) {
	ConvolutionTester()
		.inputSize(4, 4)
		.iterations(100)
		.errorLimit(1.0e-4)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

/*
 * Test that the implementation handles multi-tile inputs
 */

TEST(FT8x8, multi_tile) {
	ConvolutionTester()
		.inputSize(13, 13)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, multi_tile_with_relu) {
	ConvolutionTester()
		.inputSize(13, 13)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, multi_tile) {
	ConvolutionTester()
		.inputSize(29, 29)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, multi_tile_with_relu) {
	ConvolutionTester()
		.inputSize(29, 29)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, multi_tile) {
	ConvolutionTester()
		.inputSize(13, 13)
		.iterations(100)
		.errorLimit(1.0e-3)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, multi_tile_with_relu) {
	ConvolutionTester()
		.inputSize(13, 13)
		.iterations(100)
		.errorLimit(1.0e-3)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

/*
 * Test that the implementation handles implicit padding of input
 */

TEST(FT8x8, implicit_padding) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.kernelSize(5, 5)
		.iterations(5)
		.errorLimit(5.0e-2);
	for (size_t paddingTop = 0; paddingTop < tester.kernelHeight(); paddingTop++) {
		for (size_t paddingRight = 0; paddingRight < tester.kernelWidth(); paddingRight++) {
			for (size_t paddingLeft = 0; paddingLeft < tester.kernelWidth(); paddingLeft++) {
				for (size_t paddingBottom = 0; paddingBottom < tester.kernelHeight(); paddingBottom++) {
					tester.inputPadding(paddingTop, paddingRight, paddingBottom, paddingLeft)
						.testOutput(nnp_convolution_algorithm_ft8x8);
				}
			}
		}
	}
}

TEST(FT8x8, implicit_padding_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.kernelSize(5, 5)
		.iterations(5)
		.errorLimit(5.0e-2);
	for (size_t paddingTop = 0; paddingTop < tester.kernelHeight(); paddingTop++) {
		for (size_t paddingRight = 0; paddingRight < tester.kernelWidth(); paddingRight++) {
			for (size_t paddingLeft = 0; paddingLeft < tester.kernelWidth(); paddingLeft++) {
				for (size_t paddingBottom = 0; paddingBottom < tester.kernelHeight(); paddingBottom++) {
					tester.inputPadding(paddingTop, paddingRight, paddingBottom, paddingLeft)
						.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
				}
			}
		}
	}
}

TEST(FT16x16, implicit_padding) {
	ConvolutionTester tester;
	tester.inputSize(16, 16)
		.kernelSize(5, 5)
		.iterations(5)
		.errorLimit(5.0e-2);
	for (size_t paddingTop = 0; paddingTop < tester.kernelHeight(); paddingTop++) {
		for (size_t paddingRight = 0; paddingRight < tester.kernelWidth(); paddingRight++) {
			for (size_t paddingLeft = 0; paddingLeft < tester.kernelWidth(); paddingLeft++) {
				for (size_t paddingBottom = 0; paddingBottom < tester.kernelHeight(); paddingBottom++) {
					tester.inputPadding(paddingTop, paddingRight, paddingBottom, paddingLeft)
						.testOutput(nnp_convolution_algorithm_ft16x16);
				}
			}
		}
	}
}

TEST(FT16x16, implicit_padding_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(16, 16)
		.kernelSize(5, 5)
		.iterations(5)
		.errorLimit(5.0e-2);
	for (size_t paddingTop = 0; paddingTop < tester.kernelHeight(); paddingTop++) {
		for (size_t paddingRight = 0; paddingRight < tester.kernelWidth(); paddingRight++) {
			for (size_t paddingLeft = 0; paddingLeft < tester.kernelWidth(); paddingLeft++) {
				for (size_t paddingBottom = 0; paddingBottom < tester.kernelHeight(); paddingBottom++) {
					tester.inputPadding(paddingTop, paddingRight, paddingBottom, paddingLeft)
						.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
				}
			}
		}
	}
}

TEST(WT8x8, implicit_padding) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.kernelSize(3, 3)
		.iterations(15)
		.errorLimit(1.0e-1);
	for (size_t paddingTop = 0; paddingTop < tester.kernelHeight(); paddingTop++) {
		for (size_t paddingRight = 0; paddingRight < tester.kernelWidth(); paddingRight++) {
			for (size_t paddingLeft = 0; paddingLeft < tester.kernelWidth(); paddingLeft++) {
				for (size_t paddingBottom = 0; paddingBottom < tester.kernelHeight(); paddingBottom++) {
					tester.inputPadding(paddingTop, paddingRight, paddingBottom, paddingLeft)
						.testOutput(nnp_convolution_algorithm_wt8x8);
				}
			}
		}
	}
}

TEST(WT8x8, implicit_padding_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.kernelSize(3, 3)
		.iterations(15)
		.errorLimit(1.0e-1);
	for (size_t paddingTop = 0; paddingTop < tester.kernelHeight(); paddingTop++) {
		for (size_t paddingRight = 0; paddingRight < tester.kernelWidth(); paddingRight++) {
			for (size_t paddingLeft = 0; paddingLeft < tester.kernelWidth(); paddingLeft++) {
				for (size_t paddingBottom = 0; paddingBottom < tester.kernelHeight(); paddingBottom++) {
					tester.inputPadding(paddingTop, paddingRight, paddingBottom, paddingLeft)
						.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
				}
			}
		}
	}
}

/*
 * Test that the implementation can handle small non-unit batch size
 */

TEST(FT8x8, small_batch) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t batchSize = 2; batchSize <= 5; batchSize++) {
		tester.batchSize(batchSize).testOutput(nnp_convolution_algorithm_ft8x8);
	}
}

TEST(FT8x8, small_batch_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t batchSize = 2; batchSize <= 5; batchSize++) {
		tester.batchSize(batchSize)
			.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
	}
}

TEST(FT16x16, small_batch) {
	ConvolutionTester tester;
	tester.inputSize(16, 16)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t batchSize = 2; batchSize <= 5; batchSize++) {
		tester.batchSize(batchSize).testOutput(nnp_convolution_algorithm_ft16x16);
	}
}

TEST(FT16x16, small_batch_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(16, 16)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t batchSize = 2; batchSize <= 5; batchSize++) {
		tester.batchSize(batchSize)
			.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
	}
}

TEST(WT8x8, small_batch) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-3);
	for (size_t batchSize = 2; batchSize <= 5; batchSize++) {
		tester.batchSize(batchSize).testOutput(nnp_convolution_algorithm_wt8x8);
	}
}

TEST(WT8x8, small_batch_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-3);
	for (size_t batchSize = 2; batchSize <= 5; batchSize++) {
		tester.batchSize(batchSize)
			.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
	}
}

/*
 * Test that the implementation can handle small non-unit number of input channels
 */

TEST(FT8x8, few_input_channels) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t inputChannels = 2; inputChannels <= 5; inputChannels++) {
		tester.inputChannels(inputChannels).testOutput(nnp_convolution_algorithm_ft8x8);
	}
}

TEST(FT8x8, few_input_channels_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t inputChannels = 2; inputChannels <= 5; inputChannels++) {
		tester.inputChannels(inputChannels)
			.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
	}
}

TEST(FT16x16, few_input_channels) {
	ConvolutionTester tester;
	tester.inputSize(16, 16)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t inputChannels = 2; inputChannels <= 5; inputChannels++) {
		tester.inputChannels(inputChannels).testOutput(nnp_convolution_algorithm_ft16x16);
	}
}

TEST(FT16x16, few_input_channels_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(16, 16)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t inputChannels = 2; inputChannels <= 5; inputChannels++) {
		tester.inputChannels(inputChannels)
			.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
	}
}

TEST(WT8x8, few_input_channels) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-3);
	for (size_t inputChannels = 2; inputChannels <= 5; inputChannels++) {
		tester.inputChannels(inputChannels).testOutput(nnp_convolution_algorithm_wt8x8);
	}
}

TEST(WT8x8, few_input_channels_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-3);
	for (size_t inputChannels = 2; inputChannels <= 5; inputChannels++) {
		tester.inputChannels(inputChannels)
			.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
	}
}

/*
 * Test that the implementation can handle small non-unit number of output channels
 */

TEST(FT8x8, few_output_channels) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t outputChannels = 2; outputChannels <= 5; outputChannels++) {
		tester.outputChannels(outputChannels).testOutput(nnp_convolution_algorithm_ft8x8);
	}
}

TEST(FT8x8, few_output_channels_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t outputChannels = 2; outputChannels <= 5; outputChannels++) {
		tester.outputChannels(outputChannels)
			.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
	}
}

TEST(FT16x16, few_output_channels) {
	ConvolutionTester tester;
	tester.inputSize(16, 16)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t outputChannels = 2; outputChannels <= 5; outputChannels++) {
		tester.outputChannels(outputChannels).testOutput(nnp_convolution_algorithm_ft16x16);
	}
}

TEST(FT16x16, few_output_channels_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(16, 16)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t outputChannels = 2; outputChannels <= 5; outputChannels++) {
		tester.outputChannels(outputChannels)
			.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
	}
}

TEST(WT8x8, few_output_channels) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-3);
	for (size_t outputChannels = 2; outputChannels <= 5; outputChannels++) {
		tester.outputChannels(outputChannels).testOutput(nnp_convolution_algorithm_wt8x8);
	}
}

TEST(WT8x8, few_output_channels_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-3);
	for (size_t outputChannels = 2; outputChannels <= 5; outputChannels++) {
		tester.outputChannels(outputChannels)
			.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
	}
}

/*
 * Test that the implementation can handle non-square kernels
 */

TEST(FT8x8, non_square_kernel) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.kernelSize(2, 3)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, non_square_kernel_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(8, 8)
		.kernelSize(2, 3)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, non_square_kernel) {
	ConvolutionTester tester;
	tester.inputSize(16, 16)
		.kernelSize(2, 3)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, non_square_kernel_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(16, 16)
		.kernelSize(2, 3)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

/*
 * Test that the implementation can handle non-square images
 */

TEST(FT8x8, non_square_image) {
	ConvolutionTester tester;
	tester.inputSize(9, 10)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, non_square_image_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(9, 10)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, non_square_image) {
	ConvolutionTester tester;
	tester.inputSize(17, 19)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, non_square_image_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(17, 19)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, non_square_image) {
	ConvolutionTester tester;
	tester.inputSize(9, 10)
		.iterations(100)
		.errorLimit(1.0e-3)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, non_square_image_with_relu) {
	ConvolutionTester tester;
	tester.inputSize(9, 10)
		.iterations(100)
		.errorLimit(1.0e-3)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
