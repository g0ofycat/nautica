#include <cstddef>
#include <cstring>
#include <omp.h>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "../../utility/nn_utility.hpp"
#include "../tensors/tensor.hpp"
#include "./cnn.hpp"

// ====== PUBLIC FUNCTIONS ======

/// @brief Apply a convolutional kernel to the input tensor
/// @param input The input tensor (2D)
/// @param kernel The Kernel to apply
/// @param kernel_size Size of the kernel
/// @param row Row index
/// @param col Column index
/// @param divisor Divisor for normalization
/// @return Result of applying the kernel
double convolutional_neural_network::apply_kernel(const Tensor &input, const std::vector<float> &kernel,
                                                  size_t kernel_size, size_t row, size_t col, double divisor) {
    if (input.shape.size() != 2)
        throw std::invalid_argument("Input must be a 2D Tensor");

    double result = 0.0;

    size_t half_kernel = kernel_size / 2;

    for (size_t k_row = 0; k_row < kernel_size; ++k_row) {
        for (size_t k_col = 0; k_col < kernel_size; ++k_col) {
            int in_row = static_cast<int>(row) + static_cast<int>(k_row) - static_cast<int>(half_kernel);
            int in_col = static_cast<int>(col) + static_cast<int>(k_col) - static_cast<int>(half_kernel);

            if (in_row >= 0 && in_row < static_cast<int>(input.shape[0]) && in_col >= 0 &&
                in_col < static_cast<int>(input.shape[1])) {
                result += input.at({static_cast<size_t>(in_row), static_cast<size_t>(in_col)}) *
                          kernel[k_row * kernel_size + k_col];
            }
        }
    }

    return divisor != 0 ? result / divisor : result;
}

/// @brief Apply one 3D filter at a specific position
/// @param input Input tensor (H x W x C_in)
/// @param filter Filter tensor (K x K x C_in)
/// @param row Starting row position
/// @param col Starting column position
/// @param divisor Divisor for normalization
/// @return Convolution result (single value)
double convolutional_neural_network::apply_filter_3d(const Tensor &input, const Tensor &filter, size_t row, size_t col,
                                                     double divisor) {
    if (input.shape.size() != 3 || filter.shape.size() != 3)
        throw std::invalid_argument("Input and filter must be 3D");

    if (input.shape[2] != filter.shape[2])
        throw std::invalid_argument("Input and filter channel count must match");

    size_t kernel_h = filter.shape[0];
    size_t kernel_w = filter.shape[1];

    size_t channels = input.shape[2];

    size_t half_h = kernel_h / 2;
    size_t half_w = kernel_w / 2;

    const double *input_data = input.data.data();
    const double *filter_data = filter.data.data();

    size_t input_w = input.shape[1];
    size_t input_stride = input_w * channels;

    double result = 0.0;

    for (size_t k_row = 0; k_row < kernel_h; ++k_row) {
        for (size_t k_col = 0; k_col < kernel_w; ++k_col) {
            int in_row = static_cast<int>(row) + static_cast<int>(k_row) - static_cast<int>(half_h);
            int in_col = static_cast<int>(col) + static_cast<int>(k_col) - static_cast<int>(half_w);

            if (in_row >= 0 && in_row < static_cast<int>(input.shape[0]) && in_col >= 0 &&
                in_col < static_cast<int>(input.shape[1])) {
                size_t input_offset = in_row * input_stride + in_col * channels;
                size_t filter_offset = (k_row * kernel_w + k_col) * channels;

                size_t c = 0;

#ifdef __AVX2__
                __m256d sum = _mm256_setzero_pd();

                for (; c + 4 <= channels; c += 4) {
                    __m256d in = _mm256_loadu_pd(&input_data[input_offset + c]);
                    __m256d fil = _mm256_loadu_pd(&filter_data[filter_offset + c]);
                    sum = _mm256_add_pd(sum, _mm256_mul_pd(in, fil));
                }

                double temp[4];
                _mm256_storeu_pd(temp, sum);
                result += temp[0] + temp[1] + temp[2] + temp[3];
#endif
                for (; c < channels; ++c) {
                    result += input_data[input_offset + c] * filter_data[filter_offset + c];
                }
            }
        }
    }

    return divisor != 0.0 ? result / divisor : result;
}

/// @brief Perform 2D convolution on input tensor
/// @param input The input tensor (2D)
/// @param kernel The kernel to apply
/// @param kernel_size Size of the kernel (assumes square)
/// @param stride Stride for the convolution
/// @param divisor Divisor for normalization
/// @return Convolved output tensor
Tensor convolutional_neural_network::convolve_2d(const Tensor &input, const std::vector<float> &kernel,
                                                 size_t kernel_size, size_t stride, double divisor) {
    if (input.shape.size() != 2)
        throw std::invalid_argument("Input must be a 2D Tensor");

    if (input.shape[0] < kernel_size || input.shape[1] < kernel_size)
        throw std::invalid_argument("Input dimensions must be >= kernel size");

    size_t out_h = (input.shape[0] - kernel_size) / stride + 1;
    size_t out_w = (input.shape[1] - kernel_size) / stride + 1;

    Tensor output(std::vector<size_t>{out_h, out_w});

    for (size_t row = 0; row < out_h; ++row) {
        for (size_t col = 0; col < out_w; ++col) {
            size_t in_row = row * stride;
            size_t in_col = col * stride;

            output.at({row, col}) = apply_kernel(input, kernel, kernel_size, in_row, in_col, divisor);
        }
    }

    return output;
}

/// @brief Perform 3D convolution (handles multi-channel input)
/// @param input Input tensor (H x W x C_in)
/// @param filters Vector of filters, each (K x K x C_in)
/// @param stride Stride for convolution
/// @param divisor Divisor for normalization
/// @return Output tensor (H_out x W_out x num_filters)
Tensor convolutional_neural_network::convolve_3d(const Tensor &input, const std::vector<Tensor> &filters, size_t stride,
                                                 double divisor) {
    if (input.shape.size() != 3)
        throw std::invalid_argument("Input must be 3D (H x W x C)");

    if (filters.empty())
        throw std::invalid_argument("Must provide at least one filter");

    if (filters[0].shape.size() != 3)
        throw std::invalid_argument("Filters must be 3D");

    if (filters[0].shape[2] != input.shape[2])
        throw std::invalid_argument("Filter channels must match input");

    size_t kernel_h = filters[0].shape[0];
    size_t kernel_w = filters[0].shape[1];

    if (input.shape[0] < kernel_h || input.shape[1] < kernel_w)
        throw std::invalid_argument("Input dimensions must be >= kernel size");

    size_t num_filters = filters.size();

    size_t out_h = (input.shape[0] - kernel_h) / stride + 1;
    size_t out_w = (input.shape[1] - kernel_w) / stride + 1;

    Tensor output(std::vector<size_t>{out_h, out_w, num_filters});

#pragma omp parallel for collapse(3)
    for (size_t f = 0; f < num_filters; ++f) {
        for (size_t i = 0; i < out_h; ++i) {
            for (size_t j = 0; j < out_w; ++j) {
                size_t in_row = i * stride;
                size_t in_col = j * stride;

                output.at({i, j, f}) = apply_filter_3d(input, filters[f], in_row, in_col, divisor);
            }
        }
    }

    return output;
}

/// @brief Add padding to a 2D tensor to align with kernel operations
/// @param input The input tensor (2D)
/// @param pad The padding size
/// @return Padded tensor
Tensor convolutional_neural_network::add_padding(const Tensor &input, size_t pad) {
    if (input.shape.size() != 2)
        throw std::invalid_argument("Input must be a 2D Tensor");

    size_t old_h = input.shape[0];
    size_t old_w = input.shape[1];
    size_t new_h = old_h + 2 * pad;
    size_t new_w = old_w + 2 * pad;

    Tensor padded(std::vector<size_t>{new_h, new_w});

    const double *src = input.data.data();
    double *dst = padded.data.data();

    std::fill(dst, dst + padded.numel(), 0.0);

    for (size_t i = 0; i < old_h; ++i) {
        size_t src_offset = i * old_w;
        size_t dst_offset = (i + pad) * new_w + pad;
        std::memcpy(&dst[dst_offset], &src[src_offset], old_w * sizeof(double));
    }

    return padded;
}

/// @brief Max Pooling operation on a 2D tensor
/// @param input The input tensor (2D)
/// @param pool_size Size of the pooling window
/// @param stride Stride for the pooling operation
/// @return Pooled output tensor
Tensor convolutional_neural_network::max_pool(const Tensor &input, size_t pool_size, size_t stride) {
    if (input.shape.size() != 2)
        throw std::invalid_argument("Input must be a 2D Tensor");

    if (input.shape[0] < pool_size || input.shape[1] < pool_size)
        throw std::invalid_argument("Input dimensions must be >= pool size");

    size_t out_h = (input.shape[0] - pool_size) / stride + 1;
    size_t out_w = (input.shape[1] - pool_size) / stride + 1;

    Tensor output(std::vector<size_t>{out_h, out_w});

    const double *input_data = input.data.data();
    double *output_data = output.data.data();
    size_t input_w = input.shape[1];

    for (size_t i = 0; i < out_h; ++i) {
        for (size_t j = 0; j < out_w; ++j) {
            double max_val = -std::numeric_limits<double>::infinity();
            size_t base_row = i * stride;
            size_t base_col = j * stride;

            for (size_t pi = 0; pi < pool_size; ++pi) {
                size_t row_offset = (base_row + pi) * input_w;
                for (size_t pj = 0; pj < pool_size; ++pj) {
                    max_val = std::max(max_val, input_data[row_offset + base_col + pj]);
                }
            }

            output_data[i * out_w + j] = max_val;
        }
    }

    return output;
}