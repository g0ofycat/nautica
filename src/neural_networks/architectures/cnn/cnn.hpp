#ifndef CNN_HPP
#define CNN_HPP

#include <cstddef>
#include <vector>

#include "../../utility/nn_utility.hpp"
#include "../tensors/tensor.hpp"

// ====== convolutional_neural_network ======

class convolutional_neural_network {
  public:
    // ====== PUBLIC FUNCTIONS ======

    /// @brief Apply a convolutional kernel to the input tensor
    /// @param input The input tensor (2D)
    /// @param kernel The Kernel to apply
    /// @param kernel_size Size of the kernel (assumes square)
    /// @param row Row index
    /// @param col Column index
    /// @param divisor Divisor for normalization
    /// @return Result of applying the kernel
    static double apply_kernel(const Tensor &input, const std::vector<int> &kernel, size_t kernel_size, size_t row,
                               size_t col, double divisor = 1);

    /// @brief Apply one 3D filter at a specific position
    /// @param input Input tensor (H x W x C_in)
    /// @param filter Filter tensor (K x K x C_in)
    /// @param row Starting row position
    /// @param col Starting column position
    /// @param divisor Divisor for normalization
    /// @return Convolution result (single value)
    static double apply_filter_3d(const Tensor &input, const Tensor &filter, size_t row, size_t col,
                                  double divisor = 1);

    /// @brief Perform 2D convolution on input tensor
    /// @param input The input tensor (2D)
    /// @param kernel The kernel to apply
    /// @param kernel_size Size of the kernel (assumes square)
    /// @param stride Stride for the convolution
    /// @param divisor Divisor for normalization
    /// @return Convolved output tensor
    static Tensor convolve_2d(const Tensor &input, const std::vector<int> &kernel, size_t kernel_size,
                              size_t stride = 1, double divisor = 1);

    /// @brief Perform 3D convolution (handles multi-channel input)
    /// @param input Input tensor (H x W x C_in)
    /// @param filters Vector of filters, each (K x K x C_in)
    /// @param stride Stride for convolution
    /// @param divisor Divisor for normalization
    /// @return Output tensor (H_out x W_out x num_filters)
    static Tensor convolve_3d(const Tensor &input, const std::vector<Tensor> &filters, size_t stride = 1,
                              double divisor = 1);

    /// @brief Add padding to a 2D tensor to align with kernel operations
    /// @param input The input tensor (2D)
    /// @param pad The padding size
    /// @return Padded tensor
    static Tensor add_padding(const Tensor &input, size_t pad);

    /// @brief Max Pooling operation on a 2D tensor
    /// @param input The input tensor (2D)
    /// @param pool_size Size of the pooling window
    /// @param stride Stride for the pooling operation
    /// @return Pooled output tensor
    static Tensor max_pool(const Tensor &input, size_t pool_size, size_t stride);
};

#endif