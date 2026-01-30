#ifndef NN_INTERFACE_HPP
#define NN_INTERFACE_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "../neural_networks/tensors/tensor.hpp"

// ====== nn_interface ======

class nn_interface {
  public:
    // ====== PUBLIC FUNCTIONS ======

    /// @brief Convolute an image given a path and number of convolutions (RGB)
    /// @param image_path The path to the image file
    /// @param convolutions How many convolution layers to apply
    /// @return The convolved image tensor
    static Tensor convolute_image(const std::string &image_path, size_t convolutions);

    /// @brief Apply multiple 2D convolutions with randomly selected kernels
    /// @param input_tensor The input tensor (2D)
    /// @param convolutions Number of different kernels to apply
    /// @param stride Stride for convolution
    /// @return Vector of output feature maps (one per randomly selected kernel)
    static std::vector<Tensor> convolute_tensor(const Tensor &input_tensor, size_t convolutions, size_t stride);
};

#endif
