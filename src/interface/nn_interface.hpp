#ifndef NN_INTERFACE_HPP
#define NN_INTERFACE_HPP

#include <cstddef>

#include "../processing/images/image_extractor.hpp"
#include "../neural_networks/architectures/cnn/cnn.hpp"
#include "../neural_networks/tensors/tensor.hpp"

// ====== nn_interface ======

class nn_interface
{
public:
    // ====== PUBLIC FUNCTIONS ======

    /// @brief Convolute an image given a path and number of convolutions
    /// @param image_path The path to the image file
    /// @param convolutions How many convolution layers to apply
    /// @return The convolved image tensor
    static Tensor convolute_image(const std::string &image_path, size_t convolutions);
};

#endif