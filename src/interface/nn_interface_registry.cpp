#include <cstddef>

#include "./nn_interface.hpp"
#include "../processing/images/image_extractor.hpp"
#include "../neural_networks/architectures/cnn/cnn.hpp"
#include "../neural_networks/tensors/tensor.hpp"

// ====== PUBLIC FUNCTIONS ======

/// @brief Convolute an image given a path and number of convolutions (RGB)
/// @param image_path The path to the image file
/// @param convolutions How many convolution layers to apply
/// @return The convolved image tensor
Tensor nn_interface::convolute_image(const std::string &image_path, size_t convolutions)
{
    Tensor image_tensor = image_extractor::load_image(image_path);
    image_extractor::normalize(image_tensor);

    std::vector<Tensor> filters(convolutions);

#pragma omp parallel for
    for (size_t i = 0; i < convolutions; ++i)
    {
        filters[i] = Tensor::random_normal({3, 3, 3});
    }

    return convolutional_neural_network::convolve_3d(image_tensor, filters, 1);
}