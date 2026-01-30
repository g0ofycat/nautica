#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "../neural_networks/architectures/cnn/cnn.hpp"
#include "../neural_networks/architectures/cnn/kernels/kernels.hpp"
#include "../neural_networks/tensors/tensor.hpp"
#include "../processing/images/image_extractor.hpp"
#include "./nn_interface.hpp"

// ====== PUBLIC FUNCTIONS ======

/// @brief Convolute an image given a path and number of convolutions (RGB)
/// @param image_path The path to the image file
/// @param convolutions How many convolution layers to apply
/// @return The convolved image tensor
Tensor nn_interface::convolute_image(const std::string &image_path, size_t convolutions) {
    Tensor image_tensor = image_extractor::load_image(image_path);
    image_extractor::normalize(image_tensor);

    std::vector<Tensor> filters(convolutions);

#pragma omp parallel for
    for (size_t i = 0; i < convolutions; ++i) {
        filters[i] = Tensor::random_normal({3, 3, 3});
    }

    return convolutional_neural_network::convolve_3d(image_tensor, filters);
}

/// @brief Apply multiple 2D convolutions with randomly selected kernels
/// @param input_tensor The input tensor (2D)
/// @param convolutions Number of different kernels to apply
/// @param stride Stride for convolution
/// @return Vector of output feature maps (one per randomly selected kernel)
std::vector<Tensor> nn_interface::convolute_tensor(const Tensor &input_tensor, size_t convolutions, size_t stride) {
    std::vector<Tensor> results(convolutions);
    std::mt19937 &rng = nn_utility::get_rng();
    std::uniform_int_distribution<size_t> dist(0, Kernels::all_kernels.size() - 1);

#pragma omp parallel for
    for (size_t i = 0; i < convolutions; ++i) {
        size_t kernel_idx = dist(rng);

        const auto &kernel = Kernels::all_kernels[kernel_idx];
        size_t kernel_size = static_cast<size_t>(std::sqrt(kernel.size()));

        results[i] = convolutional_neural_network::convolve_2d(input_tensor, kernel, kernel_size, stride);
    }

    return results;
}
