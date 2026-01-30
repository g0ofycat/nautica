#include <cmath>
#include <cstddef>
#include <optional>
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
/// @param kernel Optional Kernel to apply instead of a random normal 3x3x3 Tensor
/// @return The convolved image tensor
Tensor nn_interface::convolute_image(const std::string &image_path, const size_t convolutions,
                                     const std::optional<std::vector<int>> kernel) {
    Tensor image_tensor = image_extractor::load_image(image_path);
    image_extractor::normalize(image_tensor);

    Tensor result = image_tensor;

    for (size_t conv_idx = 0; conv_idx < convolutions; ++conv_idx) {
        size_t in_channels = result.shape[2];
        std::vector<Tensor> filters(3);

        if (kernel.has_value()) {
            size_t kernel_size = std::sqrt(kernel->size());

#pragma omp parallel for
            for (size_t i = 0; i < 3; ++i) {
                Tensor filter(std::vector<size_t>{kernel_size, kernel_size, in_channels});

                for (size_t kr = 0; kr < kernel_size; ++kr) {
                    for (size_t kc = 0; kc < kernel_size; ++kc) {
                        double val = static_cast<double>((*kernel)[kr * kernel_size + kc]);

                        for (size_t ch = 0; ch < in_channels; ++ch) {
                            filter.at({kr, kc, ch}) = val;
                        }
                    }
                }

                filters[i] = filter;
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < 3; ++i) {
                filters[i] = Tensor::random_normal({3, 3, in_channels}, 0.0, 0.1);
            }
        }

        result = convolutional_neural_network::convolve_3d(result, filters);
    }

    return result;
}

/// @brief Apply multiple 2D convolutions with randomly selected kernels
/// @param input_tensor The input tensor (2D)
/// @param convolutions Number of different kernels to apply
/// @param stride Stride for convolution
/// @return Vector of output feature maps (one per randomly selected kernel)
std::vector<Tensor> nn_interface::convolute_tensor(const Tensor &input_tensor, const size_t convolutions,
                                                   const size_t stride) {
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