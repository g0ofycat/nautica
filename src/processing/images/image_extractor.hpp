#ifndef IMAGE_EXTRACTOR_HPP
#define IMAGE_EXTRACTOR_HPP

#include <string>

#include "../../neural_networks/tensors/tensor.hpp"
#include "image_processing/stb_image.h"

// ====== image_extractor ======

class image_extractor {
  public:
    // ====== PUBLIC FUNCTIONS ======

    /// @brief Load an image file into a 3D tensor (H x W x C)
    /// @param filepath Path to the image file
    /// @param desired_channels Number of channels (1 = grayscale, 3 = RGB, 4 = RGBA)
    /// @return Tensor with shape (height, width, channels)
    static Tensor load_image(const std::string &filepath, int desired_channels = 3);

    /// @brief Save a tensor as an image file
    /// @param filepath Output path (supports .png, .jpg, .bmp)
    /// @param tensor The tensor to save (H x W x C)
    /// @return True if successful
    static bool save_image(const std::string &filepath, const Tensor &tensor);

    /// @brief Normalize pixel values from [0, 255] to [0, 1]
    /// @param tensor The tensor to normalize
    static void normalize(Tensor &tensor);

    /// @brief Resize image to target dimensions
    /// @param tensor Input tensor
    /// @param target_h Target height
    /// @param target_w Target width
    /// @return Resized tensor
    static Tensor resize(const Tensor &tensor, size_t target_h, size_t target_w);
};

#endif