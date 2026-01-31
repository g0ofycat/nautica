#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <vector>

// ====== Kernels ======

namespace Kernels
{
    // ====== EDGE DETECTION ======

    extern const std::vector<float> sobel_x;
    extern const std::vector<float> sobel_y;
    extern const std::vector<float> prewitt_x;
    extern const std::vector<float> prewitt_y;
    extern const std::vector<float> scharr_x;
    extern const std::vector<float> scharr_y;
    extern const std::vector<float> laplacian_4;
    extern const std::vector<float> laplacian_8;

    // ====== BLURRING ======

    extern const std::vector<float> box_blur_3x3;
    extern const std::vector<float> gaussian_3x3;
    extern const std::vector<float> gaussian_5x5;

    // ====== SHARPENING ======

    extern const std::vector<float> sharpen;
    extern const std::vector<float> edge_enhance;
    extern const std::vector<float> unsharp_mask;

    // ====== OTHER ======

    extern const std::vector<float> identity;
    extern const std::vector<float> emboss;
    extern const std::vector<float> outline;
    extern const std::vector<float> ridge_horizontal;
    extern const std::vector<float> ridge_vertical;

    // ====== COLLECTIONS ======

    /// @brief All available kernels
    extern const std::vector<std::vector<float>> all_kernels;

    /// @brief Edge detection kernels only
    extern const std::vector<std::vector<float>> edge_detection_kernels;

    /// @brief Blurring kernels only
    extern const std::vector<std::vector<float>> blur_kernels;

    /// @brief Sharpening kernels only
    extern const std::vector<std::vector<float>> sharpen_kernels;
}

#endif