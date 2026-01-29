#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <vector>

// ====== Kernels ======

namespace Kernels
{
    // ====== EDGE DETECTION ======

    extern const std::vector<int> sobel_x;
    extern const std::vector<int> sobel_y;
    extern const std::vector<int> prewitt_x;
    extern const std::vector<int> prewitt_y;
    extern const std::vector<int> scharr_x;
    extern const std::vector<int> scharr_y;
    extern const std::vector<int> laplacian_4;
    extern const std::vector<int> laplacian_8;

    // ====== BLURRING ======

    extern const std::vector<int> box_blur_3x3;
    extern const std::vector<int> gaussian_3x3;
    extern const std::vector<int> gaussian_5x5;

    // ====== SHARPENING ======

    extern const std::vector<int> sharpen;
    extern const std::vector<int> edge_enhance;
    extern const std::vector<int> unsharp_mask;

    // ====== OTHER ======

    extern const std::vector<int> identity;
    extern const std::vector<int> emboss;
    extern const std::vector<int> outline;
    extern const std::vector<int> ridge_horizontal;
    extern const std::vector<int> ridge_vertical;
}

#endif