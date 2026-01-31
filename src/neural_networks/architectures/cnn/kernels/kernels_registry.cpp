#include "./kernels.hpp"

// ====== Kernels ======

namespace Kernels
{
    // ====== EDGE DETECTION ======

    const std::vector<float> sobel_x = {
        -1.f, 0.f, 1.f,
        -2.f, 0.f, 2.f,
        -1.f, 0.f, 1.f};

    const std::vector<float> sobel_y = {
        -1.f, -2.f, -1.f,
         0.f,  0.f,  0.f,
         1.f,  2.f,  1.f};

    const std::vector<float> prewitt_x = {
        -1.f, 0.f, 1.f,
        -1.f, 0.f, 1.f,
        -1.f, 0.f, 1.f};

    const std::vector<float> prewitt_y = {
        -1.f, -1.f, -1.f,
         0.f,  0.f,  0.f,
         1.f,  1.f,  1.f};

    const std::vector<float> scharr_x = {
        -3.f,  0.f,  3.f,
        -10.f, 0.f, 10.f,
        -3.f,  0.f,  3.f};

    const std::vector<float> scharr_y = {
        -3.f, -10.f, -3.f,
         0.f,   0.f,  0.f,
         3.f,  10.f,  3.f};

    const std::vector<float> laplacian_4 = {
         0.f,  1.f,  0.f,
         1.f, -4.f,  1.f,
         0.f,  1.f,  0.f};

    const std::vector<float> laplacian_8 = {
         1.f,  1.f,  1.f,
         1.f, -8.f,  1.f,
         1.f,  1.f,  1.f};

    // ====== BLURRING (NORMALIZED) ======

    const std::vector<float> box_blur_3x3 = {
        1.f/9, 1.f/9, 1.f/9,
        1.f/9, 1.f/9, 1.f/9,
        1.f/9, 1.f/9, 1.f/9};

    const std::vector<float> gaussian_3x3 = {
        1.f/16, 2.f/16, 1.f/16,
        2.f/16, 4.f/16, 2.f/16,
        1.f/16, 2.f/16, 1.f/16};

    const std::vector<float> gaussian_5x5 = {
         1.f/256,  4.f/256,  6.f/256,  4.f/256, 1.f/256,
         4.f/256, 16.f/256, 24.f/256, 16.f/256, 4.f/256,
         6.f/256, 24.f/256, 36.f/256, 24.f/256, 6.f/256,
         4.f/256, 16.f/256, 24.f/256, 16.f/256, 4.f/256,
         1.f/256,  4.f/256,  6.f/256,  4.f/256, 1.f/256};

    // ====== SHARPENING ======

    const std::vector<float> sharpen = {
         0.f, -1.f,  0.f,
        -1.f,  5.f, -1.f,
         0.f, -1.f,  0.f};

    const std::vector<float> edge_enhance = {
        -1.f, -1.f, -1.f,
        -1.f,  9.f, -1.f,
        -1.f, -1.f, -1.f};

    const std::vector<float> unsharp_mask = {
        -1.f/16, -2.f/16, -1.f/16,
        -2.f/16, 20.f/16, -2.f/16,
        -1.f/16, -2.f/16, -1.f/16};

    // ====== OTHER ======

    const std::vector<float> identity = {
         0.f, 0.f, 0.f,
         0.f, 1.f, 0.f,
         0.f, 0.f, 0.f};

    const std::vector<float> emboss = {
        -2.f, -1.f,  0.f,
        -1.f,  1.f,  1.f,
         0.f,  1.f,  2.f};

    const std::vector<float> outline = {
        -1.f, -1.f, -1.f,
        -1.f,  8.f, -1.f,
        -1.f, -1.f, -1.f};

    const std::vector<float> ridge_horizontal = {
        -1.f, -1.f, -1.f,
         2.f,  2.f,  2.f,
        -1.f, -1.f, -1.f};

    const std::vector<float> ridge_vertical = {
        -1.f,  2.f, -1.f,
        -1.f,  2.f, -1.f,
        -1.f,  2.f, -1.f};

    // ====== COLLECTIONS ======

    const std::vector<std::vector<float>> all_kernels = {
        sobel_x, sobel_y,
        prewitt_x, prewitt_y,
        scharr_x, scharr_y,
        laplacian_4, laplacian_8,
        box_blur_3x3, gaussian_3x3, gaussian_5x5,
        sharpen, edge_enhance, unsharp_mask,
        identity, emboss, outline,
        ridge_horizontal, ridge_vertical};

    const std::vector<std::vector<float>> edge_detection_kernels = {
        sobel_x, sobel_y,
        prewitt_x, prewitt_y,
        scharr_x, scharr_y,
        laplacian_4, laplacian_8};

    const std::vector<std::vector<float>> blur_kernels = {
        box_blur_3x3,
        gaussian_3x3,
        gaussian_5x5};

    const std::vector<std::vector<float>> sharpen_kernels = {
        sharpen,
        edge_enhance,
        unsharp_mask};
}