#include "./kernels.hpp"

namespace Kernels
{
    // ====== EDGE DETECTION ======

    const std::vector<int> sobel_x = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1};

    const std::vector<int> sobel_y = {
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1};

    const std::vector<int> prewitt_x = {
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1};

    const std::vector<int> prewitt_y = {
        -1, -1, -1,
        0, 0, 0,
        1, 1, 1};

    const std::vector<int> scharr_x = {
        -3, 0, 3,
        -10, 0, 10,
        -3, 0, 3};

    const std::vector<int> scharr_y = {
        -3, -10, -3,
        0, 0, 0,
        3, 10, 3};

    const std::vector<int> laplacian_4 = {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0};

    const std::vector<int> laplacian_8 = {
        1, 1, 1,
        1, -8, 1,
        1, 1, 1};

    // ====== BLURRING ======

    const std::vector<int> box_blur_3x3 = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1};

    const std::vector<int> gaussian_3x3 = {
        1, 2, 1,
        2, 4, 2,
        1, 2, 1};

    const std::vector<int> gaussian_5x5 = {
        1, 4, 6, 4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1, 4, 6, 4, 1};

    // ====== SHARPENING ======

    const std::vector<int> sharpen = {
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0};

    const std::vector<int> edge_enhance = {
        -1, -1, -1,
        -1, 9, -1,
        -1, -1, -1};

    const std::vector<int> unsharp_mask = {
        1, 4, 1,
        4, 20, 4,
        1, 4, 1};

    // ====== OTHER ======

    const std::vector<int> identity = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0};

    const std::vector<int> emboss = {
        -2, -1, 0,
        -1, 1, 1,
        0, 1, 2};

    const std::vector<int> outline = {
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1};

    const std::vector<int> ridge_horizontal = {
        -1, -1, -1,
        2, 2, 2,
        -1, -1, -1};

    const std::vector<int> ridge_vertical = {
        -1, 2, -1,
        -1, 2, -1,
        -1, 2, -1};
}