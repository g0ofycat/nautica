#include <string>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <execution>
#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include <functional>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize2.h"

#include "./image_extractor.hpp"
#include "../../neural_networks/tensors/tensor.hpp"

// ====== PUBLIC FUNCTIONS ======

/// @brief Load an image file into a 3D tensor (H x W x C)
/// @param filepath Path to the image file
/// @param desired_channels Number of channels (1 = grayscale, 3 = RGB, 4 = RGBA)
/// @return Tensor with shape (height, width, channels)
Tensor image_extractor::load_image(const std::string &filepath, int desired_channels)
{
    int width, height, channels;
    unsigned char *data = stbi_load(filepath.c_str(), &width, &height, &channels, desired_channels);

    if (!data)
        throw std::runtime_error("Failed to load image: " + filepath);

    int actual_channels = desired_channels > 0 ? desired_channels : channels;

    Tensor tensor(std::vector<size_t>{
        static_cast<size_t>(height),
        static_cast<size_t>(width),
        static_cast<size_t>(actual_channels)});

    std::memcpy(tensor.data.data(), data, tensor.numel() * sizeof(unsigned char));

    stbi_image_free(data);

    return tensor;
}

/// @brief Save a tensor as an image file
/// @param filepath Output path (supports .png, .jpg, .bmp)
/// @param tensor The tensor to save (H x W x C)
/// @return True if successful
bool image_extractor::save_image(const std::string &filepath, const Tensor &tensor)
{
    if (tensor.shape.size() != 3)
        throw std::invalid_argument("Tensor must be 3D (H x W x C)");

    size_t height = tensor.shape[0];
    size_t width = tensor.shape[1];
    size_t channels = tensor.shape[2];

    std::vector<unsigned char> data(tensor.numel());
    const size_t num_elements = tensor.numel();

    const double *src = tensor.data.data();
    unsigned char *dst = data.data();

    size_t i = 0;

#ifdef __AVX2__
    for (; i + 4 <= num_elements; i += 4)
    {
        __m256d v = _mm256_loadu_pd(&src[i]);
        __m128i v_int = _mm256_cvtpd_epi32(v);

        int temp[4];
        _mm_storeu_si128((__m128i *)temp, v_int);

        dst[i] = static_cast<unsigned char>(temp[0]);
        dst[i + 1] = static_cast<unsigned char>(temp[1]);
        dst[i + 2] = static_cast<unsigned char>(temp[2]);
        dst[i + 3] = static_cast<unsigned char>(temp[3]);
    }
#endif

    for (; i < num_elements; ++i)
        dst[i] = static_cast<unsigned char>(src[i]);

    std::string ext = std::filesystem::path(filepath).extension().string();

    static const std::unordered_map<std::string, std::function<int(const char *, int, int, int, const void *, int)>> writers = {
        {".png", [](const char *f, int w, int h, int c, const void *d, int s)
         {
             return stbi_write_png(f, w, h, c, d, w * c);
         }},
        {".jpg", [](const char *f, int w, int h, int c, const void *d, int)
         {
             return stbi_write_jpg(f, w, h, c, d, 90);
         }},
        {".jpeg", [](const char *f, int w, int h, int c, const void *d, int)
         {
             return stbi_write_jpg(f, w, h, c, d, 90);
         }},
        {".bmp", [](const char *f, int w, int h, int c, const void *d, int)
         {
             return stbi_write_bmp(f, w, h, c, d);
         }}};

    auto it = writers.find(ext);

    if (it == writers.end())
        throw std::invalid_argument("Unsupported file format: " + ext);

    return it->second(filepath.c_str(), width, height, channels, data.data(), 0);
}

/// @brief Normalize pixel values from [0, 255] to [0, 1]
/// @param tensor The tensor to normalize
void image_extractor::normalize(Tensor &tensor)
{
    std::transform(tensor.data.begin(), tensor.data.end(), tensor.data.begin(),
                   [](double val)
                   { return val / 255.0; });
}

/// @brief Resize image to target dimensions
/// @param tensor Input tensor
/// @param target_h Target height
/// @param target_w Target width
/// @return Resized tensor
Tensor image_extractor::resize(const Tensor &tensor, size_t target_h, size_t target_w)
{
    if (tensor.shape.size() != 3)
        throw std::invalid_argument("Tensor must be 3D");

    size_t src_h = tensor.shape[0];
    size_t src_w = tensor.shape[1];
    size_t channels = tensor.shape[2];

    std::vector<unsigned char> src_data(tensor.numel());

    std::transform(std::execution::par_unseq,
                   tensor.data.begin(), tensor.data.end(),
                   src_data.begin(),
                   [](double val)
                   { return static_cast<unsigned char>(val); });

    std::vector<unsigned char> dst_data(target_h * target_w * channels);

    stbir_resize_uint8_linear(
        src_data.data(), src_w, src_h, 0,
        dst_data.data(), target_w, target_h, 0,
        static_cast<stbir_pixel_layout>(channels));

    Tensor resized(std::vector<size_t>{target_h, target_w, channels});

    std::memcpy(resized.data.data(), dst_data.data(), resized.numel() * sizeof(unsigned char));

    return resized;
}