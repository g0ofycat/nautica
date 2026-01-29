#include <random>
#include <cmath>
#include <vector>
#include <algorithm>

#include "./nn_utility.hpp"
#include "../tensors/tensor.hpp"

// ====== MAIN ======

/// @brief Get RNG Device
/// @return static std::mt19937&
std::mt19937 &nn_utility::get_rng()
{
    static thread_local std::mt19937 gen{std::random_device{}()};

    return gen;
}

// ====== ACTIVATION FUNCTIONS ======

/// @brief Sigmoid activation function
/// @param input The input to apply
/// @return int
double nn_utility::sigmoid(double input)
{
    return 1.0 / (1.0 + std::exp(-input));
}

/// @brief Sigmoid derivative activation function
/// @param input The input to apply
/// @return int
double nn_utility::sigmoid_derivative(double input)
{
    return input * (1.0 - input);
}

/// @brief Softmax activation function (In-Place)
/// @param logits The logits to softmax
/// @param output The tensor to output the results to
void nn_utility::softmax_inplace(const Tensor &logits, Tensor &output)
{
    double max_logit = *std::max_element(logits.data.begin(), logits.data.end());
    double sum_exp = 0.0;

    for (size_t i = 0; i < logits.data.size(); ++i)
    {
        output.data[i] = std::exp(logits.data[i] - max_logit);
        sum_exp += output.data[i];
    }

    for (auto &v : output.data)
        v /= sum_exp;
}