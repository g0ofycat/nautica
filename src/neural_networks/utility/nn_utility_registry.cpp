#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "../tensors/tensor.hpp"
#include "./nn_utility.hpp"

// ====== MAIN ======

/// @brief Get RNG Device
/// @return static std::mt19937&
std::mt19937 &nn_utility::get_rng() {
    static thread_local std::mt19937 gen{std::random_device{}()};

    return gen;
}

// ====== ACTIVATION FUNCTIONS ======

/// @brief Sigmoid activation function
/// @param input The input to apply
/// @return int
double nn_utility::sigmoid(double input) { return 1.0 / (1.0 + std::exp(-input)); }

/// @brief Sigmoid derivative activation function
/// @param input The input to apply
/// @return int
double nn_utility::sigmoid_derivative(double input) { return input * (1.0 - input); }

/// @brief Softmax activation function (In-Place)
/// @param logits The logits to softmax
/// @param output The tensor to output the results to
void nn_utility::softmax_inplace(const Tensor &logits, Tensor &output) {
    double max_logit = *std::max_element(logits.data.begin(), logits.data.end());
    double sum_exp = 0.0;

    for (size_t i = 0; i < logits.data.size(); ++i) {
        output.data[i] = std::exp(logits.data[i] - max_logit);
        sum_exp += output.data[i];
    }

    for (auto &v : output.data)
        v /= sum_exp;
}

/// @brief Cross-Entropy Loss Calculation
/// @param output The output probabilities
/// @param target The target probabilities
/// @return The computed cross-entropy loss
double nn_utility::cross_entropy_loss(const std::vector<double> &output, const std::vector<double> &target) {
    if (output.size() != target.size()) {
        throw std::invalid_argument("Output and target sizes do not match for Cross-Entropy calculation.");
    }

    double loss = 0.0;

    for (size_t i = 0; i < output.size(); ++i) {
        loss -= target[i] * std::log(output[i] + 1e-15);
    }

    return loss / static_cast<double>(output.size());
}