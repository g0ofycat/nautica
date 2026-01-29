#ifndef NN_UTILITY_HPP
#define NN_UTILITY_HPP

#include <random>

// ====== FORWARD DECLARATIONS ======

class Tensor;

// ====== nn_utility ======

class nn_utility
{
public:
    // ====== MAIN ======

    /// @brief Get RNG Device
    /// @return static std::mt19937&
    static std::mt19937 &get_rng();

    // ====== ACTIVATION FUNCTIONS ======

    /// @brief Sigmoid activation function
    /// @param input The input to apply
    /// @return int
    static double sigmoid(double input);

    /// @brief Sigmoid derivative activation function
    /// @param input The input to apply
    /// @return int
    static double sigmoid_derivative(double input);

    /// @brief Softmax activation function (In-Place)
    /// @param logits The logits to softmax
    /// @param output The tensor to output the results to
    static void softmax_inplace(const Tensor &logits, Tensor &output);
};

#endif