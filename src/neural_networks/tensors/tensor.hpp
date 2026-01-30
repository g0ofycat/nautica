#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstddef>
#include <iostream>
#include <vector>

#include "../utility/nn_utility.hpp"

// ====== Tensor ======

class Tensor {
  public:
    // ====== PUBLIC DATA ======

    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::vector<double> data;

    // ====== CONSTRUCTORS ======

    /// @brief Default Constructor
    Tensor() = default;

    /// @brief 1D Tensor support; (List)
    /// @param one_dim_tensor The 1D initializer list
    Tensor(std::initializer_list<double> one_dim_tensor);

    /// @brief 2D Tensor support; (Matrix)
    /// @param two_dim_tensor The 2D initializer list
    Tensor(std::initializer_list<std::initializer_list<double>> two_dim_tensor);

    /// @brief Dynamic Shape Creation; tensor({3, 4, 5})
    /// @param dims The dimensions of the tensor
    Tensor(const std::vector<size_t> &dims);

    // ====== STATIC FUNCTIONS ======

    /// @brief Call constructor for a tensor
    /// @param dims The tensor dimensions
    /// @return Tensor
    static Tensor zeros(const std::vector<size_t> &dims);

    /// @brief Random Tensor based on Uniform Real Distribution
    /// @param dims The tensor dimensions
    /// @param min The minimum value
    /// @param max The maximum value
    /// @return Tensor
    static Tensor random_tensor(const std::vector<size_t> &dims, double min = 0.0, double max = 1.0);

    /// @brief Random Tensor based on Normal Distribution
    /// @param dims The tensor dimensions
    /// @param mean The mean value
    /// @param std_dev The standard deviation
    /// @return Tensor
    static Tensor random_normal(const std::vector<size_t> &dims, double mean = 0.0, double std_dev = 1.0);

    /// @brief Extract a specific row from a 2D Tensor
    /// @param tensor The tensor to extract from
    /// @param row_idx The index of the row to extract
    /// @return The extracted row as a tensor
    static Tensor extract_row(const Tensor &tensor, size_t row_idx);

    /// @brief Matrix multiplication of two tensors
    /// @param a The first tensor
    /// @param b The second tensor
    /// @param block_size The block size for optimization
    /// @return The resulting tensor
    static Tensor matmul(const Tensor &a, const Tensor &b, size_t block_size = 64);

    // ====== PUBLIC API ======

    /// @brief Get the Tensor Number of Dimensions
    /// @return The number of dimensions
    size_t ndim() const;

    /// @brief Get the Total Number of Elements in the Tensor
    /// @return The total number of elements
    size_t numel() const;

    /// @brief Get the index of a given multi-dimensional index
    /// @param idx The multi-dimensional index
    /// @return The index of the element
    size_t index_of(const std::vector<size_t> &idx) const;

    /// @brief Assign data from a source vector to the tensors data
    /// @param src The source vector
    /// @return A reference to the current tensor
    Tensor &assign_data(const std::vector<double> &src);

    /// @brief Get the sum of the axis
    /// @param axis The current axis
    /// @return Tensor
    Tensor sum(size_t axis) const;

    /// @brief Reshape the tensor to a new shape
    /// @param new_shape The new shape dimensions
    void reshape(const std::vector<size_t> &new_shape);

    /// @brief Flatten a tensor starting from a specific dimension
    /// @param start_dim The starting dimension
    /// @return The flattened tensor
    Tensor flatten(size_t start_dim = 0) const;

    /// @brief Compute the strides for the tensor based on its shape
    void compute_strides();

    /// @brief Fill the tensor with a specific value
    /// @param v The value to fill the tensor with
    void fill(double v);

    /// @brief Access tensor element using a vector of indices (non-const)
    /// @param idx The vector of indices
    /// @return Reference to the tensor element
    double &at(const std::vector<size_t> &idx);

    /// @brief Access tensor element using a vector of indices (const)
    /// @param idx The vector of indices
    /// @return The tensor element
    double at(const std::vector<size_t> &idx) const;

    // ====== OPERATORS ======

    /// @brief Add two tensors element-wise
    /// @param other The tensor to add
    /// @return The resulting tensor
    Tensor operator+(const Tensor &other) const;

    /// @brief Multiply two tensors element-wise
    /// @param other The tensor to multiply
    /// @return The resulting tensor
    Tensor operator*(const Tensor &other) const;

    /// @brief Operator overload for outputting tensor to ostream
    /// @param os The output stream
    /// @param t The tensor to output
    /// @return std::ostream&
    friend std::ostream &operator<<(std::ostream &os, const Tensor &t);
};

#endif