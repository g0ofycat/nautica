#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <random>
#include <numeric>
#include <omp.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "./tensor.hpp"
#include "../utility/nn_utility.hpp"

// ====== CONSTRUCTORS ======

/// @brief 1D Tensor support; (List)
Tensor::Tensor(std::initializer_list<double> one_dim_tensor)
{
    data.assign(one_dim_tensor.begin(), one_dim_tensor.end());
    shape = {one_dim_tensor.size()};
    compute_strides();
}

/// @brief 2D Tensor support; (Matrix)
Tensor::Tensor(std::initializer_list<std::initializer_list<double>> two_dim_tensor)
{
    size_t rows = two_dim_tensor.size();

    if (rows == 0)
    {
        shape = {0, 0};
        return;
    }

    size_t cols = two_dim_tensor.begin()->size();

    for (auto &row : two_dim_tensor)
    {
        if (row.size() != cols)
            throw std::invalid_argument("All rows must have same length");

        data.insert(data.end(), row.begin(), row.end());
    }

    shape = {rows, cols};
    compute_strides();
}

/// @brief Dynamic Shape Creation; tensor({3, 4, 5})
Tensor::Tensor(const std::vector<size_t> &dims) : shape(dims)
{
    compute_strides();
    data.assign(numel(), 0.0);
}

// ====== STATIC FUNCTIONS ======

/// @brief Call constructor for a tensor
/// @param dims The tensor dimensions
/// @return Tensor
Tensor Tensor::zeros(const std::vector<size_t> &dims)
{
    return Tensor(dims);
}

/// @brief Random Tensor based on Uniform Real Distribution
/// @param dims The tensor dimensions
/// @param min The minimum value
/// @param max The maximum value
/// @return Tensor
Tensor Tensor::random_tensor(const std::vector<size_t> &dims, double min, double max)
{
    Tensor t(dims);
    std::mt19937 &gen = nn_utility::get_rng();
    std::uniform_real_distribution<double> dist(min, max);

    for (auto &v : t.data)
        v = dist(gen);

    return t;
}

/// @brief Random Tensor based on Normal Distribution
/// @param dims The tensor dimensions
/// @param mean The mean value
/// @param std_dev The standard deviation
/// @return Tensor
Tensor Tensor::random_normal(const std::vector<size_t> &dims, double mean, double std_dev)
{
    Tensor t(dims);
    std::mt19937 &gen = nn_utility::get_rng();
    std::normal_distribution<> d(mean, std_dev);

    for (auto &v : t.data)
        v = d(gen);

    return t;
}

/// @brief Extract a specific row from a 2D Tensor
/// @param tensor The tensor to extract from
/// @param row_idx The index of the row to extract
/// @return The extracted row as a tensor
Tensor Tensor::extract_row(const Tensor &t, size_t row_idx)
{
    if (t.shape.size() != 2)
        throw std::invalid_argument("extract_row(): Tensor must be 2D");

    size_t row_size = t.shape[1];
    size_t start_idx = row_idx * row_size;

    Tensor result;
    result.data.assign(t.data.begin() + start_idx, t.data.begin() + start_idx + row_size);
    result.shape = {row_size};
    result.compute_strides();

    return result;
}

/// @brief Matrix multiplication of two tensors
/// @param a The first tensor
/// @param b The second tensor
/// @param block_size The block size for optimization
/// @return The resulting tensor
Tensor Tensor::matmul(const Tensor &a, const Tensor &b, size_t block_size)
{
    if (a.shape.size() != 2 || b.shape.size() != 2)
        throw std::invalid_argument("Both tensors must be 2D");
    if (a.shape[1] != b.shape[0])
        throw std::invalid_argument("Inner dimensions must match");

    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1];

    Tensor result(std::vector<size_t>{M, N});
    result.fill(0.0);

#pragma omp parallel for
    for (size_t ii = 0; ii < M; ii += block_size)
    {
        for (size_t jj = 0; jj < N; jj += block_size)
        {
            for (size_t kk = 0; kk < K; kk += block_size)
            {

                size_t i_max = std::min(ii + block_size, M);
                size_t j_max = std::min(jj + block_size, N);
                size_t k_max = std::min(kk + block_size, K);

                for (size_t i = ii; i < i_max; ++i)
                {
                    for (size_t k = kk; k < k_max; ++k)
                    {
                        double a_ik = a.data[i * K + k];
                        for (size_t j = jj; j < j_max; ++j)
                            result.data[i * N + j] += a_ik * b.data[k * N + j];
                    }
                }
            }
        }
    }

    return result;
}

// ====== UTILITY ======

/// @brief Get the Tensor Number of Dimensions
/// @return The number of dimensions
size_t Tensor::ndim() const
{
    return shape.size();
}

/// @brief Get the Total Number of Elements in the Tensor
/// @return The total number of elements
size_t Tensor::numel() const
{
    return shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
}

/// @brief Get the index of a given multi-dimensional index
/// @param idx The multi-dimensional index
/// @return The index of the element
size_t Tensor::index_of(const std::vector<size_t> &idx) const
{
    if (idx.size() != shape.size())
        throw std::out_of_range("index_of(): Index rank mismatch");

    size_t offs = 0;

    for (size_t i = 0; i < idx.size(); ++i)
    {
        if (idx[i] >= shape[i])
            throw std::out_of_range("index_of(): Index out of range");

        offs += idx[i] * strides[i];
    }

    return offs;
}

/// @brief Assign data from a source vector to the tensors data
/// @param src The source vector
/// @return A reference to the current tensor
Tensor &Tensor::assign_data(const std::vector<double> &src)
{
    data = src;
    return *this;
}

/// @brief Get the sum of the axis
/// @param axis The current axis
/// @return tensor
Tensor Tensor::sum(size_t axis) const
{
    if (axis >= ndim())
        throw std::out_of_range("sum(): Axis out of range");

    std::vector<size_t> out_shape;

    out_shape.reserve(shape.size() - 1);

    for (size_t d = 0; d < shape.size(); ++d)
        if (d != axis)
            out_shape.push_back(shape[d]);

    if (out_shape.empty())
        out_shape = {1};

    Tensor out(out_shape);
    out.fill(0.0);

    size_t outer = 1;
    size_t inner = 1;

    for (size_t d = 0; d < axis; ++d)
        outer *= shape[d];
    for (size_t d = axis + 1; d < shape.size(); ++d)
        inner *= shape[d];

    size_t axis_size = shape[axis];

    for (size_t o = 0; o < outer; ++o)
    {
        for (size_t i = 0; i < inner; ++i)
        {
            double sum = 0.0;
            size_t src_base = o * axis_size * inner + i;

            for (size_t a = 0; a < axis_size; ++a)
                sum += data[src_base + a * inner];

            out.data[o * inner + i] = sum;
        }
    }

    return out;
}

/// @brief Reshape the tensor to a new shape
/// @param new_shape The new shape dimensions
void Tensor::reshape(const std::vector<size_t> &new_shape)
{
    size_t new_num = std::accumulate(new_shape.begin(), new_shape.end(), (size_t)1, std::multiplies<size_t>());

    if (new_num != numel())
        throw std::invalid_argument("reshape(): Reshape size mismatch");

    shape = new_shape;

    compute_strides();
}

/// @brief Flatten a tensor starting from a specific dimension
/// @param start_dim The starting dimension
/// @return The flattened tensor
Tensor Tensor::flatten(size_t start_dim) const
{
    if (shape.empty())
        return Tensor(std::vector<size_t>{0});

    size_t flat_size = 1;

    for (size_t i = start_dim; i < shape.size(); ++i)
        flat_size *= shape[i];

    std::vector<size_t> new_shape(shape.begin(), shape.begin() + start_dim);
    new_shape.push_back(flat_size);

    Tensor out;
    out.data = data;
    out.shape = new_shape;

    out.compute_strides();

    return out;
}

/// @brief Compute the strides for the tensor based on its shape
void Tensor::compute_strides()
{
    strides.assign(shape.size(), 1);

    if (!shape.empty())
    {
        for (int i = (int)shape.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];
    }
}

/// @brief Fill the tensor with a specific value
/// @param v The value to fill the tensor with
void Tensor::fill(double v)
{
    std::fill(data.begin(), data.end(), v);
}

/// @brief Access tensor element using a vector of indices (non-const)
/// @param idx The vector of indices
/// @return Reference to the tensor element
double &Tensor::at(const std::vector<size_t> &idx)
{
    return data[index_of(idx)];
}

/// @brief Access tensor element using a vector of indices (const)
/// @param idx The vector of indices
/// @return The tensor element
double Tensor::at(const std::vector<size_t> &idx) const
{
    return data[index_of(idx)];
}

// ====== OPERATORS ======

/// @brief Add two tensors element-wise
/// @param other The tensor to add
/// @return The resulting tensor
Tensor Tensor::operator+(const Tensor &other) const
{
    if (shape != other.shape)
        throw std::invalid_argument("Shape mismatch");

    Tensor result(shape);
    size_t n = numel();
    size_t i = 0;

#ifdef __AVX2__
    for (; i + 4 <= n; i += 4)
    {
        __m256d a = _mm256_loadu_pd(&data[i]);
        __m256d b = _mm256_loadu_pd(&other.data[i]);
        __m256d r = _mm256_add_pd(a, b);
        _mm256_storeu_pd(&result.data[i], r);
    }
#endif

    for (; i < n; ++i)
        result.data[i] = data[i] + other.data[i];

    return result;
}

/// @brief Multiply two tensors element-wise
/// @param other The tensor to multiply
/// @return The resulting tensor
Tensor Tensor::operator*(const Tensor &other) const
{
    if (shape != other.shape)
        throw std::invalid_argument("Shape mismatch");

    Tensor result(shape);
    size_t n = numel();
    size_t i = 0;

#ifdef __AVX2__
    for (; i + 4 <= n; i += 4)
    {
        __m256d a = _mm256_loadu_pd(&data[i]);
        __m256d b = _mm256_loadu_pd(&other.data[i]);
        __m256d r = _mm256_mul_pd(a, b);
        _mm256_storeu_pd(&result.data[i], r);
    }
#endif

    for (; i < n; ++i)
        result.data[i] = data[i] * other.data[i];

    return result;
}

/// @brief Operator overload for outputting tensor to ostream
/// @param os The output stream
/// @param t The tensor to output
/// @return std::ostream&
std::ostream &operator<<(std::ostream &os, const Tensor &t)
{
    if (t.shape.empty() || t.numel() == 0)
    {
        os << "Tensor([])";
        return os;
    }

    std::function<void(size_t, size_t, const std::string &)> print_recursive;

    print_recursive = [&](size_t dim, size_t offset, const std::string &indent)
    {
        if (dim == t.shape.size() - 1)
        {
            os << "[";
            for (size_t i = 0; i < t.shape[dim]; ++i)
            {
                if (i > 0)
                    os << ", ";
                os << t.data[offset + i];
            }
            os << "]";
        }
        else
        {
            os << "[";
            size_t stride = t.strides[dim];
            for (size_t i = 0; i < t.shape[dim]; ++i)
            {
                if (i > 0)
                {
                    os << ",\n"
                       << indent << " ";
                }
                print_recursive(dim + 1, offset + i * stride, indent + " ");
            }
            os << "]";
        }
    };

    os << "Tensor(shape=[";

    for (size_t i = 0; i < t.shape.size(); ++i)
    {
        if (i > 0)
            os << ", ";
        os << t.shape[i];
    }

    os << "])\n";
    print_recursive(0, 0, "");

    return os;
}