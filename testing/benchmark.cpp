#include <iostream>
#include <vector>
#include <benchmark/benchmark.h>

#include "../src/interface/nn_interface.hpp"
#include "../src/neural_networks/architectures/cnn/kernels/kernels.hpp"

static void BM_Tensor(benchmark::State &state)
{
    Tensor test_tensor = {{1.0, 0.0, 0.0},
                          {0.0, 1.0, 0.0},
                          {0.0, 0.0, 1.0}};

    for (auto _ : state)
    {
        std::vector<Tensor> convoluted_tensors = nn_interface::convolute_tensor(test_tensor, 1, 1);

        benchmark::DoNotOptimize(convoluted_tensors);
    }
}

BENCHMARK(BM_Tensor);

BENCHMARK_MAIN();