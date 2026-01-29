#include <iostream>
#include <benchmark/benchmark.h>

#include "../src/interface/nn_interface.hpp"

static void BM_Tensor(benchmark::State &state)
{
    for (auto _ : state)
    {
        Tensor convoluted_img_tensor = nn_interface::convolute_image("testing/images/img_1.jpg", 1);

        benchmark::DoNotOptimize(convoluted_img_tensor);
    }
}

BENCHMARK(BM_Tensor);

BENCHMARK_MAIN();