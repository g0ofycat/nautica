#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>

#include "../src/interface/nn_interface.hpp"
#include "../src/neural_networks/architectures/cnn/kernels/kernels.hpp"
#include "../src/processing/images/image_extractor.hpp"

/*
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
*/

int main() {
    Tensor img_tensor_conv = nn_interface::convolute_image("testing/images/img_1.jpg", 1);

    image_extractor::save_image("testing/images/img_1_conv.jpg", img_tensor_conv);

    return 0;
}