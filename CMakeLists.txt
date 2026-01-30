cmake_minimum_required(VERSION 3.16)
project(LaTeX_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-mavx2 -mfma)
elseif(MSVC)
    add_compile_options(/arch:AVX2)
endif()

find_package(benchmark CONFIG REQUIRED)

add_executable(main
    testing/benchmark.cpp
    src/interface/nn_interface_registry.cpp
    src/neural_networks/architectures/cnn/cnn_registry.cpp
    src/neural_networks/tensors/tensor_registry.cpp
    src/neural_networks/utility/nn_utility_registry.cpp
    src/neural_networks/architectures/cnn/kernels/kernels_registry.cpp
    src/processing/images/image_extractor_registry.cpp
    src/stb/stb_implementation.cpp
)

target_compile_options(main PRIVATE -fverbose-asm -save-temps=obj)

target_include_directories(main PRIVATE
    src/neural_networks
    src/neural_networks/architectures
    src/neural_networks/architectures/cnn
    src/neural_networks/architectures/cnn/kernels
    src/neural_networks/enums
    src/neural_networks/tensors
    src/neural_networks/utility
    src/interface
    src/processing
    src/processing/images
    src/stb
    dependencies
)

target_link_libraries(main PRIVATE benchmark::benchmark shlwapi)

add_definitions(-DBENCHMARK_STATIC_DEFINE)

target_link_options(main PRIVATE
    -static-libgcc
    -static-libstdc++
)

add_executable(bench
    testing/benchmark.cpp
    src/interface/nn_interface_registry.cpp
    src/neural_networks/architectures/cnn/cnn_registry.cpp
    src/neural_networks/tensors/tensor_registry.cpp
    src/neural_networks/utility/nn_utility_registry.cpp
    src/neural_networks/architectures/cnn/kernels/kernels_registry.cpp
    src/processing/images/image_extractor_registry.cpp
    src/stb/stb_implementation.cpp
)

target_include_directories(bench PRIVATE
    src/neural_networks
    src/neural_networks/architectures
    src/neural_networks/architectures/cnn
    src/neural_networks/architectures/cnn/kernels
    src/neural_networks/enums
    src/neural_networks/tensors
    src/neural_networks/utility
    src/interface
    src/processing
    src/processing/images
    src/stb
    dependencies
)

target_link_libraries(bench PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    shlwapi
)

target_link_options(bench PRIVATE -static-libgcc -static-libstdc++)
