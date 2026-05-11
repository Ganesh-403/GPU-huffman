cmake_minimum_required(VERSION 3.18)
project(GPU_Huffman_Pro VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# CUDA configuration
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89) # Support Turing, Ampere, Ada
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # Useful for profiling

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark
)

# Source files
set(SOURCES
    main.cpp
    core/huffman_tree.cpp
    cpu/cpu_frequency_counter.cpp
    cuda/cuda_frequency_counter.cu
    benchmark/benchmark.cpp
)

# Executable
add_executable(gpu_huffman ${SOURCES})

# Target properties
target_link_libraries(gpu_huffman PRIVATE OpenMP::OpenMP_CXX)

# Set output directory
set_target_properties(gpu_huffman PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Install rules (optional)
install(TARGETS gpu_huffman RUNTIME DESTINATION bin)
