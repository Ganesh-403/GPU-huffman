cmake_minimum_required(VERSION 3.18)

project(GPU_Huffman_Pro VERSION 1.0.0 LANGUAGES CXX)

option(ENABLE_CUDA "Build CUDA acceleration when the host toolchain supports it" ON)

set(HAS_CUDA OFF)
if(ENABLE_CUDA)
    if(WIN32 AND NOT MSVC)
        message(WARNING "CUDA on Windows requires MSVC as the host compiler. Building the CPU fallback instead.")
    elseif(WIN32 AND MSVC)
        # MSVC_VERSION 1950 = VS 2026. CUDA 13.1 only supports up to 1927 (VS 2022).
        # CUDA 13.2+ supports VS 2026.
        if(MSVC_VERSION GREATER 1927)
            message(WARNING "Your CUDA version may not support MSVC ${MSVC_VERSION} (VS 2026). "
                            "CUDA 13.1 is limited to VS 2019–2022. "
                            "Upgrade to CUDA 13.2+ for full VS 2026 support. "
                            "Building the CPU fallback instead.")
        else()
            # Initialize CUDA flags before the language is enabled so CMake's compiler-id
            # probe sees the unsupported-compiler override on newer MSVC toolsets.
            set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} -lineinfo -allow-unsupported-compiler")
            enable_language(CUDA)
            set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89) # Support Turing, Ampere, Ada
            set(HAS_CUDA ON)
        endif()
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark
)

# Source files
set(SOURCES
    src/main.cpp
    src/core/huffman_tree.cpp
    src/cpu/cpu_frequency_counter.cpp
    src/benchmark/benchmark.cpp
)

if(HAS_CUDA)
    list(APPEND SOURCES src/cuda/cuda_frequency_counter.cu)
else()
    list(APPEND SOURCES src/cuda/cuda_frequency_counter.cpp)
endif()

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
