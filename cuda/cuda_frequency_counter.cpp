#include "cuda_frequency_counter.h"
#include "../cpu/cpu_frequency_counter.h"

#include <chrono>
#include <cstring>

namespace gpu_huffman {

CudaFrequencyCounter::CudaFrequencyCounter(CudaKernelType type) : type_(type) {}

std::string CudaFrequencyCounter::name() const {
    switch (type_) {
        case CudaKernelType::NAIVE_GLOBAL:   return "CUDA Fallback (CPU Serial)";
        case CudaKernelType::SHARED_MEMORY:  return "CUDA Fallback (CPU Shared)";
        case CudaKernelType::WARP_OPTIMIZED: return "CUDA Fallback (CPU Warp)";
        case CudaKernelType::MULTI_STREAM:   return "CUDA Fallback (CPU Multi-Stream)";
        case CudaKernelType::UNIFIED_MEMORY: return "CUDA Fallback (CPU Unified)";
        default:                             return "CUDA Fallback (CPU)";
    }
}

FrequencyResult CudaFrequencyCounter::count(const std::string& text) {
    FrequencyResult result;
    std::memset(result.frequencies, 0, sizeof(result.frequencies));
    result.method_name = name();

    auto start = std::chrono::high_resolution_clock::now();

    for (unsigned char c : text) {
        result.frequencies[c]++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<float, std::milli>(end - start).count();

    return result;
}

} // namespace gpu_huffman