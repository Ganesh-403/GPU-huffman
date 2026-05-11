#include "cpu_frequency_counter.h"
#include <chrono>
#include <cstring>
#include <omp.h>

namespace gpu_huffman {

FrequencyResult CpuFrequencyCounter::count(const std::string& text) {
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

FrequencyResult OpenMpFrequencyCounter::count(const std::string& text) {
    FrequencyResult result;
    std::memset(result.frequencies, 0, sizeof(result.frequencies));
    result.method_name = name();

    auto start = std::chrono::high_resolution_clock::now();

    // Use a local frequency table per thread to avoid contention/false sharing
    #pragma omp parallel
    {
        uint32_t local_freq[256] = {0};
        #pragma omp for nowait
        for (size_t i = 0; i < text.size(); ++i) {
            local_freq[static_cast<unsigned char>(text[i])]++;
        }

        #pragma omp critical
        {
            for (int i = 0; i < 256; ++i) {
                result.frequencies[i] += local_freq[i];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<float, std::milli>(end - start).count();

    return result;
}

} // namespace gpu_huffman
