#pragma once

#include "../core/frequency_counter.h"
#include <string>

namespace gpu_huffman {

enum class CudaKernelType {
    NAIVE_GLOBAL,
    SHARED_MEMORY,
    WARP_OPTIMIZED,
    MULTI_STREAM,
    UNIFIED_MEMORY
};

class CudaFrequencyCounter : public IFrequencyCounter {
public:
    explicit CudaFrequencyCounter(CudaKernelType type = CudaKernelType::SHARED_MEMORY);
    
    FrequencyResult count(const std::string& text) override;
    std::string name() const override;

private:
    CudaKernelType type_;
};

} // namespace gpu_huffman
