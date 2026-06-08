#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace gpu_huffman {

/**
 * @brief Results of a frequency counting operation.
 */
struct FrequencyResult {
    uint32_t frequencies[256];
    float execution_time_ms;
    std::string method_name;
};

/**
 * @brief Abstract base class for frequency counting.
 */
class IFrequencyCounter {
public:
    virtual ~IFrequencyCounter() = default;

    /**
     * @brief Count character frequencies in the input text.
     * @param text Input string.
     * @return FrequencyResult containing counts and timing.
     */
    virtual FrequencyResult count(const std::string& text) = 0;

    /**
     * @brief Returns the name of the implementation.
     */
    virtual std::string name() const = 0;
};

} // namespace gpu_huffman
