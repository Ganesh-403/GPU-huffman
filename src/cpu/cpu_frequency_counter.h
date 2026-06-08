#pragma once

#include "../core/frequency_counter.h"

namespace gpu_huffman {

class CpuFrequencyCounter : public IFrequencyCounter {
public:
    FrequencyResult count(const std::string& text) override;
    std::string name() const override { return "CPU Serial"; }
};

class OpenMpFrequencyCounter : public IFrequencyCounter {
public:
    FrequencyResult count(const std::string& text) override;
    std::string name() const override { return "CPU OpenMP (Multithreaded)"; }
};

} // namespace gpu_huffman
