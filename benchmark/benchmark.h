#pragma once

#include "../core/frequency_counter.h"
#include <vector>
#include <string>
#include <memory>

namespace gpu_huffman {

struct BenchmarkConfig {
    std::string dataset_name;
    size_t iterations = 5;
    bool warmup = true;
};

struct BenchmarkEntry {
    std::string method_name;
    float avg_time_ms;
    float min_time_ms;
    float max_time_ms;
    float throughput_gbs; // GB/s
};

class BenchmarkHarness {
public:
    void add_counter(std::unique_ptr<IFrequencyCounter> counter);
    void run(const std::string& text, const BenchmarkConfig& config);
    void print_results() const;
    void export_csv(const std::string& filename) const;

private:
    std::vector<std::unique_ptr<IFrequencyCounter>> counters_;
    std::vector<BenchmarkEntry> results_;
    size_t current_data_size_ = 0;
};

} // namespace gpu_huffman
