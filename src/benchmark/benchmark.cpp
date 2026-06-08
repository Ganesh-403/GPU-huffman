#include "benchmark.h"
#include "../utils/logger.h"
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iomanip>

namespace gpu_huffman {

void BenchmarkHarness::add_counter(std::unique_ptr<IFrequencyCounter> counter) {
    counters_.push_back(std::move(counter));
}

void BenchmarkHarness::run(const std::string& text, const BenchmarkConfig& config) {
    results_.clear();
    current_data_size_ = text.size();
    
    Logger::info("Starting benchmark on dataset: " + config.dataset_name + " (" + std::to_string(text.size()) + " bytes)");

    for (const auto& counter : counters_) {
        Logger::info("Benchmarking: " + counter->name() + "...");
        
        if (config.warmup) {
            counter->count(text); // Warmup run
        }

        std::vector<float> times;
        for (size_t i = 0; i < config.iterations; ++i) {
            auto result = counter->count(text);
            times.push_back(result.execution_time_ms);
        }

        float sum = std::accumulate(times.begin(), times.end(), 0.0f);
        float avg = sum / times.size();
        float min = *std::min_element(times.begin(), times.end());
        float max = *std::max_element(times.begin(), times.end());
        
        // Throughput = (Size in GB) / (Time in seconds)
        // GB = Bytes / 10^9
        // s = ms / 1000
        double size_gb = static_cast<double>(text.size()) / 1e9;
        double time_s = static_cast<double>(avg) / 1000.0;
        float throughput = (time_s > 0) ? static_cast<float>(size_gb / time_s) : 0.0f;

        results_.push_back({counter->name(), avg, min, max, throughput});
    }
}

void BenchmarkHarness::print_results() const {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << std::left << std::setw(30) << "Method" 
              << std::setw(15) << "Avg Time (ms)" 
              << std::setw(15) << "Throughput" 
              << "Speedup" << "\n";
    std::cout << std::string(80, '-') << "\n";

    float baseline_time = results_.empty() ? 1.0f : results_[0].avg_time_ms;

    for (const auto& entry : results_) {
        float speedup = baseline_time / entry.avg_time_ms;
        std::cout << std::left << std::setw(30) << entry.method_name 
                  << std::setw(15) << std::fixed << std::setprecision(3) << entry.avg_time_ms 
                  << std::setw(15) << std::fixed << std::setprecision(2) << entry.throughput_gbs << " GB/s"
                  << std::fixed << std::setprecision(2) << speedup << "x" << "\n";
    }
    std::cout << std::string(80, '=') << "\n\n";
}

void BenchmarkHarness::export_csv(const std::string& filename) const {
    std::ofstream file(filename);
    file << "Method,AvgTimeMs,MinTimeMs,MaxTimeMs,ThroughputGbs\n";
    for (const auto& entry : results_) {
        file << entry.method_name << "," 
             << entry.avg_time_ms << "," 
             << entry.min_time_ms << "," 
             << entry.max_time_ms << "," 
             << entry.throughput_gbs << "\n";
    }
    Logger::success("Results exported to " + filename);
}

} // namespace gpu_huffman
