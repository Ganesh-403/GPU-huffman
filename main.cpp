#include "utils/logger.h"
#include "utils/cuda_utils.h"
#include "cpu/cpu_frequency_counter.h"
#include "cuda/cuda_frequency_counter.h"
#include "core/huffman_tree.h"
#include "benchmark/benchmark.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>

using namespace gpu_huffman;

void print_header() {
    std::cout << "\n\033[1;35m";
    std::cout << "============================================================\n";
    std::cout << "      GPU-ACCELERATED PARALLEL COMPRESSION ENGINE          \n";
    std::cout << "          Performance Engineering Showcase                 \n";
    std::cout << "============================================================\n";
    std::cout << "\033[0m\n";
}

int main(int argc, char** argv) {
    print_header();

    std::string input_file = "big.txt";
    if (argc > 1) {
        input_file = argv[1];
    }

    // 1. Load Data
    Logger::info("Loading input file: " + input_file);
    std::ifstream file(input_file, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Failed to open " + input_file);
        return 1;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    Logger::success("Loaded " + std::to_string(text.size()) + " bytes.");

    // 2. Setup Benchmark Harness
    BenchmarkHarness harness;
    harness.add_counter(std::make_unique<CpuFrequencyCounter>());
    harness.add_counter(std::make_unique<OpenMpFrequencyCounter>());
    harness.add_counter(std::make_unique<CudaFrequencyCounter>(CudaKernelType::NAIVE_GLOBAL));
    harness.add_counter(std::make_unique<CudaFrequencyCounter>(CudaKernelType::SHARED_MEMORY));
    harness.add_counter(std::make_unique<CudaFrequencyCounter>(CudaKernelType::WARP_OPTIMIZED));

    // 3. Run Benchmark
    BenchmarkConfig config;
    config.dataset_name = input_file;
    config.iterations = 10;
    harness.run(text, config);
    harness.print_results();
    harness.export_csv("benchmark_results.csv");

    // 4. Build Huffman Tree and Calculate Compression Ratio
    Logger::info("Building Huffman Tree and calculating compression metrics...");
    CpuFrequencyCounter cpu_counter;
    auto freq_result = cpu_counter.count(text);
    
    HuffmanTree tree;
    tree.build(freq_result.frequencies);
    auto stats = tree.compute_stats(freq_result.frequencies);

    std::cout << "\n--- Compression Results ---\n";
    std::cout << "Original Size   : " << stats.original_size_bits / 8 << " bytes\n";
    std::cout << "Compressed Size : " << (stats.compressed_size_bits + 7) / 8 << " bytes\n";
    std::cout << "Savings         : " << std::fixed << std::setprecision(2) << stats.savings_percent << "%\n";
    std::cout << std::string(30, '-') << "\n\n";

    Logger::success("Program completed successfully.");
    return 0;
}
