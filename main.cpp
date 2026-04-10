// ============================================================
// main.cpp
// Entry point for the Huffman Encoding GPU vs CPU demo.
//
// Program Flow:
//   1.  Read input text file (sample.txt)
//   2.  Count frequencies on CPU  (std::chrono timing)
//   3.  Count frequencies on GPU  (cudaEvent timing)
//   4.  Verify CPU and GPU results match
//   5.  Build Huffman Tree on CPU
//   6.  Generate Huffman Codes
//   7.  Display frequency table, codes, and performance metrics
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <vector>
#include <utility>

#include "huffman_cpu.h"
#include "gpu_kernels.h"

// ------------------------------------------------------------
// Helper: returns a printable display string for a character
// ------------------------------------------------------------
static std::string charDisplay(int ascii) {
    if (ascii == ' ')  return "SPACE";
    if (ascii == '\n') return "NEWLINE";
    if (ascii == '\r') return "CR";
    if (ascii == '\t') return "TAB";
    return std::string(1, static_cast<char>(ascii));
}

// ------------------------------------------------------------
// Helper: print a horizontal divider line
// ------------------------------------------------------------
static void divider(int width = 52) {
    std::cout << std::string(width, '=') << "\n";
}
static void thinDivider(int width = 52) {
    std::cout << std::string(width, '-') << "\n";
}

int main() {

    // ----------------------------------------------------------
    // STEP 1: Read the input file into a std::string
    // ----------------------------------------------------------
    const std::string filename = "sample.txt";
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open \"" << filename << "\".\n"
                  << "        Make sure sample.txt is in the same\n"
                  << "        directory as the executable.\n";
        return 1;
    }

    // Read entire file content into a single string
    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    file.close();

    if (text.empty()) {
        std::cerr << "[ERROR] Input file is empty.\n";
        return 1;
    }

    divider();
    std::cout << "   Huffman Encoding: GPU vs CPU Comparison\n";
    divider();
    std::cout << "Input file    : " << filename << "\n";
    std::cout << "Total chars   : " << text.size() << "\n\n";

    // ----------------------------------------------------------
    // STEP 2: CPU Frequency Count  (std::chrono high-res timer)
    // ----------------------------------------------------------
    int cpuFreq[256] = {0};

    // Record wall-clock time before and after the CPU function
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpuCountFrequency(text, cpuFreq);
    auto cpuEnd   = std::chrono::high_resolution_clock::now();

    // Convert duration to milliseconds (double precision)
    double cpuTimeMs =
        std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    // ----------------------------------------------------------
    // STEP 3: GPU Frequency Count  (cudaEvent timer)
    // The GPU timing is returned via the gpuTimeMs parameter.
    // It measures only the kernel execution — not memory copies —
    // which gives the fairest comparison of compute performance.
    // ----------------------------------------------------------
    int   gpuFreq[256] = {0};
    float gpuTimeMs    = 0.0f;

    // Also time total GPU work (includes memory transfers)
    auto gpuTotalStart = std::chrono::high_resolution_clock::now();
    gpuCountFrequency(text, gpuFreq, gpuTimeMs);
    auto gpuTotalEnd   = std::chrono::high_resolution_clock::now();

    double gpuTotalMs =
        std::chrono::duration<double, std::milli>(gpuTotalStart - gpuTotalEnd).count();
    gpuTotalMs = std::chrono::duration<double, std::milli>(
                     gpuTotalEnd - gpuTotalStart).count();

    // ----------------------------------------------------------
    // STEP 4: Verify that CPU and GPU produce identical results
    // ----------------------------------------------------------
    bool match = (memcmp(cpuFreq, gpuFreq, 256 * sizeof(int)) == 0);
    std::cout << "CPU vs GPU frequency match: "
              << (match ? "YES [PASS]" : "NO  [FAIL]") << "\n\n";

    // ----------------------------------------------------------
    // STEP 5: Display Character Frequency Table
    // ----------------------------------------------------------
    std::cout << "--- Character Frequency Table ---\n";
    std::cout << std::left
              << std::setw(10) << "Char"
              << std::setw(8)  << "ASCII"
              << std::setw(12) << "Frequency"
              << "\n";
    thinDivider(30);

    // Collect characters that actually appear in the text
    std::vector<std::pair<int,int>> freqList; // (ascii, count)
    for (int i = 0; i < 256; i++) {
        if (cpuFreq[i] > 0) {
            freqList.emplace_back(i, cpuFreq[i]);
        }
    }

    // Sort by descending frequency for readability
    std::sort(freqList.begin(), freqList.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });

    for (auto& [ascii, count] : freqList) {
        std::cout << std::left
                  << std::setw(10) << charDisplay(ascii)
                  << std::setw(8)  << ascii
                  << std::setw(12) << count
                  << "\n";
    }

    // ----------------------------------------------------------
    // STEP 6: Build Huffman Tree (CPU only — tree construction
    //         is inherently sequential due to data dependencies)
    // ----------------------------------------------------------
    HuffmanNode* root = buildHuffmanTree(cpuFreq);

    // ----------------------------------------------------------
    // STEP 7: Generate Huffman Codes via recursive tree traversal
    // ----------------------------------------------------------
    std::unordered_map<char, std::string> codes;
    generateCodes(root, "", codes);

    // ----------------------------------------------------------
    // STEP 8: Display Huffman Codes
    // ----------------------------------------------------------
    std::cout << "\n--- Huffman Codes (sorted by frequency) ---\n";
    std::cout << std::left
              << std::setw(10) << "Char"
              << std::setw(10) << "Frequency"
              << std::setw(8)  << "Bits"
              << "Code\n";
    thinDivider(48);

    // Calculate total compressed bits for efficiency metric
    long long totalBits = 0;
    for (auto& [ascii, count] : freqList) {
        char c = static_cast<char>(ascii);
        const std::string& code = codes[c];
        totalBits += static_cast<long long>(count) * code.size();

        std::cout << std::left
                  << std::setw(10) << charDisplay(ascii)
                  << std::setw(10) << count
                  << std::setw(8)  << code.size()
                  << code << "\n";
    }

    // Original size if stored as 8-bit ASCII
    long long originalBits = static_cast<long long>(text.size()) * 8;
    double compressionRatio =
        static_cast<double>(totalBits) / static_cast<double>(originalBits) * 100.0;

    std::cout << "\nOriginal size  : " << originalBits << " bits ("
              << text.size() << " bytes)\n";
    std::cout << "Compressed size: " << totalBits << " bits ("
              << (totalBits + 7) / 8 << " bytes)\n";
    std::cout << "Space savings  : "
              << std::fixed << std::setprecision(2)
              << (100.0 - compressionRatio) << "%\n";

    // ----------------------------------------------------------
    // STEP 9: Performance Comparison
    // ----------------------------------------------------------
    std::cout << "\n";
    divider();
    std::cout << "           Performance Comparison\n";
    divider();
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Input size         : " << text.size() << " characters\n\n";

    std::cout << "CPU Time (chrono)  : " << cpuTimeMs     << " ms\n";
    std::cout << "GPU Kernel Time    : " << gpuTimeMs     << " ms  (kernel only)\n";
    std::cout << "GPU Total Time     : " << gpuTotalMs    << " ms  (incl. memcpy)\n\n";

    // Speedup: how many times faster is GPU kernel vs CPU?
    double kernelSpeedup = cpuTimeMs / static_cast<double>(gpuTimeMs);
    double totalSpeedup  = cpuTimeMs / gpuTotalMs;

    std::cout << "Kernel Speedup     : " << std::setprecision(4)
              << kernelSpeedup << "x\n";
    std::cout << "End-to-End Speedup : " << totalSpeedup  << "x\n\n";

    // Interpretation for viva
    std::cout << "--- Interpretation ---\n";
    if (kernelSpeedup >= 1.0) {
        std::cout << "[GPU] Kernel is " << kernelSpeedup
                  << "x faster than CPU for frequency counting.\n";
    } else {
        std::cout << "[NOTE] GPU kernel appears slower than CPU here.\n";
        std::cout << "       This is expected for SMALL inputs because:\n";
        std::cout << "       - CUDA has fixed overhead (kernel launch ~5-10 us)\n";
        std::cout << "       - Memory transfers add latency\n";
        std::cout << "       - CPU is optimised for sequential small loops\n";
        std::cout << "       GPU advantage becomes clear at 10MB+ input sizes.\n";
    }

    std::cout << "\n[GPU Parallelism] Input split across "
              << "multiple threads:\n";
    std::cout << "   Threads per block : 256\n";
    int blocks = (static_cast<int>(text.size()) + 255) / 256;
    std::cout << "   Blocks launched   : " << blocks << "\n";
    std::cout << "   Total GPU threads : " << (blocks * 256) << "\n";
    std::cout << "   Each thread       : processes 1 character simultaneously\n";

    // ----------------------------------------------------------
    // Cleanup
    // ----------------------------------------------------------
    freeTree(root);

    divider();
    std::cout << "Program completed successfully.\n";
    divider();

    return 0;
}
