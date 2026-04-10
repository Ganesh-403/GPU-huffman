#pragma once

// ============================================================
// gpu_kernels.h
// Declares the host-callable wrapper for the GPU frequency
// counting kernel.  Only standard C++ types are used here so
// that main.cpp (compiled by the C++ compiler) can include
// this header without needing CUDA headers.
// ============================================================

#include <string>

// ------------------------------------------------------------
// gpuCountFrequency
//
// Launches a CUDA kernel to count character frequencies in
// parallel on the GPU.
//
// Parameters:
//   text      - input string to analyse
//   freq[256] - output array filled with character frequencies
//   gpuTimeMs - [out] kernel execution time in milliseconds
//               (measured with CUDA Events, excludes mem copy)
// ------------------------------------------------------------
void gpuCountFrequency(const std::string& text,
                       int freq[256],
                       float& gpuTimeMs);
