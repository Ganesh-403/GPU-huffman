# Huffman Encoding using GPU Parallelization (CUDA)
### Final Year Mini Project | Computer Engineering | SPPU 2019 Pattern

---

## Project Overview

This project demonstrates how **GPU parallelization using CUDA** can accelerate
the character **frequency counting** step of Huffman Encoding.

| Component         | Technology                     |
|-------------------|--------------------------------|
| Language          | C++ with CUDA (.cu)            |
| Compiler          | nvcc (NVIDIA CUDA Toolkit)     |
| Platform          | Windows 10/11                  |
| GPU               | NVIDIA RTX 3050 (or any CUDA)  |
| Timing (CPU)      | std::chrono high_resolution_clock |
| Timing (GPU)      | cudaEventRecord / ElapsedTime  |

---

## Project Structure

```
GPU huffman/
├── main.cpp           — Entry point; orchestrates CPU/GPU implementations, timing & validation
├── huffman_cpu.h      — Data structures & CPU function declarations
├── huffman_cpu.cpp    — CPU Huffman: frequency count, tree building, code generation
├── gpu_kernels.h      — GPU kernel function declarations
├── gpu_kernels.cu     — CUDA kernels + device memory management
├── sample.txt         — Small test input (base file for generating larger datasets)
├── big.txt            — Large test file (25MB+); use for demonstrating GPU speedup
├── huffman.exe        — Compiled executable
├── .git/              — Git repository metadata
├── .gitignore         — Git ignore rules
└── README.md          — This file
```

---

## Prerequisites

1. **NVIDIA CUDA Toolkit** — Download from https://developer.nvidia.com/cuda-downloads
   - Recommended: CUDA 11.x or 12.x
2. **Visual Studio** (Community edition is free) — Required by nvcc on Windows
   - Install with "Desktop development with C++" workload
3. **GPU Driver** — Update to latest from https://www.nvidia.com/drivers

Verify installation:
```
nvcc --version
nvidia-smi
```

---

## Compilation Command

Open **x64 Native Tools Command Prompt for VS** (or Developer PowerShell) and run:

```bash
nvcc main.cpp huffman_cpu.cpp gpu_kernels.cu -o huffman.exe -std=c++17 -O2 -allow-unsupported-compiler
```

### Flag Explanations
| Flag          | Purpose                                       |
|---------------|-----------------------------------------------|
| `main.cpp`    | C++ entry point                               |
| `huffman_cpu.cpp` | CPU Huffman implementation                |
| `gpu_kernels.cu`  | CUDA kernel file (compiled by nvcc)       |
| `-o huffman.exe`  | Output executable name                    |
| `-std=c++17`      | Use C++17 standard                        |
| `-O2`             | Compiler optimization level 2            |
| `-allow-unsupported-compiler` | Allow MSVC versions newer than officially supported |

---

## How to Run

### Quick Test (Small File)
```bash
huffman.exe
```
By default, the program reads `sample.txt`. This verifies correctness but shows minimal GPU speedup due to small data size.

### Full Test (Large File - Recommended for GPU Speedup Demonstration)

**Generate `big.txt` (25MB) using Command Prompt** (NOT PowerShell—PowerShell injects CR characters):
```cmd
cd /d "D:\BE\SEM 8\HPC Mini Project\GPU huffman"
del big.txt
for /L %i in (1,1,5000) do type sample.txt >> big.txt
```

Then recompile and run:
```bash
nvcc main.cpp huffman_cpu.cpp gpu_kernels.cu -o huffman.exe -std=c++17 -O2 -allow-unsupported-compiler
huffman.exe
```

With `big.txt` (25MB):
- CPU time: ~7-8ms
- GPU kernel time: ~2-3ms
- Expected speedup: **~3x-4x faster on GPU**

### Important Notes
- ⚠️ **Use Command Prompt (cmd.exe)** for file generation, NOT PowerShell
  - PowerShell's file encoding adds CR (carriage return) characters, corrupting test data
- ✓ **Modify `main.cpp` line** to read `big.txt` instead of `sample.txt` for larger tests
- ✓ Test with different file sizes to observe when GPU parallelism becomes beneficial

---

## Expected Output

```
====================================================
   Huffman Encoding: GPU vs CPU Comparison
====================================================
Input file    : sample.txt
Total chars   : 3521

CPU vs GPU frequency match: YES [PASS]

--- Character Frequency Table ---
Char      ASCII   Frequency
------------------------------
SPACE     32      612
e         101     389
t         116     267
...

--- Huffman Codes ---
Char      Frequency  Bits    Code
------------------------------------------------
SPACE     612        4       0101
e         389        4       1101
...

====================================================
           Performance Comparison
====================================================
Input size         : 3521 characters

CPU Time (chrono)  : 0.012300 ms
GPU Kernel Time    : 0.008500 ms
GPU Total Time     : 1.234000 ms (incl. memcpy)

Kernel Speedup     : 1.4470x
End-to-End Speedup : 0.0099x
```

---

## Understanding the Performance Results

### Why GPU kernel is faster (for large inputs):
- The GPU launches **one thread per character**
- Thousands of characters are processed **simultaneously**
- On RTX 3050: ~2048 CUDA cores run in parallel

### Why CPU might win for small files:
| Factor                  | Impact                          |
|-------------------------|---------------------------------|
| CUDA kernel launch      | ~5–10 µs fixed overhead         |
| cudaMemcpy H→D + D→H    | Adds latency for small data     |
| CPU is optimised for    | Sequential memory access        |

### Rule of thumb:
- **< 100 KB** → CPU is faster (overhead dominates)
- **> 1 MB**   → GPU kernel becomes significantly faster
- **> 10 MB**  → GPU speedup is clearly visible

---

## How GPU Parallelization Works (Kernel Explanation)

```
Input text: "hello world" (11 characters)

Thread 0  → 'h' → atomicAdd(&freq['h'], 1)
Thread 1  → 'e' → atomicAdd(&freq['e'], 1)
Thread 2  → 'l' → atomicAdd(&freq['l'], 1)
Thread 3  → 'l' → atomicAdd(&freq['l'], 1)  ← same bucket, safe!
Thread 4  → 'o' → atomicAdd(&freq['o'], 1)
...
All threads execute SIMULTANEOUSLY on the GPU

Grid layout:
  Threads per block : 256
  Total blocks      : ceil(n / 256)
  Total threads     : blocks × 256
```

**atomicAdd** ensures that when Thread 2 and Thread 3 both update
`freq['l']` at the same time, neither update is lost — each is
applied atomically (indivisibly) to prevent data races.

---

## Viva Questions & Answers

**Q: Why use atomicAdd instead of a regular increment?**
A: Multiple GPU threads may try to update the same frequency bucket
   simultaneously. atomicAdd ensures thread-safe, serialised updates
   to shared memory, preventing race conditions.

**Q: Why is tree construction done on CPU?**
A: Building the Huffman tree requires repeatedly finding and combining
   the two minimum-frequency nodes — a sequential, data-dependent
   operation not suited to parallel GPU execution.

**Q: What is the time complexity?**
A: CPU counting: O(n). GPU counting: O(n/P) where P = number of threads.
   Tree building: O(k log k) where k = distinct characters ≤ 256.

**Q: What is cudaMemcpy direction?**
A: cudaMemcpyHostToDevice = CPU RAM → GPU VRAM
   cudaMemcpyDeviceToHost = GPU VRAM → CPU RAM

**Q: What are CUDA events used for?**
A: cudaEventRecord inserts timestamp markers into the GPU command queue.
   cudaEventElapsedTime calculates the precise kernel execution time
   in milliseconds, independent of CPU-side timing overhead.
