// ============================================================
// gpu_kernels.cu
// Contains the CUDA kernel and its host-callable wrapper for
// parallel character frequency counting on the GPU.
//
// Key CUDA concepts demonstrated:
//   - Thread indexing  (blockIdx, threadIdx, blockDim)
//   - atomicAdd        (thread-safe shared counter updates)
//   - cudaMalloc/Free  (device memory management)
//   - cudaMemcpy       (host <-> device data transfer)
//   - cudaEvent        (high-resolution GPU timing)
// ============================================================

#include "gpu_kernels.h"
#include <cuda_runtime.h>
#include <cstring>   // memset
#include <cstdio>    // fprintf
#include <algorithm> // min

// ------------------------------------------------------------
// CUDA error-checking helper macro.
// Wraps every CUDA call and prints a descriptive message if
// anything goes wrong, then exits the program.
// ------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d  ->  %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// KERNEL: countFrequenciesKernel
//
// Each GPU thread handles ONE character from the input array.
// All threads run in parallel, so an n-character string is
// processed in O(n / total_threads) time steps instead of O(n).
//
// Parameters:
//   d_data  - device pointer to the input character array
//   n       - total number of characters
//   d_freq  - device pointer to the 256-element frequency array
//
// Thread indexing formula:
//   global_idx = blockIdx.x * blockDim.x + threadIdx.x
//   ──────────────────────────────────────────────────
//   blockIdx.x  = which block this thread belongs to  (0 … gridDim.x-1)
//   blockDim.x  = threads per block (e.g., 256)
//   threadIdx.x = local index within the block         (0 … blockDim.x-1)
//
// Why atomicAdd?
//   Multiple threads may map to the SAME character (e.g., many
//   threads could all read 'e' from different positions).
//   Without atomicAdd, concurrent reads and writes to d_freq[idx]
//   would cause a data race and corrupt the count.
//   atomicAdd guarantees each increment is performed atomically.
// ============================================================
__global__ void countFrequenciesKernel(const char* d_data,
                                        int          n,
                                        int*         d_freq)
{
    __shared__ int blockFreq[256];

    int tid = threadIdx.x;
    if (tid < 256) {
        blockFreq[tid] = 0;
    }
    __syncthreads();

    int idx    = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop allows each thread to process multiple chars.
    while (idx < n) {
        unsigned char c = static_cast<unsigned char>(d_data[idx]);
        atomicAdd(&blockFreq[c], 1);
        idx += stride;
    }
    __syncthreads();

    // Merge per-block histogram into global histogram once per bucket.
    if (tid < 256 && blockFreq[tid] > 0) {
        atomicAdd(&d_freq[tid], blockFreq[tid]);
    }
}

// ============================================================
// HOST WRAPPER: gpuCountFrequency
//
// Called from main.cpp. Handles all memory management, kernel
// launch configuration, timing, and result retrieval.
// ============================================================
void gpuCountFrequency(const std::string& text,
                       int freq[256],
                       float& gpuTimeMs)
{
    int n = static_cast<int>(text.size());

    // ----------------------------------------------------------
    // Step 1: Declare device (GPU) memory pointers
    // ----------------------------------------------------------
    char* d_data = nullptr;   // Device copy of the input string
    int*  d_freq = nullptr;   // Device frequency table [256]

    // ----------------------------------------------------------
    // Step 2: Allocate device memory
    //   cudaMalloc(pointer, bytes) — equivalent of malloc() on GPU
    // ----------------------------------------------------------
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data),
                          n * sizeof(char)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_freq),
                          256 * sizeof(int)));

    // ----------------------------------------------------------
    // Step 3: Zero-initialise the device frequency table
    //   cudaMemset fills device memory with a byte value (0 here)
    // ----------------------------------------------------------
    CUDA_CHECK(cudaMemset(d_freq, 0, 256 * sizeof(int)));

    // ----------------------------------------------------------
    // Step 4: Copy input data from Host (CPU RAM) → Device (VRAM)
    //   cudaMemcpy(dst, src, bytes, direction)
    // ----------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_data,
                          text.c_str(),
                          n * sizeof(char),
                          cudaMemcpyHostToDevice));

    // ----------------------------------------------------------
    // Step 5: Create CUDA Events for precise GPU timing
    //   cudaEventRecord marks a timestamp in the GPU command queue.
    //   The elapsed time between two events is the kernel duration.
    // ----------------------------------------------------------
    cudaEvent_t startEvt, stopEvt;
    CUDA_CHECK(cudaEventCreate(&startEvt));
    CUDA_CHECK(cudaEventCreate(&stopEvt));

    // ----------------------------------------------------------
    // Step 6: Configure kernel launch parameters
    //
    //   threadsPerBlock: how many threads run in one block.
    //                    256 is a standard, efficient choice for
    //                    most NVIDIA GPUs (multiple of warp size 32).
    //
    //   blocksPerGrid:   how many blocks are needed to cover all
    //                    n characters. The formula:
    //                    ceil(n / threadsPerBlock)
    //                    ensures we don't miss any characters even
    //                    when n is not a multiple of threadsPerBlock.
    // ----------------------------------------------------------
    int threadsPerBlock = 256;
    int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid       = std::min(blocksPerGrid, 4096);

    // ----------------------------------------------------------
    // Step 7: Warm up once, then record start/stop around one
    //         real kernel launch so counts are not duplicated.
    // ----------------------------------------------------------
    countFrequenciesKernel<<<1, 1>>>(d_data, 1, d_freq);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_freq, 0, 256 * sizeof(int)));

    CUDA_CHECK(cudaEventRecord(startEvt));

    // *** KERNEL LAUNCH ***
    // Syntax: kernelName<<<gridDim, blockDim>>>(args...)
    countFrequenciesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_data, n, d_freq
    );

    CUDA_CHECK(cudaEventRecord(stopEvt));
    // Wait until the GPU has finished executing the kernel
    CUDA_CHECK(cudaEventSynchronize(stopEvt));

    // Calculate elapsed time in milliseconds
    CUDA_CHECK(cudaEventElapsedTime(&gpuTimeMs, startEvt, stopEvt));

    // ----------------------------------------------------------
    // Step 8: Copy results from Device (VRAM) → Host (CPU RAM)
    // ----------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(freq,
                          d_freq,
                          256 * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // ----------------------------------------------------------
    // Step 9: Release all GPU resources
    // ----------------------------------------------------------
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_freq));
    CUDA_CHECK(cudaEventDestroy(startEvt));
    CUDA_CHECK(cudaEventDestroy(stopEvt));
}
