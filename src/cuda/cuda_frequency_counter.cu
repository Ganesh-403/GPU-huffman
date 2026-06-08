#include "cuda_frequency_counter.h"
#include "../utils/cuda_utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <memory>

namespace gpu_huffman {

// 1. Baseline: Naive Global Memory Atomics
__global__ void count_naive_kernel(const char* data, size_t n, uint32_t* freq) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        unsigned char c = static_cast<unsigned char>(data[i]);
        atomicAdd(&freq[c], 1);
    }
}

// 2. Shared Memory Optimization (Per-block histograms)
__global__ void count_shared_kernel(const char* data, size_t n, uint32_t* freq) {
    __shared__ uint32_t local_freq[256];

    // Initialize shared memory
    int tid = threadIdx.x;
    if (tid < 256) {
        local_freq[tid] = 0;
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        unsigned char c = static_cast<unsigned char>(data[i]);
        atomicAdd(&local_freq[c], 1);
    }
    __syncthreads();

    // Merge to global memory
    if (tid < 256) {
        if (local_freq[tid] > 0) {
            atomicAdd(&freq[tid], local_freq[tid]);
        }
    }
}

// 3. Warp-Level Optimized (Warp-aggregated atomics)
// Note: On modern architectures, atomicAdd on shared memory is very fast.
// But for "research" flavor, we can show a warp-shuffle based approach if buckets were fewer.
// For 256 buckets, shared memory is standard. We can implement a version that uses 
// multiple shared memory banks to reduce contention.
__global__ void count_warp_optimized_kernel(const char* data, size_t n, uint32_t* freq) {
    // Each warp or set of warps can have its own shared memory sub-histogram to reduce bank conflicts
    // However, for simplicity and effectiveness, we'll use a multi-bank shared memory approach.
    __shared__ uint32_t local_freq[4][256]; // 4 sub-histograms to reduce contention

    int tid = threadIdx.x;
    int sub_idx = tid % 4;

    if (tid < 256) {
        local_freq[0][tid] = 0;
        local_freq[1][tid] = 0;
        local_freq[2][tid] = 0;
        local_freq[3][tid] = 0;
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        unsigned char c = static_cast<unsigned char>(data[i]);
        atomicAdd(&local_freq[sub_idx][c], 1);
    }
    __syncthreads();

    // Merge sub-histograms to global
    if (tid < 256) {
        uint32_t sum = local_freq[0][tid] + local_freq[1][tid] + local_freq[2][tid] + local_freq[3][tid];
        if (sum > 0) {
            atomicAdd(&freq[tid], sum);
        }
    }
}

CudaFrequencyCounter::CudaFrequencyCounter(CudaKernelType type) : type_(type) {}

std::string CudaFrequencyCounter::name() const {
    switch (type_) {
        case CudaKernelType::NAIVE_GLOBAL:   return "CUDA Naive (Global Atomics)";
        case CudaKernelType::SHARED_MEMORY:  return "CUDA Shared Memory (Optimized)";
        case CudaKernelType::WARP_OPTIMIZED: return "CUDA Warp-Aggregated (Multi-Bank)";
        case CudaKernelType::MULTI_STREAM:   return "CUDA Multi-Stream Async";
        case CudaKernelType::UNIFIED_MEMORY: return "CUDA Unified Memory";
        default:                             return "CUDA Unknown";
    }
}

FrequencyResult CudaFrequencyCounter::count(const std::string& text) {
    FrequencyResult result;
    std::memset(result.frequencies, 0, sizeof(result.frequencies));
    result.method_name = name();

    size_t n = text.size();
    if (n == 0) return result;

    if (type_ == CudaKernelType::UNIFIED_MEMORY) {
        UnifiedBuffer<char> unified_data(n);
        UnifiedBuffer<uint32_t> unified_freq(256);

        std::memcpy(unified_data.get(), text.c_str(), n);
        unified_freq.zero();

        GpuEvent start, stop;

        int threadsPerBlock = 256;
        int blocksPerGrid = static_cast<int>((n + threadsPerBlock - 1) / threadsPerBlock);
        blocksPerGrid = std::min(blocksPerGrid, 2048);

        start.record();

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        unified_data.prefetch_to_device(device);
        unified_freq.prefetch_to_device(device);

        count_shared_kernel<<<blocksPerGrid, threadsPerBlock>>>(unified_data.get(), n, unified_freq.get());

        unified_freq.prefetch_to_host();
        stop.record();
        stop.synchronize();

        result.execution_time_ms = GpuEvent::elapsed_time(start, stop);
        std::memcpy(result.frequencies, unified_freq.get(), 256 * sizeof(uint32_t));

        return result;
    }

    if (type_ == CudaKernelType::MULTI_STREAM) {
        const int num_streams = 4;
        std::vector<std::vector<uint32_t>> host_freq(num_streams, std::vector<uint32_t>(256, 0));

        GpuBuffer<char> d_data(n);
        std::vector<std::unique_ptr<GpuBuffer<uint32_t>>> d_freq_buffers;
        std::vector<std::unique_ptr<GpuStream>> streams;

        for (int i = 0; i < num_streams; ++i) {
            d_freq_buffers.push_back(std::make_unique<GpuBuffer<uint32_t>>(256));
            d_freq_buffers[i]->zero();
            streams.push_back(std::make_unique<GpuStream>());
        }

        GpuEvent start, stop;
        start.record();

        size_t chunk_size = n / num_streams;
        for (int i = 0; i < num_streams; ++i) {
            size_t offset = i * chunk_size;
            size_t size = (i == num_streams - 1) ? (n - offset) : chunk_size;

            if (size == 0) continue;

            CUDA_CHECK(cudaMemcpyAsync(d_data.get() + offset, text.c_str() + offset, size * sizeof(char), 
                                       cudaMemcpyHostToDevice, streams[i]->get()));

            int threadsPerBlock = 256;
            int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
            blocksPerGrid = std::min(blocksPerGrid, 512);

            count_shared_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]->get()>>>(
                d_data.get() + offset, size, d_freq_buffers[i]->get()
            );

            CUDA_CHECK(cudaMemcpyAsync(host_freq[i].data(), d_freq_buffers[i]->get(), 256 * sizeof(uint32_t), 
                                       cudaMemcpyDeviceToHost, streams[i]->get()));
        }

        for (int i = 0; i < num_streams; ++i) {
            streams[i]->synchronize();
        }

        stop.record();
        stop.synchronize();

        result.execution_time_ms = GpuEvent::elapsed_time(start, stop);

        for (int i = 0; i < num_streams; ++i) {
            for (int j = 0; j < 256; ++j) {
                result.frequencies[j] += host_freq[i][j];
            }
        }

        return result;
    }

    // Default modes: NAIVE, SHARED, WARP
    GpuBuffer<char> d_data(n);
    GpuBuffer<uint32_t> d_freq(256);

    d_data.copy_to_device(text.c_str());
    d_freq.zero();

    GpuEvent start, stop;

    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((n + threadsPerBlock - 1) / threadsPerBlock);
    blocksPerGrid = std::min(blocksPerGrid, 2048);

    start.record();

    switch (type_) {
        case CudaKernelType::NAIVE_GLOBAL:
            count_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data.get(), n, d_freq.get());
            break;
        case CudaKernelType::SHARED_MEMORY:
            count_shared_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data.get(), n, d_freq.get());
            break;
        case CudaKernelType::WARP_OPTIMIZED:
            count_warp_optimized_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data.get(), n, d_freq.get());
            break;
        default:
            count_shared_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data.get(), n, d_freq.get());
            break;
    }

    stop.record();
    stop.synchronize();

    result.execution_time_ms = GpuEvent::elapsed_time(start, stop);
    d_freq.copy_to_host(result.frequencies);

    return result;
}

} // namespace gpu_huffman
