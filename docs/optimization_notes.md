# CUDA Optimization Notes & Bottleneck Analysis

This document provides a deep dive into the engineering decisions and performance optimizations implemented in this project.

## 1. Frequency Counting: The Contention Problem

The core challenge in parallelizing Huffman frequency counting is **memory contention**. Many characters (like 'e', 't', ' ', '\n') occur with very high frequency. In a naive GPU implementation, thousands of threads might attempt to update the same counter simultaneously.

### Optimization 1: Global Atomics (Naive)
- **Implementation**: Every thread calls `atomicAdd` on global memory directly.
- **Bottleneck**: Massive L2 cache and DRAM contention. The atomic operations are serialized at the memory controller, leading to very low memory bandwidth utilization.

### Optimization 2: Shared Memory Histograms
- **Implementation**: Each block maintains its own 256-bucket histogram in shared memory.
- **Why it works**: Shared memory atomics are significantly faster (order of magnitude) than global memory atomics because they happen on-chip.
- **Result**: Drastic reduction in global memory traffic. The only global atomics occur during the final "merge" phase (one update per block per character).

### Optimization 3: Multi-Bank Aggregation (Warp-Level)
- **Implementation**: To further reduce **shared memory bank conflicts**, we can use multiple sub-histograms within shared memory.
- **Details**: Since shared memory is divided into 32 banks, having multiple threads in a warp hit the same bucket (e.g., character 'A') causes a bank conflict. By interleaving 4 sub-histograms, we distribute the traffic.

## 2. Occupancy & Grid-Stride Loops

Instead of launching exactly one thread per character, we use **Grid-Stride Loops**.
```cpp
for (size_t i = idx; i < n; i += stride) { ... }
```
- **Benefits**:
    1.  **Scalability**: Handles inputs larger than the maximum grid size.
    2.  **Performance**: Allows threads to reuse registers and stay "hot" in the pipeline, reducing the overhead of thread creation/destruction.
    3.  **Flexibility**: Allows easy tuning of block and grid dimensions to find the "sweet spot" for occupancy.

## 3. The CPU-GPU Trade-off

### Latency vs. Throughput
- **Small Files (< 1MB)**: The time taken to allocate GPU memory (`cudaMalloc`) and transfer data over PCIe dominates. The CPU is often faster for these "bursty" tasks.
- **Large Files (> 10MB)**: The massive throughput of the GPU (tens of GB/s) overcomes the fixed launch overhead.

### Data Transfer Bottlenecks
The project uses **Pinned Memory** (via future expansion or manual optimization) and **Async Memcpy** where possible. Overlapping the compression of chunk `N` with the data transfer of chunk `N+1` is the key to reaching peak system performance.

## 4. Future Roadmap
- [ ] **CUDA Graphs**: Reduce kernel launch overhead for repeated small compressions.
- [ ] **Warp Shuffle Primitives**: Use `__shfl_down_sync` for even faster intra-warp reductions before hitting shared memory.
- [ ] **Multi-GPU Scaling**: Distribute massive file chunks (e.g., 10GB+) across multiple GPUs using NCCL or Peer-to-Peer copies.
