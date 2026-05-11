#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <memory>

/**
 * @brief Macro for checking CUDA errors.
 */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error("CUDA Error at " + std::string(__FILE__) + \
                                     ":" + std::to_string(__LINE__) +       \
                                     " -> " + cudaGetErrorString(err));     \
        }                                                                   \
    } while (0)

namespace gpu_huffman {

/**
 * @brief RAII wrapper for GPU memory.
 */
template <typename T>
class GpuBuffer {
public:
    explicit GpuBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    ~GpuBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Disable copy
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // Enable move
    GpuBuffer(GpuBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t count() const { return count_; }
    size_t size_bytes() const { return count_ * sizeof(T); }

    void copy_to_device(const T* host_ptr) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, count_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* host_ptr) const {
        CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void zero() {
        CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

/**
 * @brief RAII wrapper for CUDA Events.
 */
class GpuEvent {
public:
    GpuEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~GpuEvent() {
        cudaEventDestroy(event_);
    }

    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    static float elapsed_time(const GpuEvent& start, const GpuEvent& stop) {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, stop.event_));
        return ms;
    }

private:
    cudaEvent_t event_;
};

/**
 * @brief RAII wrapper for CUDA Streams.
 */
class GpuStream {
public:
    GpuStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~GpuStream() {
        cudaStreamDestroy(stream_);
    }

    cudaStream_t get() const { return stream_; }
    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
    cudaStream_t stream_;
};

} // namespace gpu_huffman
