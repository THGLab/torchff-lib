#pragma once

#include <cuda_runtime.h>

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void block_reduce_sum(T v, T* __restrict__ out) {
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");
    static_assert(BLOCK_SIZE >= 32, "BLOCK_SIZE must be >= 32");
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE must be <= 1024");

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ T warp_sums[NUM_WARPS];

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);

    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    if (wid == 0) {
        T v = (lane < NUM_WARPS) ? warp_sums[lane] : T(0);
        v = warp_reduce_sum(v);
        if (lane == 0) { atomicAdd(out, v); }
    }
}


/**
 * @brief Computes the rank (0-based index) of the current thread's true value 
 * among all threads in the warp that have a true value.
 * * @param my_bool The boolean condition for the calling thread.
 * @return The 0-based index if my_bool is true, otherwise -1.
 */
__device__ __forceinline__ void count_true_values_in_warp(bool& boolean, int& rank, int& count, unsigned int lane) {
    // Synchronize all threads in the warp and collect their boolean results into a 32-bit mask.
    // Each bit i in the mask corresponds to the boolean value of lane i.
    unsigned int warp_mask = __ballot_sync(0xFFFFFFFFu, boolean);

    // Generate a bitmask where all bits less than the current lane_id are set to 1.
    // Example: if lane_id is 3, prefix_mask is 000...0111 (binary)
    unsigned int prefix_mask = (1u << lane) - 1;

    // Calculate the number of set bits (true values) that occurred before this lane.
    // This is the intersection of the total warp mask and the prefix mask.
    rank = (boolean ? __popc(warp_mask & prefix_mask) : -1);
    count = __popc(warp_mask);
}


template <typename T_INPUT, typename T_OUTPUT, int N, int BUFFER_SIZE, int WARP_SIZE=32>
__device__ __forceinline__ void flush_warp_buffer(
    T_INPUT* buffer,
    int& curr_buffer_size, 
    T_OUTPUT* out,
    int* out_count,
    int max_count,
    int lane_id,
    bool force_flush
) {
    // // Ensure all threads in the warp see the same 'curr_buffer_size' before proceeding
    // // This prevents race conditions where threads might read inconsistent size values
    __syncwarp(); 

    // Check if the buffer has reached the threshold to trigger a flush
    if (curr_buffer_size + WARP_SIZE > BUFFER_SIZE || force_flush) {

        int start = 0;
        if ( lane_id == 0 ) {
            start = atomicAdd(out_count, curr_buffer_size);
        }
        start = __shfl_sync(0xFFFFFFFFu, start, 0);

        // Cooperative move: Each thread in the warp processes a subset of the buffer.
        // This maximizes memory throughput by utilizing coalesced global memory access.
        for (int i = lane_id; i < curr_buffer_size; i += WARP_SIZE) {
            int start_idx = start + i;
            if ( start_idx < max_count ) {
                #pragma unroll
                for (int j = 0; j < N; j++) {
                    // Transfer data from the local/shared buffer to the output destination
                    out[start_idx * N + j] = static_cast<T_OUTPUT>(buffer[i * N + j]);
                }
            }
        }
        curr_buffer_size = 0; 
    }
}