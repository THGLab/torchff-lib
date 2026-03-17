#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"
#include "nblist/exclusions.cuh"


#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef NBLIST_NSQUARED_BLOCK_SIZE
#define NBLIST_NSQUARED_BLOCK_SIZE 256
#endif

#ifndef NBLIST_NSQUARED_BUFFER_SIZE
#define NBLIST_NSQUARED_BUFFER_SIZE 256
#endif


template <typename scalar_t, bool PERIODIC, int BUFFER_SIZE, int BLOCK_SIZE>
__global__ void build_neighbor_list_nsquared_kernel(
    scalar_t* coords,
    scalar_t* g_box,
    scalar_t cutoff2,
    int64_t* pairs,
    int32_t* npairs,
    int64_t natoms,
    int64_t max_npairs,
    const int64_t* excl_row_ptr,
    const int64_t* excl_col_indices,
    bool include_self
)
{
    __shared__ scalar_t box[9];
    __shared__ scalar_t box_inv[9];

    if constexpr (PERIODIC) {
        if (threadIdx.x < 9) {
            box[threadIdx.x] = g_box[threadIdx.x];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            invert_box_3x3<scalar_t>(box, box_inv);
        }
        __syncthreads();
    }

    int idxInWarp = threadIdx.x & (WARP_SIZE - 1);
    int warpIdxInBlock = threadIdx.x / WARP_SIZE;
    __shared__ int64_t pairs_buffer[BUFFER_SIZE * 2 * BLOCK_SIZE / WARP_SIZE];
    int64_t* pairs_buffer_warp = pairs_buffer + warpIdxInBlock * BUFFER_SIZE * 2;
    int curr_buffer_size = 0;

    int64_t maxv = natoms * (natoms + 1) / 2;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t maxv_padded = ((maxv + stride - 1) / stride) * stride;

    for ( int64_t index = threadIdx.x+blockIdx.x*blockDim.x; 
        index < maxv_padded; 
        index += stride ) {
        bool include = false;
        int64_t i = 0, j = 0;

        if ( index < maxv ) {
            i = (int64_t)floor((sqrt(((double)index) * 8 + 1) - 1) / 2);
            j = (int64_t)index - (i * (i + 1)) / 2;

            scalar_t drvec[3];
            diff_vec3(&coords[i * 3], &coords[j * 3], drvec);

            if constexpr (PERIODIC) {
                apply_pbc_triclinic(drvec, box, box_inv, drvec);
            }

            scalar_t dist2 = drvec[0] * drvec[0] + drvec[1] * drvec[1] + drvec[2] * drvec[2];
            include = (dist2 <= cutoff2) && (i != j || include_self);
            if ( include ) {
                include = !is_excluded_csr(i, j, excl_row_ptr, excl_col_indices);
            }
        }

        int rank, count;
        count_true_values_in_warp(include, rank, count, idxInWarp);
        if ( count > 0 ) {
            // int start = 0;
            // if ( idxInWarp == 0 ) {
            //     start = atomicAdd(npairs, count);
            // }
            // start = __shfl_sync(0xFFFFFFFFu, start, 0);
            // int start_idx = start + rank;
            // if (include && start_idx < max_npairs) {
            //     pairs[start_idx * 2] = i;
            //     pairs[start_idx * 2 + 1] = j;
            // }
            int start_idx = curr_buffer_size + rank;
            if ( include ) {
                pairs_buffer_warp[start_idx * 2] = i;
                pairs_buffer_warp[start_idx * 2 + 1] = j;
            }
            curr_buffer_size += count;
            flush_warp_buffer<int64_t, int64_t, 2, BUFFER_SIZE, WARP_SIZE>(pairs_buffer_warp, curr_buffer_size, pairs, npairs, max_npairs, idxInWarp, false);
        }
    }

    flush_warp_buffer<int64_t, int64_t, 2, BUFFER_SIZE, WARP_SIZE>(pairs_buffer_warp, curr_buffer_size, pairs, npairs, max_npairs, idxInWarp, true);
}


at::Tensor build_neighbor_list_nsquared_out_cuda(
    const at::Tensor& coords,
    const c10::optional<at::Tensor> box,
    const at::Scalar& cutoff,
    at::Tensor pairs,
    c10::optional<at::Tensor> excl_row_ptr,
    c10::optional<at::Tensor> excl_col_indices,
    bool include_self
);

std::tuple<at::Tensor, at::Tensor> build_neighbor_list_nsquared_cuda(
    const at::Tensor& coords,
    const c10::optional<at::Tensor> box,
    const at::Scalar& cutoff,
    const at::Scalar& max_npairs,
    c10::optional<at::Tensor> excl_row_ptr,
    c10::optional<at::Tensor> excl_col_indices,
    bool include_self
)
{
    int64_t natoms = coords.size(0);
    int64_t max_npairs_ = (max_npairs.toLong() < 0) ? natoms * (natoms + 1) / 2 : max_npairs.toLong();
    at::Tensor pairs = at::empty({max_npairs_, 2}, coords.options().dtype(at::kLong));
    at::Tensor npairs = build_neighbor_list_nsquared_out_cuda(
        coords, box, cutoff, pairs, excl_row_ptr, excl_col_indices, include_self
    );
    return std::make_tuple(pairs, npairs);
}


at::Tensor build_neighbor_list_nsquared_out_cuda(
    const at::Tensor& coords,
    const c10::optional<at::Tensor> box,
    const at::Scalar& cutoff,
    at::Tensor pairs,
    c10::optional<at::Tensor> excl_row_ptr,
    c10::optional<at::Tensor> excl_col_indices,
    bool include_self
)
{
    int64_t natoms = coords.size(0);
    int64_t max_npairs_ = pairs.size(0);

    pairs.fill_(-1);

    const int64_t* row_ptr = (excl_row_ptr.has_value() && excl_row_ptr.value().defined())
        ? excl_row_ptr.value().data_ptr<int64_t>() : nullptr;
    const int64_t* col_ind = (excl_col_indices.has_value() && excl_col_indices.value().defined())
        ? excl_col_indices.value().data_ptr<int64_t>() : nullptr;

    bool periodic = box.has_value() && box.value().defined();

    at::Tensor npairs = at::zeros({1}, coords.options().dtype(at::kInt));

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    int32_t GRID_SIZE = props->maxBlocksPerMultiProcessor*props->multiProcessorCount;

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "build_neighbor_list_nsquared_out_cuda", ([&] {
        scalar_t cutoff2 = static_cast<scalar_t>(cutoff.toDouble() * cutoff.toDouble());
        scalar_t* box_ptr = periodic ? box.value().data_ptr<scalar_t>() : nullptr;

        DISPATCH_BOOL(periodic, PERIODIC, [&] {
            build_neighbor_list_nsquared_kernel<scalar_t, PERIODIC, NBLIST_NSQUARED_BUFFER_SIZE, NBLIST_NSQUARED_BLOCK_SIZE><<<GRID_SIZE, NBLIST_NSQUARED_BLOCK_SIZE, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                box_ptr,
                cutoff2,
                pairs.data_ptr<int64_t>(),
                npairs.data_ptr<int32_t>(),
                natoms,
                max_npairs_,
                row_ptr,
                col_ind,
                include_self
            );
        });
    }));

    return npairs;
}


TORCH_LIBRARY_IMPL(torchff, CUDA, m) {
    m.impl("build_neighbor_list_nsquared", build_neighbor_list_nsquared_cuda);
    m.impl("build_neighbor_list_nsquared_out", build_neighbor_list_nsquared_out_cuda);
}
