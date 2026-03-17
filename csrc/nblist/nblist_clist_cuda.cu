#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"
#include "common/reduce.cuh"
#include "nblist/exclusions.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef NBLIST_CLIST_BLOCK_SIZE
#define NBLIST_CLIST_BLOCK_SIZE 256
#endif

#ifndef NBLIST_CLIST_BUFFER_SIZE
#define NBLIST_CLIST_BUFFER_SIZE 256
#endif


template <typename scalar_t>
__device__ __forceinline__ scalar_t dist_from_fractional_coords(
    scalar_t* fcrd_i, scalar_t* fcrd_j, scalar_t* box
) {
    scalar_t dfcrd[3];
    diff_vec3(fcrd_i, fcrd_j, dfcrd);
    dfcrd[0] -= round_(dfcrd[0]);
    dfcrd[1] -= round_(dfcrd[1]);
    dfcrd[2] -= round_(dfcrd[2]);
    scalar_t x = dfcrd[0] * box[0] + dfcrd[1] * box[3] + dfcrd[2] * box[6];
    scalar_t y = dfcrd[0] * box[1] + dfcrd[1] * box[4] + dfcrd[2] * box[7];
    scalar_t z = dfcrd[0] * box[2] + dfcrd[1] * box[5] + dfcrd[2] * box[8];
    return norm3d_(x, y, z);
}


// Step 1: Compute fractional coordinates and assign each atom to a cell.
template <typename scalar_t>
__global__ void assign_cell_index_kernel(
    scalar_t* coords,
    scalar_t* box,
    int ncells,
    int natoms,
    scalar_t* f_coords,
    int64_t* cell_indices
)
{
    __shared__ scalar_t box_inv[9];
    if (threadIdx.x == 0) {
        invert_box_3x3<scalar_t>(box, box_inv);
    }
    __syncthreads();

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= natoms) {
        return;
    }

    scalar_t rx = coords[index * 3];
    scalar_t ry = coords[index * 3 + 1];
    scalar_t rz = coords[index * 3 + 2];

    scalar_t fx = box_inv[0]*rx + box_inv[3]*ry + box_inv[6]*rz; fx -= floor_(fx);
    scalar_t fy = box_inv[1]*rx + box_inv[4]*ry + box_inv[7]*rz; fy -= floor_(fy);
    scalar_t fz = box_inv[2]*rx + box_inv[5]*ry + box_inv[8]*rz; fz -= floor_(fz);

    int cx = min(static_cast<int>(fx * ncells), ncells - 1);
    int cy = min(static_cast<int>(fy * ncells), ncells - 1);
    int cz = min(static_cast<int>(fz * ncells), ncells - 1);

    f_coords[index*3]   = fx;
    f_coords[index*3+1] = fy;
    f_coords[index*3+2] = fz;

    cell_indices[index] = cx * ncells * ncells + cy * ncells + cz;
}


// Step 3: Compute bounding box (center + radius) for each 32-atom cluster
// in fractional coordinate space.
template <typename scalar_t>
__global__ void compute_bounding_box_kernel(
    scalar_t* f_coords_sorted,
    int32_t natoms,
    scalar_t* cluster_centers,
    scalar_t* cluster_sizes
)
{
    int32_t totalWarps = blockDim.x * gridDim.x / WARP_SIZE;
    int32_t warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int32_t idxInWarp = threadIdx.x & (WARP_SIZE - 1);

    int32_t nclusters = (natoms + WARP_SIZE - 1) / WARP_SIZE;
    for (int32_t pos = warpIdx; pos < nclusters; pos += totalWarps) {
        scalar_t crd[3] = {0.0, 0.0, 0.0};

        int32_t i = pos * WARP_SIZE + idxInWarp;
        if (i < natoms) {
            crd[0] = f_coords_sorted[i*3];
            crd[1] = f_coords_sorted[i*3+1];
            crd[2] = f_coords_sorted[i*3+2];
        }

        scalar_t anchor[3];
        anchor[0] = __shfl_sync(0xFFFFFFFFu, crd[0], 0);
        anchor[1] = __shfl_sync(0xFFFFFFFFu, crd[1], 0);
        anchor[2] = __shfl_sync(0xFFFFFFFFu, crd[2], 0);

        diff_vec3(crd, anchor, crd);
        crd[0] -= round_(crd[0]);
        crd[1] -= round_(crd[1]);
        crd[2] -= round_(crd[2]);

        scalar_t mincrd[3] = {1.0, 1.0, 1.0};
        scalar_t maxcrd[3] = {-1.0, -1.0, -1.0};

        if (i < natoms) {
            mincrd[0] = maxcrd[0] = crd[0];
            mincrd[1] = maxcrd[1] = crd[1];
            mincrd[2] = maxcrd[2] = crd[2];
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            mincrd[0] = min_(mincrd[0], __shfl_down_sync(0xFFFFFFFFu, mincrd[0], offset));
            mincrd[1] = min_(mincrd[1], __shfl_down_sync(0xFFFFFFFFu, mincrd[1], offset));
            mincrd[2] = min_(mincrd[2], __shfl_down_sync(0xFFFFFFFFu, mincrd[2], offset));
            maxcrd[0] = max_(maxcrd[0], __shfl_down_sync(0xFFFFFFFFu, maxcrd[0], offset));
            maxcrd[1] = max_(maxcrd[1], __shfl_down_sync(0xFFFFFFFFu, maxcrd[1], offset));
            maxcrd[2] = max_(maxcrd[2], __shfl_down_sync(0xFFFFFFFFu, maxcrd[2], offset));
        }
        if (idxInWarp == 0) {
            cluster_centers[pos*3]   = (mincrd[0] + maxcrd[0]) / 2 + anchor[0];
            cluster_centers[pos*3+1] = (mincrd[1] + maxcrd[1]) / 2 + anchor[1];
            cluster_centers[pos*3+2] = (mincrd[2] + maxcrd[2]) / 2 + anchor[2];
            scalar_t bbx = (maxcrd[0] - mincrd[0]) / 2;
            scalar_t bby = (maxcrd[1] - mincrd[1]) / 2;
            scalar_t bbz = (maxcrd[2] - mincrd[2]) / 2;
            cluster_sizes[pos] = norm3d_(bbx, bby, bbz);
        }
    }
}


// Step 4: Find pairs of clusters whose bounding spheres overlap within
// the fractional-space cutoff.
template <typename scalar_t>
__global__ void find_interacting_clusters_kernel(
    scalar_t* cluster_centers,
    scalar_t* cluster_sizes,
    int nclusters,
    scalar_t* box,
    scalar_t cutoff,
    int32_t* interacting_clusters,
    int32_t* num_interacting_clusters,
    int64_t max_interacting_clusters
)
{
    __shared__ scalar_t s_f_cutoff;
    if (threadIdx.x == 0) {
        scalar_t a = norm3d_(box[0], box[1], box[2]);
        scalar_t b = norm3d_(box[3], box[4], box[5]);
        scalar_t c = norm3d_(box[6], box[7], box[8]);
        s_f_cutoff = cutoff / min_(a, min_(b, c));
    }
    __syncthreads();

    scalar_t f_cutoff = s_f_cutoff;
    int idxInWarp = threadIdx.x & (WARP_SIZE - 1);

    int64_t nclusters_int64 = (int64_t)nclusters;
    int64_t maxv = nclusters_int64 * (nclusters_int64 + 1) / 2;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t maxv_padded = ((maxv + stride - 1) / stride) * stride;

    for (int64_t index = threadIdx.x + blockIdx.x * blockDim.x;
         index < maxv_padded;
         index += stride) {

        bool include = false;
        int64_t ci = 0, cj = 0;

        if (index < maxv) {
            ci = (int64_t)floor((sqrt(((double)index) * 8 + 1) - 1) / 2);
            cj = index - (ci * (ci + 1)) / 2;

            if (ci == cj) {
                include = true;
            } else {
                scalar_t dr[3];
                diff_vec3(&cluster_centers[ci*3], &cluster_centers[cj*3], dr);
                dr[0] -= round_(dr[0]);
                dr[1] -= round_(dr[1]);
                dr[2] -= round_(dr[2]);
                scalar_t dist = norm3d_(dr[0], dr[1], dr[2]);
                scalar_t cri = cluster_sizes[ci];
                scalar_t crj = cluster_sizes[cj];
                include = (dist <= cri + crj + f_cutoff);
            }
        }

        int rank, count;
        count_true_values_in_warp(include, rank, count, idxInWarp);
        if (count > 0) {
            int start = 0;
            if (idxInWarp == 0) {
                start = atomicAdd(num_interacting_clusters, count);
            }
            start = __shfl_sync(0xFFFFFFFFu, start, 0);
            int start_idx = start + rank;
            if (include && start_idx < max_interacting_clusters) {
                interacting_clusters[start_idx * 2] = (int32_t)ci;
                interacting_clusters[start_idx * 2 + 1] = (int32_t)cj;
            }
        }
    }
}


// Step 5: For each interacting cluster pair, compute pairwise distances
// using warp shuffles (32x32) and emit valid neighbor pairs.
template <typename scalar_t, int BUFFER_SIZE, int BLOCK_SIZE>
__global__ void build_neighbor_list_cell_list_kernel(
    scalar_t* f_coords_sorted,
    scalar_t* g_box,
    scalar_t cutoff,
    int64_t* sorted_atom_indices,
    int32_t* interacting_blocks,
    const int32_t* num_interacting_ptr,
    int64_t natoms,
    int64_t max_npairs,
    int64_t* pairs,
    int32_t* npairs,
    const int64_t* excl_row_ptr,
    const int64_t* excl_col_indices,
    bool include_self
)
{
    __shared__ scalar_t box[9];
    __shared__ int32_t s_num_interacting;
    if (threadIdx.x < 9) {
        box[threadIdx.x] = g_box[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        s_num_interacting = *num_interacting_ptr;
    }
    __syncthreads();

    int32_t num_interacting = s_num_interacting;
    int32_t warpIdxInBlock = threadIdx.x / WARP_SIZE;
    int32_t idxInWarp = threadIdx.x & (WARP_SIZE - 1);
    int32_t totalWarps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;
    int32_t warpIdx = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpIdxInBlock;

    __shared__ int64_t pairs_buffer[BUFFER_SIZE * 2 * (BLOCK_SIZE / WARP_SIZE)];
    int64_t* my_buffer = pairs_buffer + warpIdxInBlock * BUFFER_SIZE * 2;
    int curr_buffer_size = 0;

    for (int32_t blk = warpIdx; blk < num_interacting; blk += totalWarps) {
        int32_t cx = interacting_blocks[blk * 2];
        int32_t cy = interacting_blocks[blk * 2 + 1];

        int64_t idx_i = (int64_t)cx * WARP_SIZE + idxInWarp;
        scalar_t fcrd_i[3] = {0.0, 0.0, 0.0};
        int64_t atom_i = -1;
        if (idx_i < natoms) {
            fcrd_i[0] = f_coords_sorted[idx_i*3];
            fcrd_i[1] = f_coords_sorted[idx_i*3+1];
            fcrd_i[2] = f_coords_sorted[idx_i*3+2];
            atom_i = sorted_atom_indices[idx_i];
        }

        if (cx == cy) {
            // Diagonal block: compare atoms within the same cluster.
            for (int32_t srcLane = 0; srcLane < WARP_SIZE; ++srcLane) {
                scalar_t fcrd_src[3];
                fcrd_src[0] = __shfl_sync(0xFFFFFFFFu, fcrd_i[0], srcLane);
                fcrd_src[1] = __shfl_sync(0xFFFFFFFFu, fcrd_i[1], srcLane);
                fcrd_src[2] = __shfl_sync(0xFFFFFFFFu, fcrd_i[2], srcLane);
                int64_t atom_src = __shfl_sync(0xFFFFFFFFu, atom_i, srcLane);

                bool valid = false;
                if (atom_i >= 0 && atom_src >= 0) {
                    if (atom_i == atom_src) {
                        valid = include_self;
                    } else if (atom_i > atom_src) {
                        scalar_t dr = dist_from_fractional_coords(fcrd_i, fcrd_src, box);
                        if (dr <= cutoff && !is_excluded_csr(atom_i, atom_src, excl_row_ptr, excl_col_indices)) {
                            valid = true;
                        }
                    }
                }

                int rank, count;
                count_true_values_in_warp(valid, rank, count, idxInWarp);
                if (count > 0) {
                    int buf_idx = curr_buffer_size + rank;
                    if (valid) {
                        int64_t a = atom_i, b = atom_src;
                        if (a == b) { /* self-pair */ }
                        else if (a < b) { int64_t t = a; a = b; b = t; }
                        my_buffer[buf_idx * 2] = a;
                        my_buffer[buf_idx * 2 + 1] = b;
                    }
                    curr_buffer_size += count;
                    flush_warp_buffer<int64_t, int64_t,2, BUFFER_SIZE, WARP_SIZE>(
                        my_buffer, curr_buffer_size, pairs, npairs, max_npairs, idxInWarp, false);
                }
            }
        } else {
            // Off-diagonal block: compare atoms from cluster cx vs cluster cy.
            int64_t idx_j = (int64_t)cy * WARP_SIZE + idxInWarp;
            scalar_t fcrd_j[3] = {0.0, 0.0, 0.0};
            int64_t atom_j = -1;
            if (idx_j < natoms) {
                fcrd_j[0] = f_coords_sorted[idx_j*3];
                fcrd_j[1] = f_coords_sorted[idx_j*3+1];
                fcrd_j[2] = f_coords_sorted[idx_j*3+2];
                atom_j = sorted_atom_indices[idx_j];
            }

            for (int32_t srcLane = 0; srcLane < WARP_SIZE; ++srcLane) {
                scalar_t fcrd_src[3];
                fcrd_src[0] = __shfl_sync(0xFFFFFFFFu, fcrd_i[0], srcLane);
                fcrd_src[1] = __shfl_sync(0xFFFFFFFFu, fcrd_i[1], srcLane);
                fcrd_src[2] = __shfl_sync(0xFFFFFFFFu, fcrd_i[2], srcLane);
                int64_t atom_src = __shfl_sync(0xFFFFFFFFu, atom_i, srcLane);

                bool valid = false;
                if (atom_src >= 0 && atom_j >= 0) {
                    scalar_t dr = dist_from_fractional_coords(fcrd_j, fcrd_src, box);
                    if (dr <= cutoff && !is_excluded_csr(atom_j, atom_src, excl_row_ptr, excl_col_indices)) {
                        valid = true;
                    }
                }

                int rank, count;
                count_true_values_in_warp(valid, rank, count, idxInWarp);
                if (count > 0) {
                    int buf_idx = curr_buffer_size + rank;
                    if (valid) {
                        int64_t a = atom_src, b = atom_j;
                        if (a < b) { int64_t t = a; a = b; b = t; }
                        my_buffer[buf_idx * 2] = a;
                        my_buffer[buf_idx * 2 + 1] = b;
                    }
                    curr_buffer_size += count;
                    flush_warp_buffer<int64_t, int64_t, 2, BUFFER_SIZE, WARP_SIZE>(
                        my_buffer, curr_buffer_size, pairs, npairs, max_npairs, idxInWarp, false);
                }
            }
        }
    }

    flush_warp_buffer<int64_t, int64_t, 2, BUFFER_SIZE, WARP_SIZE>(
        my_buffer, curr_buffer_size, pairs, npairs, max_npairs, idxInWarp, true);
}


// Forward declaration
std::tuple<at::Tensor, at::Tensor> build_neighbor_list_cell_list_out_cuda(
    const at::Tensor& coords,
    const c10::optional<at::Tensor> box,
    const at::Scalar& cutoff,
    at::Tensor pairs,
    c10::optional<at::Tensor> excl_row_ptr,
    c10::optional<at::Tensor> excl_col_indices,
    bool include_self
);


std::tuple<at::Tensor, at::Tensor> build_neighbor_list_cell_list_cuda(
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
    return build_neighbor_list_cell_list_out_cuda(
        coords, box, cutoff, pairs, excl_row_ptr, excl_col_indices, include_self
    );
}


std::tuple<at::Tensor, at::Tensor> build_neighbor_list_cell_list_out_cuda(
    const at::Tensor& coords,
    const c10::optional<at::Tensor> box,
    const at::Scalar& cutoff,
    at::Tensor pairs,
    c10::optional<at::Tensor> excl_row_ptr,
    c10::optional<at::Tensor> excl_col_indices,
    bool include_self
)
{
    TORCH_CHECK(box.has_value() && box.value().defined(),
                "Cell-list neighbor list requires a periodic box");

    int64_t natoms = coords.size(0);
    int64_t max_npairs_ = pairs.size(0);

    pairs.fill_(-1);

    const int64_t* row_ptr = (excl_row_ptr.has_value() && excl_row_ptr.value().defined())
        ? excl_row_ptr.value().data_ptr<int64_t>() : nullptr;
    const int64_t* col_ind = (excl_col_indices.has_value() && excl_col_indices.value().defined())
        ? excl_col_indices.value().data_ptr<int64_t>() : nullptr;

    int32_t ncells = std::max(1, (int32_t)std::cbrt(static_cast<double>(natoms / WARP_SIZE)));
    int32_t nclusters = (natoms + WARP_SIZE - 1) / WARP_SIZE;
    int64_t max_interacting = std::min<int64_t>(max_npairs_, (int64_t)nclusters * (nclusters + 1) / 2);
    // (int64_t)nclusters * (nclusters + 1) / 2;

    at::Tensor f_coords = at::empty_like(coords);
    at::Tensor cell_indices = at::empty({natoms}, coords.options().dtype(at::kLong));
    at::Tensor npairs = at::zeros({1}, coords.options().dtype(at::kInt));

    at::Tensor cluster_centers = at::empty({nclusters, 3}, coords.options());
    at::Tensor cluster_sizes = at::empty({nclusters}, coords.options());

    at::Tensor num_interacting_blocks = at::zeros({1}, coords.options().dtype(at::kInt));
    at::Tensor interacting_blocks = at::full({max_interacting, 2}, -1, coords.options().dtype(at::kInt));

    auto stream = at::cuda::getCurrentCUDAStream();
    auto props = at::cuda::getCurrentDeviceProperties();

    // Step 1: Compute fractional coords and assign cell indices
    constexpr int STEP1_BLOCK_SIZE = 256;
    int STEP1_GRID_SIZE = (natoms + STEP1_BLOCK_SIZE - 1) / STEP1_BLOCK_SIZE;
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "assign_cell_index", ([&] {
        assign_cell_index_kernel<scalar_t><<<STEP1_GRID_SIZE, STEP1_BLOCK_SIZE, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            box.value().data_ptr<scalar_t>(),
            ncells,
            (int)natoms,
            f_coords.data_ptr<scalar_t>(),
            cell_indices.data_ptr<int64_t>()
        );
    }));

    // Step 2: Sort atoms by cell index for spatial coherence
    at::Tensor sorted_cell_indices, sort_perm;
    std::tie(sorted_cell_indices, sort_perm) = at::sort(cell_indices);
    at::Tensor f_coords_sorted = f_coords.index_select(0, sort_perm);
    at::Tensor sorted_atom_indices = sort_perm;

    // Step 3: Compute bounding boxes for each 32-atom cluster
    constexpr int STEP3_BLOCK_SIZE = 256;
    int STEP3_GRID_SIZE = std::min(
        (int)((nclusters * WARP_SIZE + STEP3_BLOCK_SIZE - 1) / STEP3_BLOCK_SIZE),
        props->maxBlocksPerMultiProcessor * props->multiProcessorCount
    );
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_bounding_box", ([&] {
        compute_bounding_box_kernel<scalar_t><<<STEP3_GRID_SIZE, STEP3_BLOCK_SIZE, 0, stream>>>(
            f_coords_sorted.data_ptr<scalar_t>(),
            (int32_t)natoms,
            cluster_centers.data_ptr<scalar_t>(),
            cluster_sizes.data_ptr<scalar_t>()
        );
    }));

    // Step 4: Find interacting cluster pairs
    int32_t STEP4_BLOCK_SIZE_val = NBLIST_CLIST_BLOCK_SIZE;
    int32_t STEP4_GRID_SIZE = std::min(
        (int)((max_interacting + STEP4_BLOCK_SIZE_val - 1) / STEP4_BLOCK_SIZE_val),
        props->maxBlocksPerMultiProcessor * props->multiProcessorCount
    );
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "find_interacting_clusters", ([&] {
        scalar_t cutoff_val = static_cast<scalar_t>(cutoff.toDouble());
        find_interacting_clusters_kernel<scalar_t><<<STEP4_GRID_SIZE, STEP4_BLOCK_SIZE_val, 0, stream>>>(
            cluster_centers.data_ptr<scalar_t>(),
            cluster_sizes.data_ptr<scalar_t>(),
            nclusters,
            box.value().data_ptr<scalar_t>(),
            cutoff_val,
            interacting_blocks.data_ptr<int32_t>(),
            num_interacting_blocks.data_ptr<int32_t>(),
            max_interacting
        );
    }));

    // Step 5: Build atom-pair neighbor list from interacting cluster pairs.
    // Use max_interacting (host-computable) for the grid size to avoid a
    // device-to-host sync that would break CUDA graph capture.  The actual
    // count is read from num_interacting_blocks inside the kernel.
    if (max_interacting > 0) {
        int64_t warps_per_block = NBLIST_CLIST_BLOCK_SIZE / WARP_SIZE;
        int64_t needed_blocks = (max_interacting + warps_per_block - 1) / warps_per_block;
        int32_t STEP5_GRID_SIZE = (int32_t)std::min(
            needed_blocks,
            (int64_t)(props->maxBlocksPerMultiProcessor * props->multiProcessorCount)
        );
        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "build_neighbor_list_cell_list", ([&] {
            scalar_t cutoff_val = static_cast<scalar_t>(cutoff.toDouble());
            build_neighbor_list_cell_list_kernel<scalar_t, NBLIST_CLIST_BUFFER_SIZE, NBLIST_CLIST_BLOCK_SIZE>
                <<<STEP5_GRID_SIZE, NBLIST_CLIST_BLOCK_SIZE, 0, stream>>>(
                f_coords_sorted.data_ptr<scalar_t>(),
                box.value().data_ptr<scalar_t>(),
                cutoff_val,
                sorted_atom_indices.data_ptr<int64_t>(),
                interacting_blocks.data_ptr<int32_t>(),
                num_interacting_blocks.data_ptr<int32_t>(),
                natoms,
                max_npairs_,
                pairs.data_ptr<int64_t>(),
                npairs.data_ptr<int32_t>(),
                row_ptr,
                col_ind,
                include_self
            );
        }));
    }

    return std::make_tuple(pairs, npairs);
}


TORCH_LIBRARY_IMPL(torchff, CUDA, m) {
    m.impl("build_neighbor_list_cell_list", build_neighbor_list_cell_list_cuda);
    m.impl("build_neighbor_list_cell_list_out", build_neighbor_list_cell_list_out_cuda);
}
