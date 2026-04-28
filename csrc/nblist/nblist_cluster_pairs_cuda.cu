#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"
#include "nblist/exclusions.cuh"

#define BLOCK_DIM 256
#define BUFFER_SIZE 256


template <typename scalar_t> 
__global__ void assign_cell_index_kernel(
    scalar_t* coords,
    scalar_t* box_inv,
    scalar_t fcrx, scalar_t fcry, scalar_t fcrz, // cell size in fractional coords
    int32_t ncx, int32_t ncy, int32_t ncz, // number of cells in one dimension
    int32_t natoms,
    int32_t* cell_indices,
    scalar_t* f_coords
)
{
    __shared__ scalar_t s_box_inv[9];
    if ( threadIdx.x < 9 ) {
        s_box_inv[threadIdx.x] = box_inv[threadIdx.x];
    }
    __syncthreads();

    int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index >= natoms ) {
        return;
    }

    scalar_t crd[3];
    crd[0] = coords[index * 3];
    crd[1] = coords[index * 3 + 1];
    crd[2] = coords[index * 3 + 2];

    // compute fractional coords
    // TODO: need to change it to adapt row-major box
    scalar_t fx = dot_vec3(s_box_inv, crd);
    scalar_t fy = dot_vec3(s_box_inv+3, crd);
    scalar_t fz = dot_vec3(s_box_inv+6, crd);

    // shift to [0, 1]
    fx -= floor_(fx);
    fy -= floor_(fy);
    fz -= floor_(fz);

    // compute cell index
    int32_t cx = (int32_t)(fx / fcrx) % ncx;
    int32_t cy = (int32_t)(fy / fcry) % ncy;
    int32_t cz = (int32_t)(fz / fcrz) % ncz;
    int32_t c = (cx * ncy + cy) * ncz + cz;

    cell_indices[index] = c;

    f_coords[index*3]   = fx;
    f_coords[index*3+1] = fy;
    f_coords[index*3+2] = fz;
}


template <typename scalar_t>
__global__ void compute_bounding_box_kernel(
    scalar_t* f_coords_sorted,
    int32_t natoms,
    scalar_t* cluster_centers,
    scalar_t* cluster_bounding_boxes
)
{
    // Each warp process a cluster
    int32_t totalWarps = blockDim.x * gridDim.x / warpSize;
    int32_t warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int32_t idxInWarp = threadIdx.x & (warpSize - 1);

    int32_t nclusters = (natoms + warpSize - 1) / warpSize;
    for (int32_t pos = warpIdx; pos < nclusters; pos += totalWarps) {
        scalar_t crd[3] = {0.0, 0.0, 0.0};

        int32_t i = pos*warpSize+idxInWarp;
        if ( i < natoms ) {
            crd[0] = f_coords_sorted[i*3];
            crd[1] = f_coords_sorted[i*3+1];
            crd[2] = f_coords_sorted[i*3+2];
        }

        // choose the first atom as a anchor
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
        
        if ( i < natoms ) {
            mincrd[0] = maxcrd[0] = crd[0]; 
            mincrd[1] = maxcrd[1] = crd[1]; 
            mincrd[2] = maxcrd[2] = crd[2];
        }

        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            mincrd[0] = min_(mincrd[0], __shfl_down_sync(0xFFFFFFFFu, mincrd[0], offset));
            mincrd[1] = min_(mincrd[1], __shfl_down_sync(0xFFFFFFFFu, mincrd[1], offset));
            mincrd[2] = min_(mincrd[2], __shfl_down_sync(0xFFFFFFFFu, mincrd[2], offset));
            maxcrd[0] = max_(maxcrd[0], __shfl_down_sync(0xFFFFFFFFu, maxcrd[0], offset));
            maxcrd[1] = max_(maxcrd[1], __shfl_down_sync(0xFFFFFFFFu, maxcrd[1], offset));
            maxcrd[2] = max_(maxcrd[2], __shfl_down_sync(0xFFFFFFFFu, maxcrd[2], offset));
        }
        if ( idxInWarp == 0 ) {
            cluster_centers[pos*3]   = (mincrd[0] + maxcrd[0]) / 2 + anchor[0];
            cluster_centers[pos*3+1] = (mincrd[1] + maxcrd[1]) / 2 + anchor[1];
            cluster_centers[pos*3+2] = (mincrd[2] + maxcrd[2]) / 2 + anchor[2];
            cluster_bounding_boxes[pos*3]   = (maxcrd[0] - mincrd[0]) / 2;
            cluster_bounding_boxes[pos*3+1] = (maxcrd[1] - mincrd[1]) / 2;
            cluster_bounding_boxes[pos*3+2] = (maxcrd[2] - mincrd[2]) / 2;
        }
    }
}


template <typename scalar_t>
__global__ void find_interacting_clusters_kernel(
    scalar_t* f_coords_sorted,
    int32_t* sorted_atom_indices,
    int32_t natoms,
    scalar_t frx, scalar_t fry, scalar_t frz,
    scalar_t* cluster_centers,
    scalar_t* cluster_bounding_boxes,
    int32_t* cluster_exclusions_crow_indices,
    int32_t* cluster_exclusions_col_indices,
    int32_t max_exclusions_per_cluster,
    int32_t* interacting_clusters,
    int32_t* interacting_atoms,
    int32_t* num_interacting_clusters
)
{
    int32_t totalWarps = blockDim.x * gridDim.x / warpSize;
    int32_t warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int32_t idxInWarp = threadIdx.x & (warpSize - 1);
    uint32_t warpMask = (1u << idxInWarp) - 1;

    int32_t warpIdxInBlock = threadIdx.x / warpSize;
    __shared__ int32_t interacting_atoms_buffer_block[BUFFER_SIZE*BLOCK_DIM/32];
    extern __shared__ int32_t shared_excl[];

    int32_t nclusters = (natoms + warpSize - 1) /  warpSize;
    int32_t* interacting_atoms_buffer = interacting_atoms_buffer_block + warpIdxInBlock * BUFFER_SIZE;
    int32_t* x_excl = shared_excl + warpIdxInBlock * max_exclusions_per_cluster;

    // Cutoff distance in fractional coordinates
    scalar_t fr = max_(max_(frx, fry), frz);

    // Each warp process a cluster X
    for (int32_t cx = warpIdx; cx < nclusters; cx += totalWarps) {
        // Center of cluster X
        scalar_t ctrx[3];
        ctrx[0] = cluster_centers[cx*3]; ctrx[1] = cluster_centers[cx*3+1]; ctrx[2] = cluster_centers[cx*3+2];
        // Bounding box of cluster X
        scalar_t bbx[3];
        bbx[0] = cluster_bounding_boxes[cx*3]; bbx[1] = cluster_bounding_boxes[cx*3+1]; bbx[2] = cluster_bounding_boxes[cx*3+2];
        scalar_t brx = sqrt_(bbx[0]*bbx[0]+bbx[1]*bbx[1]+bbx[2]*bbx[2]);

        // if (idxInWarp==0) {
        //     printf("Cluster %d bounding box: %f\n", cx, brx);
        // }

        // Number of interacting atoms in the buffer
        int32_t num_atoms_in_buffer = 0;

        // Load exclusions into shared memory
        int32_t excl_start = cluster_exclusions_crow_indices[cx];
        int32_t excl_end = cluster_exclusions_crow_indices[cx+1];
        int32_t nexcl = excl_end - excl_start;
        for (int32_t n = idxInWarp; n < nexcl; n += warpSize) {
            x_excl[n] = cluster_exclusions_col_indices[excl_start+n];
        }

        // Each thread in this warp process another cluster (Y)
        for (int32_t cy_start = cx+1; cy_start < nclusters; cy_start += warpSize) {
            int32_t cy = cy_start + idxInWarp;
            scalar_t ctry[3];
            scalar_t bby[3];
            bool include = false;

            // X and Y are not excluded, check if their bounding box overlap
            if ( cy < nclusters && (!in_list(cy, x_excl, nexcl, -1)) ) {
                ctry[0] = cluster_centers[cy*3]; ctry[1] = cluster_centers[cy*3+1]; ctry[2] = cluster_centers[cy*3+2];
                bby[0] = cluster_bounding_boxes[cy*3]; bby[1] = cluster_bounding_boxes[cy*3+1]; bby[2] = cluster_bounding_boxes[cy*3+2];
                scalar_t bry = sqrt_(bby[0]*bby[0]+bby[1]*bby[1]+bby[2]*bby[2]);
                scalar_t dr[3];
                diff_vec3(ctrx, ctry, dr);
                dr[0] -= round(dr[0]); dr[1] -= round(dr[1]); dr[2] -= round(dr[2]);
                // include = include || ( abs_(dr[0]) <= (bbx[0] + bby[0] + frx) && abs_(dr[1]) <= (bbx[1] + bby[1] + fry) && abs_(dr[2]) <= (bbx[2] + bby[2] + frz) );
                include = include || ((dr[0]*dr[0]+dr[1]*dr[1]+dr[2]*dr[2]) <= (brx+bry+fr)*(brx+bry+fr));
                // if (!include) {
                //     printf("#Excluded between %d and %d\n", cx, cy);
                // }
            }

            // Loop over all interacting cluster Ys and check their atoms
            uint32_t includeBallot = __ballot_sync(0xFFFFFFFFu, include);
            while ( includeBallot > 0 ) {
                uint32_t k = __ffs(includeBallot)-1;   // This fetch the index of the first non-zero bit
                includeBallot &= ~(1u << k);   // This set the k-th bit to zero
                cy = cy_start + k;    
                
                // if (idxInWarp==0) {
                //     printf("Cluster %d and %d are not excl but interacting\n", cx, cy);
                // }

                // Check if this atom in cluster Y interacts with cluster X
                int32_t atom = cy*warpSize+idxInWarp;
                scalar_t crd[3];
                bool interact = false;
                if ( atom < natoms ){
                    crd[0] = f_coords_sorted[atom*3];
                    crd[1] = f_coords_sorted[atom*3+1];
                    crd[2] = f_coords_sorted[atom*3+2];
                    scalar_t dr[3];
                    diff_vec3(crd, ctrx, dr);
                    dr[0] -= round_(dr[0]); dr[1] -= round_(dr[1]); dr[2] -= round_(dr[2]);
                    // interact = abs_(dr[0]) <= (bbx[0] + frx) && abs_(dr[1]) <= (bbx[1] + fry) && abs_(dr[2]) <= (bbx[2] + frz);
                    interact = ( (dr[0]*dr[0]+dr[1]*dr[1]+dr[2]*dr[2]) <= (brx+fr)*(brx+fr) );
                }

                // if ( sorted_atom_indices[atom] == 137 && cx == 1) {
                //     printf("$$CHECK---Cluster 1 interact with atom 137: %s\n", interact?"true":"false");
                //     printf("Cluster 1 bounding box in frac coords: %f,%f,%f\n", bbx[0], bbx[1], bbx[2]);
                //     printf("Cluster 1 center in frac coords: %f,%f,%f\n", ctrx[0], ctrx[1], ctrx[2]);
                //     printf("Atom 137 frac coords: %f,%f,%f\n", crd[0], crd[1], crd[2]);
                // }
               
                // Gather intearcting atoms and dump them in the buffer
                // We actively flush the buffer to ensure there are always enough available slots
                uint32_t interactBallot = __ballot_sync(0xFFFFFFFFu, interact);                
                if (interact) {
                    int32_t p = num_atoms_in_buffer+__popc(interactBallot&warpMask);
                    interacting_atoms_buffer[p] = sorted_atom_indices[atom];
                    // printf("Cluster %d interacts with %d. write to %d+%d\n", cx, sorted_atom_indices[atom],num_atoms_in_buffer,__popc(interactBallot&warpMask) );
                }
                num_atoms_in_buffer += __popc(interactBallot);
                
                // Flush the buffer when the number of remaining slots in the buffer are not enough to fill another entire warp (32)
                if (num_atoms_in_buffer > (BUFFER_SIZE - warpSize)) {
                    int32_t num = num_atoms_in_buffer / warpSize;
                    int32_t start_index;
                    if (idxInWarp == 0) {
                        start_index = atomicAdd(num_interacting_clusters, num);
                    }
                    start_index = __shfl_sync(0xFFFFFFFFu, start_index, 0);
                    for (int32_t c = 0; c < num; ++c) {
                        interacting_clusters[start_index+c] = cx;
                        interacting_atoms[(start_index+c)*warpSize+idxInWarp] = interacting_atoms_buffer[c*warpSize+idxInWarp];
                    }
                    // Remaining some atoms, move them to the front and wait for the next flush
                    if (idxInWarp+num*warpSize < num_atoms_in_buffer) {
                        interacting_atoms_buffer[idxInWarp] = interacting_atoms_buffer[idxInWarp+num*warpSize];
                    }
                    num_atoms_in_buffer -= num * warpSize;
                }
            }
        }

        // Forcibly flush the buffer
        if ( num_atoms_in_buffer > 0 ) {
            int32_t num = (num_atoms_in_buffer + warpSize - 1) / warpSize;
            int32_t start_index;
            if (idxInWarp == 0) {
                start_index = atomicAdd(num_interacting_clusters, num);
            }
            start_index = __shfl_sync(0xFFFFFFFFu, start_index, 0);
            for (int32_t c = 0; c < num-1; ++c) {
                interacting_clusters[start_index+c] = cx;
                interacting_atoms[(start_index+c)*warpSize+idxInWarp] = interacting_atoms_buffer[c*warpSize+idxInWarp];
            }
            interacting_clusters[start_index+num-1] = cx;
            if (idxInWarp+(num-1)*warpSize < num_atoms_in_buffer) {
                interacting_atoms[(start_index+num-1)*warpSize+idxInWarp] = interacting_atoms_buffer[(num-1)*warpSize+idxInWarp];
            }
            else {
                interacting_atoms[(start_index+num-1)*warpSize+idxInWarp] = -1; // padding the unfilled with -1
            }
        }
        num_atoms_in_buffer = 0;
   }

}


__global__ void set_bitmask_exclusions_kernel(
    int32_t* sorted_atom_indices,
    int32_t natoms,
    // cluster exclusions in COO format
    int32_t* cluster_exclusions,
    int32_t num_cluster_exclusions,
    // atom exclusions in CSR format
    int32_t* atom_exclusions_crows,
    int32_t* atom_exclusions_cols,
    uint32_t* bitmask_exclusions
)
{
    int32_t totalWarps = blockDim.x * gridDim.x / warpSize;
    int32_t warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int32_t idxInWarp = threadIdx.x & (warpSize - 1);

    for (int32_t pos = warpIdx; pos < num_cluster_exclusions; pos += totalWarps) {
        int32_t x = cluster_exclusions[pos];
        int32_t y = cluster_exclusions[pos+num_cluster_exclusions];

        if ( x < 0 || y < 0 ) {
            continue;
        }

        int32_t index_i = x * warpSize + idxInWarp;
        int32_t index_j = y * warpSize + idxInWarp;
        int32_t i = -1;
        int32_t j = -1;
        int32_t j_shfl = -1;

        int32_t* i_excl = nullptr;
        int32_t nexcl = 0;
        if ( index_i < natoms ) {
            i = sorted_atom_indices[index_i];
            i_excl = atom_exclusions_cols + atom_exclusions_crows[i];
            nexcl = atom_exclusions_crows[i+1] - atom_exclusions_crows[i];
        }
        if ( index_j < natoms ) {
            j = sorted_atom_indices[index_j];
        }

        uint32_t excl = 0;
        for ( int srcLane = 0; srcLane < warpSize; ++srcLane ) {
            j_shfl = __shfl_sync(0xFFFFFFFFu, j, srcLane);
            if ( i == -1 || j_shfl == -1 || in_list(j_shfl, i_excl, nexcl, -1) ) {
                excl |= (1u << srcLane);
            }
        }

        // for the same cluster, count only i-j pair (not j-i) to avoid double counting
        if ( x == y ) {
            for ( int srcLane = 0; srcLane <= idxInWarp; ++srcLane ) {
                excl |= (1u << srcLane);
            }
        }
        bitmask_exclusions[pos * warpSize + idxInWarp] = excl;
    }
}


template <typename scalar_t>
__global__ void decode_cluster_pairs_kernel(
    scalar_t* coords,
    scalar_t* g_box,
    scalar_t* g_box_inv,
    scalar_t cutoff2,
    int32_t* sorted_atom_indices,
    int32_t natoms,
    int32_t* cluster_exclusions,
    int32_t num_cluster_exclusions,
    uint32_t* bitmask_exclusions,
    int32_t* interacting_clusters,
    int32_t* interacting_atoms,
    int32_t num_interacting_clusters,
    int32_t max_npairs,
    int32_t* pairs,
    int32_t* npairs
)
{
    __shared__ scalar_t box[9];
    __shared__ scalar_t box_inv[9];

    if (threadIdx.x < 9) {
        box[threadIdx.x] = g_box[threadIdx.x];
        box_inv[threadIdx.x] = g_box_inv[threadIdx.x];
    }

    __syncthreads();

    int32_t totalWarps = blockDim.x * gridDim.x / warpSize;
    int32_t warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int32_t idxInWarp = threadIdx.x & (warpSize - 1);

    // First loop: clusters with exclusions
    for (int32_t pos = warpIdx; pos < num_cluster_exclusions; pos += totalWarps) {
        int32_t x = cluster_exclusions[pos];
        int32_t y = cluster_exclusions[pos+num_cluster_exclusions];

        if ( x < 0 || y < 0 || x > y ) {
            continue;
        }

        // if (idxInWarp==0) {
        //     printf("Process cluster exclusion %d and %d\n", x, y);
        // }
        
        int32_t index_i = x * warpSize + idxInWarp;
        int32_t index_j = y * warpSize + idxInWarp;
        int32_t i = -1;
        int32_t j = -1;
        int32_t j_shfl = -1;

        scalar_t crd_i[3] = {0.0, 0.0, 0.0};
        scalar_t crd_j[3] = {0.0, 0.0, 0.0};
        scalar_t crd_j_shfl[3] = {0.0, 0.0, 0.0};

        if ( index_i < natoms ) {
            i = sorted_atom_indices[index_i];
            crd_i[0] = coords[i*3];
            crd_i[1] = coords[i*3+1];
            crd_i[2] = coords[i*3+2];
        }
        if ( index_j < natoms ) {
            j = sorted_atom_indices[index_j];
            crd_j[0] = coords[j*3];
            crd_j[1] = coords[j*3+1];
            crd_j[2] = coords[j*3+2];
        }

        uint32_t excl = bitmask_exclusions[pos * warpSize + idxInWarp];
        for (int32_t srcLane = 0; srcLane < warpSize; ++srcLane) {
            j_shfl = __shfl_sync(0xFFFFFFFFu, j, srcLane);
            crd_j_shfl[0] = __shfl_sync(0xFFFFFFFFu, crd_j[0], srcLane);
            crd_j_shfl[1] = __shfl_sync(0xFFFFFFFFu, crd_j[1], srcLane);
            crd_j_shfl[2] = __shfl_sync(0xFFFFFFFFu, crd_j[2], srcLane);
            
            scalar_t rij[3];
            diff_vec3(crd_i, crd_j_shfl, rij);
            apply_pbc_triclinic(rij, box, box_inv, rij);
            
            if ( i >= 0 && j_shfl >= 0 && !(excl & 0x1) && (rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]) <= cutoff2) {
                // TODO: this part might be buffered
                int32_t i_curr_pair = atomicAdd(npairs, 1) % max_npairs;
                pairs[i_curr_pair * 2] = i;
                pairs[i_curr_pair * 2 + 1] = j_shfl;
            }
            excl >>= 1;
        }
    }

    // Second loop: atoms without exclusions, read through a neighbor list
    for (int32_t pos = warpIdx; pos < num_interacting_clusters; pos += totalWarps) {
        int32_t x = interacting_clusters[pos];
        if ( x == -1 ) {
            continue;
        }
        int32_t index_i = x * warpSize + idxInWarp;

        int32_t i = -1;
        int32_t j = -1;
        int32_t j_shfl = -1;

        scalar_t crd_i[3] = {0.0, 0.0, 0.0};
        scalar_t crd_j[3] = {0.0, 0.0, 0.0};
        scalar_t crd_j_shfl[3] = {0.0, 0.0, 0.0};

        if ( index_i < natoms ) {
            i = sorted_atom_indices[index_i];
            crd_i[0] = coords[i*3];
            crd_i[1] = coords[i*3+1];
            crd_i[2] = coords[i*3+2];
        }

        j = interacting_atoms[pos*warpSize+idxInWarp];
        if ( j >= 0 ) {
            crd_j[0] = coords[j*3];
            crd_j[1] = coords[j*3+1];
            crd_j[2] = coords[j*3+2];
        }

        // printf("Pos %d read cluster %d interact with atom %d\n", pos, x, j);

        for (int32_t srcLane = 0; srcLane < warpSize; ++srcLane) {
            j_shfl = __shfl_sync(0xFFFFFFFFu, j, srcLane);
            crd_j_shfl[0] = __shfl_sync(0xFFFFFFFFu, crd_j[0], srcLane);
            crd_j_shfl[1] = __shfl_sync(0xFFFFFFFFu, crd_j[1], srcLane);
            crd_j_shfl[2] = __shfl_sync(0xFFFFFFFFu, crd_j[2], srcLane);
            
            scalar_t rij[3];
            diff_vec3(crd_i, crd_j_shfl, rij);
            apply_pbc_triclinic(rij, box, box_inv, rij);
            
            if ( i < 0 || j_shfl < 0 || (rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]) > cutoff2 ) {
                continue;
            }

            // TODO: this part might be buffered
            int32_t i_curr_pair = atomicAdd(npairs, 1) % max_npairs;
            pairs[i_curr_pair * 2] = i;
            pairs[i_curr_pair * 2 + 1] = j_shfl;
        }
    }
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> build_cluster_pairs_cuda(
    at::Tensor& coords,
    at::Tensor& box,
    at::Scalar cutoff,
    at::Tensor& exclusions,
    at::Scalar cell_size,
    at::Scalar max_num_interacting_clusters
)
{
    at::Tensor box_inv = at::linalg_inv(box);
    int32_t natoms = coords.size(0);
    
    at::Tensor box_cpu = box.to(at::kCPU);
    at::Tensor box_len = at::linalg_norm(box_cpu, 2, 0);
    at::Tensor cell_size_tensor = at::tensor({cell_size.toDouble(), cell_size.toDouble(), cell_size.toDouble()/10}, box_len.options());
    at::Tensor f_cell_size = cell_size_tensor / box_len;
    at::Tensor nc = at::floor(box_len / cell_size_tensor).to(at::kInt);

    int32_t ncx = nc[0].item<int32_t>();
    int32_t ncy = nc[1].item<int32_t>();
    int32_t ncz = nc[2].item<int32_t>();

    at::Tensor f_coords = at::empty_like(coords);
    at::Tensor cell_indices = at::empty({natoms}, coords.options().dtype(at::kInt));
    
    auto stream = at::cuda::getCurrentCUDAStream();

    // Step 1: Assign cell index for each atom
    int32_t block_dim = BLOCK_DIM;
    int32_t grid_dim = (natoms + block_dim - 1) / block_dim;
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "assign_cell_index", ([&] {
        scalar_t* fcr = f_cell_size.data_ptr<scalar_t>();
        scalar_t fcrx = fcr[0];
        scalar_t fcry = fcr[1];
        scalar_t fcrz = fcr[2];
        assign_cell_index_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            box_inv.data_ptr<scalar_t>(),
            fcrx, fcry, fcrz,
            ncx, ncy, ncz,
            natoms,
            cell_indices.data_ptr<int32_t>(),
            f_coords.data_ptr<scalar_t>()
        );
    }));

    // Step 2: Sort atoms according to cell indices
    at::Tensor sorted_cell_indices;
    at::Tensor sorted_atom_indices_long;
    std::tie(sorted_cell_indices, sorted_atom_indices_long) = at::sort(cell_indices);
    at::Tensor sorted_atom_indices = sorted_atom_indices_long.to(at::kInt);
    at::Tensor f_coords_sorted = f_coords.index_select(0, sorted_atom_indices);

    // Step 3: Compute bounding boxes
    int32_t num_clusters = (natoms + 31) / 32;
    // grid_dim = (num_clusters * (num_clusters + 1) / 2 + block_dim - 1) / block_dim;
    grid_dim = 4 * 108;
    at::Tensor cluster_centers = at::zeros({num_clusters, 3}, coords.options());
    at::Tensor cluster_bounding_boxes = at::zeros({num_clusters, 3}, coords.options());
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_bounding_box", ([&] {
        compute_bounding_box_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            f_coords_sorted.data_ptr<scalar_t>(),
            natoms,
            cluster_centers.data_ptr<scalar_t>(),
            cluster_bounding_boxes.data_ptr<scalar_t>()
        );
    }));

    // Step 4: Construct cluster pairs with exclusions & set bitmasks
    at::Tensor atom_exclusions_coo = at::sparse_coo_tensor(exclusions, at::zeros({exclusions.size(1)}, exclusions.options()), {natoms, natoms}).coalesce();
    at::Tensor atom_exclusions_csr = atom_exclusions_coo.to_sparse_csr();
    at::Tensor atom_exclusions_crows = atom_exclusions_csr.crow_indices().to(at::kInt);
    at::Tensor atom_exclusions_cols = atom_exclusions_csr.col_indices().to(at::kInt);
    // std::cout << "Atom exclusions set" << std::endl;

    at::Tensor cluster_indices = at::floor_divide(
        at::arange(natoms, coords.options().dtype(at::kLong)),
        32
    ).index_select(0, at::argsort(sorted_atom_indices));
    at::Tensor cluster_exclusions_indices = cluster_indices.index({exclusions});

    at::Tensor cluster_exclusions_coo = at::sparse_coo_tensor(
        cluster_exclusions_indices,
        at::ones({cluster_exclusions_indices.size(1)}, coords.options().dtype(at::kInt)),
        {num_clusters, num_clusters}
    ).coalesce();
    at::Tensor cluster_exclusions_csr = cluster_exclusions_coo.to_sparse_csr();
    at::Tensor cluster_exclusions_crow_indices = cluster_exclusions_csr.crow_indices().to(at::kInt);
    at::Tensor cluster_exclusions_col_indices = cluster_exclusions_csr.col_indices().to(at::kInt);
    at::Tensor cluster_exclusions = cluster_exclusions_coo.indices().to(at::kInt);
    // std::cout << "Cluster exclusions set" << std::endl;

    at::Tensor max_exclusions_per_cluster_tensor = at::max(cluster_exclusions_crow_indices.slice(0, 1, num_clusters+1) - cluster_exclusions_crow_indices.slice(0, 0, num_clusters));
    int32_t max_exclusions_per_cluster = max_exclusions_per_cluster_tensor.item().toInt();

    int32_t num_cluster_exclusions = cluster_exclusions.size(1);
    at::Tensor bitmask_exclusions = at::empty({num_cluster_exclusions, 32}, coords.options().dtype(at::kUInt32));
    set_bitmask_exclusions_kernel<<<grid_dim, block_dim, 0, stream>>>(
        sorted_atom_indices.data_ptr<int32_t>(),
        natoms,
        cluster_exclusions.data_ptr<int32_t>(),
        num_cluster_exclusions,
        atom_exclusions_crows.data_ptr<int32_t>(),
        atom_exclusions_cols.data_ptr<int32_t>(),
        bitmask_exclusions.data_ptr<uint32_t>()
    );
    // std::cout << "Bitmask exclusions set" << std::endl;
    
    // Step 5: Find interacting atoms
    int32_t max_num_interacting_clusters_ = num_clusters * (num_clusters + 1) / 2;
    if ( max_num_interacting_clusters.toInt() > 0 ) {
        max_num_interacting_clusters_ = std::min(max_num_interacting_clusters_, max_num_interacting_clusters.toInt());
    }
    at::Tensor num_interacting_clusters = at::zeros({1}, coords.options().dtype(at::kInt));
    at::Tensor interacting_clusters = at::full(
        {max_num_interacting_clusters_}, -1, coords.options().dtype(at::kInt)
    );
    at::Tensor interacting_atoms = at::full(
        {max_num_interacting_clusters_, 32}, -1, coords.options().dtype(at::kInt)
    );
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "find_interacting_clusters", ([&] {
        at::Tensor fr = cutoff / box_len;
        scalar_t frx = fr.data_ptr<scalar_t>()[0];
        scalar_t fry = fr.data_ptr<scalar_t>()[1];
        scalar_t frz = fr.data_ptr<scalar_t>()[2];
        find_interacting_clusters_kernel<scalar_t><<<grid_dim, block_dim, max_exclusions_per_cluster*sizeof(int32_t), stream>>>(
            f_coords_sorted.data_ptr<scalar_t>(),
            sorted_atom_indices.data_ptr<int32_t>(),
            natoms,
            frx, fry, frz,
            cluster_centers.data_ptr<scalar_t>(),
            cluster_bounding_boxes.data_ptr<scalar_t>(),
            cluster_exclusions_crow_indices.data_ptr<int32_t>(),
            cluster_exclusions_col_indices.data_ptr<int32_t>(),
            max_exclusions_per_cluster,
            interacting_clusters.data_ptr<int32_t>(),
            interacting_atoms.data_ptr<int32_t>(),
            num_interacting_clusters.data_ptr<int32_t>()
        );
    }));

    return std::make_tuple(
        sorted_atom_indices,
        cluster_exclusions,
        bitmask_exclusions,
        interacting_clusters,
        interacting_atoms
    );
}


std::tuple<at::Tensor, at::Tensor> decode_cluster_pairs_cuda(
    at::Tensor& coords,
    at::Tensor& box,
    at::Tensor& sorted_atom_indices,
    at::Tensor& cluster_exclusions,
    at::Tensor& bitmask_exclusions,
    at::Tensor& interacting_clusters,
    at::Tensor& interacting_atoms,
    at::Scalar cutoff,
    at::Scalar max_npairs,
    bool padding
)
{
    at::Tensor box_inv = at::linalg_inv(box);
    int32_t natoms = coords.size(0);
    int32_t max_npairs_ = max_npairs.toInt();
    max_npairs_ = ( max_npairs_ == -1 ) ? natoms * (natoms - 1) / 2 : max_npairs_;
    at::Tensor pairs = at::empty({max_npairs_, 2}, coords.options().dtype(at::kInt));
    at::Tensor npairs = at::zeros({1}, coords.options().dtype(at::kInt));

    auto stream = at::cuda::getCurrentCUDAStream();
    int32_t block_dim = BLOCK_DIM;
    int32_t grid_dim = 4 * 108;
    
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "decode_cluster_pairs", ([&] {
        decode_cluster_pairs_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            box.data_ptr<scalar_t>(),
            box_inv.data_ptr<scalar_t>(),
            static_cast<scalar_t>(cutoff.toDouble() * cutoff.toDouble()),
            sorted_atom_indices.data_ptr<int32_t>(),
            natoms,
            cluster_exclusions.data_ptr<int32_t>(),
            (int32_t)cluster_exclusions.size(1),
            bitmask_exclusions.data_ptr<uint32_t>(),
            interacting_clusters.data_ptr<int32_t>(),
            interacting_atoms.data_ptr<int32_t>(),
            (int32_t)interacting_clusters.size(0),
            max_npairs_,
            pairs.data_ptr<int32_t>(),
            npairs.data_ptr<int32_t>()
        );
    }));

    if ( !padding ) {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

        // check if the number of pairs exceeds the capacity
        int32_t npairs_found = npairs[0].item<int32_t>();
        TORCH_CHECK(npairs_found <= max_npairs_, "Too many neighbor pairs found. Maximum is " + std::to_string(max_npairs_), " but found " + std::to_string(npairs_found));
        return std::make_tuple(pairs.index({at::indexing::Slice(0, npairs_found), at::indexing::Slice()}), npairs);

    }
    else {
        return std::make_tuple(pairs, npairs);
    }
}

TORCH_LIBRARY_IMPL(torchff, CUDA, m) {
    m.impl("build_cluster_pairs", build_cluster_pairs_cuda);
    m.impl("decode_cluster_pairs", decode_cluster_pairs_cuda);
}