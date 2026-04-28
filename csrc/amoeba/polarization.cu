#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <tuple>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"
#include "common/reduce.cuh"
#include "common/dispatch.cuh"
#include "ewald/damps.cuh"
#include "amoeba/thole.cuh"
#include "multipoles/multipoles_forward.cuh"
#include "multipoles/multipoles_backward.cuh"


constexpr int BLOCK_SIZE = 256;


template <typename scalar_t, int BLOCK_SIZE, int RANK, bool DO_EWALD=false, bool DO_ENERGY=false, bool DO_COORD_GRAD=false, bool DO_MPOLE_GRAD=true>
__global__ void amoeba_induced_field_forward_kernel(
    scalar_t* coords,
    scalar_t* box,
    int64_t* pairs,
    const int64_t* pairs_excl,
    int64_t npairs,
    int64_t npairs_excl,
    scalar_t cutoff,
    scalar_t ewald_alpha,
    scalar_t prefactor,
    const scalar_t* thole,
    scalar_t* polarity,
    scalar_t* q,
    scalar_t* p,
    scalar_t* t,
    scalar_t* ene_out,
    scalar_t* coord_grad,
    scalar_t* q_grad,
    scalar_t* p_grad,
    scalar_t* t_grad
) {

    __shared__ scalar_t box_inv[9];
    if (threadIdx.x == 0) {
        // Compute reciprocal box matrix once per block
        invert_box_3x3(box, box_inv);
    }
    __syncthreads();

    scalar_t ene = static_cast<scalar_t>(0.0);
    for (int64_t index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
         index < npairs;
         index += BLOCK_SIZE * gridDim.x) {
        int64_t i = pairs[index * 2];
        int64_t j = pairs[index * 2 + 1];
        if (i < 0 || j < 0) {
            continue;
        }

        scalar_t drvec[3];
        diff_vec3(&coords[j * 3], &coords[i * 3], drvec);
        apply_pbc_triclinic(drvec, box, box_inv, drvec);

        scalar_t dr = norm3d_(drvec[0], drvec[1], drvec[2]);
        if (dr > cutoff) {
            continue;
        }

        CartesianExpansion<scalar_t, RANK> mpi;
        CartesianExpansion<scalar_t, RANK> mpj;

        mpi.s = q[i];
        mpj.s = q[j];

        if constexpr (RANK >= 1) {
            mpi.x = p[i * 3];
            mpi.y = p[i * 3 + 1];
            mpi.z = p[i * 3 + 2];
            mpj.x = p[j * 3];
            mpj.y = p[j * 3 + 1];
            mpj.z = p[j * 3 + 2];
        }

        if constexpr (RANK >= 2) {
            mpi.xx = t[i * 9 + 0] * scalar_t(1/3.0);
            mpi.xy = (t[i * 9 + 1] + t[i * 9 + 3]) * scalar_t(1/3.0);
            mpi.xz = (t[i * 9 + 2] + t[i * 9 + 6]) * scalar_t(1/3.0);
            mpi.yy = t[i * 9 + 4] * scalar_t(1/3.0);
            mpi.yz = (t[i * 9 + 5] + t[i * 9 + 7]) * scalar_t(1/3.0);
            mpi.zz = t[i * 9 + 8] * scalar_t(1/3.0);

            mpj.xx = t[j * 9 + 0] * scalar_t(1/3.0);
            mpj.xy = (t[j * 9 + 1] + t[j * 9 + 3]) * scalar_t(1/3.0);
            mpj.xz = (t[j * 9 + 2] + t[j * 9 + 6]) * scalar_t(1/3.0);
            mpj.yy = t[j * 9 + 4] * scalar_t(1/3.0);
            mpj.yz = (t[j * 9 + 5] + t[j * 9 + 7]) * scalar_t(1/3.0);
            mpj.zz = t[j * 9 + 8] * scalar_t(1/3.0);
        }

        CartesianExpansion<scalar_t, RANK> gpi{};
        CartesianExpansion<scalar_t, RANK> gpj{};
        scalar_t damps[RANK * 2 + 2] = {};
        if constexpr (DO_EWALD) {
            ewald_erfc_damps<scalar_t, 4 * RANK + 3>(dr, ewald_alpha, damps);
        }
        else {
            #pragma unroll
            for (int k = 0; k < RANK*2+2; ++k) {
                damps[k] = scalar_t(1.0);
            }
        }
        scalar_t factor = 1 / pow_(polarity[i] * polarity[j], scalar_t(1.0 / 6.0));
        scalar_t thole_pair = (thole[i] < thole[j]) ? thole[i] : thole[j];
        thole_damps<scalar_t, 4 * RANK + 3>(dr, thole_pair, factor, damps);

        scalar_t dr_g_buf[3] = {};
        scalar_t* dr_g_ptr = ( DO_COORD_GRAD ) ? dr_g_buf : nullptr;
        scalar_t* damps_ptr = damps;
        CartesianExpansion<scalar_t, RANK>* gpi_ptr = ( DO_MPOLE_GRAD ) ? &gpi : nullptr;
        CartesianExpansion<scalar_t, RANK>* gpj_ptr = ( DO_MPOLE_GRAD ) ? &gpj : nullptr;

        pairwise_multipole_kernel_with_grad<scalar_t, RANK, true, DO_ENERGY, DO_COORD_GRAD, DO_MPOLE_GRAD>(
            mpi, mpj, gpi_ptr, gpj_ptr, drvec[0], drvec[1], drvec[2], dr, damps_ptr, &ene, dr_g_ptr, nullptr);

        if constexpr ( DO_COORD_GRAD ) {
            atomicAdd(&coord_grad[i * 3],     -dr_g_buf[0] * prefactor);
            atomicAdd(&coord_grad[i * 3 + 1], -dr_g_buf[1] * prefactor);
            atomicAdd(&coord_grad[i * 3 + 2], -dr_g_buf[2] * prefactor);
            atomicAdd(&coord_grad[j * 3],      dr_g_buf[0] * prefactor);
            atomicAdd(&coord_grad[j * 3 + 1],  dr_g_buf[1] * prefactor);
            atomicAdd(&coord_grad[j * 3 + 2],  dr_g_buf[2] * prefactor);
        }

        if constexpr ( DO_MPOLE_GRAD ) {
            atomicAdd(&q_grad[i], gpi.s * prefactor);
            atomicAdd(&q_grad[j], gpj.s * prefactor);
        }

        if constexpr ( (RANK >= 1) && DO_MPOLE_GRAD ) {
            atomicAdd(&p_grad[i * 3],     gpi.x * prefactor);
            atomicAdd(&p_grad[i * 3 + 1], gpi.y * prefactor);
            atomicAdd(&p_grad[i * 3 + 2], gpi.z * prefactor);
            atomicAdd(&p_grad[j * 3],     gpj.x * prefactor);
            atomicAdd(&p_grad[j * 3 + 1], gpj.y * prefactor);
            atomicAdd(&p_grad[j * 3 + 2], gpj.z * prefactor);
        }

        if constexpr ( (RANK >= 2) && DO_MPOLE_GRAD ){
            atomicAdd(&t_grad[i * 9 + 0], gpi.xx * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 1], gpi.xy * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 2], gpi.xz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 3], gpi.xy * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 4], gpi.yy * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 5], gpi.yz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 6], gpi.xz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 7], gpi.yz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 8], gpi.zz * scalar_t(1/3.0) * prefactor);

            atomicAdd(&t_grad[j * 9 + 0], gpj.xx * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 1], gpj.xy * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 2], gpj.xz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 3], gpj.xy * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 4], gpj.yy * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 5], gpj.yz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 6], gpj.xz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 7], gpj.yz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 8], gpj.zz * scalar_t(1/3.0) * prefactor);
        }
    }

    // Exclusion pairs: only when DO_EWALD and pairs_excl is provided (nullptr and npairs_excl=0 when not used).
    // Use damping (erfcs - 1) for exclusion correction.
    if constexpr (DO_EWALD) {
        if (pairs_excl != nullptr && npairs_excl > 0) {
            constexpr int NDAMPS = RANK * 2 + 2;
            for (int64_t index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
                 index < npairs_excl;
                 index += BLOCK_SIZE * gridDim.x) {
                int64_t i = pairs_excl[index * 2];
                int64_t j = pairs_excl[index * 2 + 1];
                if (i < 0 || j < 0) {
                    continue;
                }

                scalar_t drvec[3];
                diff_vec3(&coords[j * 3], &coords[i * 3], drvec);
                apply_pbc_triclinic(drvec, box, box_inv, drvec);

                scalar_t dr = norm3d_(drvec[0], drvec[1], drvec[2]);
                if (dr > cutoff) {
                    continue;
                }

                CartesianExpansion<scalar_t, RANK> mpi;
                CartesianExpansion<scalar_t, RANK> mpj;

                mpi.s = q[i];
                mpj.s = q[j];

                if constexpr (RANK >= 1) {
                    mpi.x = p[i * 3];
                    mpi.y = p[i * 3 + 1];
                    mpi.z = p[i * 3 + 2];
                    mpj.x = p[j * 3];
                    mpj.y = p[j * 3 + 1];
                    mpj.z = p[j * 3 + 2];
                }

                if constexpr (RANK >= 2) {
                    mpi.xx = t[i * 9 + 0] * scalar_t(1/3.0);
                    mpi.xy = (t[i * 9 + 1] + t[i * 9 + 3]) * scalar_t(1/3.0);
                    mpi.xz = (t[i * 9 + 2] + t[i * 9 + 6]) * scalar_t(1/3.0);
                    mpi.yy = t[i * 9 + 4] * scalar_t(1/3.0);
                    mpi.yz = (t[i * 9 + 5] + t[i * 9 + 7]) * scalar_t(1/3.0);
                    mpi.zz = t[i * 9 + 8] * scalar_t(1/3.0);

                    mpj.xx = t[j * 9 + 0] * scalar_t(1/3.0);
                    mpj.xy = (t[j * 9 + 1] + t[j * 9 + 3]) * scalar_t(1/3.0);
                    mpj.xz = (t[j * 9 + 2] + t[j * 9 + 6]) * scalar_t(1/3.0);
                    mpj.yy = t[j * 9 + 4] * scalar_t(1/3.0);
                    mpj.yz = (t[j * 9 + 5] + t[j * 9 + 7]) * scalar_t(1/3.0);
                    mpj.zz = t[j * 9 + 8] * scalar_t(1/3.0);
                }

                CartesianExpansion<scalar_t, RANK> gpi{};
                CartesianExpansion<scalar_t, RANK> gpj{};
                scalar_t damps[NDAMPS];
                ewald_erfc_damps<scalar_t, 4 * RANK + 3>(dr, ewald_alpha, damps);
                for (int k = 0; k < NDAMPS; k++) {
                    damps[k] -= scalar_t(1.0);
                }
                scalar_t dr_g_buf[3] = {};

                scalar_t* dr_g_ptr = ( DO_COORD_GRAD ) ? dr_g_buf : nullptr;
                CartesianExpansion<scalar_t, RANK>* gpi_ptr = ( DO_MPOLE_GRAD ) ? &gpi : nullptr;
                CartesianExpansion<scalar_t, RANK>* gpj_ptr = ( DO_MPOLE_GRAD ) ? &gpj : nullptr;

                pairwise_multipole_kernel_with_grad<scalar_t, RANK, true, DO_ENERGY, DO_COORD_GRAD, DO_MPOLE_GRAD>(
                    mpi, mpj, gpi_ptr, gpj_ptr, drvec[0], drvec[1], drvec[2], dr, damps, &ene, dr_g_ptr, nullptr);

                if constexpr ( DO_COORD_GRAD ) {
                    atomicAdd(&coord_grad[i * 3],     -dr_g_buf[0] * prefactor);
                    atomicAdd(&coord_grad[i * 3 + 1], -dr_g_buf[1] * prefactor);
                    atomicAdd(&coord_grad[i * 3 + 2], -dr_g_buf[2] * prefactor);
                    atomicAdd(&coord_grad[j * 3],      dr_g_buf[0] * prefactor);
                    atomicAdd(&coord_grad[j * 3 + 1],  dr_g_buf[1] * prefactor);
                    atomicAdd(&coord_grad[j * 3 + 2],  dr_g_buf[2] * prefactor);
                }

                if constexpr ( DO_MPOLE_GRAD ) {
                    atomicAdd(&q_grad[i], gpi.s * prefactor);
                    atomicAdd(&q_grad[j], gpj.s * prefactor);
                }

                if constexpr ( (RANK >= 1) && DO_MPOLE_GRAD ) {
                    atomicAdd(&p_grad[i * 3],     gpi.x * prefactor);
                    atomicAdd(&p_grad[i * 3 + 1], gpi.y * prefactor);
                    atomicAdd(&p_grad[i * 3 + 2], gpi.z * prefactor);
                    atomicAdd(&p_grad[j * 3],     gpj.x * prefactor);
                    atomicAdd(&p_grad[j * 3 + 1], gpj.y * prefactor);
                    atomicAdd(&p_grad[j * 3 + 2], gpj.z * prefactor);
                }

                if constexpr ( (RANK >= 2) && DO_MPOLE_GRAD ){
                    atomicAdd(&t_grad[i * 9 + 0], gpi.xx * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 1], gpi.xy * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 2], gpi.xz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 3], gpi.xy * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 4], gpi.yy * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 5], gpi.yz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 6], gpi.xz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 7], gpi.yz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 8], gpi.zz * scalar_t(1/3.0) * prefactor);

                    atomicAdd(&t_grad[j * 9 + 0], gpj.xx * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 1], gpj.xy * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 2], gpj.xz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 3], gpj.xy * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 4], gpj.yy * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 5], gpj.yz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 6], gpj.xz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 7], gpj.yz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 8], gpj.zz * scalar_t(1/3.0) * prefactor);
                }
            }
        }
    }

    if constexpr ( DO_ENERGY ) {
        ene *= prefactor;
        block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
    }
}


template <typename scalar_t, int BLOCK_SIZE, int RANK=2, bool DO_EWALD=false, bool DO_ENERGY=false, bool DO_COORD_GRAD=true, bool DO_MPOLE_GRAD=true>
__global__ void amoeba_induced_field_backward_kernel(
    const scalar_t* coords,
    const scalar_t* box,
    const int64_t* pairs,
    const int64_t* pairs_excl,
    int64_t npairs,
    int64_t npairs_excl,
    const scalar_t cutoff,
    const scalar_t ewald_alpha,
    const scalar_t prefactor,
    const scalar_t* thole,
    const scalar_t* polarity,
    const scalar_t* q,
    const scalar_t* p,
    const scalar_t* t,
    const scalar_t* d_efield,
    scalar_t* coords_grad,
    scalar_t* q_grad,
    scalar_t* p_grad,
    scalar_t* t_grad
)
{
    constexpr bool NEED_FORWARD_GRAD = DO_COORD_GRAD || DO_MPOLE_GRAD;
    __shared__ scalar_t box_inv[9];
    if (threadIdx.x == 0) {
        // Compute reciprocal box matrix once per block
        invert_box_3x3(box, box_inv);
    }
    __syncthreads();
    

    for (int64_t index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
         index < npairs;
         index += BLOCK_SIZE * gridDim.x) {
        int64_t i = pairs[index * 2];
        int64_t j = pairs[index * 2 + 1];
        if (i < 0 || j < 0) {
            continue;
        }

        scalar_t drvec[3];
        diff_vec3(&coords[j * 3], &coords[i * 3], drvec);
        apply_pbc_triclinic(drvec, box, box_inv, drvec);

        scalar_t dr = norm3d_(drvec[0], drvec[1], drvec[2]);
        if (dr > cutoff) {
            continue;
        }

        CartesianExpansion<scalar_t, RANK> mpi;
        CartesianExpansion<scalar_t, RANK> mpj;
        CartesianExpansion<scalar_t, RANK> mpi_expand;
        CartesianExpansion<scalar_t, RANK> mpj_expand;
        CartesianExpansion<scalar_t, RANK> gpi_from_ene;
        CartesianExpansion<scalar_t, RANK> gpj_from_ene;
        CartesianExpansion<scalar_t, RANK> gpi_out;
        CartesianExpansion<scalar_t, RANK> gpj_out;

        mpi.s = q[i];
        mpj.s = q[j];
        mpi_expand.s = static_cast<scalar_t>(0.0);
        mpj_expand.s = static_cast<scalar_t>(0.0);

        if constexpr (RANK >= 1) {
            mpi.x = p[i * 3];
            mpi.y = p[i * 3 + 1];
            mpi.z = p[i * 3 + 2];
            mpj.x = p[j * 3];
            mpj.y = p[j * 3 + 1];
            mpj.z = p[j * 3 + 2];
            mpi_expand.x = -d_efield[i * 3];
            mpi_expand.y = -d_efield[i * 3 + 1];
            mpi_expand.z = -d_efield[i * 3 + 2];
            mpj_expand.x = -d_efield[j * 3];
            mpj_expand.y = -d_efield[j * 3 + 1];
            mpj_expand.z = -d_efield[j * 3 + 2];
        }

        if constexpr (RANK >= 2) {
            mpi.xx = t[i * 9 + 0] * scalar_t(1/3.0);
            mpi.xy = (t[i * 9 + 1] + t[i * 9 + 3]) * scalar_t(1/3.0);
            mpi.xz = (t[i * 9 + 2] + t[i * 9 + 6]) * scalar_t(1/3.0);
            mpi.yy = t[i * 9 + 4] * scalar_t(1/3.0);
            mpi.yz = (t[i * 9 + 5] + t[i * 9 + 7]) * scalar_t(1/3.0);
            mpi.zz = t[i * 9 + 8] * scalar_t(1/3.0);

            mpj.xx = t[j * 9 + 0] * scalar_t(1/3.0);
            mpj.xy = (t[j * 9 + 1] + t[j * 9 + 3]) * scalar_t(1/3.0);
            mpj.xz = (t[j * 9 + 2] + t[j * 9 + 6]) * scalar_t(1/3.0);
            mpj.yy = t[j * 9 + 4] * scalar_t(1/3.0);
            mpj.yz = (t[j * 9 + 5] + t[j * 9 + 7]) * scalar_t(1/3.0);
            mpj.zz = t[j * 9 + 8] * scalar_t(1/3.0);
        }

        scalar_t dr_g_buf[3] = {static_cast<scalar_t>(0.0), static_cast<scalar_t>(0.0), static_cast<scalar_t>(0.0)};

        scalar_t interaction_tensor[20] = {};

        scalar_t* dr_g_ptr = (DO_COORD_GRAD) ? dr_g_buf : nullptr;
        scalar_t* interaction_tensor_ptr = (DO_COORD_GRAD || DO_MPOLE_GRAD) ? interaction_tensor : nullptr;
        CartesianExpansion<scalar_t, RANK>* gpi_ptr = NEED_FORWARD_GRAD ? &gpi_from_ene : nullptr;
        CartesianExpansion<scalar_t, RANK>* gpj_ptr = NEED_FORWARD_GRAD ? &gpj_from_ene : nullptr;

        scalar_t damps[RANK * 2 + 2] = {};
        if constexpr (DO_EWALD) {
            ewald_erfc_damps<scalar_t, 4 * RANK + 3>(dr, ewald_alpha, damps);
        }
        else {
            #pragma unroll
            for (int k = 0; k < RANK*2+2; ++k) {
                damps[k] = scalar_t(1.0);
            }
            
        }
        scalar_t factor = 1 / pow_(polarity[i] * polarity[j], scalar_t(1.0 / 6.0));
        scalar_t thole_pair = (thole[i] < thole[j]) ? thole[i] : thole[j];
        thole_damps<scalar_t, 4 * RANK + 3>(dr, thole_pair, factor, damps);

        pairwise_multipole_kernel_with_grad<scalar_t, RANK, true, false, false, NEED_FORWARD_GRAD>(
            mpi, mpj, gpi_ptr, gpj_ptr,
            drvec[0], drvec[1], drvec[2], dr,
            damps,
            nullptr, nullptr,
            interaction_tensor_ptr
        );
    
        if constexpr (DO_COORD_GRAD || DO_MPOLE_GRAD) {
            pairwise_multipole_backward_add_contribution_from_field<scalar_t, RANK, true, true, DO_COORD_GRAD, DO_MPOLE_GRAD>(
                mpi_expand, mpj_expand, interaction_tensor_ptr, gpi_from_ene, gpj_from_ene,
                (DO_MPOLE_GRAD) ? &gpi_out : nullptr,
                (DO_MPOLE_GRAD) ? &gpj_out : nullptr,
                (DO_COORD_GRAD) ? dr_g_buf : nullptr
            );
        }

        if constexpr (DO_COORD_GRAD) {
            atomicAdd(&coords_grad[i * 3],     -dr_g_buf[0] * prefactor);
            atomicAdd(&coords_grad[i * 3 + 1], -dr_g_buf[1] * prefactor);
            atomicAdd(&coords_grad[i * 3 + 2], -dr_g_buf[2] * prefactor);
            atomicAdd(&coords_grad[j * 3],      dr_g_buf[0] * prefactor);
            atomicAdd(&coords_grad[j * 3 + 1],  dr_g_buf[1] * prefactor);
            atomicAdd(&coords_grad[j * 3 + 2],  dr_g_buf[2] * prefactor);
        }

        if constexpr (DO_MPOLE_GRAD) {
            atomicAdd(&q_grad[i], gpi_out.s * prefactor);
            atomicAdd(&q_grad[j], gpj_out.s * prefactor);
        }

        if constexpr ((RANK >= 1) && DO_MPOLE_GRAD) {
            atomicAdd(&p_grad[i * 3],     gpi_out.x * prefactor);
            atomicAdd(&p_grad[i * 3 + 1], gpi_out.y * prefactor);
            atomicAdd(&p_grad[i * 3 + 2], gpi_out.z * prefactor);
            atomicAdd(&p_grad[j * 3],     gpj_out.x * prefactor);
            atomicAdd(&p_grad[j * 3 + 1], gpj_out.y * prefactor);
            atomicAdd(&p_grad[j * 3 + 2], gpj_out.z * prefactor);
        }

        if constexpr ((RANK >= 2) && DO_MPOLE_GRAD) {
            atomicAdd(&t_grad[i * 9 + 0], gpi_out.xx * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 1], gpi_out.xy * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 2], gpi_out.xz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 3], gpi_out.xy * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 4], gpi_out.yy * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 5], gpi_out.yz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 6], gpi_out.xz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 7], gpi_out.yz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[i * 9 + 8], gpi_out.zz * scalar_t(1/3.0) * prefactor);

            atomicAdd(&t_grad[j * 9 + 0], gpj_out.xx * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 1], gpj_out.xy * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 2], gpj_out.xz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 3], gpj_out.xy * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 4], gpj_out.yy * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 5], gpj_out.yz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 6], gpj_out.xz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 7], gpj_out.yz * scalar_t(1/3.0) * prefactor);
            atomicAdd(&t_grad[j * 9 + 8], gpj_out.zz * scalar_t(1/3.0) * prefactor);
        }
    }

    // Exclusion pairs: only when DO_EWALD and pairs_excl is provided. Use damping (erfcs - 1).
    if constexpr (DO_EWALD) {
        if (pairs_excl != nullptr && npairs_excl > 0) {
            constexpr int NDAMPS = RANK * 2 + 2;
            for (int64_t index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
                 index < npairs_excl;
                 index += BLOCK_SIZE * gridDim.x) {
                int64_t i = pairs_excl[index * 2];
                int64_t j = pairs_excl[index * 2 + 1];
                if (i < 0 || j < 0) {
                    continue;
                }

                scalar_t drvec[3];
                diff_vec3(&coords[j * 3], &coords[i * 3], drvec);
                apply_pbc_triclinic(drvec, box, box_inv, drvec);

                scalar_t dr = norm3d_(drvec[0], drvec[1], drvec[2]);
                if (dr > cutoff) {
                    continue;
                }

                CartesianExpansion<scalar_t, RANK> mpi;
                CartesianExpansion<scalar_t, RANK> mpj;
                CartesianExpansion<scalar_t, RANK> mpi_expand;
                CartesianExpansion<scalar_t, RANK> mpj_expand;
                CartesianExpansion<scalar_t, RANK> gpi_from_ene;
                CartesianExpansion<scalar_t, RANK> gpj_from_ene;
                CartesianExpansion<scalar_t, RANK> gpi_out;
                CartesianExpansion<scalar_t, RANK> gpj_out;

                mpi.s = q[i];
                mpj.s = q[j];
                mpi_expand.s = static_cast<scalar_t>(0.0);
                mpj_expand.s = static_cast<scalar_t>(0.0);

                if constexpr (RANK >= 1) {
                    mpi.x = p[i * 3];
                    mpi.y = p[i * 3 + 1];
                    mpi.z = p[i * 3 + 2];
                    mpj.x = p[j * 3];
                    mpj.y = p[j * 3 + 1];
                    mpj.z = p[j * 3 + 2];
                    mpi_expand.x = -d_efield[i * 3];
                    mpi_expand.y = -d_efield[i * 3 + 1];
                    mpi_expand.z = -d_efield[i * 3 + 2];
                    mpj_expand.x = -d_efield[j * 3];
                    mpj_expand.y = -d_efield[j * 3 + 1];
                    mpj_expand.z = -d_efield[j * 3 + 2];
                }

                if constexpr (RANK >= 2) {
                    mpi.xx = t[i * 9 + 0] * scalar_t(1/3.0);
                    mpi.xy = (t[i * 9 + 1] + t[i * 9 + 3]) * scalar_t(1/3.0);
                    mpi.xz = (t[i * 9 + 2] + t[i * 9 + 6]) * scalar_t(1/3.0);
                    mpi.yy = t[i * 9 + 4] * scalar_t(1/3.0);
                    mpi.yz = (t[i * 9 + 5] + t[i * 9 + 7]) * scalar_t(1/3.0);
                    mpi.zz = t[i * 9 + 8] * scalar_t(1/3.0);

                    mpj.xx = t[j * 9 + 0] * scalar_t(1/3.0);
                    mpj.xy = (t[j * 9 + 1] + t[j * 9 + 3]) * scalar_t(1/3.0);
                    mpj.xz = (t[j * 9 + 2] + t[j * 9 + 6]) * scalar_t(1/3.0);
                    mpj.yy = t[j * 9 + 4] * scalar_t(1/3.0);
                    mpj.yz = (t[j * 9 + 5] + t[j * 9 + 7]) * scalar_t(1/3.0);
                    mpj.zz = t[j * 9 + 8] * scalar_t(1/3.0);
                }

                scalar_t dr_g_buf[3] = {static_cast<scalar_t>(0.0), static_cast<scalar_t>(0.0), static_cast<scalar_t>(0.0)};

                scalar_t interaction_tensor[20] = {};

                scalar_t* dr_g_ptr = (DO_COORD_GRAD) ? dr_g_buf : nullptr;
                scalar_t* interaction_tensor_ptr = (DO_COORD_GRAD || DO_MPOLE_GRAD) ? interaction_tensor : nullptr;
                CartesianExpansion<scalar_t, RANK>* gpi_ptr = &gpi_from_ene;
                CartesianExpansion<scalar_t, RANK>* gpj_ptr = &gpj_from_ene;

                scalar_t damps[NDAMPS];
                ewald_erfc_damps<scalar_t, 4 * RANK + 3>(dr, ewald_alpha, damps);
                for (int k = 0; k < NDAMPS; k++) {
                    damps[k] -= scalar_t(1.0);
                }
                pairwise_multipole_kernel_with_grad<scalar_t, RANK, true, false, false, true>(
                    mpi, mpj, gpi_ptr, gpj_ptr,
                    drvec[0], drvec[1], drvec[2], dr,
                    damps,
                    nullptr, nullptr,
                    interaction_tensor_ptr
                );

                if constexpr (DO_COORD_GRAD || DO_MPOLE_GRAD) {
                    pairwise_multipole_backward_add_contribution_from_field<scalar_t, RANK, true, true, DO_COORD_GRAD, DO_MPOLE_GRAD>(
                        mpi_expand, mpj_expand, interaction_tensor_ptr, gpi_from_ene, gpj_from_ene,
                        (DO_MPOLE_GRAD) ? &gpi_out : nullptr,
                        (DO_MPOLE_GRAD) ? &gpj_out : nullptr,
                        (DO_COORD_GRAD) ? dr_g_buf : nullptr
                    );
                }

                if constexpr (DO_COORD_GRAD) {
                    atomicAdd(&coords_grad[i * 3],     -dr_g_buf[0] * prefactor);
                    atomicAdd(&coords_grad[i * 3 + 1], -dr_g_buf[1] * prefactor);
                    atomicAdd(&coords_grad[i * 3 + 2], -dr_g_buf[2] * prefactor);
                    atomicAdd(&coords_grad[j * 3],      dr_g_buf[0] * prefactor);
                    atomicAdd(&coords_grad[j * 3 + 1],  dr_g_buf[1] * prefactor);
                    atomicAdd(&coords_grad[j * 3 + 2],  dr_g_buf[2] * prefactor);
                }

                if constexpr (DO_MPOLE_GRAD) {
                    atomicAdd(&q_grad[i], gpi_out.s * prefactor);
                    atomicAdd(&q_grad[j], gpj_out.s * prefactor);
                }

                if constexpr ((RANK >= 1) && DO_MPOLE_GRAD) {
                    atomicAdd(&p_grad[i * 3],     gpi_out.x * prefactor);
                    atomicAdd(&p_grad[i * 3 + 1], gpi_out.y * prefactor);
                    atomicAdd(&p_grad[i * 3 + 2], gpi_out.z * prefactor);
                    atomicAdd(&p_grad[j * 3],     gpj_out.x * prefactor);
                    atomicAdd(&p_grad[j * 3 + 1], gpj_out.y * prefactor);
                    atomicAdd(&p_grad[j * 3 + 2], gpj_out.z * prefactor);
                }

                if constexpr ((RANK >= 2) && DO_MPOLE_GRAD) {
                    atomicAdd(&t_grad[i * 9 + 0], gpi_out.xx * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 1], gpi_out.xy * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 2], gpi_out.xz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 3], gpi_out.xy * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 4], gpi_out.yy * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 5], gpi_out.yz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 6], gpi_out.xz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 7], gpi_out.yz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[i * 9 + 8], gpi_out.zz * scalar_t(1/3.0) * prefactor);

                    atomicAdd(&t_grad[j * 9 + 0], gpj_out.xx * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 1], gpj_out.xy * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 2], gpj_out.xz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 3], gpj_out.xy * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 4], gpj_out.yy * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 5], gpj_out.yz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 6], gpj_out.xz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 7], gpj_out.yz * scalar_t(1/3.0) * prefactor);
                    atomicAdd(&t_grad[j * 9 + 8], gpj_out.zz * scalar_t(1/3.0) * prefactor);
                }
            }
        }
    }
}


class AmoebaInducedFieldFunctionCuda : public torch::autograd::Function<AmoebaInducedFieldFunctionCuda> {

public:

static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& coords,
    at::Tensor& box,
    at::Tensor& pairs,
    c10::optional<at::Tensor> pairs_excl,
    at::Tensor& q,
    at::Tensor& p,
    c10::optional<at::Tensor> t,
    at::Tensor& polarity,
    at::Tensor& thole,
    at::Scalar cutoff,
    at::Scalar ewald_alpha,
    at::Scalar prefactor
) {

    int64_t rank = 1;
    if (t.has_value()) {
        rank = 2;
    }

    int64_t npairs = pairs.size(0);
    int64_t npairs_excl = (pairs_excl.has_value() && pairs_excl.value().defined())
        ? pairs_excl.value().size(0) : 0;
    const int64_t* pairs_excl_ptr = (pairs_excl.has_value() && pairs_excl.value().defined() && npairs_excl > 0)
        ? pairs_excl.value().data_ptr<int64_t>() : nullptr;

    bool do_ewald = (ewald_alpha.toDouble() >= 0);

    at::Tensor ene = at::zeros({1}, coords.options());
    at::Tensor q_grad = at::zeros_like(q);
    at::Tensor p_grad = at::zeros_like(p);
    at::Tensor t_grad = (rank >= 2) ? at::zeros_like(t.value()) : at::Tensor();

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t max_pairs = npairs_excl > npairs ? npairs_excl : npairs;
    int grid_dim = std::min(
        static_cast<int>((max_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE),
        props->maxBlocksPerMultiProcessor * props->multiProcessorCount
    );

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "amoeba_induced_field_forward_kernel", ([&] {
        const scalar_t cutoff_val = cutoff.to<scalar_t>();
        const scalar_t ewald_alpha_val = ewald_alpha.to<scalar_t>();
        const scalar_t prefactor_val = prefactor.to<scalar_t>();

        scalar_t* t_ptr = (rank >= 2) ? t.value().data_ptr<scalar_t>() : nullptr;
        scalar_t* t_grad_ptr = (rank >= 2) ? t_grad.data_ptr<scalar_t>() : nullptr;

        const scalar_t* thole_ptr = thole.data_ptr<scalar_t>();
        scalar_t* polarity_ptr = polarity.data_ptr<scalar_t>();

        DISPATCH_RANK(rank, RANK, [&] {
            DISPATCH_BOOL(do_ewald, DO_EWALD, [&] {
                amoeba_induced_field_forward_kernel<
                    scalar_t, BLOCK_SIZE, RANK, DO_EWALD, false, false, true>
                <<<grid_dim, BLOCK_SIZE, 0, stream>>>(
                    coords.data_ptr<scalar_t>(),
                    box.data_ptr<scalar_t>(),
                    pairs.data_ptr<int64_t>(),
                    pairs_excl_ptr,
                    npairs,
                    npairs_excl,
                    cutoff_val,
                    ewald_alpha_val,
                    prefactor_val,
                    thole_ptr,
                    polarity_ptr,
                    q.data_ptr<scalar_t>(),
                    p.data_ptr<scalar_t>(),
                    t_ptr,
                    nullptr,
                    nullptr,
                    q_grad.data_ptr<scalar_t>(),
                    p_grad.data_ptr<scalar_t>(),
                    t_grad_ptr
                );
            });
        });
    }));

    bool need_coord_grad = coords.requires_grad();
    bool need_q_grad = q.requires_grad();
    bool need_p_grad = p.requires_grad();
    bool need_t_grad = t.has_value() && t.value().requires_grad();
    ctx->saved_data["need_coord_grad"] = need_coord_grad;
    ctx->saved_data["need_q_grad"] = need_q_grad;
    ctx->saved_data["need_p_grad"] = need_p_grad;
    ctx->saved_data["need_t_grad"] = need_t_grad;

    ctx->saved_data["rank"] = rank;
    ctx->saved_data["cutoff"] = cutoff.to<double>();
    ctx->saved_data["ewald_alpha"] = ewald_alpha.to<double>();
    ctx->saved_data["prefactor"] = prefactor.to<double>();
    at::Tensor pairs_excl_t = (pairs_excl.has_value() && pairs_excl.value().defined())
        ? pairs_excl.value() : at::empty({0, 2}, coords.options().dtype(at::kLong));
    ctx->save_for_backward({
        coords, box, pairs, pairs_excl_t, 
        q, 
        p,
        (rank >= 2) ? t.value() : at::Tensor(),
        polarity,
        thole
    });
    return -p_grad;
}

static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<at::Tensor> grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    at::Tensor coords = saved[0];
    at::Tensor box = saved[1];
    at::Tensor pairs = saved[2];
    at::Tensor pairs_excl_saved = saved[3];
    
    int rank = ctx->saved_data["rank"].toInt();
    at::Tensor q = saved[4];
    at::Tensor p = ( rank >= 1 ) ? saved[5] : at::zeros({coords.size(0), 3}, coords.options());
    at::Tensor t = ( rank >= 2 ) ? saved[6] : at::zeros({coords.size(0), 3, 3}, coords.options());
    at::Tensor polarity = saved[7];
    at::Tensor thole = saved[8];

    double cutoff_val = ctx->saved_data["cutoff"].toDouble();
    double ewald_alpha_val = ctx->saved_data["ewald_alpha"].toDouble();
    double prefactor_val = ctx->saved_data["prefactor"].toDouble();

    int64_t npairs = pairs.size(0);
    int64_t npairs_excl = (pairs_excl_saved.defined() && pairs_excl_saved.numel() > 0)
        ? pairs_excl_saved.size(0) : 0;
    const int64_t* pairs_excl_ptr = (pairs_excl_saved.defined() && npairs_excl > 0)
        ? pairs_excl_saved.data_ptr<int64_t>() : nullptr;

    bool do_ewald = (ewald_alpha_val >= 0);
    bool need_coord_grad = ctx->saved_data["need_coord_grad"].toBool();
    bool need_q_grad = ctx->saved_data["need_q_grad"].toBool();
    bool need_p_grad = ctx->saved_data["need_p_grad"].toBool();
    bool need_t_grad = ctx->saved_data["need_t_grad"].toBool();
    bool need_mpole_grad = need_q_grad || need_p_grad || need_t_grad;

    at::Tensor coord_grad = need_coord_grad ? at::zeros_like(coords) : at::Tensor();
    at::Tensor q_grad = need_mpole_grad ? at::zeros_like(q) : at::Tensor();
    at::Tensor p_grad = need_mpole_grad ? at::zeros_like(p) : at::Tensor();
    at::Tensor t_grad = need_mpole_grad ? at::zeros_like(t) : at::Tensor();

    // Forward returns a single Tensor: the induced field.
    at::Tensor d_efield_tensor = grad_outputs[0].defined() ? grad_outputs[0].contiguous() : at::zeros_like(p);

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t max_pairs = npairs_excl > npairs ? npairs_excl : npairs;
    int grid_dim = std::min(
        static_cast<int>((max_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE),
        props->maxBlocksPerMultiProcessor * props->multiProcessorCount
    );

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "amoeba_induced_field_backward_cuda", ([&] {
        const scalar_t cutoff = static_cast<scalar_t>(cutoff_val);
        const scalar_t ewald_alpha = static_cast<scalar_t>(ewald_alpha_val);
        const scalar_t prefactor = static_cast<scalar_t>(prefactor_val);
        const scalar_t* thole_ptr = thole.data_ptr<scalar_t>();

        scalar_t* coord_grad_ptr = need_coord_grad ? coord_grad.data_ptr<scalar_t>() : nullptr;
        scalar_t* q_grad_ptr = need_mpole_grad ? q_grad.data_ptr<scalar_t>() : nullptr;
        scalar_t* p_grad_ptr = need_mpole_grad ? p_grad.data_ptr<scalar_t>() : nullptr;
        scalar_t* t_grad_ptr = need_mpole_grad ? t_grad.data_ptr<scalar_t>() : nullptr;

        DISPATCH_RANK(rank, RANK, [&] {
            DISPATCH_BOOL(do_ewald, DO_EWALD, [&] {
                DISPATCH_BOOL(need_coord_grad, DO_COORD_GRAD, [&] {
                    DISPATCH_BOOL(need_mpole_grad, DO_MPOLE_GRAD, [&] {
                        amoeba_induced_field_backward_kernel<
                            scalar_t, BLOCK_SIZE, 2, DO_EWALD, false, DO_COORD_GRAD, DO_MPOLE_GRAD>
                        <<<grid_dim, BLOCK_SIZE, 0, stream>>>(
                            coords.data_ptr<scalar_t>(),
                            box.data_ptr<scalar_t>(),
                            pairs.data_ptr<int64_t>(),
                            pairs_excl_ptr,
                            npairs,
                            npairs_excl,
                            cutoff,
                            ewald_alpha,
                            prefactor,
                            thole_ptr,
                            polarity.data_ptr<scalar_t>(),
                            q.data_ptr<scalar_t>(),
                            p.data_ptr<scalar_t>(),
                            t.data_ptr<scalar_t>(),
                            d_efield_tensor.data_ptr<scalar_t>(),
                            coord_grad_ptr,
                            q_grad_ptr,
                            p_grad_ptr,
                            t_grad_ptr
                        );
                    });
                });
            });
        });
    }));

    at::Tensor ignore;
    std::vector<at::Tensor> grads(12);
    grads[0] = need_coord_grad ? coord_grad : ignore;  // coords
    grads[1] = ignore;                                 // box
    grads[2] = ignore;                                 // pairs
    grads[3] = ignore;                                 // pairs_excl
    grads[4] = need_q_grad ? q_grad : ignore;          // q
    grads[5] = need_p_grad ? p_grad : ignore;          // p
    grads[6] = need_t_grad ? t_grad : ignore;          // t
    grads[7] = ignore;                                 // polarity (no gradient)
    grads[8] = ignore;                                 // cutoff
    grads[9] = ignore;                                 // ewald_alpha
    grads[10] = ignore;                                // thole
    grads[11] = ignore;                                // prefactor
    return grads;
}

};


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_amoeba_induced_field_from_atom_pairs",
        [](const at::Tensor& coords,
            const at::Tensor& box,
            const at::Tensor& pairs,
            c10::optional<at::Tensor> pairs_excl,
            const at::Tensor& q,
            const at::Tensor& p,
            c10::optional<at::Tensor> t,
            const at::Tensor& polarity,
            const at::Tensor& thole,
            at::Scalar cutoff,
            at::Scalar ewald_alpha,
            at::Scalar prefactor) -> at::Tensor {
            auto field = AmoebaInducedFieldFunctionCuda::apply(
                const_cast<at::Tensor&>(coords), const_cast<at::Tensor&>(box),
                const_cast<at::Tensor&>(pairs), pairs_excl, const_cast<at::Tensor&>(q),
                const_cast<at::Tensor&>(p), t, const_cast<at::Tensor&>(polarity),
                const_cast<at::Tensor&>(thole), cutoff, ewald_alpha, prefactor);
            return field;
        });
}