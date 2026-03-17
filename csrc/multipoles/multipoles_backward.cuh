#ifndef TORCHFF_MULTIPOLES_BACKWARD_CUH
#define TORCHFF_MULTIPOLES_BACKWARD_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"
#include "storage.cuh"
#include "multipoles_forward.cuh"


template <typename scalar_t, int RANK=2, bool USE_I=true, bool USE_J=true, bool DO_COORD_GRAD=true, bool DO_MPOLE_GRAD=true>
__device__ __forceinline__ void pairwise_multipole_backward_add_contribution_from_field(
    const CartesianExpansion<scalar_t, RANK>& mpi_expand,
    const CartesianExpansion<scalar_t, RANK>& mpj_expand,
    const scalar_t* interaction_tensor,
    const CartesianExpansion<scalar_t, RANK>& gpi_from_ene,
    const CartesianExpansion<scalar_t, RANK>& gpj_from_ene,
    CartesianExpansion<scalar_t, RANK>* gpi_out,
    CartesianExpansion<scalar_t, RANK>* gpj_out,
    scalar_t* dr_g
)
{
    const scalar_t& drinv = interaction_tensor[0];
    const scalar_t& tx   = interaction_tensor[1];
    const scalar_t& ty   = interaction_tensor[2];
    const scalar_t& tz   = interaction_tensor[3];
    const scalar_t& txx  = interaction_tensor[4];
    const scalar_t& txy  = interaction_tensor[5];
    const scalar_t& txz  = interaction_tensor[6];
    const scalar_t& tyy  = interaction_tensor[7];
    const scalar_t& tyz  = interaction_tensor[8];
    const scalar_t& tzz  = interaction_tensor[9];
    const scalar_t& txxx = interaction_tensor[10];
    const scalar_t& txxy = interaction_tensor[11];
    const scalar_t& txxz = interaction_tensor[12];
    const scalar_t& tyyy = interaction_tensor[13];
    const scalar_t& tyyx = interaction_tensor[14];
    const scalar_t& tyyz = interaction_tensor[15];
    const scalar_t& tzzz = interaction_tensor[16];
    const scalar_t& tzzx = interaction_tensor[17];
    const scalar_t& tzzy = interaction_tensor[18];
    const scalar_t& txyz = interaction_tensor[19];

    if (USE_I) {
        if constexpr (DO_COORD_GRAD) {
            dr_g[0] += -gpi_from_ene.x*mpi_expand.s - gpi_from_ene.xx*mpi_expand.x - gpi_from_ene.xy*mpi_expand.y - gpi_from_ene.xz*mpi_expand.z;
            dr_g[1] += -gpi_from_ene.y*mpi_expand.s - gpi_from_ene.xy*mpi_expand.x - gpi_from_ene.yy*mpi_expand.y - gpi_from_ene.yz*mpi_expand.z;
            dr_g[2] += -gpi_from_ene.z*mpi_expand.s - gpi_from_ene.xz*mpi_expand.x - gpi_from_ene.yz*mpi_expand.y - gpi_from_ene.zz*mpi_expand.z;
        }
        if constexpr (DO_MPOLE_GRAD) {
            gpi_out->s  +=  mpj_expand.s*drinv + mpj_expand.x*tx + mpj_expand.y*ty + mpj_expand.z*tz;
            gpi_out->x  += -mpj_expand.s*tx - mpj_expand.x*txx - mpj_expand.y*txy - mpj_expand.z*txz;
            gpi_out->y  += -mpj_expand.s*ty - mpj_expand.x*txy - mpj_expand.y*tyy - mpj_expand.z*tyz;
            gpi_out->z  += -mpj_expand.s*tz - mpj_expand.x*txz - mpj_expand.y*tyz - mpj_expand.z*tzz;
            gpi_out->xx +=  mpj_expand.s*txx + mpj_expand.x*txxx + mpj_expand.y*txxy + mpj_expand.z*txxz;
            gpi_out->xy +=  mpj_expand.s*txy + mpj_expand.x*txxy + mpj_expand.y*tyyx + mpj_expand.z*txyz;
            gpi_out->xz +=  mpj_expand.s*txz + mpj_expand.x*txxz + mpj_expand.y*txyz + mpj_expand.z*tzzx;
            gpi_out->yy +=  mpj_expand.s*tyy + mpj_expand.x*tyyx + mpj_expand.y*tyyy + mpj_expand.z*tyyz;
            gpi_out->yz +=  mpj_expand.s*tyz + mpj_expand.x*txyz + mpj_expand.y*tyyz + mpj_expand.z*tzzy;
            gpi_out->zz +=  mpj_expand.s*tzz + mpj_expand.x*tzzx + mpj_expand.y*tzzy + mpj_expand.z*tzzz;
        }
    }
    if (USE_J) {
        if constexpr (DO_COORD_GRAD) {
            dr_g[0] += gpj_from_ene.x*mpj_expand.s + gpj_from_ene.xx*mpj_expand.x + gpj_from_ene.xy*mpj_expand.y + gpj_from_ene.xz*mpj_expand.z;
            dr_g[1] += gpj_from_ene.y*mpj_expand.s + gpj_from_ene.xy*mpj_expand.x + gpj_from_ene.yy*mpj_expand.y + gpj_from_ene.yz*mpj_expand.z;
            dr_g[2] += gpj_from_ene.z*mpj_expand.s + gpj_from_ene.xz*mpj_expand.x + gpj_from_ene.yz*mpj_expand.y + gpj_from_ene.zz*mpj_expand.z;
        }
        if constexpr (DO_MPOLE_GRAD) {
            gpj_out->s  += mpi_expand.s*drinv - mpi_expand.x*tx - mpi_expand.y*ty - mpi_expand.z*tz;
            gpj_out->x  += mpi_expand.s*tx - mpi_expand.x*txx - mpi_expand.y*txy - mpi_expand.z*txz;
            gpj_out->y  += mpi_expand.s*ty - mpi_expand.x*txy - mpi_expand.y*tyy - mpi_expand.z*tyz;
            gpj_out->z  += mpi_expand.s*tz - mpi_expand.x*txz - mpi_expand.y*tyz - mpi_expand.z*tzz;
            gpj_out->xx += mpi_expand.s*txx - mpi_expand.x*txxx - mpi_expand.y*txxy - mpi_expand.z*txxz;
            gpj_out->xy += mpi_expand.s*txy - mpi_expand.x*txxy - mpi_expand.y*tyyx - mpi_expand.z*txyz;
            gpj_out->xz += mpi_expand.s*txz - mpi_expand.x*txxz - mpi_expand.y*txyz - mpi_expand.z*tzzx;
            gpj_out->yy += mpi_expand.s*tyy - mpi_expand.x*tyyx - mpi_expand.y*tyyy - mpi_expand.z*tyyz;
            gpj_out->yz += mpi_expand.s*tyz - mpi_expand.x*txyz - mpi_expand.y*tyyz - mpi_expand.z*tzzy;
            gpj_out->zz += mpi_expand.s*tzz - mpi_expand.x*tzzx - mpi_expand.y*tzzy - mpi_expand.z*tzzz;
        }
    }
}


template <typename scalar_t, int BLOCK_SIZE, int RANK=2, bool DO_EWALD=false, bool DO_ENERGY=true, bool DO_COORD_GRAD=true, bool DO_MPOLE_GRAD=true>
__global__ void pairwise_multipole_with_fields_backward_kernel(
    const scalar_t* coords,
    const scalar_t* box,
    const int64_t* pairs,
    const int64_t* pairs_excl,
    int64_t npairs,
    int64_t npairs_excl,
    const scalar_t cutoff,
    const scalar_t ewald_alpha,
    const scalar_t prefactor,
    const scalar_t* q,
    const scalar_t* p,
    const scalar_t* t,
    const scalar_t* d_energy,
    const scalar_t* d_epot,
    const scalar_t* d_efield,
    scalar_t* coords_grad,
    scalar_t* q_grad,
    scalar_t* p_grad,
    scalar_t* t_grad
)
{
    __shared__ scalar_t box_inv[9];
    if (threadIdx.x == 0) {
        // Compute reciprocal box matrix once per block
        invert_box_3x3(box, box_inv);
    }
    __syncthreads();
    
    scalar_t dene = ( d_energy ) ? d_energy[0] : static_cast<scalar_t>(0.0);

    constexpr bool NEED_FORWARD_GRAD = DO_COORD_GRAD || DO_MPOLE_GRAD;

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
        mpi_expand.s = d_epot[i];
        mpj_expand.s = d_epot[j];

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

        if constexpr (DO_EWALD) {
            scalar_t damps[RANK * 2 + 2];
            ewald_erfc_damps<scalar_t, 4 * RANK + 3>(dr, ewald_alpha, damps);
            pairwise_multipole_kernel_with_grad<scalar_t, RANK, true, false, DO_COORD_GRAD, NEED_FORWARD_GRAD>(
                mpi, mpj, gpi_ptr, gpj_ptr,
                drvec[0], drvec[1], drvec[2], dr,
                damps,
                nullptr, dr_g_ptr,
                interaction_tensor_ptr
            );
        } else {
            pairwise_multipole_kernel_with_grad<scalar_t, RANK, false, false, DO_COORD_GRAD, NEED_FORWARD_GRAD>(
                mpi, mpj, gpi_ptr, gpj_ptr,
                drvec[0], drvec[1], drvec[2], dr,
                nullptr,
                nullptr, dr_g_ptr,
                interaction_tensor_ptr
            );
        }
        if constexpr (DO_COORD_GRAD) {
            dr_g_buf[0] *= dene;
            dr_g_buf[1] *= dene;
            dr_g_buf[2] *= dene;
        }
        if constexpr (DO_COORD_GRAD || DO_MPOLE_GRAD) {
            pairwise_multipole_backward_add_contribution_from_field<scalar_t, RANK, true, true, DO_COORD_GRAD, DO_MPOLE_GRAD>(
                mpi_expand, mpj_expand, interaction_tensor_ptr, gpi_from_ene, gpj_from_ene,
                (DO_MPOLE_GRAD) ? &gpi_out : nullptr,
                (DO_MPOLE_GRAD) ? &gpj_out : nullptr,
                (DO_COORD_GRAD) ? dr_g_buf : nullptr
            );
        }

        if constexpr (DO_MPOLE_GRAD) {
            gpi_out.s  += dene * gpi_from_ene.s;
            gpj_out.s  += dene * gpj_from_ene.s;
            if constexpr (RANK >= 1) {
                gpi_out.x  += dene * gpi_from_ene.x;
                gpi_out.y  += dene * gpi_from_ene.y;
                gpi_out.z  += dene * gpi_from_ene.z;
                gpj_out.x  += dene * gpj_from_ene.x;
                gpj_out.y  += dene * gpj_from_ene.y;
                gpj_out.z  += dene * gpj_from_ene.z;
            }
            if constexpr (RANK >= 2) {
                gpi_out.xx += dene * gpi_from_ene.xx;
                gpi_out.xy += dene * gpi_from_ene.xy;
                gpi_out.xz += dene * gpi_from_ene.xz;
                gpi_out.yy += dene * gpi_from_ene.yy;
                gpi_out.yz += dene * gpi_from_ene.yz;
                gpi_out.zz += dene * gpi_from_ene.zz;
                gpj_out.xx += dene * gpj_from_ene.xx;
                gpj_out.xy += dene * gpj_from_ene.xy;
                gpj_out.xz += dene * gpj_from_ene.xz;
                gpj_out.yy += dene * gpj_from_ene.yy;
                gpj_out.yz += dene * gpj_from_ene.yz;
                gpj_out.zz += dene * gpj_from_ene.zz;
            }
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
                mpi_expand.s = d_epot[i];
                mpj_expand.s = d_epot[j];

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
                pairwise_multipole_kernel_with_grad<scalar_t, RANK, true, false, DO_COORD_GRAD, NEED_FORWARD_GRAD>(
                    mpi, mpj, gpi_ptr, gpj_ptr,
                    drvec[0], drvec[1], drvec[2], dr,
                    damps,
                    nullptr, dr_g_ptr,
                    interaction_tensor_ptr
                );

                if constexpr (DO_COORD_GRAD) {
                    dr_g_buf[0] *= dene;
                    dr_g_buf[1] *= dene;
                    dr_g_buf[2] *= dene;
                }
                if constexpr (DO_COORD_GRAD || DO_MPOLE_GRAD) {
                    pairwise_multipole_backward_add_contribution_from_field<scalar_t, RANK, true, true, DO_COORD_GRAD, DO_MPOLE_GRAD>(
                        mpi_expand, mpj_expand, interaction_tensor_ptr, gpi_from_ene, gpj_from_ene,
                        (DO_MPOLE_GRAD) ? &gpi_out : nullptr,
                        (DO_MPOLE_GRAD) ? &gpj_out : nullptr,
                        (DO_COORD_GRAD) ? dr_g_buf : nullptr
                    );
                }

                if constexpr (DO_MPOLE_GRAD) {
                    gpi_out.s  += dene * gpi_from_ene.s;
                    gpj_out.s  += dene * gpj_from_ene.s;
                    if constexpr (RANK >= 1) {
                        gpi_out.x  += dene * gpi_from_ene.x;
                        gpi_out.y  += dene * gpi_from_ene.y;
                        gpi_out.z  += dene * gpi_from_ene.z;
                        gpj_out.x  += dene * gpj_from_ene.x;
                        gpj_out.y  += dene * gpj_from_ene.y;
                        gpj_out.z  += dene * gpj_from_ene.z;
                    }
                    if constexpr (RANK >= 2) {
                        gpi_out.xx += dene * gpi_from_ene.xx;
                        gpi_out.xy += dene * gpi_from_ene.xy;
                        gpi_out.xz += dene * gpi_from_ene.xz;
                        gpi_out.yy += dene * gpi_from_ene.yy;
                        gpi_out.yz += dene * gpi_from_ene.yz;
                        gpi_out.zz += dene * gpi_from_ene.zz;
                        gpj_out.xx += dene * gpj_from_ene.xx;
                        gpj_out.xy += dene * gpj_from_ene.xy;
                        gpj_out.xz += dene * gpj_from_ene.xz;
                        gpj_out.yy += dene * gpj_from_ene.yy;
                        gpj_out.yz += dene * gpj_from_ene.yz;
                        gpj_out.zz += dene * gpj_from_ene.zz;
                    }
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


#endif