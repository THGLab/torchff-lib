#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"
#include "common/reduce.cuh"
#include "multipoles.cuh"
#include "damps.cuh"
#include "ewald/damps.cuh"
#include "common/switch.cuh"


template <typename scalar_t, int BLOCK_SIZE>
__global__ void cmm_elec_from_pairs_forward_kernel(
    scalar_t* coords,
    scalar_t* g_box,
    int64_t* pairs,
    int64_t* pairs_excl,
    scalar_t* multipoles,
    scalar_t* Z, scalar_t* b_elec_ij, scalar_t* b_elec,
    scalar_t ewald_alpha,
    scalar_t rcut_sr,
    scalar_t rcut_lr,
    scalar_t rcut_switch_buf,
    int64_t npairs,
    int64_t npairs_excl,
    scalar_t* ene_out,
    scalar_t* epot,
    scalar_t* efield
)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ene_out[0] = scalar_t(0.0);
    }

    __shared__ scalar_t box[9];
    __shared__ scalar_t box_inv[9];
    if (threadIdx.x < 9) {
        box[threadIdx.x] = g_box[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        invert_box_3x3<scalar_t>(box, box_inv);
    }
    __syncthreads();

    constexpr scalar_t ZERO = scalar_t(0.0);
    constexpr scalar_t ONE  = scalar_t(1.0);

    scalar_t ene = scalar_t(0.0);

    int64_t start = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    for (int64_t index = start; index < npairs; index += gridDim.x * BLOCK_SIZE) {
        int64_t i = pairs[index * 2];
        int64_t j = pairs[index * 2 + 1];
        if ( i < 0 || j < 0 ) {
            continue;
        }
        scalar_t e = scalar_t(0.0);
        scalar_t etmp = scalar_t(0.0);
        scalar_t mi[10]; 
        scalar_t mj[10]; 

        #pragma unroll
        for (int n = 0; n < 10; n++) {
            mi[n] = multipoles[i*10+n];
            mj[n] = multipoles[j*10+n];
        }

        scalar_t mi_grad[10] = {};
        scalar_t mj_grad[10] = {};

        scalar_t mi_grad_tmp[10];
        scalar_t mj_grad_tmp[10];
        scalar_t drx_grad_tmp ;
        scalar_t dry_grad_tmp ;
        scalar_t drz_grad_tmp ;

        scalar_t damps[6];
        scalar_t zi = Z[i];
        scalar_t zj = Z[j];

        scalar_t rij[3], tmp_vec[3];
        diff_vec3(&coords[j*3], &coords[i*3], tmp_vec);
        apply_pbc_triclinic(tmp_vec, box, box_inv, rij);
        scalar_t drx = rij[0];
        scalar_t dry = rij[1];
        scalar_t drz = rij[2];
        scalar_t dr = sqrt_(drx*drx+dry*dry+drz*drz);

        scalar_t epot_i = scalar_t(0.0);
        scalar_t epot_j = scalar_t(0.0);
        scalar_t efield_i[3] = {};
        scalar_t efield_j[3] = {};

        if ( dr < rcut_sr ) {
            // shell-shell
            two_center_damps(dr, b_elec_ij[index], damps);
            pairwise_multipole_kernel_with_grad(
                mi[0]-zi, mi[1], mi[2], mi[3], mi[4], mi[5], mi[6], mi[7], mi[8], mi[9],
                mj[0]-zj, mj[1], mj[2], mj[3], mj[4], mj[5], mj[6], mj[7], mj[8], mj[9],
                drx, dry, drz,
                -damps[0], -damps[1], -damps[2], -damps[3], -damps[4], -damps[5],
                &etmp,
                mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
                mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9
            );
            e += etmp;

            /* core-j shell-i */
            one_center_damps(dr, b_elec[i], damps);
            pairwise_multipole_kernel_with_grad(
                mi[0]-zi, mi[1], mi[2], mi[3], mi[4], mi[5], mi[6], mi[7], mi[8], mi[9],
                zj, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,
                drx, dry, drz,
                -damps[0], -damps[1], -damps[2], -damps[3], -damps[4], -damps[5],
                &etmp,
                mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
                mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9
            );
            e += etmp; 
            epot_j += mj_grad_tmp[0];
            efield_j[0] -= mj_grad_tmp[1];
            efield_j[1] -= mj_grad_tmp[2];
            efield_j[2] -= mj_grad_tmp[3];
            
            /* core-i shell-j */
            one_center_damps(dr, b_elec[j], damps);
            pairwise_multipole_kernel_with_grad(
                zi, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,
                mj[0]-zj, mj[1], mj[2], mj[3], mj[4], mj[5], mj[6], mj[7], mj[8], mj[9],
                drx, dry, drz,
                -damps[0], -damps[1], -damps[2], -damps[3], -damps[4], -damps[5],
                &etmp,
                mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
                mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9
            );
            e += etmp;
            epot_i += mi_grad_tmp[0];
            efield_i[0] -= mi_grad_tmp[1];
            efield_i[1] -= mi_grad_tmp[2];
            efield_i[2] -= mi_grad_tmp[3];
            
            // switching functions
            scalar_t s, sg;
            s = clamp_((dr - rcut_sr + rcut_switch_buf) / rcut_switch_buf, ZERO, ONE);
            s = SWITCH(s);
            e = e * s;
            #pragma unroll
            for (int n = 0; n < 10; n++) {
                mi_grad[n] *= s;
                mj_grad[n] *= s;
            }
        }

        if ( dr < rcut_lr ) {
            // Ewald-real space
            ewald_erfc_damps<scalar_t, 11>(dr, ewald_alpha, damps);
            pairwise_multipole_kernel_with_grad(
                mi[0], mi[1], mi[2], mi[3], mi[4], mi[5], mi[6], mi[7], mi[8], mi[9],
                mj[0], mj[1], mj[2], mj[3], mj[4], mj[5], mj[6], mj[7], mj[8], mj[9],
                drx, dry, drz,
                damps[0], damps[1], damps[2], damps[3], damps[4], damps[5],
                &etmp,
                mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
                mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9
            );
            e += etmp;
            
            epot_i += mi_grad_tmp[0];
            epot_j += mj_grad_tmp[0];

            efield_i[0] -= mi_grad_tmp[1];
            efield_i[1] -= mi_grad_tmp[2];
            efield_i[2] -= mi_grad_tmp[3];

            efield_j[0] -= mj_grad_tmp[1];
            efield_j[1] -= mj_grad_tmp[2];
            efield_j[2] -= mj_grad_tmp[3];
            
        }

        ene += e;

        atomicAdd(&epot[i], epot_i);
        atomicAdd(&epot[j], epot_j);

        atomicAdd(&efield[i*3], efield_i[0]);
        atomicAdd(&efield[i*3+1], efield_i[1]);
        atomicAdd(&efield[i*3+2], efield_i[2]);

        atomicAdd(&efield[j*3], efield_j[0]);
        atomicAdd(&efield[j*3+1], efield_j[1]);
        atomicAdd(&efield[j*3+2], efield_j[2]);
    }

    for (int64_t index = start; index < npairs_excl; index += gridDim.x * BLOCK_SIZE) {
        int64_t i = pairs_excl[index * 2];
        int64_t j = pairs_excl[index * 2 + 1];

        scalar_t e = scalar_t(0.0);
        scalar_t mi[10]; 
        scalar_t mj[10]; 

        #pragma unroll
        for (int n = 0; n < 10; n++) {
            mi[n] = multipoles[i*10+n];
            mj[n] = multipoles[j*10+n];
        }

        scalar_t mi_grad[10] = {};
        scalar_t mj_grad[10] = {};

        scalar_t damps[6];

        scalar_t rij[3], tmp_vec[3];
        diff_vec3(&coords[j*3], &coords[i*3], tmp_vec);
        apply_pbc_triclinic(tmp_vec, box, box_inv, rij);
        scalar_t drx = rij[0];
        scalar_t dry = rij[1];
        scalar_t drz = rij[2];
        scalar_t dr = sqrt_(drx*drx+dry*dry+drz*drz);

        ewald_erfc_damps<scalar_t, 11>(dr, ewald_alpha, damps);
        pairwise_multipole_kernel_with_grad(
            mi[0], mi[1], mi[2], mi[3], mi[4], mi[5], mi[6], mi[7], mi[8], mi[9],
            mj[0], mj[1], mj[2], mj[3], mj[4], mj[5], mj[6], mj[7], mj[8], mj[9],
            drx, dry, drz,
            damps[0]-ONE, damps[1]-ONE, damps[2]-ONE, damps[3]-ONE, damps[4]-ONE, damps[5]-ONE,
            &e,
            mi_grad, mi_grad+1, mi_grad+2, mi_grad+3, mi_grad+4, mi_grad+5, mi_grad+6, mi_grad+7, mi_grad+8, mi_grad+9,
            mj_grad, mj_grad+1, mj_grad+2, mj_grad+3, mj_grad+4, mj_grad+5, mj_grad+6, mj_grad+7, mj_grad+8, mj_grad+9
        );

        ene += e;

        atomicAdd(&epot[i], mi_grad[0]);
        atomicAdd(&epot[j], mj_grad[0]);

        atomicAdd(&efield[i*3], -mi_grad[1]);
        atomicAdd(&efield[i*3+1], -mi_grad[2]);
        atomicAdd(&efield[i*3+2], -mi_grad[3]);

        atomicAdd(&efield[j*3], -mj_grad[1]);
        atomicAdd(&efield[j*3+1], -mj_grad[2]);
        atomicAdd(&efield[j*3+2], -mj_grad[3]);
    }

    block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
}


template <typename scalar_t, int BLOCK_SIZE>
__global__ void cmm_elec_from_pairs_backward_kernel(
    scalar_t* coords,
    scalar_t* g_box,
    int64_t* pairs,
    int64_t* pairs_excl,
    scalar_t* multipoles,
    scalar_t* Z, scalar_t* b_elec_ij, scalar_t* b_elec,
    scalar_t ewald_alpha,
    scalar_t rcut_sr,
    scalar_t rcut_lr,
    scalar_t rcut_switch_buf,
    int64_t npairs,
    int64_t npairs_excl,
    scalar_t* ene_grad,
    scalar_t* epot_grad,
    scalar_t* efield_grad,
    scalar_t* coords_grad,
    scalar_t* multipoles_grad
)
{
    __shared__ scalar_t box[9];
    __shared__ scalar_t box_inv[9];
    if (threadIdx.x < 9) {
        box[threadIdx.x] = g_box[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        invert_box_3x3<scalar_t>(box, box_inv);
    }
    __syncthreads();

    constexpr scalar_t ZERO = scalar_t(0.0);
    constexpr scalar_t ONE  = scalar_t(1.0);
    
    // interaction tensor
    scalar_t interaction_tensor[35] = {};

    scalar_t& drinv = interaction_tensor[0];

    scalar_t& tx   = interaction_tensor[1];
    scalar_t& ty   = interaction_tensor[2];
    scalar_t& tz   = interaction_tensor[3];
    
    scalar_t& txx  = interaction_tensor[4];
    scalar_t& txy  = interaction_tensor[5];
    scalar_t& txz  = interaction_tensor[6];
    scalar_t& tyy  = interaction_tensor[7];
    scalar_t& tyz  = interaction_tensor[8];
    scalar_t& tzz  = interaction_tensor[9];
    
    scalar_t& txxx = interaction_tensor[10];
    scalar_t& txxy = interaction_tensor[11];
    scalar_t& txxz = interaction_tensor[12];
    scalar_t& tyyy = interaction_tensor[13];
    scalar_t& tyyx = interaction_tensor[14];
    scalar_t& tyyz = interaction_tensor[15];
    scalar_t& tzzz = interaction_tensor[16];
    scalar_t& tzzx = interaction_tensor[17];
    scalar_t& tzzy = interaction_tensor[18];
    scalar_t& txyz = interaction_tensor[19];

    scalar_t& txxxx = interaction_tensor[20];
    scalar_t& txxxy = interaction_tensor[21];
    scalar_t& txxxz = interaction_tensor[22];
    scalar_t& txxyy = interaction_tensor[23];
    scalar_t& txxzz = interaction_tensor[24];
    scalar_t& txxyz = interaction_tensor[25];
    scalar_t& tyyyy = interaction_tensor[26];
    scalar_t& tyyyx = interaction_tensor[27];
    scalar_t& tyyyz = interaction_tensor[28];
    scalar_t& tyyzz = interaction_tensor[29];
    scalar_t& tyyxz = interaction_tensor[30];
    scalar_t& tzzzz = interaction_tensor[31];
    scalar_t& tzzzx = interaction_tensor[32];
    scalar_t& tzzzy = interaction_tensor[33];
    scalar_t& tzzxy = interaction_tensor[34];

    scalar_t mi_grad[10] ;
    scalar_t mj_grad[10] ;
    scalar_t drx_grad ;
    scalar_t dry_grad ;
    scalar_t drz_grad ;

    scalar_t mi_grad_tmp[10];
    scalar_t mj_grad_tmp[10];
    scalar_t drx_grad_tmp ;
    scalar_t dry_grad_tmp ;
    scalar_t drz_grad_tmp ;

    scalar_t damps[6];

    scalar_t e_grad = ene_grad[0];
    scalar_t tmp;
    scalar_t mi[10]; 
    scalar_t mj[10]; 

    int64_t start = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    for (int64_t index = start; index < npairs; index += gridDim.x * BLOCK_SIZE) {
        int64_t i = pairs[index * 2];
        int64_t j = pairs[index * 2 + 1];
        if ( i < 0 || j < 0 ) {
            continue;
        }
        
        #pragma unroll
        for (int n = 0; n < 10; n++) {
            mi[n] = multipoles[i*10+n];
            mj[n] = multipoles[j*10+n];
            mi_grad[n] = ZERO;
            mj_grad[n] = ZERO;
        }
        drx_grad = ZERO; dry_grad = ZERO; drz_grad = ZERO;

        scalar_t zi = Z[i];
        scalar_t zj = Z[j];

        scalar_t rij[3], tmp_vec[3];
        diff_vec3(&coords[j*3], &coords[i*3], tmp_vec);
        apply_pbc_triclinic(tmp_vec, box, box_inv, rij);
        scalar_t drx = rij[0];
        scalar_t dry = rij[1];
        scalar_t drz = rij[2];
        scalar_t dr = sqrt_(drx*drx+dry*dry+drz*drz);

        scalar_t epot_grad_i = epot_grad[i];
        scalar_t epot_grad_j = epot_grad[j];
        scalar_t efield_x_grad_i = efield_grad[i*3];
        scalar_t efield_y_grad_i = efield_grad[i*3+1];
        scalar_t efield_z_grad_i = efield_grad[i*3+2];
        scalar_t efield_x_grad_j = efield_grad[j*3];
        scalar_t efield_y_grad_j = efield_grad[j*3+1];
        scalar_t efield_z_grad_j = efield_grad[j*3+2];

        if ( dr < rcut_sr ) {
            // switching functions
            scalar_t s, sg;
            s = clamp_((dr - rcut_sr + rcut_switch_buf) / rcut_switch_buf, ZERO, ONE);
            sg = SWITCH_GRAD(s) / rcut_switch_buf / dr;
            s = SWITCH(s);

            // shell-shell
            two_center_damps(dr, b_elec_ij[index], damps);
            pairwise_multipole_kernel_with_grad(
                mi[0]-zi, mi[1], mi[2], mi[3], mi[4], mi[5], mi[6], mi[7], mi[8], mi[9],
                mj[0]-zj, mj[1], mj[2], mj[3], mj[4], mj[5], mj[6], mj[7], mj[8], mj[9],
                drx, dry, drz,
                -damps[0], -damps[1], -damps[2], -damps[3], -damps[4], -damps[5],
                &tmp,
                mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
                mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9,
                &drx_grad_tmp, &dry_grad_tmp, &drz_grad_tmp
            );

            #pragma unroll
            for (int n = 0; n < 10; n++) {
                mi_grad[n] += e_grad*mi_grad_tmp[n]*s;
                mj_grad[n] += e_grad*mj_grad_tmp[n]*s;
            }
            drx_grad += e_grad * (drx_grad_tmp * s + tmp * sg * drx);
            dry_grad += e_grad * (dry_grad_tmp * s + tmp * sg * dry);
            drz_grad += e_grad * (drz_grad_tmp * s + tmp * sg * drz);

            /* core-j shell-i */
            one_center_damps(dr, b_elec[i], damps);
            pairwise_multipole_kernel_with_grad(
                mi[0]-zi, mi[1], mi[2], mi[3], mi[4], mi[5], mi[6], mi[7], mi[8], mi[9],
                zj, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,
                drx, dry, drz,
                -damps[0], -damps[1], -damps[2], -damps[3], -damps[4], -damps[5],
                &tmp,
                mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
                mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9,
                &drx_grad_tmp, &dry_grad_tmp, &drz_grad_tmp,
                interaction_tensor
            );

            // multipole gradient
            mi_grad[0] += e_grad*mi_grad_tmp[0]*s + epot_grad_j*drinv - efield_x_grad_j*tx - efield_y_grad_j*ty - efield_z_grad_j*tz;
            mi_grad[1] += e_grad*mi_grad_tmp[1]*s - epot_grad_j*tx + efield_x_grad_j*txx + efield_y_grad_j*txy + efield_z_grad_j*txz;
            mi_grad[2] += e_grad*mi_grad_tmp[2]*s - epot_grad_j*ty + efield_x_grad_j*txy + efield_y_grad_j*tyy + efield_z_grad_j*tyz;
            mi_grad[3] += e_grad*mi_grad_tmp[3]*s - epot_grad_j*tz + efield_x_grad_j*txz + efield_y_grad_j*tyz + efield_z_grad_j*tzz;
            mi_grad[4] += e_grad*mi_grad_tmp[4]*s + epot_grad_j*txx - efield_x_grad_j*txxx - efield_y_grad_j*txxy - efield_z_grad_j*txxz;
            mi_grad[5] += e_grad*mi_grad_tmp[5]*s + epot_grad_j*txy - efield_x_grad_j*txxy - efield_y_grad_j*tyyx - efield_z_grad_j*txyz;
            mi_grad[6] += e_grad*mi_grad_tmp[6]*s + epot_grad_j*txz - efield_x_grad_j*txxz - efield_y_grad_j*txyz - efield_z_grad_j*tzzx;
            mi_grad[7] += e_grad*mi_grad_tmp[7]*s + epot_grad_j*tyy - efield_x_grad_j*tyyx - efield_y_grad_j*tyyy - efield_z_grad_j*tyyz;
            mi_grad[8] += e_grad*mi_grad_tmp[8]*s + epot_grad_j*tyz - efield_x_grad_j*txyz - efield_y_grad_j*tyyz - efield_z_grad_j*tzzy;
            mi_grad[9] += e_grad*mi_grad_tmp[9]*s + epot_grad_j*tzz - efield_x_grad_j*tzzx - efield_y_grad_j*tzzy - efield_z_grad_j*tzzz;

            // coordinate gradient - contribution from energy
            drx_grad += e_grad * (drx_grad_tmp * s + tmp * sg * drx);
            dry_grad += e_grad * (dry_grad_tmp * s + tmp * sg * dry);
            drz_grad += e_grad * (drz_grad_tmp * s + tmp * sg * drz);

            // coordinate gradient - contribution from electric potential
            drx_grad += mj_grad_tmp[1]*epot_grad_j;
            dry_grad += mj_grad_tmp[2]*epot_grad_j;
            drz_grad += mj_grad_tmp[3]*epot_grad_j;

            // coordinate gradient - contribution from electric field
            drx_grad += -mj_grad_tmp[4]*efield_x_grad_j -mj_grad_tmp[5]*efield_y_grad_j -mj_grad_tmp[6]*efield_z_grad_j;
            dry_grad += -mj_grad_tmp[5]*efield_x_grad_j -mj_grad_tmp[7]*efield_y_grad_j -mj_grad_tmp[8]*efield_z_grad_j;
            drz_grad += -mj_grad_tmp[6]*efield_x_grad_j -mj_grad_tmp[8]*efield_y_grad_j -mj_grad_tmp[9]*efield_z_grad_j;
            
            /* core-i shell-j */
            one_center_damps(dr, b_elec[j], damps);
            pairwise_multipole_kernel_with_grad(
                zi, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,
                mj[0]-zj, mj[1], mj[2], mj[3], mj[4], mj[5], mj[6], mj[7], mj[8], mj[9],
                drx, dry, drz,
                -damps[0], -damps[1], -damps[2], -damps[3], -damps[4], -damps[5],
                &tmp,
                mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
                mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9,
                &drx_grad_tmp, &dry_grad_tmp, &drz_grad_tmp,
                interaction_tensor
            );

            // multipole gradient
            mj_grad[0] += e_grad*mj_grad_tmp[0]*s + epot_grad_i*drinv + efield_x_grad_i*tx + efield_y_grad_i*ty + efield_z_grad_i*tz;
            mj_grad[1] += e_grad*mj_grad_tmp[1]*s + epot_grad_i*tx + efield_x_grad_i*txx + efield_y_grad_i*txy + efield_z_grad_i*txz;
            mj_grad[2] += e_grad*mj_grad_tmp[2]*s + epot_grad_i*ty + efield_x_grad_i*txy + efield_y_grad_i*tyy + efield_z_grad_i*tyz;
            mj_grad[3] += e_grad*mj_grad_tmp[3]*s + epot_grad_i*tz + efield_x_grad_i*txz + efield_y_grad_i*tyz + efield_z_grad_i*tzz;
            mj_grad[4] += e_grad*mj_grad_tmp[4]*s + epot_grad_i*txx + efield_x_grad_i*txxx + efield_y_grad_i*txxy + efield_z_grad_i*txxz;
            mj_grad[5] += e_grad*mj_grad_tmp[5]*s + epot_grad_i*txy + efield_x_grad_i*txxy + efield_y_grad_i*tyyx + efield_z_grad_i*txyz;
            mj_grad[6] += e_grad*mj_grad_tmp[6]*s + epot_grad_i*txz + efield_x_grad_i*txxz + efield_y_grad_i*txyz + efield_z_grad_i*tzzx;
            mj_grad[7] += e_grad*mj_grad_tmp[7]*s + epot_grad_i*tyy + efield_x_grad_i*tyyx + efield_y_grad_i*tyyy + efield_z_grad_i*tyyz;
            mj_grad[8] += e_grad*mj_grad_tmp[8]*s + epot_grad_i*tyz + efield_x_grad_i*txyz + efield_y_grad_i*tyyz + efield_z_grad_i*tzzy;
            mj_grad[9] += e_grad*mj_grad_tmp[9]*s + epot_grad_i*tzz + efield_x_grad_i*tzzx + efield_y_grad_i*tzzy + efield_z_grad_i*tzzz;

            // coordinate gradient - contribution from energy
            drx_grad += e_grad * (drx_grad_tmp * s + tmp * sg * drx);
            dry_grad += e_grad * (dry_grad_tmp * s + tmp * sg * dry);
            drz_grad += e_grad * (drz_grad_tmp * s + tmp * sg * drz);
            
            // coordinate gradient - contribution from electric potential
            drx_grad += -mi_grad_tmp[1]*epot_grad_i ;
            dry_grad += -mi_grad_tmp[2]*epot_grad_i ;
            drz_grad += -mi_grad_tmp[3]*epot_grad_i ;

            // coordinate gradient - contribution from electric field
            drx_grad += mi_grad_tmp[4]*efield_x_grad_i + mi_grad_tmp[5]*efield_y_grad_i + mi_grad_tmp[6]*efield_z_grad_i;
            dry_grad += mi_grad_tmp[5]*efield_x_grad_i + mi_grad_tmp[7]*efield_y_grad_i + mi_grad_tmp[8]*efield_z_grad_i;
            drz_grad += mi_grad_tmp[6]*efield_x_grad_i + mi_grad_tmp[8]*efield_y_grad_i + mi_grad_tmp[9]*efield_z_grad_i;
        }
        
        if ( dr < rcut_lr ) {
            // Ewald-real space
            ewald_erfc_damps<scalar_t, 11>(dr, ewald_alpha, damps);
            pairwise_multipole_kernel_with_grad(
                mi[0], mi[1], mi[2], mi[3], mi[4], mi[5], mi[6], mi[7], mi[8], mi[9],
                mj[0], mj[1], mj[2], mj[3], mj[4], mj[5], mj[6], mj[7], mj[8], mj[9],
                drx, dry, drz,
                damps[0], damps[1], damps[2], damps[3], damps[4], damps[5],
                &tmp,
                mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
                mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9,
                &drx_grad_tmp, &dry_grad_tmp, &drz_grad_tmp,
                interaction_tensor
            );
            
            // multipole gradient
            mi_grad[0] += e_grad*mi_grad_tmp[0] + epot_grad_j*drinv - efield_x_grad_j*tx - efield_y_grad_j*ty - efield_z_grad_j*tz;
            mi_grad[1] += e_grad*mi_grad_tmp[1] - epot_grad_j*tx + efield_x_grad_j*txx + efield_y_grad_j*txy + efield_z_grad_j*txz;
            mi_grad[2] += e_grad*mi_grad_tmp[2] - epot_grad_j*ty + efield_x_grad_j*txy + efield_y_grad_j*tyy + efield_z_grad_j*tyz;
            mi_grad[3] += e_grad*mi_grad_tmp[3] - epot_grad_j*tz + efield_x_grad_j*txz + efield_y_grad_j*tyz + efield_z_grad_j*tzz;
            mi_grad[4] += e_grad*mi_grad_tmp[4] + epot_grad_j*txx - efield_x_grad_j*txxx - efield_y_grad_j*txxy - efield_z_grad_j*txxz;
            mi_grad[5] += e_grad*mi_grad_tmp[5] + epot_grad_j*txy - efield_x_grad_j*txxy - efield_y_grad_j*tyyx - efield_z_grad_j*txyz;
            mi_grad[6] += e_grad*mi_grad_tmp[6] + epot_grad_j*txz - efield_x_grad_j*txxz - efield_y_grad_j*txyz - efield_z_grad_j*tzzx;
            mi_grad[7] += e_grad*mi_grad_tmp[7] + epot_grad_j*tyy - efield_x_grad_j*tyyx - efield_y_grad_j*tyyy - efield_z_grad_j*tyyz;
            mi_grad[8] += e_grad*mi_grad_tmp[8] + epot_grad_j*tyz - efield_x_grad_j*txyz - efield_y_grad_j*tyyz - efield_z_grad_j*tzzy;
            mi_grad[9] += e_grad*mi_grad_tmp[9] + epot_grad_j*tzz - efield_x_grad_j*tzzx - efield_y_grad_j*tzzy - efield_z_grad_j*tzzz;

            mj_grad[0] += e_grad*mj_grad_tmp[0] + epot_grad_i*drinv + efield_x_grad_i*tx + efield_y_grad_i*ty + efield_z_grad_i*tz;
            mj_grad[1] += e_grad*mj_grad_tmp[1] + epot_grad_i*tx + efield_x_grad_i*txx + efield_y_grad_i*txy + efield_z_grad_i*txz;
            mj_grad[2] += e_grad*mj_grad_tmp[2] + epot_grad_i*ty + efield_x_grad_i*txy + efield_y_grad_i*tyy + efield_z_grad_i*tyz;
            mj_grad[3] += e_grad*mj_grad_tmp[3] + epot_grad_i*tz + efield_x_grad_i*txz + efield_y_grad_i*tyz + efield_z_grad_i*tzz;
            mj_grad[4] += e_grad*mj_grad_tmp[4] + epot_grad_i*txx + efield_x_grad_i*txxx + efield_y_grad_i*txxy + efield_z_grad_i*txxz;
            mj_grad[5] += e_grad*mj_grad_tmp[5] + epot_grad_i*txy + efield_x_grad_i*txxy + efield_y_grad_i*tyyx + efield_z_grad_i*txyz;
            mj_grad[6] += e_grad*mj_grad_tmp[6] + epot_grad_i*txz + efield_x_grad_i*txxz + efield_y_grad_i*txyz + efield_z_grad_i*tzzx;
            mj_grad[7] += e_grad*mj_grad_tmp[7] + epot_grad_i*tyy + efield_x_grad_i*tyyx + efield_y_grad_i*tyyy + efield_z_grad_i*tyyz;
            mj_grad[8] += e_grad*mj_grad_tmp[8] + epot_grad_i*tyz + efield_x_grad_i*txyz + efield_y_grad_i*tyyz + efield_z_grad_i*tzzy;
            mj_grad[9] += e_grad*mj_grad_tmp[9] + epot_grad_i*tzz + efield_x_grad_i*tzzx + efield_y_grad_i*tzzy + efield_z_grad_i*tzzz;

            // coordinate gradient - contribution from energy
            drx_grad += e_grad * drx_grad_tmp;
            dry_grad += e_grad * dry_grad_tmp;
            drz_grad += e_grad * drz_grad_tmp;

            // coordinate gradient - contribution from electric potential
            drx_grad += -mi_grad_tmp[1]*epot_grad_i + mj_grad_tmp[1]*epot_grad_j;
            dry_grad += -mi_grad_tmp[2]*epot_grad_i + mj_grad_tmp[2]*epot_grad_j;
            drz_grad += -mi_grad_tmp[3]*epot_grad_i + mj_grad_tmp[3]*epot_grad_j;

            // coordinate gradient - contribution from electric field
            drx_grad += mi_grad_tmp[4]*efield_x_grad_i-mj_grad_tmp[4]*efield_x_grad_j + mi_grad_tmp[5]*efield_y_grad_i-mj_grad_tmp[5]*efield_y_grad_j + mi_grad_tmp[6]*efield_z_grad_i-mj_grad_tmp[6]*efield_z_grad_j;
            dry_grad += mi_grad_tmp[5]*efield_x_grad_i-mj_grad_tmp[5]*efield_x_grad_j + mi_grad_tmp[7]*efield_y_grad_i-mj_grad_tmp[7]*efield_y_grad_j + mi_grad_tmp[8]*efield_z_grad_i-mj_grad_tmp[8]*efield_z_grad_j;
            drz_grad += mi_grad_tmp[6]*efield_x_grad_i-mj_grad_tmp[6]*efield_x_grad_j + mi_grad_tmp[8]*efield_y_grad_i-mj_grad_tmp[8]*efield_y_grad_j + mi_grad_tmp[9]*efield_z_grad_i-mj_grad_tmp[9]*efield_z_grad_j;
        }

        // dr = coords[j] - coords[i], so grad wrt j is +, wrt i is -
        atomicAdd(&coords_grad[j*3],   drx_grad);
        atomicAdd(&coords_grad[j*3+1], dry_grad);
        atomicAdd(&coords_grad[j*3+2], drz_grad);
        atomicAdd(&coords_grad[i*3],   -drx_grad);
        atomicAdd(&coords_grad[i*3+1], -dry_grad);
        atomicAdd(&coords_grad[i*3+2], -drz_grad);

        #pragma unroll
        for (int n = 0; n < 10; n++) {
            atomicAdd(&multipoles_grad[i*10+n], mi_grad[n]);
            atomicAdd(&multipoles_grad[j*10+n], mj_grad[n]);
        }
    }

    for (int64_t index = start; index < npairs_excl; index += gridDim.x * BLOCK_SIZE) {
        int64_t i = pairs_excl[index * 2];
        int64_t j = pairs_excl[index * 2 + 1];

        #pragma unroll
        for (int n = 0; n < 10; n++) {
            mi[n] = multipoles[i*10+n];
            mj[n] = multipoles[j*10+n];
        }

        scalar_t rij[3], tmp_vec[3];
        diff_vec3(&coords[j*3], &coords[i*3], tmp_vec);
        apply_pbc_triclinic(tmp_vec, box, box_inv, rij);
        scalar_t drx = rij[0];
        scalar_t dry = rij[1];
        scalar_t drz = rij[2];
        scalar_t dr = sqrt_(drx*drx+dry*dry+drz*drz);

        scalar_t epot_grad_i = epot_grad[i];
        scalar_t epot_grad_j = epot_grad[j];
        scalar_t efield_x_grad_i = efield_grad[i*3];
        scalar_t efield_y_grad_i = efield_grad[i*3+1];
        scalar_t efield_z_grad_i = efield_grad[i*3+2];
        scalar_t efield_x_grad_j = efield_grad[j*3];
        scalar_t efield_y_grad_j = efield_grad[j*3+1];
        scalar_t efield_z_grad_j = efield_grad[j*3+2];

        ewald_erfc_damps<scalar_t, 11>(dr, ewald_alpha, damps);
        pairwise_multipole_kernel_with_grad(
            mi[0], mi[1], mi[2], mi[3], mi[4], mi[5], mi[6], mi[7], mi[8], mi[9],
            mj[0], mj[1], mj[2], mj[3], mj[4], mj[5], mj[6], mj[7], mj[8], mj[9],
            drx, dry, drz,
            damps[0]-ONE, damps[1]-ONE, damps[2]-ONE, damps[3]-ONE, damps[4]-ONE, damps[5]-ONE,
            &tmp,
            mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
            mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9,
            &drx_grad_tmp, &dry_grad_tmp, &drz_grad_tmp,
            interaction_tensor
        );

        mi_grad[0] = e_grad*mi_grad_tmp[0] + epot_grad_j*drinv - efield_x_grad_j*tx - efield_y_grad_j*ty - efield_z_grad_j*tz;
        mi_grad[1] = e_grad*mi_grad_tmp[1] - epot_grad_j*tx + efield_x_grad_j*txx + efield_y_grad_j*txy + efield_z_grad_j*txz;
        mi_grad[2] = e_grad*mi_grad_tmp[2] - epot_grad_j*ty + efield_x_grad_j*txy + efield_y_grad_j*tyy + efield_z_grad_j*tyz;
        mi_grad[3] = e_grad*mi_grad_tmp[3] - epot_grad_j*tz + efield_x_grad_j*txz + efield_y_grad_j*tyz + efield_z_grad_j*tzz;
        mi_grad[4] = e_grad*mi_grad_tmp[4] + epot_grad_j*txx - efield_x_grad_j*txxx - efield_y_grad_j*txxy - efield_z_grad_j*txxz;
        mi_grad[5] = e_grad*mi_grad_tmp[5] + epot_grad_j*txy - efield_x_grad_j*txxy - efield_y_grad_j*tyyx - efield_z_grad_j*txyz;
        mi_grad[6] = e_grad*mi_grad_tmp[6] + epot_grad_j*txz - efield_x_grad_j*txxz - efield_y_grad_j*txyz - efield_z_grad_j*tzzx;
        mi_grad[7] = e_grad*mi_grad_tmp[7] + epot_grad_j*tyy - efield_x_grad_j*tyyx - efield_y_grad_j*tyyy - efield_z_grad_j*tyyz;
        mi_grad[8] = e_grad*mi_grad_tmp[8] + epot_grad_j*tyz - efield_x_grad_j*txyz - efield_y_grad_j*tyyz - efield_z_grad_j*tzzy;
        mi_grad[9] = e_grad*mi_grad_tmp[9] + epot_grad_j*tzz - efield_x_grad_j*tzzx - efield_y_grad_j*tzzy - efield_z_grad_j*tzzz;

        mj_grad[0] = e_grad*mj_grad_tmp[0] + epot_grad_i*drinv + efield_x_grad_i*tx + efield_y_grad_i*ty + efield_z_grad_i*tz;
        mj_grad[1] = e_grad*mj_grad_tmp[1] + epot_grad_i*tx + efield_x_grad_i*txx + efield_y_grad_i*txy + efield_z_grad_i*txz;
        mj_grad[2] = e_grad*mj_grad_tmp[2] + epot_grad_i*ty + efield_x_grad_i*txy + efield_y_grad_i*tyy + efield_z_grad_i*tyz;
        mj_grad[3] = e_grad*mj_grad_tmp[3] + epot_grad_i*tz + efield_x_grad_i*txz + efield_y_grad_i*tyz + efield_z_grad_i*tzz;
        mj_grad[4] = e_grad*mj_grad_tmp[4] + epot_grad_i*txx + efield_x_grad_i*txxx + efield_y_grad_i*txxy + efield_z_grad_i*txxz;
        mj_grad[5] = e_grad*mj_grad_tmp[5] + epot_grad_i*txy + efield_x_grad_i*txxy + efield_y_grad_i*tyyx + efield_z_grad_i*txyz;
        mj_grad[6] = e_grad*mj_grad_tmp[6] + epot_grad_i*txz + efield_x_grad_i*txxz + efield_y_grad_i*txyz + efield_z_grad_i*tzzx;
        mj_grad[7] = e_grad*mj_grad_tmp[7] + epot_grad_i*tyy + efield_x_grad_i*tyyx + efield_y_grad_i*tyyy + efield_z_grad_i*tyyz;
        mj_grad[8] = e_grad*mj_grad_tmp[8] + epot_grad_i*tyz + efield_x_grad_i*txyz + efield_y_grad_i*tyyz + efield_z_grad_i*tzzy;
        mj_grad[9] = e_grad*mj_grad_tmp[9] + epot_grad_i*tzz + efield_x_grad_i*tzzx + efield_y_grad_i*tzzy + efield_z_grad_i*tzzz;

        drx_grad = e_grad * drx_grad_tmp;
        dry_grad = e_grad * dry_grad_tmp;
        drz_grad = e_grad * drz_grad_tmp;

        drx_grad += -mi_grad_tmp[1]*epot_grad_i + mj_grad_tmp[1]*epot_grad_j;
        dry_grad += -mi_grad_tmp[2]*epot_grad_i + mj_grad_tmp[2]*epot_grad_j;
        drz_grad += -mi_grad_tmp[3]*epot_grad_i + mj_grad_tmp[3]*epot_grad_j;

        drx_grad += mi_grad_tmp[4]*efield_x_grad_i-mj_grad_tmp[4]*efield_x_grad_j + mi_grad_tmp[5]*efield_y_grad_i-mj_grad_tmp[5]*efield_y_grad_j + mi_grad_tmp[6]*efield_z_grad_i-mj_grad_tmp[6]*efield_z_grad_j;
        dry_grad += mi_grad_tmp[5]*efield_x_grad_i-mj_grad_tmp[5]*efield_x_grad_j + mi_grad_tmp[7]*efield_y_grad_i-mj_grad_tmp[7]*efield_y_grad_j + mi_grad_tmp[8]*efield_z_grad_i-mj_grad_tmp[8]*efield_z_grad_j;
        drz_grad += mi_grad_tmp[6]*efield_x_grad_i-mj_grad_tmp[6]*efield_x_grad_j + mi_grad_tmp[8]*efield_y_grad_i-mj_grad_tmp[8]*efield_y_grad_j + mi_grad_tmp[9]*efield_z_grad_i-mj_grad_tmp[9]*efield_z_grad_j;

        atomicAdd(&coords_grad[j*3],   drx_grad);
        atomicAdd(&coords_grad[j*3+1], dry_grad);
        atomicAdd(&coords_grad[j*3+2], drz_grad);
        atomicAdd(&coords_grad[i*3],   -drx_grad);
        atomicAdd(&coords_grad[i*3+1], -dry_grad);
        atomicAdd(&coords_grad[i*3+2], -drz_grad);

        #pragma unroll
        for (int n = 0; n < 10; n++) {
            atomicAdd(&multipoles_grad[i*10+n], mi_grad[n]);
            atomicAdd(&multipoles_grad[j*10+n], mj_grad[n]);
        }
    }
}


class CMMElectrostaticsFromPairsFunctionCuda: public torch::autograd::Function<CMMElectrostaticsFromPairsFunctionCuda> {

public: 

static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& coords, at::Tensor& box,
    at::Tensor& pairs, at::Tensor& pairs_excl,
    at::Tensor& multipoles,
    at::Tensor& Z, at::Tensor& b_elec_ij, at::Tensor& b_elec,
    at::Scalar ewald_alpha,
    at::Scalar rcut_sr,
    at::Scalar rcut_lr,
    at::Scalar rcut_switch_buf
)
{
    int64_t npairs = pairs.size(0);
    int64_t npairs_excl = pairs_excl.size(0);

    auto opts = coords.options();
    at::Tensor ene = at::zeros({}, opts);
    at::Tensor epot = at::zeros({multipoles.size(0)}, opts);
    at::Tensor efield = at::zeros({multipoles.size(0), 3}, opts);

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int BLOCK_SIZE = 128;
    int64_t grid_dim = std::min(
        static_cast<int64_t>(props->maxBlocksPerMultiProcessor * props->multiProcessorCount),
        (npairs + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "cmm_elec_from_pairs_forward_kernel", ([&] {
        cmm_elec_from_pairs_forward_kernel<scalar_t, BLOCK_SIZE><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            box.data_ptr<scalar_t>(),
            pairs.data_ptr<int64_t>(),
            pairs_excl.data_ptr<int64_t>(),
            multipoles.data_ptr<scalar_t>(),
            Z.data_ptr<scalar_t>(),
            b_elec_ij.data_ptr<scalar_t>(),
            b_elec.data_ptr<scalar_t>(),
            static_cast<scalar_t>(ewald_alpha.toDouble()),
            static_cast<scalar_t>(rcut_sr.toDouble()),
            static_cast<scalar_t>(rcut_lr.toDouble()),
            static_cast<scalar_t>(rcut_switch_buf.toDouble()),
            npairs,
            npairs_excl,
            ene.data_ptr<scalar_t>(),
            epot.data_ptr<scalar_t>(),
            efield.data_ptr<scalar_t>()
        );
    }));
    
    ctx->save_for_backward({
        coords, box, pairs, pairs_excl,
        multipoles, Z, b_elec_ij, b_elec
    });
    ctx->saved_data["rcut_sr"] = rcut_sr;
    ctx->saved_data["rcut_lr"] = rcut_lr;
    ctx->saved_data["rcut_switch_buf"] = rcut_switch_buf;
    ctx->saved_data["ewald_alpha"] = ewald_alpha;

    std::vector<at::Tensor> outs;
    outs.reserve(3);
    outs.push_back(ene);
    outs.push_back(epot);
    outs.push_back(efield);

    return outs;
}

static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<at::Tensor> grad_outputs
)
{
    auto saved = ctx->get_saved_variables();

    at::Tensor coords_grad = at::zeros_like(saved[0], saved[0].options());
    at::Tensor multipoles_grad = at::zeros_like(saved[4], saved[4].options());

    int64_t npairs = saved[2].size(0);
    int64_t npairs_excl = saved[3].size(0);

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int BLOCK_SIZE = 128;
    int64_t grid_dim = std::min(
        static_cast<int64_t>(props->maxBlocksPerMultiProcessor * props->multiProcessorCount),
        (npairs + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(saved[0].scalar_type(), "cmm_elec_from_pairs_backward_kernel", ([&] {
        cmm_elec_from_pairs_backward_kernel<scalar_t, BLOCK_SIZE><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
            saved[0].data_ptr<scalar_t>(),
            saved[1].data_ptr<scalar_t>(),
            saved[2].data_ptr<int64_t>(),
            saved[3].data_ptr<int64_t>(),
            saved[4].data_ptr<scalar_t>(),
            saved[5].data_ptr<scalar_t>(),
            saved[6].data_ptr<scalar_t>(),
            saved[7].data_ptr<scalar_t>(),
            static_cast<scalar_t>(ctx->saved_data["ewald_alpha"].toDouble()),
            static_cast<scalar_t>(ctx->saved_data["rcut_sr"].toDouble()),
            static_cast<scalar_t>(ctx->saved_data["rcut_lr"].toDouble()),
            static_cast<scalar_t>(ctx->saved_data["rcut_switch_buf"].toDouble()),
            npairs,
            npairs_excl,
            grad_outputs[0].contiguous().data_ptr<scalar_t>(),
            grad_outputs[1].contiguous().data_ptr<scalar_t>(),
            grad_outputs[2].contiguous().data_ptr<scalar_t>(),
            coords_grad.data_ptr<scalar_t>(),
            multipoles_grad.data_ptr<scalar_t>()
        );
    }));

    at::Tensor ignore;
    return {
        coords_grad, // coords grad
        ignore, // box
        ignore, // pairs
        ignore, // pairs_excl
        multipoles_grad, 
        ignore, ignore, ignore, 
        ignore, ignore, ignore, ignore
    };
}

};


std::tuple<at::Tensor, at::Tensor, at::Tensor> cmm_elec_from_pairs_cuda(
    at::Tensor& coords, at::Tensor& box,
    at::Tensor& pairs, at::Tensor& pairs_excl,
    at::Tensor& multipoles,
    at::Tensor& Z, at::Tensor& b_elec_ij, at::Tensor& b_elec,
    at::Scalar ewald_alpha,
    at::Scalar rcut_sr,
    at::Scalar rcut_lr,
    at::Scalar rcut_switch_buf
){
    
    auto outs = CMMElectrostaticsFromPairsFunctionCuda::apply(
        coords, box,
        pairs, pairs_excl,
        multipoles,
        Z, b_elec_ij, b_elec,
        ewald_alpha, rcut_sr, rcut_lr, rcut_switch_buf
    );
    return {outs[0], outs[1], outs[2]};
}

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("cmm_elec_from_pairs", cmm_elec_from_pairs_cuda);
}
