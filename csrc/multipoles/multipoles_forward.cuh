#ifndef TORCHFF_MULTIPOLES_CUH
#define TORCHFF_MULTIPOLES_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"
#include "storage.cuh"
#include "ewald/damps.cuh"


template <typename scalar_t, int RANK=0, bool USE_DAMPS=true, bool DO_ENERGY=true, bool DO_COORD_GRAD=true, bool DO_MPOLE_GRAD=true>
__device__ __forceinline__ void pairwise_multipole_kernel_with_grad(
    const CartesianExpansion<scalar_t, RANK>& mpi,
    const CartesianExpansion<scalar_t, RANK>& mpj,
    CartesianExpansion<scalar_t, RANK>* gpi_ptr,
    CartesianExpansion<scalar_t, RANK>* gpj_ptr,
    scalar_t drx, scalar_t dry, scalar_t drz, scalar_t dr,
    scalar_t* damps, // 1,3,5,7,9,11
    scalar_t* ene,
    scalar_t* dr_g,
    scalar_t* interaction_tensor
)
{

    scalar_t drinvs[RANK * 2 + 2];
    drinvs[0] = scalar_t(1.0) / dr;
    scalar_t drinv2 = drinvs[0] * drinvs[0];

    #pragma unroll
    for (int i = 1; i < RANK * 2 + 2; i++) {
        drinvs[i] = drinvs[i-1] * drinv2;
    }

    if ( damps ) {
        #pragma unroll
        for (int i = 0; i < RANK * 2 + 2; i++) {
            drinvs[i] *= damps[i];
        }
    }

    scalar_t& drinv = drinvs[0];
    scalar_t& drinv3 = drinvs[1];

    scalar_t tx = -drx * drinv3;
    scalar_t ty = -dry * drinv3;
    scalar_t tz = -drz * drinv3;

    // energy
    scalar_t c0prod = mpi.s * mpj.s;
    if constexpr ( DO_ENERGY ) { (*ene) += c0prod * drinv; }

    // charge gradient;
    if constexpr ( DO_MPOLE_GRAD ) {
        gpi_ptr->s = drinv * mpj.s;
        gpj_ptr->s = drinv * mpi.s;
    }

    if constexpr ( DO_COORD_GRAD ) {
        dr_g[0] = c0prod * tx;
        dr_g[1] = c0prod * ty;
        dr_g[2] = c0prod * tz;
    }

    if ( interaction_tensor ) {
        interaction_tensor[0] = drinv;
        interaction_tensor[1] = tx; interaction_tensor[2] = ty; interaction_tensor[3] = tz;
    }

    if constexpr (RANK >= 1) {
        scalar_t& drinv5 = drinvs[2];
        scalar_t& drinv7 = drinvs[3];

        scalar_t x2 = drx * drx;
        scalar_t xy = drx * dry;
        scalar_t xz = drx * drz;
        scalar_t y2 = dry * dry;
        scalar_t yz = dry * drz;
        scalar_t z2 = drz * drz;
        scalar_t xyz = drx * dry * drz;

        scalar_t txx = 3 * x2 * drinv5 - drinv3;
        scalar_t txy = 3 * xy * drinv5;
        scalar_t txz = 3 * xz * drinv5;
        scalar_t tyy = 3 * y2 * drinv5 - drinv3;
        scalar_t tyz = 3 * yz * drinv5;
        scalar_t tzz = 3 * z2 * drinv5 - drinv3;

        scalar_t txxx = -15 * x2 * drx * drinv7 + 9 * drx * drinv5;
        scalar_t txxy = -15 * x2 * dry * drinv7 + 3 * dry * drinv5;
        scalar_t txxz = -15 * x2 * drz * drinv7 + 3 * drz * drinv5;
        scalar_t tyyy = -15 * y2 * dry * drinv7 + 9 * dry * drinv5;
        scalar_t tyyx = -15 * y2 * drx * drinv7 + 3 * drx * drinv5;
        scalar_t tyyz = -15 * y2 * drz * drinv7 + 3 * drz * drinv5;
        scalar_t tzzz = -15 * z2 * drz * drinv7 + 9 * drz * drinv5;
        scalar_t tzzx = -15 * z2 * drx * drinv7 + 3 * drx * drinv5;
        scalar_t tzzy = -15 * z2 * dry * drinv7 + 3 * dry * drinv5;
        scalar_t txyz = -15 * xyz * drinv7;

        if ( interaction_tensor ) {
            interaction_tensor[4] = txx; interaction_tensor[5] = txy; interaction_tensor[6] = txz;
            interaction_tensor[7] = tyy; interaction_tensor[8] = tyz; interaction_tensor[9] = tzz;
            interaction_tensor[10] = txxx; interaction_tensor[11] = txxy; interaction_tensor[12] = txxz;
            interaction_tensor[13] = tyyy; interaction_tensor[14] = tyyx; interaction_tensor[15] = tyyz;
            interaction_tensor[16] = tzzz; interaction_tensor[17] = tzzx; interaction_tensor[18] = tzzy;
            interaction_tensor[19] = txyz;
        }

        scalar_t tx_g = mpi.s * mpj.x - mpj.s * mpi.x;
        scalar_t ty_g = mpi.s * mpj.y - mpj.s * mpi.y;
        scalar_t tz_g = mpi.s * mpj.z - mpj.s * mpi.z;

        scalar_t txx_g = - mpi.x * mpj.x;
        scalar_t txy_g = - mpi.x * mpj.y - mpj.x * mpi.y;
        scalar_t txz_g = - mpi.x * mpj.z - mpj.x * mpi.z;
        scalar_t tyy_g = - mpi.y * mpj.y;
        scalar_t tyz_g = - mpi.y * mpj.z - mpj.y * mpi.z;
        scalar_t tzz_g = - mpi.z * mpj.z;

        if constexpr (RANK >= 2) {
            txx_g += mpi.s * mpj.xx + mpj.s * mpi.xx;
            txy_g += mpi.s * mpj.xy + mpj.s * mpi.xy;
            txz_g += mpi.s * mpj.xz + mpj.s * mpi.xz;
            tyy_g += mpi.s * mpj.yy + mpj.s * mpi.yy;
            tyz_g += mpi.s * mpj.yz + mpj.s * mpi.yz;
            tzz_g += mpi.s * mpj.zz + mpj.s * mpi.zz;
        }

        // energy (charge-dipole + dipole-dipole / charge-quad)
        if constexpr ( DO_ENERGY ) {
            *ene += tx * tx_g + ty * ty_g + tz * tz_g +
                    txx * txx_g + txy * txy_g + txz * txz_g + tyy * tyy_g + tyz * tyz_g + tzz * tzz_g;

        }

        if constexpr ( DO_MPOLE_GRAD ) {
            // charge gradient - electric potential
            gpi_ptr->s +=  tx * mpj.x + ty * mpj.y + tz * mpj.z;
            gpj_ptr->s += -tx * mpi.x - ty * mpi.y - tz * mpi.z;

            // dipole gradient - electric field
            gpi_ptr->x = -mpj.s * tx - txx * mpj.x - txy * mpj.y - txz * mpj.z;
            gpi_ptr->y = -mpj.s * ty - txy * mpj.x - tyy * mpj.y - tyz * mpj.z;
            gpi_ptr->z = -mpj.s * tz - txz * mpj.x - tyz * mpj.y - tzz * mpj.z;

            gpj_ptr->x = mpi.s * tx - txx * mpi.x - txy * mpi.y - txz * mpi.z;
            gpj_ptr->y = mpi.s * ty - txy * mpi.x - tyy * mpi.y - tyz * mpi.z;
            gpj_ptr->z = mpi.s * tz - txz * mpi.x - tyz * mpi.y - tzz * mpi.z;
        }


        // coordinate gradient - force
        if constexpr ( DO_COORD_GRAD ) {
            dr_g[0] += tx_g * txx + ty_g * txy + tz_g * txz + txxx * txx_g + txxy * txy_g + txxz * txz_g + tyyx * tyy_g + txyz * tyz_g + tzzx * tzz_g;
            dr_g[1] += tx_g * txy + ty_g * tyy + tz_g * tyz + txxy * txx_g + tyyx * txy_g + txyz * txz_g + tyyy * tyy_g + tyyz * tyz_g + tzzy * tzz_g;
            dr_g[2] += tx_g * txz + ty_g * tyz + tz_g * tzz + txxz * txx_g + txyz * txy_g + tzzx * txz_g + tyyz * tyy_g + tzzy * tyz_g + tzzz * tzz_g;
        }

        if constexpr (RANK >= 2) {
            scalar_t& drinv9 = drinvs[4];

            scalar_t txxxx = 105 * x2 * x2 * drinv9 - 90 * x2 * drinv7 + 9 * drinv5;
            scalar_t txxxy = 105 * x2 * xy * drinv9 - 45 * xy * drinv7;
            scalar_t txxxz = 105 * x2 * xz * drinv9 - 45 * xz * drinv7;
            scalar_t txxyy = 105 * x2 * y2 * drinv9 - 15 * (x2 + y2) * drinv7 + 3 * drinv5;
            scalar_t txxzz = 105 * x2 * z2 * drinv9 - 15 * (x2 + z2) * drinv7 + 3 * drinv5;
            scalar_t txxyz = 105 * x2 * yz * drinv9 - 15 * yz * drinv7;

            scalar_t tyyyy = 105 * y2 * y2 * drinv9 - 90 * y2 * drinv7 + 9 * drinv5;
            scalar_t tyyyx = 105 * y2 * xy * drinv9 - 45 * xy * drinv7;
            scalar_t tyyyz = 105 * y2 * yz * drinv9 - 45 * yz * drinv7;
            scalar_t tyyzz = 105 * y2 * z2 * drinv9 - 15 * (y2 + z2) * drinv7 + 3 * drinv5;
            scalar_t tyyxz = 105 * y2 * xz * drinv9 - 15 * xz * drinv7;

            scalar_t tzzzz = 105 * z2 * z2 * drinv9 - 90 * z2 * drinv7 + 9 * drinv5;
            scalar_t tzzzx = 105 * z2 * xz * drinv9 - 45 * xz * drinv7;
            scalar_t tzzzy = 105 * z2 * yz * drinv9 - 45 * yz * drinv7;
            scalar_t tzzxy = 105 * z2 * xy * drinv9 - 15 * xy * drinv7;

            // if ( interaction_tensor ) {
            //     interaction_tensor[20] = txxxx; interaction_tensor[21] = txxxy; interaction_tensor[22] = txxxz;
            //     interaction_tensor[23] = txxyy; interaction_tensor[24] = txxzz; interaction_tensor[25] = txxyz;
            //     interaction_tensor[26] = tyyyy; interaction_tensor[27] = tyyyx; interaction_tensor[28] = tyyyz;
            //     interaction_tensor[29] = tyyzz; interaction_tensor[30] = tyyxz;
            //     interaction_tensor[31] = tzzzz; interaction_tensor[32] = tzzzx; interaction_tensor[33] = tzzzy;
            //     interaction_tensor[34] = tzzxy;
            // }

            scalar_t txxx_g = mpi.xx * mpj.x - mpj.xx * mpi.x;
            scalar_t txxy_g = mpi.xx * mpj.y - mpj.xx * mpi.y + mpi.xy * mpj.x - mpj.xy * mpi.x;
            scalar_t txxz_g = mpi.xx * mpj.z - mpj.xx * mpi.z + mpi.xz * mpj.x - mpj.xz * mpi.x;
            scalar_t tyyy_g = mpi.yy * mpj.y - mpj.yy * mpi.y;
            scalar_t tyyx_g = mpi.yy * mpj.x - mpj.yy * mpi.x + mpi.xy * mpj.y - mpj.xy * mpi.y;
            scalar_t tyyz_g = mpi.yy * mpj.z - mpj.yy * mpi.z + mpi.yz * mpj.y - mpj.yz * mpi.y;
            scalar_t tzzz_g = mpi.zz * mpj.z - mpj.zz * mpi.z;
            scalar_t tzzx_g = mpi.zz * mpj.x - mpj.zz * mpi.x + mpi.xz * mpj.z - mpj.xz * mpi.z;
            scalar_t tzzy_g = mpi.zz * mpj.y - mpj.zz * mpi.y + mpi.yz * mpj.z - mpj.yz * mpi.z;
            scalar_t txyz_g = mpi.xy * mpj.z - mpj.xy * mpi.z + mpi.xz * mpj.y - mpj.xz * mpi.y + mpi.yz * mpj.x - mpj.yz * mpi.x;

            scalar_t txxxx_g = mpi.xx * mpj.xx;
            scalar_t txxxy_g = mpi.xx * mpj.xy + mpi.xy * mpj.xx;
            scalar_t txxxz_g = mpi.xx * mpj.xz + mpi.xz * mpj.xx;
            scalar_t txxyy_g = mpi.xx * mpj.yy + mpi.yy * mpj.xx + mpi.xy * mpj.xy;
            scalar_t txxzz_g = mpi.xx * mpj.zz + mpi.zz * mpj.xx + mpi.xz * mpj.xz;
            scalar_t txxyz_g = mpi.xx * mpj.yz + mpi.yz * mpj.xx + mpi.xy * mpj.xz + mpi.xz * mpj.xy;

            scalar_t tyyyy_g = mpi.yy * mpj.yy;
            scalar_t tyyyx_g = mpi.yy * mpj.xy + mpi.xy * mpj.yy;
            scalar_t tyyyz_g = mpi.yy * mpj.yz + mpi.yz * mpj.yy;
            scalar_t tyyzz_g = mpi.yy * mpj.zz + mpi.zz * mpj.yy + mpi.yz * mpj.yz;
            scalar_t tyyxz_g = mpi.yy * mpj.xz + mpi.xz * mpj.yy + mpi.xy * mpj.yz + mpi.yz * mpj.xy;

            scalar_t tzzzz_g = mpi.zz * mpj.zz;
            scalar_t tzzzx_g = mpi.zz * mpj.xz + mpi.xz * mpj.zz;
            scalar_t tzzzy_g = mpi.zz * mpj.yz + mpi.yz * mpj.zz;
            scalar_t tzzxy_g = mpi.zz * mpj.xy + mpi.xy * mpj.zz + mpi.xz * mpj.yz + mpi.yz * mpj.xz;

            // energy
            if constexpr ( DO_ENERGY ) {
                *ene += txxx * txxx_g + txxy * txxy_g + txxz * txxz_g + tyyy * tyyy_g + tyyx * tyyx_g + tyyz * tyyz_g + tzzz * tzzz_g + tzzx * tzzx_g + tzzy * tzzy_g + txyz * txyz_g
                + txxxx * txxxx_g
                + txxxy * txxxy_g
                + txxxz * txxxz_g
                + txxyy * txxyy_g
                + txxzz * txxzz_g
                + txxyz * txxyz_g
                + tyyyy * tyyyy_g
                + tyyyx * tyyyx_g
                + tyyyz * tyyyz_g
                + tyyzz * tyyzz_g
                + tyyxz * tyyxz_g
                + tzzzz * tzzzz_g
                + tzzzx * tzzzx_g
                + tzzzy * tzzzy_g
                + tzzxy * tzzxy_g;
            }

            if constexpr ( DO_MPOLE_GRAD ) {
                // charge gradient - electric potential
                gpi_ptr->s += txx * mpj.xx + txy * mpj.xy + txz * mpj.xz + tyy * mpj.yy + tyz * mpj.yz + tzz * mpj.zz;
                gpj_ptr->s += txx * mpi.xx + txy * mpi.xy + txz * mpi.xz + tyy * mpi.yy + tyz * mpi.yz + tzz * mpi.zz;

                // dipole gradient - electric field
                gpi_ptr->x += - txxx * mpj.xx - txxy * mpj.xy - txxz * mpj.xz - tyyx * mpj.yy - tzzx * mpj.zz - txyz * mpj.yz;
                gpi_ptr->y += - txxy * mpj.xx - tyyx * mpj.xy - txyz * mpj.xz - tyyy * mpj.yy - tzzy * mpj.zz - tyyz * mpj.yz;
                gpi_ptr->z += - txxz * mpj.xx - txyz * mpj.xy - tzzx * mpj.xz - tyyz * mpj.yy - tzzz * mpj.zz - tzzy * mpj.yz;

                gpj_ptr->x += txxx * mpi.xx + txxy * mpi.xy + txxz * mpi.xz + tyyx * mpi.yy + tzzx * mpi.zz + txyz * mpi.yz;
                gpj_ptr->y += txxy * mpi.xx + tyyx * mpi.xy + txyz * mpi.xz + tyyy * mpi.yy + tzzy * mpi.zz + tyyz * mpi.yz;
                gpj_ptr->z += txxz * mpi.xx + txyz * mpi.xy + tzzx * mpi.xz + tyyz * mpi.yy + tzzz * mpi.zz + tzzy * mpi.yz;

                // quadrupole gradient - electric field graident
                gpi_ptr->xx += mpj.s * txx + txxx * mpj.x + txxy * mpj.y + txxz * mpj.z + txxxx * mpj.xx + txxxy * mpj.xy + txxxz * mpj.xz + txxyy * mpj.yy + txxyz * mpj.yz + txxzz * mpj.zz;
                gpi_ptr->xy += mpj.s * txy + txxy * mpj.x + tyyx * mpj.y + txyz * mpj.z + txxxy * mpj.xx + txxyy * mpj.xy + txxyz * mpj.xz + tyyyx * mpj.yy + tyyxz * mpj.yz + tzzxy * mpj.zz;
                gpi_ptr->xz += mpj.s * txz + txxz * mpj.x + txyz * mpj.y + tzzx * mpj.z + txxxz * mpj.xx + txxyz * mpj.xy + txxzz * mpj.xz + tyyxz * mpj.yy + tzzxy * mpj.yz + tzzzx * mpj.zz;
                gpi_ptr->yy += mpj.s * tyy + tyyx * mpj.x + tyyy * mpj.y + tyyz * mpj.z + txxyy * mpj.xx + tyyyx * mpj.xy + tyyxz * mpj.xz + tyyyy * mpj.yy + tyyyz * mpj.yz + tyyzz * mpj.zz;
                gpi_ptr->yz += mpj.s * tyz + txyz * mpj.x + tyyz * mpj.y + tzzy * mpj.z + txxyz * mpj.xx + tyyxz * mpj.xy + tzzxy * mpj.xz + tyyyz * mpj.yy + tyyzz * mpj.yz + tzzzy * mpj.zz;
                gpi_ptr->zz += mpj.s * tzz + tzzx * mpj.x + tzzy * mpj.y + tzzz * mpj.z + txxzz * mpj.xx + tzzxy * mpj.xy + tzzzx * mpj.xz + tyyzz * mpj.yy + tzzzy * mpj.yz + tzzzz * mpj.zz;

                gpj_ptr->xx += mpi.s * txx - txxx * mpi.x - txxy * mpi.y - txxz * mpi.z + txxxx * mpi.xx + txxxy * mpi.xy + txxxz * mpi.xz + txxyy * mpi.yy + txxyz * mpi.yz + txxzz * mpi.zz;
                gpj_ptr->xy += mpi.s * txy - txxy * mpi.x - tyyx * mpi.y - txyz * mpi.z + txxxy * mpi.xx + txxyy * mpi.xy + txxyz * mpi.xz + tyyyx * mpi.yy + tyyxz * mpi.yz + tzzxy * mpi.zz;
                gpj_ptr->xz += mpi.s * txz - txxz * mpi.x - txyz * mpi.y - tzzx * mpi.z + txxxz * mpi.xx + txxyz * mpi.xy + txxzz * mpi.xz + tyyxz * mpi.yy + tzzxy * mpi.yz + tzzzx * mpi.zz;
                gpj_ptr->yy += mpi.s * tyy - tyyx * mpi.x - tyyy * mpi.y - tyyz * mpi.z + txxyy * mpi.xx + tyyyx * mpi.xy + tyyxz * mpi.xz + tyyyy * mpi.yy + tyyyz * mpi.yz + tyyzz * mpi.zz;
                gpj_ptr->yz += mpi.s * tyz - txyz * mpi.x - tyyz * mpi.y - tzzy * mpi.z + txxyz * mpi.xx + tyyxz * mpi.xy + tzzxy * mpi.xz + tyyyz * mpi.yy + tyyzz * mpi.yz + tzzzy * mpi.zz;
                gpj_ptr->zz += mpi.s * tzz - tzzx * mpi.x - tzzy * mpi.y - tzzz * mpi.z + txxzz * mpi.xx + tzzxy * mpi.xy + tzzzx * mpi.xz + tyyzz * mpi.yy + tzzzy * mpi.yz + tzzzz * mpi.zz;
            }

            // dr gradient - forces
            if constexpr ( DO_COORD_GRAD ) {
                scalar_t& drinv11 = drinvs[5];

                scalar_t c945dr11 = -945 * drinv11;
                scalar_t c105dr9 = 105 * drinv9;
                scalar_t c15dr7 = 15 * drinv7;

                scalar_t t5x = c945dr11 * x2 * x2 * drx + 10 * c105dr9 * x2 * drx - 15 * c15dr7 * drx;
                scalar_t t5y = c945dr11 * y2 * y2 * dry + 10 * c105dr9 * y2 * dry - 15 * c15dr7 * dry;
                scalar_t t5z = c945dr11 * z2 * z2 * drz + 10 * c105dr9 * z2 * drz - 15 * c15dr7 * drz;
                scalar_t t4x1y = c945dr11 * x2 * x2 * dry + 6 * c105dr9 * x2 * dry - 3 * c15dr7 * dry;
                scalar_t t4x1z = c945dr11 * x2 * x2 * drz + 6 * c105dr9 * x2 * drz - 3 * c15dr7 * drz;
                scalar_t t4y1x = c945dr11 * y2 * y2 * drx + 6 * c105dr9 * y2 * drx - 3 * c15dr7 * drx;
                scalar_t t4z1x = c945dr11 * z2 * z2 * drx + 6 * c105dr9 * z2 * drx - 3 * c15dr7 * drx;
                scalar_t t4y1z = c945dr11 * y2 * y2 * drz + 6 * c105dr9 * y2 * drz - 3 * c15dr7 * drz;
                scalar_t t4z1y = c945dr11 * z2 * z2 * dry + 6 * c105dr9 * z2 * dry - 3 * c15dr7 * dry;
                scalar_t t3x1y1z = c945dr11 * x2 * xyz + 3 * c105dr9 * xyz;
                scalar_t t3y1x1z = c945dr11 * y2 * xyz + 3 * c105dr9 * xyz;
                scalar_t t3z1x1y = c945dr11 * z2 * xyz + 3 * c105dr9 * xyz;
                scalar_t t3x2y = c945dr11 * x2 * drx * y2 + c105dr9 * drx * (3 * y2 + x2) - 3 * c15dr7 * drx;
                scalar_t t3y2x = c945dr11 * y2 * dry * x2 + c105dr9 * dry * (3 * x2 + y2) - 3 * c15dr7 * dry;
                scalar_t t3x2z = c945dr11 * x2 * drx * z2 + c105dr9 * drx * (3 * z2 + x2) - 3 * c15dr7 * drx;
                scalar_t t3z2x = c945dr11 * z2 * drz * x2 + c105dr9 * drz * (3 * x2 + z2) - 3 * c15dr7 * drz;
                scalar_t t3y2z = c945dr11 * y2 * dry * z2 + c105dr9 * dry * (3 * z2 + y2) - 3 * c15dr7 * dry;
                scalar_t t3z2y = c945dr11 * z2 * drz * y2 + c105dr9 * drz * (3 * y2 + z2) - 3 * c15dr7 * drz;
                scalar_t t2x2y1z = c945dr11 * xy * xyz + c105dr9 * drz * (x2 + y2) - c15dr7 * drz;
                scalar_t t2x2z1y = c945dr11 * xz * xyz + c105dr9 * dry * (x2 + z2) - c15dr7 * dry;
                scalar_t t2y2z1x = c945dr11 * yz * xyz + c105dr9 * drx * (y2 + z2) - c15dr7 * drx;

                dr_g[0] += txxxx * txxx_g + txxxy * txxy_g + txxxz * txxz_g + tyyyx * tyyy_g + txxyy * tyyx_g + tyyxz * tyyz_g + tzzzx * tzzz_g + txxzz * tzzx_g + tzzxy * tzzy_g + txxyz * txyz_g
                        + t5x * txxxx_g
                        + t4x1y * txxxy_g
                        + t4x1z * txxxz_g
                        + t3x2y * txxyy_g
                        + t3x2z * txxzz_g
                        + t3x1y1z * txxyz_g
                        + t4y1x * tyyyy_g
                        + t3y2x * tyyyx_g
                        + t3y1x1z * tyyyz_g
                        + t2y2z1x * tyyzz_g
                        + t2x2y1z * tyyxz_g
                        + t4z1x * tzzzz_g
                        + t3z2x * tzzzx_g
                        + t3z1x1y * tzzzy_g
                        + t2x2z1y * tzzxy_g;

                dr_g[1] += txxxy * txxx_g + txxyy * txxy_g + txxyz * txxz_g + tyyyy * tyyy_g + tyyyx * tyyx_g + tyyyz * tyyz_g + tzzzy * tzzz_g + tzzxy * tzzx_g + tyyzz * tzzy_g + tyyxz * txyz_g
                        + t4x1y * txxxx_g
                        + t3x2y * txxxy_g
                        + t3x1y1z * txxxz_g
                        + t3y2x * txxyy_g
                        + t2x2z1y * txxzz_g
                        + t2x2y1z * txxyz_g
                        + t5y * tyyyy_g
                        + t4y1x * tyyyx_g
                        + t4y1z * tyyyz_g
                        + t3y2z * tyyzz_g
                        + t3y1x1z * tyyxz_g
                        + t4z1y * tzzzz_g
                        + t3z1x1y * tzzzx_g
                        + t3z2y * tzzzy_g
                        + t2y2z1x * tzzxy_g;

                dr_g[2] += txxxz * txxx_g + txxyz * txxy_g + txxzz * txxz_g + tyyyz * tyyy_g + tyyxz * tyyx_g + tyyzz * tyyz_g + tzzzz * tzzz_g + tzzzx * tzzx_g + tzzzy * tzzy_g + tzzxy * txyz_g
                        + t4x1z * txxxx_g
                        + t3x1y1z * txxxy_g
                        + t3x2z * txxxz_g
                        + t2x2y1z * txxyy_g
                        + t3z2x * txxzz_g
                        + t2x2z1y * txxyz_g
                        + t4y1z * tyyyy_g
                        + t3y1x1z * tyyyx_g
                        + t3y2z * tyyyz_g
                        + t3z2y * tyyzz_g
                        + t2y2z1x * tyyxz_g
                        + t5z * tzzzz_g
                        + t4z1x * tzzzx_g
                        + t4z1y * tzzzy_g
                        + t3z1x1y * tzzxy_g;
            }
        }  // RANK >= 2

    }  // RANK >= 1
}

template <typename scalar_t, int BLOCK_SIZE, int RANK, bool DO_EWALD=false, bool DO_ENERGY=true, bool DO_COORD_GRAD=true, bool DO_MPOLE_GRAD=true>
__global__ void multipolar_interaction_atom_pairs_kernel(
    scalar_t* coords,
    scalar_t* box,
    int64_t* pairs,
    const int64_t* pairs_excl,
    int64_t npairs,
    int64_t npairs_excl,
    scalar_t cutoff,
    scalar_t ewald_alpha,
    scalar_t prefactor,
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
        scalar_t damps[RANK * 2 + 2];
        if constexpr (DO_EWALD) {
            ewald_erfc_damps<scalar_t, 4 * RANK + 3>(dr, ewald_alpha, damps);
        }
        scalar_t dr_g_buf[3] = {};

        scalar_t* dr_g_ptr = ( DO_COORD_GRAD ) ? dr_g_buf : nullptr;
        scalar_t* damps_ptr = ( DO_EWALD ) ? damps : nullptr;
        CartesianExpansion<scalar_t, RANK>* gpi_ptr = ( DO_MPOLE_GRAD ) ? &gpi : nullptr;
        CartesianExpansion<scalar_t, RANK>* gpj_ptr = ( DO_MPOLE_GRAD ) ? &gpj : nullptr;

        pairwise_multipole_kernel_with_grad<scalar_t, RANK, DO_EWALD, DO_ENERGY, DO_COORD_GRAD, DO_MPOLE_GRAD>(
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

#endif