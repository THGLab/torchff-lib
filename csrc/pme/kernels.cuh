#ifndef TOCHFF_PME_KERNELS_CUH
#define TOCHFF_PME_KERNELS_CUH

#include <cuda_runtime.h>
#include "bsplines.cuh"
#include "common/constants.cuh"
#include "common/pbc.cuh"
#include "multipoles/storage.cuh"
#include <c10/util/complex.h>


template <typename T, int RANK>
__global__ void spread_q_kernel(
    const T* __restrict__ coords,
    const T* __restrict__ q,
    const T* __restrict__ p,
    const T* __restrict__ Q,
    const T* __restrict__ box,
    T* __restrict__ grid,
    int N_atoms,
    int K1, int K2, int K3
) {
    __shared__ T s_box_inv[9];
    __shared__ T cart2frac[3][3];
    if (threadIdx.x == 0) {
        invert_box_3x3(box, s_box_inv);
        cart2frac[0][0] = s_box_inv[0 * 3 + 0] * K1;
        cart2frac[0][1] = s_box_inv[1 * 3 + 0] * K1;
        cart2frac[0][2] = s_box_inv[2 * 3 + 0] * K1;
        cart2frac[1][0] = s_box_inv[0 * 3 + 1] * K2;
        cart2frac[1][1] = s_box_inv[1 * 3 + 1] * K2;
        cart2frac[1][2] = s_box_inv[2 * 3 + 1] * K2;
        cart2frac[2][0] = s_box_inv[0 * 3 + 2] * K3;
        cart2frac[2][1] = s_box_inv[1 * 3 + 2] * K3;
        cart2frac[2][2] = s_box_inv[2 * 3 + 2] * K3;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_atoms) return;

    CartesianExpansion<T, RANK> mp{};
    mp.s = q[idx];
    if constexpr (RANK >= 1) {
        mp.x = p[idx * 3 + 0];
        mp.y = p[idx * 3 + 1];
        mp.z = p[idx * 3 + 2];
    }
    if constexpr (RANK >= 2) {
        mp.xx = Q[idx * 9 + 0];
        mp.xy = Q[idx * 9 + 1];
        mp.xz = Q[idx * 9 + 2];
        mp.yy = Q[idx * 9 + 4];
        mp.yz = Q[idx * 9 + 5];
        mp.zz = Q[idx * 9 + 8];
    }

    T r[3];
    r[0] = coords[idx * 3 + 0]; r[1] = coords[idx * 3 + 1]; r[2] = coords[idx * 3 + 2];

    int m_u0[3];
    T u_frac[3];

    #pragma unroll
    for(int i=0; i<3; i++) {
        T val = (cart2frac[i][0]*r[0] + cart2frac[i][1]*r[1] + cart2frac[i][2]*r[2]);
        m_u0[i] = (int)ceil(val);
        u_frac[i] = ( (T)m_u0[i] - val) + (T)3.0;
    }

    T M[3][6], dM[3][6], d2M[3][6];
    // Calculate B-Spline and derivatives
    #pragma unroll
    for(int d=0; d<3; d++) {
        #pragma unroll
        for(int k=0; k<6; k++) {
            T u_eval = u_frac[d] + (T)(k - 3);
            eval_b6_and_derivs<T>(u_eval, &M[d][k], &dM[d][k], &d2M[d][k]);
        }
    }
    #pragma unroll 6
    for (int iz = 0; iz < 6; iz++) {
        int gz = (m_u0[2] + (iz - 3) + 1000 * K3) % K3;
        T Mz = M[2][iz];
        T dMz = (RANK >= 1) ? dM[2][iz] : (T)0.0;
        T d2Mz = (RANK >= 2) ? d2M[2][iz] : (T)0.0;

        #pragma unroll 6
        for (int iy = 0; iy < 6; iy++) {
            int gy = (m_u0[1] + (iy - 3) + 1000 * K2) % K2;
            T My = M[1][iy];
            T dMy = (RANK >= 1) ? dM[1][iy] : (T)0.0;
            T d2My = (RANK >= 2) ? d2M[1][iy] : (T)0.0;

            #pragma unroll 6
            for (int ix = 0; ix < 6; ix++) {
                int gx = (m_u0[0] + (ix - 3) + 1000 * K1) % K1;
                T Mx = M[0][ix];
                T dMx = (RANK >= 1) ? dM[0][ix] : (T)0.0;
                T d2Mx = (RANK >= 2) ? d2M[0][ix] : (T)0.0;

                T theta = Mx * My * Mz;
                T term = mp.s * theta;

                if constexpr (RANK >= 1) {
                    // Initialize b-spline derivative
                    T dt_du[3];
                    dt_du[0] = dMx * My * Mz;
                    dt_du[1] = Mx * dMy * Mz;
                    dt_du[2] = Mx * My * dMz;

                    T dt_dr[3] = {(T)0,(T)0,(T)0};
                    #pragma unroll
                    for(int i=0; i<3; i++) {
                        dt_dr[0] += cart2frac[i][0] * dt_du[i];
                        dt_dr[1] += cart2frac[i][1] * dt_du[i];
                        dt_dr[2] += cart2frac[i][2] * dt_du[i];
                    }
                    term -= mp.x * dt_dr[0] + mp.y * dt_dr[1] + mp.z * dt_dr[2];
                    if constexpr (RANK >= 2) {
                        // 1. Load Cartesian Quadrupoles
                        T& Qxx = mp.xx;
                        T& Qxy = mp.xy;
                        T& Qxz = mp.xz;
                        T& Qyy = mp.yy;
                        T& Qyz = mp.yz;
                        T& Qzz = mp.zz;

                        // 3. Transform Q_cart to Q_lat
                        // Q_lat = A * Q_cart * A^T
                        T Q_cart[3][3];
                        Q_cart[0][0]=Qxx; Q_cart[0][1]=Qxy; Q_cart[0][2]=Qxz;
                        Q_cart[1][0]=Qxy; Q_cart[1][1]=Qyy; Q_cart[1][2]=Qyz;
                        Q_cart[2][0]=Qxz; Q_cart[2][1]=Qyz; Q_cart[2][2]=Qzz;

                        T Q_temp[3][3] = {(T)0};
                        #pragma unroll
                        for(int i=0; i<3; i++)
                            for(int j=0; j<3; j++)
                                for(int k=0; k<3; k++)
                                    Q_temp[i][j] += Q_cart[i][k] * cart2frac[j][k]; // Q * A^T (symmetric)

                        T Q_lat[3][3] = {(T)0};
                        #pragma unroll
                        for(int i=0; i<3; i++)
                            for(int j=0; j<3; j++)
                                for(int k=0; k<3; k++)
                                    Q_lat[i][j] += cart2frac[i][k] * Q_temp[k][j]; // A * (Q * A^T)

                        T Q_lat_xx = Q_lat[0][0]; T Q_lat_yy = Q_lat[1][1]; T Q_lat_zz = Q_lat[2][2];
                        T Q_lat_xy = Q_lat[0][1]; T Q_lat_xz = Q_lat[0][2]; T Q_lat_yz = Q_lat[1][2];

                        // 4. Compute B-Spline Lattice Hessians (Hu_vals)
                        T d2t_du2[3];
                        d2t_du2[0] = d2Mx * My * Mz;
                        d2t_du2[1] = Mx * d2My * Mz;
                        d2t_du2[2] = Mx * My * d2Mz;
                        T d2t_du_mix[3];
                        d2t_du_mix[0] = dMx * dMy * Mz;
                        d2t_du_mix[1] = dMx * My * dMz;
                        d2t_du_mix[2] = Mx * dMy * dMz;

                        T Hu_vals[3][3];
                        Hu_vals[0][0] = d2t_du2[0]; Hu_vals[1][1] = d2t_du2[1]; Hu_vals[2][2] = d2t_du2[2];
                        Hu_vals[0][1] = d2t_du_mix[0];
                        Hu_vals[0][2] = d2t_du_mix[1];
                        Hu_vals[1][2] = d2t_du_mix[2];

                        // 5. Contract Lattice Quadrupole with Lattice Hessian
                        T interaction = Q_lat_xx * Hu_vals[0][0] +
                                         Q_lat_yy * Hu_vals[1][1] +
                                         Q_lat_zz * Hu_vals[2][2] +
                                         (T)2.0 * (Q_lat_xy * Hu_vals[0][1] +
                                         Q_lat_xz * Hu_vals[0][2] +
                                         Q_lat_yz * Hu_vals[1][2]);

                        interaction *= T(1.0/3.0);
                        term += interaction;
                    }
                }

                int grid_ptr = gx * K2 * K3 + gy * K3 + gz;
                atomicAdd(&grid[grid_ptr], term);
            }
        }
    }
}



template <typename T, int RANK>
__global__ void interpolate_kernel(
    const T* __restrict__ grid,
    const T* __restrict__ coords,
    const T* __restrict__ box,
    const T* __restrict__ q,
    const T* __restrict__ p,
    const T* __restrict__ Q,
    T* __restrict__ phi_atoms,
    T* __restrict__ E_atoms,
    T* __restrict__ EG_atoms,
    T* __restrict__ force_atoms,
    T alpha,
    int N_atoms,
    int K1, int K2, int K3
) {
    __shared__ T s_box_inv[9];
    __shared__ T cart2frac[3][3];
    if (threadIdx.x == 0) {
        invert_box_3x3(box, s_box_inv);
        cart2frac[0][0] = s_box_inv[0 * 3 + 0] * K1;
        cart2frac[0][1] = s_box_inv[1 * 3 + 0] * K1;
        cart2frac[0][2] = s_box_inv[2 * 3 + 0] * K1;
        cart2frac[1][0] = s_box_inv[0 * 3 + 1] * K2;
        cart2frac[1][1] = s_box_inv[1 * 3 + 1] * K2;
        cart2frac[1][2] = s_box_inv[2 * 3 + 1] * K2;
        cart2frac[2][0] = s_box_inv[0 * 3 + 2] * K3;
        cart2frac[2][1] = s_box_inv[1 * 3 + 2] * K3;
        cart2frac[2][2] = s_box_inv[2 * 3 + 2] * K3;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_atoms) return;

    // Multipole Setup
    CartesianExpansion<T, RANK> mp{};
    mp.s = q[idx];
    if constexpr (RANK >= 1) {
        mp.x = p[idx * 3 + 0];
        mp.y = p[idx * 3 + 1];
        mp.z = p[idx * 3 + 2];
    }
    if constexpr (RANK >= 2) {
        mp.xx = Q[idx * 9 + 0];
        mp.xy = Q[idx * 9 + 1];
        mp.xz = Q[idx * 9 + 2];
        mp.yy = Q[idx * 9 + 4];
        mp.yz = Q[idx * 9 + 5];
        mp.zz = Q[idx * 9 + 8];
    }

    // --- 1. Geometry Setup ---
    T r[3] = {coords[idx*3+0], coords[idx*3+1], coords[idx*3+2]};

    int m_u0[3];
    T u_frac[3];

    #pragma unroll
    for(int i=0; i<3; i++) {
        T val = (cart2frac[i][0]*r[0] + cart2frac[i][1]*r[1] + cart2frac[i][2]*r[2]);
        m_u0[i] = (int)ceil(val);
        u_frac[i] = ((T)m_u0[i] - val) + (T)3.0;
    }

    // --- 2. B-Spline Evaluation ---
    T M[3][6], dM[3][6], d2M[3][6], d3M[3][6];
    #pragma unroll
    for(int d=0; d<3; d++) {
        #pragma unroll
        for(int k=0; k<6; k++) {
            T u_eval = u_frac[d] + (T)(k - 3);
            eval_b6_and_derivs<T>(u_eval, &M[d][k], &dM[d][k], &d2M[d][k], &d3M[d][k]);
        }
    }

    // --- 3. Multipole Setup ---

    // Dipoles
    T p_lat[3] = {(T)0.0};
    if constexpr (RANK >= 1) {
        #pragma unroll
        for(int i=0; i<3; i++)
            p_lat[i] = (mp.x * cart2frac[i][0] + mp.y * cart2frac[i][1] + mp.z * cart2frac[i][2]);
    }

    // Quadrupoles
    T Q_lat[6] = {(T)0.0};
    if constexpr (RANK >= 2) {
        // Initialize quadrupoles
        T& Qxx = mp.xx;
        T& Qxy = mp.xy;
        T& Qxz = mp.xz;
        T& Qyy = mp.yy;
        T& Qyz = mp.yz;
        T& Qzz = mp.zz;

        T Q_cart[3][3];
        Q_cart[0][0]=Qxx; Q_cart[0][1]=Qxy; Q_cart[0][2]=Qxz;
        Q_cart[1][0]=Qxy; Q_cart[1][1]=Qyy; Q_cart[1][2]=Qyz;
        Q_cart[2][0]=Qxz; Q_cart[2][1]=Qyz; Q_cart[2][2]=Qzz;


        T Q_temp[3][3] = {0};
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    Q_temp[i][j] += Q_cart[i][k] * cart2frac[j][k];

        T Q_L[3][3] = {0};
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    Q_L[i][j] += cart2frac[i][k] * Q_temp[k][j];

        Q_lat[0]=Q_L[0][0]; Q_lat[1]=Q_L[1][1]; Q_lat[2]=Q_L[2][2];
        Q_lat[3]=Q_L[0][1]; Q_lat[4]=Q_L[0][2]; Q_lat[5]=Q_L[1][2];
    }

    // --- 4. Grid Accumulation ---
    T phi_acc = (T)0.0;
    T grad_lat[3] = {(T)0.0};
    T hess_p_lat[3] = {(T)0.0};
    T grad_Q_lat[3] = {(T)0.0};
    T hess_lat[6] = {(T)0.0};

    #pragma unroll 6
    for (int iz = 0; iz < 6; iz++) {
        int gz = (m_u0[2] + (iz - 3) + 1000 * K3) % K3;
        T Mz = M[2][iz]; T dMz = dM[2][iz];
        T d2Mz = T(0.0);
        if constexpr (RANK >= 1) { d2Mz = d2M[2][iz]; }
        T d3Mz = T(0.0);
        if constexpr (RANK >= 2) { d3Mz = d3M[2][iz]; }

        #pragma unroll 6
        for (int iy = 0; iy < 6; iy++) {
            int gy = (m_u0[1] + (iy - 3) + 1000 * K2) % K2;
            T My = M[1][iy]; T dMy = dM[1][iy];
            T d2My = T(0.0);
            if constexpr (RANK >= 1) { d2My = d2M[1][iy]; }
            T d3My = T(0.0);
            if constexpr (RANK >= 2) { d3My = d3M[1][iy]; }

            #pragma unroll 6
            for (int ix = 0; ix < 6; ix++) {
                int gx = (m_u0[0] + (ix - 3) + 1000 * K1) % K1;
                T Mx = M[0][ix]; T dMx = dM[0][ix];
                T d2Mx = T(0.0);
                if constexpr (RANK >= 1) { d2Mx = d2M[0][ix]; }
                T d3Mx = T(0.0);
                if constexpr (RANK >= 2) { d3Mx = d3M[0][ix]; }

                T val = grid[gx * K2 * K3 + gy * K3 + gz];

                phi_acc      += (Mx * My * Mz) * val;
                grad_lat[0]  += (dMx * My * Mz) * val;
                grad_lat[1]  += (Mx * dMy * Mz) * val;
                grad_lat[2]  += (Mx * My * dMz) * val;

                if constexpr (RANK >= 1) {
                    T H_uu = d2Mx * My * Mz;
                    T H_vv = Mx * d2My * Mz;
                    T H_ww = Mx * My * d2Mz;
                    T H_uv = dMx * dMy * Mz;
                    T H_uw = dMx * My * dMz;
                    T H_vw = Mx * dMy * dMz;

                    hess_lat[0] += H_uu * val; hess_lat[1] += H_vv * val; hess_lat[2] += H_ww * val;
                    hess_lat[3] += H_uv * val; hess_lat[4] += H_uw * val; hess_lat[5] += H_vw * val;

                    hess_p_lat[0] += (p_lat[0]*H_uu + p_lat[1]*H_uv + p_lat[2]*H_uw) * val;
                    hess_p_lat[1] += (p_lat[0]*H_uv + p_lat[1]*H_vv + p_lat[2]*H_vw) * val;
                    hess_p_lat[2] += (p_lat[0]*H_uw + p_lat[1]*H_vw + p_lat[2]*H_ww) * val;

                    if constexpr (RANK >= 2) {
                        T T_uuu = d3Mx * My * Mz;
                        T T_vvv = Mx * d3My * Mz;
                        T T_www = Mx * My * d3Mz;
                        T T_uuv = d2Mx * dMy * Mz;
                        T T_uuw = d2Mx * My * dMz;
                        T T_uvv = dMx * d2My * Mz;
                        T T_vvw = Mx * d2My * dMz;
                        T T_uww = dMx * My * d2Mz;
                        T T_vww = Mx * dMy * d2Mz;
                        T T_uvw = dMx * dMy * dMz;

                        T dE_du = T(1.0/3.0) * (Q_lat[0]*T_uuu + Q_lat[1]*T_uvv + Q_lat[2]*T_uww +
                                              (T)2.0*(Q_lat[3]*T_uuv + Q_lat[4]*T_uuw + Q_lat[5]*T_uvw));
                        T dE_dv = T(1.0/3.0) * (Q_lat[0]*T_uuv + Q_lat[1]*T_vvv + Q_lat[2]*T_vww +
                                              (T)2.0*(Q_lat[3]*T_uvv + Q_lat[4]*T_uvw + Q_lat[5]*T_vvw));
                        T dE_dw = T(1.0/3.0) * (Q_lat[0]*T_uuw + Q_lat[1]*T_vvw + Q_lat[2]*T_www +
                                              (T)2.0*(Q_lat[3]*T_uvw + Q_lat[4]*T_uww + Q_lat[5]*T_vww));

                        grad_Q_lat[0] += dE_du * val;
                        grad_Q_lat[1] += dE_dv * val;
                        grad_Q_lat[2] += dE_dw * val;
                    }
                }
            }
        }
    }

    // --- 5. Cartesian Transform ---
    // Output Potential
    phi_atoms[idx] += phi_acc;

    // Transform Gradients (Lattice -> Cartesian)
    T grad_cart[3] = {(T)0.0};
    #pragma unroll
    for(int x=0; x<3; x++) {
        grad_cart[x] = grad_lat[0] * (cart2frac[0][x]) +
                       grad_lat[1] * (cart2frac[1][x]) +
                       grad_lat[2] * (cart2frac[2][x]);
    }

    if constexpr (RANK >= 1) {
        E_atoms[idx * 3 + 0] += grad_cart[0];
        E_atoms[idx * 3 + 1] += grad_cart[1];
        E_atoms[idx * 3 + 2] += grad_cart[2];
    }

    T grad_U_dip[3] = {(T)0.0};
    if constexpr (RANK >= 1) {
        #pragma unroll
        for(int x=0; x<3; x++) {
            grad_U_dip[x] = hess_p_lat[0] * (cart2frac[0][x]) +
                            hess_p_lat[1] * (cart2frac[1][x]) +
                            hess_p_lat[2] * (cart2frac[2][x]);
        }
    }

    T grad_U_quad[3] = {(T)0.0};
    if constexpr (RANK >= 2) {
        // Transform Force from Quadrupoles
        #pragma unroll
        for(int x=0; x<3; x++) {
            grad_U_quad[x] = grad_Q_lat[0] * (cart2frac[0][x]) +
                             grad_Q_lat[1] * (cart2frac[1][x]) +
                             grad_Q_lat[2] * (cart2frac[2][x]);
        }

        // Transform EFG
        T H_lat_mat[3][3];
        H_lat_mat[0][0]=hess_lat[0]; H_lat_mat[1][1]=hess_lat[1]; H_lat_mat[2][2]=hess_lat[2];
        H_lat_mat[0][1]=H_lat_mat[1][0]=hess_lat[3];
        H_lat_mat[0][2]=H_lat_mat[2][0]=hess_lat[4];
        H_lat_mat[1][2]=H_lat_mat[2][1]=hess_lat[5];

        T temp[3][3] = {0};
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    temp[i][j] += cart2frac[k][i] * H_lat_mat[k][j];

        T EFG[3][3] = {0};
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    EFG[i][j] += temp[i][k] * cart2frac[k][j];

        // Store full 3x3 field-gradient tensor per atom (N*9),
        // still applying the -0.5 factor used previously.
        T factor = T(-1.0);
        int base = idx * 9;
        EG_atoms[base + 0] += EFG[0][0] * factor;
        EG_atoms[base + 1] += EFG[0][1] * factor;
        EG_atoms[base + 2] += EFG[0][2] * factor;
        EG_atoms[base + 3] += EFG[1][0] * factor;
        EG_atoms[base + 4] += EFG[1][1] * factor;
        EG_atoms[base + 5] += EFG[1][2] * factor;
        EG_atoms[base + 6] += EFG[2][0] * factor;
        EG_atoms[base + 7] += EFG[2][1] * factor;
        EG_atoms[base + 8] += EFG[2][2] * factor;
    }

    force_atoms[idx*3 + 0] += (mp.s * grad_cart[0] - grad_U_dip[0] + grad_U_quad[0]);
    force_atoms[idx*3 + 1] += (mp.s * grad_cart[1] - grad_U_dip[1] + grad_U_quad[1]);
    force_atoms[idx*3 + 2] += (mp.s * grad_cart[2] - grad_U_dip[2] + grad_U_quad[2]);
}


template <typename T>
__global__ void pme_convolution_fused_kernel(
    c10::complex<T>* __restrict__ grid_recip,
    const T* __restrict__ box,
    const T* __restrict__ xmoduli,
    const T* __restrict__ ymoduli,
    const T* __restrict__ zmoduli,
    int K1, int K2, int K3,
    T alpha
) {
    __shared__ T s_box_inv[9];
    __shared__ T d_recip[9];
    __shared__ T s_volume;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        invert_box_3x3(box, s_box_inv, &s_volume);
        for (int k = 0; k < 9; k++) {
            int i = k / 3, j = k % 3;
            d_recip[k] = s_box_inv[j * 3 + i];
        }
    }
    __syncthreads();

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
    int K3_complex = K3 / 2 + 1;
    if (idx_x >= K1 || idx_y >= K2 || idx_z >= K3_complex) return;
    int flat_idx = idx_x * K2 * K3_complex + idx_y * K3_complex + idx_z;

    if (idx_x == 0 && idx_y == 0 && idx_z == 0) {
        grid_recip[flat_idx] = c10::complex<T>((T)0.0, (T)0.0); return;
    }
    constexpr T TWOPI = two_pi<T>();
    T mx = (idx_x <= K1/2) ? (T)idx_x : (T)(idx_x - K1);
    T my = (idx_y <= K2/2) ? (T)idx_y : (T)(idx_y - K2);
    T mz = (T)idx_z; 
    T kx = TWOPI * (mx * d_recip[0] + my * d_recip[3] + mz * d_recip[6]);
    T ky = TWOPI * (mx * d_recip[1] + my * d_recip[4] + mz * d_recip[7]);
    T kz = TWOPI * (mx * d_recip[2] + my * d_recip[5] + mz * d_recip[8]);
    T ksq = kx*kx + ky*ky + kz*kz;
    T C_k = ((T)2.0 * TWOPI / (s_volume * ksq)) * exp(-ksq / ((T)4.0 * alpha * alpha));
    T theta_x = xmoduli[idx_x];
    T theta_y = ymoduli[idx_y];
    T theta_z = zmoduli[idx_z];
    T theta = theta_x * theta_y * theta_z;
    T theta_sq = theta * theta;
    T scale_factor = ((T)1.0 / theta_sq);
    T factor = C_k * scale_factor;
    grid_recip[flat_idx] *= factor;
}

#endif