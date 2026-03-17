#ifndef TORCHFF_PBC_CUH
#define TORCHFF_PBC_CUH

#include "vec3.cuh"

template <typename scalar_t>
__device__ __forceinline__ void apply_pbc_triclinic(const scalar_t* vec, const scalar_t* box, const scalar_t* box_inv, scalar_t* out) {
    // box in row major
    scalar_t sa = vec[0] * box_inv[0] + vec[1] * box_inv[3] + vec[2] * box_inv[6];
    scalar_t sb = vec[0] * box_inv[1] + vec[1] * box_inv[4] + vec[2] * box_inv[7];
    scalar_t sc = vec[0] * box_inv[2] + vec[1] * box_inv[5] + vec[2] * box_inv[8];
    sa -= round_(sa);
    sb -= round_(sb);
    sc -= round_(sc);
    out[0] = sa * box[0] + sb * box[3] + sc * box[6];
    out[1] = sa * box[1] + sb * box[4] + sc * box[7];
    out[2] = sa * box[2] + sb * box[5] + sc * box[8];
}


template <typename scalar_t>
__device__ __forceinline__ void apply_pbc_orthorhombic(scalar_t* vec, scalar_t* box, scalar_t* out) {
    out[0] = vec[0] - round_(vec[0] / box[0]) * box[0];
    out[1] = vec[1] - round_(vec[1] / box[1]) * box[1];
    out[2] = vec[2] - round_(vec[2] / box[2]) * box[2];
}


template <typename scalar_t>
__device__ __forceinline__ void apply_pbc_cubic(scalar_t* vec, scalar_t box, scalar_t* out) {
    out[0] = vec[0] - round_(vec[0] / box) * box;
    out[1] = vec[1] - round_(vec[1] / box) * box;
    out[2] = vec[2] - round_(vec[2] / box) * box;
}

// Invert 3x3 row-major box matrix: box_inv = inv(box). box and box_inv are 9-element arrays.
// If volume_out is non-null, writes the box determinant (volume) to *volume_out.
template <typename scalar_t>
__device__ __forceinline__ void invert_box_3x3(const scalar_t* box, scalar_t* box_inv, scalar_t* volume_out = nullptr) {
    scalar_t m00 = box[0], m01 = box[1], m02 = box[2];
    scalar_t m10 = box[3], m11 = box[4], m12 = box[5];
    scalar_t m20 = box[6], m21 = box[7], m22 = box[8];
    scalar_t c00 = m11 * m22 - m12 * m21;
    scalar_t c01 = m12 * m20 - m10 * m22;
    scalar_t c02 = m10 * m21 - m11 * m20;
    scalar_t c10 = m02 * m21 - m01 * m22;
    scalar_t c11 = m00 * m22 - m02 * m20;
    scalar_t c12 = m01 * m20 - m00 * m21;
    scalar_t c20 = m01 * m12 - m02 * m11;
    scalar_t c21 = m02 * m10 - m00 * m12;
    scalar_t c22 = m00 * m11 - m01 * m10;
    scalar_t det = m00 * c00 + m01 * c01 + m02 * c02;
    if (volume_out) *volume_out = det;
    scalar_t inv_det = (scalar_t)1 / det;
    box_inv[0] = c00 * inv_det;
    box_inv[1] = c10 * inv_det;
    box_inv[2] = c20 * inv_det;
    box_inv[3] = c01 * inv_det;
    box_inv[4] = c11 * inv_det;
    box_inv[5] = c21 * inv_det;
    box_inv[6] = c02 * inv_det;
    box_inv[7] = c12 * inv_det;
    box_inv[8] = c22 * inv_det;
}

#endif
