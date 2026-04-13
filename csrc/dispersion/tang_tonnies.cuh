#ifndef TORCHFF_TT_CUH
#define TORCHFF_TT_CUH

#include "common/vec3.cuh"

/// Tang–Tonnies order-6 damping: f_6(u)=1-e^{-u} sum_{k=0}^6 u^k/k!, u=b*r.
/// U = -C6 f_6 / r^6. Displacement (drx,dry,drz) is minimum-image x_i - x_j (same as vdW).
/// Coordinate part: (gx,gy,gz) = grad w.r.t. x_i (add to atom i, subtract from atom j).
template <typename scalar_t> __device__ __forceinline__
void tang_tonnies_6_dispersion(
    scalar_t c6, scalar_t b,
    scalar_t dr, scalar_t drx, scalar_t dry, scalar_t drz,
    scalar_t* ene, scalar_t* gx, scalar_t* gy, scalar_t* gz,
    scalar_t* c6_grad, scalar_t* b_grad
)
{
    scalar_t rinv = 1 / dr;
    scalar_t rinv6 = rinv*rinv*rinv*rinv*rinv*rinv;
    scalar_t u = b * dr;
    scalar_t u6 = u * u * u * u * u * u;
    scalar_t expu = exp_(-u);
    scalar_t f6 = (1 - exp_(-u) * (1 + u * ( 1 + u/2 * (1 + u/3 * (1 + u/4 * (1 + u/5 * (1 + u/6)))))));
    scalar_t egrad = c6 * (6 * f6 - expu * u6 * u / 720) * rinv6 * rinv * rinv;
    *ene = -f6*c6*rinv6;
    *gx = egrad * drx;
    *gy = egrad * dry;
    *gz = egrad * drz;
    if (c6_grad) {
        *c6_grad = -f6 * rinv6;
    }
    if (b_grad) {
        scalar_t df6_du = expu * u6 / scalar_t(720.0);
        *b_grad = -c6 * rinv6 * dr * df6_du;
    }
}


#endif
