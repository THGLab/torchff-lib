#ifndef TORCHFF_THOLE_DAMPS_CUH
#define TORCHFF_THOLE_DAMPS_CUH


template <typename scalar_t, int ORDER>
__device__ __forceinline__ void thole_damps(scalar_t r, scalar_t thole, scalar_t factor, scalar_t* damps) {

    scalar_t u = r * factor;
    scalar_t x = thole * u * u * u;
    scalar_t exp_x = exp_(-x);

    damps[0] += 0.0;

    if constexpr ( ORDER >= 3 ) {
        damps[1] += -exp_x;
    }
    if constexpr ( ORDER >= 5 ) {
        damps[2] += -(1.0 + x) * exp_x;
    }
    scalar_t x2 = x * x;
    if constexpr ( ORDER >= 7 ) {
        damps[3] += -(1.0 + x + 3.0/5.0*x2) * exp_x;
    }
    if constexpr ( ORDER >= 9 ) {
        damps[4] += -(1.0 + x + 18.0/35.0*x2 + 9.0/35.0*x2*x) * exp_x;
    }
    if constexpr ( ORDER >= 11 ) {
        damps[5] += -(1.0 + x + 53.0/105.0*x2 + 6.0/35.0*x2*x + 3.0/35.0*x2*x2) * exp_x;
    }
}


#endif