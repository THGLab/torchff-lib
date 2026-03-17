#ifndef TORCHFF_VEC3_CUH
#define TORCHFF_VEC3_CUH

#include <cuda_runtime.h>

// sqrt
template <typename scalar_t> __device__ __forceinline__ scalar_t sqrt_(scalar_t x) {};
template<> __device__ __forceinline__ float sqrt_(float x) { return ::sqrtf(x); };
template<> __device__ __forceinline__ double sqrt_(double x) { return ::sqrt(x); };

// arccos
template <typename scalar_t> __device__ __forceinline__ scalar_t acos_(scalar_t x) {};
template<> __device__ __forceinline__ float acos_(float x) { return ::acosf(x); };
template<> __device__ __forceinline__ double acos_(double x) { return ::acos(x); };

// cos
template <typename scalar_t> __device__ __forceinline__ scalar_t cos_(scalar_t x) {};
template<> __device__ __forceinline__ float cos_(float x) { return ::cos(x); };
template<> __device__ __forceinline__ double cos_(double x) { return ::cosf(x); };

// sin
template <typename scalar_t> __device__ __forceinline__ scalar_t sin_(scalar_t x) {};
template<> __device__ __forceinline__ float sin_(float x) { return ::sin(x); };
template<> __device__ __forceinline__ double sin_(double x) { return ::sinf(x); };

// sincos
template <typename scalar_t> __device__ __forceinline__ void sincos_(scalar_t x, scalar_t* s, scalar_t* c) {};
template<> __device__ __forceinline__ void sincos_(float x, float* s, float* c) { ::sincosf(x, s, c); };
template<> __device__ __forceinline__ void sincos_(double x, double* s, double* c) { ::sincos(x, s, c); };

// clamp
template <typename scalar_t> __device__ __forceinline__ scalar_t clamp_(scalar_t x, scalar_t lo, scalar_t hi) {};
template<> __device__ __forceinline__ float clamp_(float x, float lo, float hi) { return ::fminf( ::fmaxf(x, lo), hi ); };
template<> __device__ __forceinline__ double clamp_(double x, double lo, double hi) { return ::fmin( ::fmax(x, lo), hi ); };

// pow
template <typename scalar_t> __device__ __forceinline__ scalar_t pow_(scalar_t x, scalar_t p) {};
template<> __device__ __forceinline__ float pow_(float x, float p) { return ::powf(x, p); };
template<> __device__ __forceinline__ double pow_(double x, double p) { return ::pow(x, p); };

// round
template <typename scalar_t> __device__ __forceinline__ scalar_t round_(scalar_t x) {};
template<> __device__ __forceinline__ float round_(float x) { return ::roundf(x); };
template<> __device__ __forceinline__ double round_(double x) { return ::round(x); };

// floor
template <typename scalar_t> __device__ __forceinline__ scalar_t floor_(scalar_t x) {};
template<> __device__ __forceinline__ float floor_(float x) { return ::floorf(x); };
template<> __device__ __forceinline__ double floor_(double x) { return ::floor(x); };

// ceil
template <typename scalar_t> __device__ __forceinline__ scalar_t ceil_(scalar_t x) {};
template<> __device__ __forceinline__ float ceil_(float x) { return ::ceilf(x); };
template<> __device__ __forceinline__ double ceil_(double x) { return ::ceil(x); };

// rsqrt
template <typename scalar_t> __device__ __forceinline__ scalar_t rsqrt_(scalar_t x) {};
template<> __device__ __forceinline__ float rsqrt_(float x) { return ::rsqrtf(x); };
template<> __device__ __forceinline__ double rsqrt_(double x) { return ::rsqrt(x); };

// abs
template <typename scalar_t> __device__ __forceinline__ scalar_t abs_(scalar_t x) {};
template<> __device__ __forceinline__ float abs_(float x) { return ::fabsf(x); };
template<> __device__ __forceinline__ double abs_(double x) { return ::fabs(x); };

// min
template <typename scalar_t> __device__ __forceinline__ scalar_t min_(scalar_t x, scalar_t y) {};
template<> __device__ __forceinline__ float min_(float x, float y) { return ::fminf( x, y ); };
template<> __device__ __forceinline__ double min_(double x, double y) { return ::fmin( x,y ); };

// max
template <typename scalar_t> __device__ __forceinline__ scalar_t max_(scalar_t x, scalar_t y) {};
template<> __device__ __forceinline__ float max_(float x, float y) { return ::fmaxf( x, y ); };
template<> __device__ __forceinline__ double max_(double x, double y) { return ::fmax( x,y ); };

// exp
template <typename scalar_t> __device__ __forceinline__ scalar_t exp_(scalar_t x) {};
template<> __device__ __forceinline__ float exp_(float x) { return ::expf( x ); };
template<> __device__ __forceinline__ double exp_(double x) { return ::exp( x ); };

// erf
template <typename scalar_t> __device__ __forceinline__ scalar_t erf_(scalar_t x) {};
template<> __device__ __forceinline__ float erf_(float x) { return ::erff( x ); };
template<> __device__ __forceinline__ double erf_(double x) { return ::erf( x ); };

// erfc
template <typename scalar_t> __device__ __forceinline__ scalar_t erfc_(scalar_t x) {};
template<> __device__ __forceinline__ float erfc_(float x) { return ::erfcf( x ); };
template<> __device__ __forceinline__ double erfc_(double x) { return ::erfc( x ); };

// norm3d
template <typename scalar_t> __device__ __forceinline__ scalar_t norm3d_(scalar_t a, scalar_t b, scalar_t c) {};
template<> __device__ __forceinline__ float norm3d_(float a, float b, float c) { return norm3df(a, b, c); };
template<> __device__ __forceinline__ double norm3d_(double a, double b, double c) { return norm3d(a, b, c); };

// rnorm3d
template <typename scalar_t> __device__ __forceinline__ scalar_t rnorm3d_(scalar_t a, scalar_t b, scalar_t c) {};
template<> __device__ __forceinline__ float rnorm3d_(float a, float b, float c) { return rnorm3df(a, b, c); };
template<> __device__ __forceinline__ double rnorm3d_(double a, double b, double c) { return rnorm3d(a, b, c); };


template <typename scalar_t>
__device__ __forceinline__ void cross_vec3(scalar_t* a, scalar_t* b, scalar_t* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename scalar_t>
__device__ __forceinline__ void diff_vec3(const scalar_t* a, const scalar_t* b, scalar_t* out) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t dot_vec3(scalar_t* a, scalar_t* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t norm_vec3(scalar_t* a) {
    return norm3d_(a[0], a[1], a[2]);
}

template <typename scalar_t>
__device__ __forceinline__ void normalize_vec3(scalar_t* a, scalar_t* out) {
    scalar_t rnorm = rnorm3d_(a[0], a[1], a[2]);
    out[0] = a[0] * rnorm;
    out[1] = a[1] * rnorm;
    out[2] = a[2] * rnorm;
}

template <typename scalar_t>
__device__ __forceinline__ void scalar_mult_vec3(scalar_t* vec, scalar_t s, scalar_t* out) {
    out[0] = vec[0] * s;
    out[1] = vec[1] * s;
    out[2] = vec[2] * s;
}

template <typename scalar_t>
__device__ __forceinline__ void add_vec3(scalar_t* a, scalar_t* b, scalar_t* out) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
}

#endif
