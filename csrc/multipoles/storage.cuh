#ifndef TORCHFF_MULTIPOLES_STORAGE_CUH
#define TORCHFF_MULTIPOLES_STORAGE_CUH

template <typename T, int RANK>
struct CartesianExpansion;

template <typename T>
struct CartesianExpansion<T,0> {
    T s;
    __device__ __host__ CartesianExpansion() : s(T(0)) {}
};

template <typename T>
struct CartesianExpansion<T,1> {
    T s, x, y, z;
    __device__ __host__ CartesianExpansion() : s(T(0)), x(T(0)), y(T(0)), z(T(0)) {}
};

template <typename T>
struct CartesianExpansion<T,2> {
    T s, x, y, z, xx, xy, xz, yy, yz, zz;
    __device__ __host__ CartesianExpansion() : s(T(0)), x(T(0)), y(T(0)), z(T(0)), xx(T(0)), xy(T(0)), xz(T(0)), yy(T(0)), yz(T(0)), zz(T(0)) {}
};

template <typename T>
struct CartesianExpansion<T,3> {
    T s, x, y, z, xx, xy, xz, yy, yz, zz, xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
    __device__ __host__ CartesianExpansion() : s(T(0)), x(T(0)), y(T(0)), z(T(0)), xx(T(0)), xy(T(0)), xz(T(0)), yy(T(0)), yz(T(0)), zz(T(0)),
        xxx(T(0)), xxy(T(0)), xxz(T(0)), xyy(T(0)), xyz(T(0)), xzz(T(0)), yyy(T(0)), yyz(T(0)), yzz(T(0)), zzz(T(0)) {}
};

template <typename T>
struct CartesianExpansion<T,4> {
    T s, x, y, z, xx, xy, xz, yy, yz, zz, xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
    T xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz, yyyy, yyyz, yyzz, yzzz, zzzz;
    __device__ __host__ CartesianExpansion() : s(T(0)), x(T(0)), y(T(0)), z(T(0)), xx(T(0)), xy(T(0)), xz(T(0)), yy(T(0)), yz(T(0)), zz(T(0)),
        xxx(T(0)), xxy(T(0)), xxz(T(0)), xyy(T(0)), xyz(T(0)), xzz(T(0)), yyy(T(0)), yyz(T(0)), yzz(T(0)), zzz(T(0)),
        xxxx(T(0)), xxxy(T(0)), xxxz(T(0)), xxyy(T(0)), xxyz(T(0)), xxzz(T(0)), xyyy(T(0)), xyyz(T(0)), xyzz(T(0)), xzzz(T(0)),
        yyyy(T(0)), yyyz(T(0)), yyzz(T(0)), yzzz(T(0)), zzzz(T(0)) {}
};

#endif