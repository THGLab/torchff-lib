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
#include "common/dispatch.cuh"


template <typename scalar_t, bool USE_LJ=true>
__device__ __forceinline__ void vdw_pairwise_kernel(
    scalar_t& drx, scalar_t& dry, scalar_t& drz, scalar_t& dr, 
    scalar_t& sigma_ij, scalar_t& epsilon_ij,
    scalar_t* ene, scalar_t* dr_grad, scalar_t* sigma_ij_grad, scalar_t* epsilon_ij_grad
) {
    if constexpr (USE_LJ) {
        scalar_t rho = sigma_ij / dr;
        scalar_t rho2 = rho * rho;
        scalar_t rho6 = rho2 * rho2 * rho2;
        scalar_t rho12 = rho6 * rho6;
        scalar_t dedrho = scalar_t(24.0) * epsilon_ij * (2*rho12 - rho6) / rho;
    
        if ( ene ) {
            *ene += scalar_t(4.0) * epsilon_ij * (rho12 - rho6);
        }
        if ( dr_grad ) {
            *dr_grad += -dedrho * rho / dr;
        }
        if ( sigma_ij_grad ) {
            *sigma_ij_grad += dedrho / dr;
        }
        if ( epsilon_ij_grad ) {
            *epsilon_ij_grad += scalar_t(4.0) * (rho12 - rho6);
        }
    }
    else {
        constexpr scalar_t dhal = scalar_t(0.07);
        constexpr scalar_t ghal = scalar_t(0.12);
        constexpr scalar_t dhal1 = scalar_t(1.07);
        constexpr scalar_t ghal1 = scalar_t(1.12);

        scalar_t rho = dr / sigma_ij;
        scalar_t rho2 = rho * rho;
        scalar_t rho6 = rho2 * rho2 * rho2;
        scalar_t rhop = rho + dhal;
        scalar_t rhop2 = rhop * rhop;
        scalar_t rhop6 = rhop2 * rhop2 * rhop2;
        scalar_t s1 = scalar_t(1.0) / (rhop6 * rhop);
        scalar_t s2 = scalar_t(1.0) / (rho6 * rho + ghal);

        scalar_t dhal1_squared = dhal1 * dhal1;
        scalar_t t1 = dhal1 * dhal1_squared * dhal1_squared * dhal1_squared * s1;
        scalar_t t2 = ghal1 * s2;
        scalar_t t2min = t2 - scalar_t(2.0);
        scalar_t dt1drho = -scalar_t(7.0) * t1 * s1 * rhop6;
        scalar_t dt2drho = -scalar_t(7.0) * t2 * s2 * rho6;
        scalar_t dedrho = epsilon_ij * (dt1drho * t2min + t1 * dt2drho);

        if ( ene ) {
            *ene += epsilon_ij * t1 * t2min;
        }
        if ( dr_grad ) {
            *dr_grad = dedrho / sigma_ij;
        }
        if ( sigma_ij_grad ) {
            *sigma_ij_grad = -dedrho * dr / sigma_ij / sigma_ij;
        }
        if ( epsilon_ij_grad ) {
            *epsilon_ij_grad = t1 * t2min;
        }
    }
}


template <typename scalar_t, int BLOCK_SIZE, bool USE_LJ=true, bool USE_TYPE_PAIRS=false>
__global__ void vdw_cuda_kernel(
    scalar_t* coords,
    int64_t* pairs,
    scalar_t* g_box,
    scalar_t cutoff,
    int64_t npairs,
    int64_t* atom_types,
    scalar_t* sigma,
    scalar_t* epsilon,
    int64_t ntypes,
    scalar_t* ene_out,
    scalar_t* coord_grad,
    scalar_t* sigma_grad,
    scalar_t* epsilon_grad
) {
    if ( ene_out && threadIdx.x == 0 && blockIdx.x == 0 ) {
        ene_out[0] = scalar_t(0.0);
    }
    __syncthreads();

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

    scalar_t ene = static_cast<scalar_t>(0.0);
    scalar_t sigma_ij_grad = static_cast<scalar_t>(0.0);
    scalar_t epsilon_ij_grad = static_cast<scalar_t>(0.0);
    scalar_t dedr = static_cast<scalar_t>(0.0);
    scalar_t sigma_ij = static_cast<scalar_t>(0.0);
    scalar_t epsilon_ij = static_cast<scalar_t>(0.0);
    
    for (int64_t index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
         index < npairs;
         index += BLOCK_SIZE * gridDim.x) {
        int64_t i = pairs[index * 2];
        int64_t j = pairs[index * 2 + 1];
        if (i < 0 || j < 0) {
            continue;
        }
        int64_t offset_i = 3 * i;
        int64_t offset_j = 3 * j;

        scalar_t rij_vec[3];
        scalar_t tmp[3];
        diff_vec3(&coords[offset_i], &coords[offset_j], tmp);
        apply_pbc_triclinic(tmp, box, box_inv, rij_vec);

        scalar_t r = norm_vec3(rij_vec);
        if (r > cutoff) {
            continue;
        }

        if constexpr (USE_TYPE_PAIRS) {
            int64_t type_index = atom_types[i] * ntypes + atom_types[j];
            sigma_ij = sigma[type_index];
            epsilon_ij = epsilon[type_index];
        }
        else {
            sigma_ij = sigma[index];
            epsilon_ij = epsilon[index];
        }

        int64_t type_index = atom_types[i] * ntypes + atom_types[j];
        vdw_pairwise_kernel<scalar_t, USE_LJ>(
            rij_vec[0], rij_vec[1], rij_vec[2], r, sigma_ij, epsilon_ij,
            &ene, 
            (coord_grad ? &dedr : nullptr), 
            (sigma_grad ? &sigma_ij_grad : nullptr), 
            (epsilon_grad ? &epsilon_ij_grad : nullptr)
        );

        // Sigma and epsilon gradients
        if constexpr (USE_TYPE_PAIRS) {
            if ( sigma_grad ) {
                atomicAdd(&sigma_grad[type_index], sigma_ij_grad);
            }
            if ( epsilon_grad ) {
                atomicAdd(&epsilon_grad[type_index], epsilon_ij_grad);
            }
        } else {
            if ( sigma_grad ) {
                sigma_grad[index] = sigma_ij_grad;
            }
            if ( epsilon_grad ) {
                epsilon_grad[index] = epsilon_ij_grad;
            }
        }

        // Coordinate and radius gradients share common tmp_force
        if ( coord_grad ) {
            scalar_t drx = dedr * rij_vec[0] / r;
            scalar_t dry = dedr * rij_vec[1] / r;
            scalar_t drz = dedr * rij_vec[2] / r;
            atomicAdd(&coord_grad[offset_i],   drx);
            atomicAdd(&coord_grad[offset_i+1], dry);
            atomicAdd(&coord_grad[offset_i+2], drz);
            atomicAdd(&coord_grad[offset_j],   -drx);
            atomicAdd(&coord_grad[offset_j+1], -dry);
            atomicAdd(&coord_grad[offset_j+2], -drz);
        }
    }

    if (ene_out) {
        block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
    }
}


class VdwFunctionCuda : public torch::autograd::Function<VdwFunctionCuda> {
public:

static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& box,
    at::Tensor& sigma,
    at::Tensor& epsilon,
    at::Scalar cutoff,
    c10::optional<at::Tensor> atom_types_optional,
    at::Scalar use_lj
) {
    int64_t npairs = pairs.size(0);

    // use atom types
    at::Tensor atom_types;
    int64_t ntypes = 0;
    bool use_type_pairs = false;
    if (atom_types_optional.has_value()) {
        atom_types = atom_types_optional.value();
        ntypes = sigma.size(0);
        use_type_pairs = true;
    }

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = std::min(
        static_cast<int>((npairs + BLOCK_SIZE - 1) / BLOCK_SIZE),
        props->multiProcessorCount * props->maxBlocksPerMultiProcessor
    );

    auto opts = coords.options();

    at::Tensor ene = at::empty({}, opts);
    at::Tensor coord_grad = at::zeros_like(coords, opts);
    at::Tensor sigma_grad = at::empty_like(sigma, opts);
    if ( sigma.requires_grad() ) {
        sigma_grad.zero_();
    }
    at::Tensor epsilon_grad = at::empty_like(epsilon, opts);
    if ( epsilon.requires_grad() ) {
        epsilon_grad.zero_();
    }

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "vdw_cuda_kernel", ([&] {
        DISPATCH_BOOL(use_lj.to<bool>(), USE_LJ, [&] {
            DISPATCH_BOOL(use_type_pairs, USE_TYPE_PAIRS, [&] {
                vdw_cuda_kernel<scalar_t, BLOCK_SIZE, USE_LJ, USE_TYPE_PAIRS><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                    coords.data_ptr<scalar_t>(),
                    pairs.data_ptr<int64_t>(),
                    box.data_ptr<scalar_t>(),
                    cutoff.to<scalar_t>(),
                    npairs,
                    use_type_pairs ? atom_types.data_ptr<int64_t>() : nullptr,
                    sigma.data_ptr<scalar_t>(),
                    epsilon.data_ptr<scalar_t>(),
                    ntypes,
                    ene.data_ptr<scalar_t>(),
                    (coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr),
                    (sigma.requires_grad() ? sigma_grad.data_ptr<scalar_t>() : nullptr),
                    (epsilon.requires_grad() ? epsilon_grad.data_ptr<scalar_t>() : nullptr)
                );
            });
        });
    }));

    ctx->save_for_backward({coord_grad, sigma_grad, epsilon_grad, sigma, epsilon});
    return ene;
}

static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<at::Tensor> grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    at::Tensor ignore;
    return {
        saved[0] * grad_outputs[0], // coords
        ignore,                     // pairs
        ignore,                     // box
        (saved[3].requires_grad() ? saved[1] * grad_outputs[0] : ignore), // sigma
        (saved[4].requires_grad() ? saved[2] * grad_outputs[0] : ignore), // epsilon
        ignore,                     // cutoff
        ignore,                     // atom_types
        ignore                      // use_lj
    };
}

};


at::Tensor compute_lennard_jones_energy_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& box,
    at::Tensor& sigma,
    at::Tensor& epsilon,
    at::Scalar cutoff,
    c10::optional<at::Tensor> atom_types_optional
) {
    at::Scalar use_lj = at::Scalar(true);
    return VdwFunctionCuda::apply(coords, pairs, box, sigma, epsilon, cutoff, atom_types_optional, use_lj);
}


at::Tensor compute_vdw_14_7_energy_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& box,
    at::Tensor& radius,
    at::Tensor& epsilon,
    at::Scalar cutoff,
    c10::optional<at::Tensor> atom_types_optional
) {
    at::Scalar use_lj = at::Scalar(false);
    return VdwFunctionCuda::apply(coords, pairs, box, radius, epsilon, cutoff, atom_types_optional, use_lj);
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_lennard_jones_energy",
        [](at::Tensor coords,
           at::Tensor pairs,
           at::Tensor box,
           at::Tensor sigma,
           at::Tensor epsilon,
           at::Scalar cutoff,
           c10::optional<at::Tensor> atom_types_optional) {
            return compute_lennard_jones_energy_cuda(coords, pairs, box, sigma, epsilon, cutoff, atom_types_optional);
        });

    m.impl("compute_vdw_14_7_energy",
        [](at::Tensor coords,
           at::Tensor pairs,
           at::Tensor box,
           at::Tensor radius,
           at::Tensor epsilon,
           at::Scalar cutoff,
           c10::optional<at::Tensor> atom_types_optional) {
            return compute_vdw_14_7_energy_cuda(coords, pairs, box, radius, epsilon, cutoff, atom_types_optional);
        });
}

