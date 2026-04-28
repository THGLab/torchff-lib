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

// E = A * P * exp(-x),  P = x^2/3 + x + 1,  x = B * r
template <typename scalar_t>
__device__ __forceinline__ void slater_pairwise_kernel(
    scalar_t& drx,
    scalar_t& dry,
    scalar_t& drz,
    scalar_t& dr,
    scalar_t& A_ij,
    scalar_t& B_ij,
    scalar_t* ene,
    scalar_t* dr_grad,
    scalar_t* A_ij_grad,
    scalar_t* B_ij_grad
) {
    scalar_t x = B_ij * dr;
    scalar_t exp_mx = exp_(-x);
    scalar_t x2 = x * x;
    scalar_t P = x2 / scalar_t(3.0) + x + scalar_t(1.0);
    scalar_t common = exp_mx * x * (x + scalar_t(1.0)) / scalar_t(3.0);

    if (ene) {
        *ene += A_ij * P * exp_mx;
    }
    if (dr_grad) {
        *dr_grad = -A_ij * B_ij * common;
    }
    if (A_ij_grad) {
        *A_ij_grad = P * exp_mx;
    }
    if (B_ij_grad) {
        *B_ij_grad = -A_ij * dr * common;
    }
}

template <typename scalar_t, int BLOCK_SIZE, bool USE_TYPE_PAIRS = false>
__global__ void slater_cuda_kernel(
    scalar_t* coords,
    int64_t* pairs,
    scalar_t* g_box,
    scalar_t cutoff,
    int64_t npairs,
    int64_t* atom_types,
    scalar_t* A,
    scalar_t* B,
    int64_t ntypes,
    scalar_t* ene_out,
    scalar_t* coord_grad,
    scalar_t* A_grad,
    scalar_t* B_grad
) {
    if (ene_out && threadIdx.x == 0 && blockIdx.x == 0) {
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

    for (int64_t index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
         index < npairs;
         index += BLOCK_SIZE * gridDim.x) {
        int64_t i = pairs[index * 2];
        int64_t j = pairs[index * 2 + 1];
        int64_t offset_i = 3 * i;
        int64_t offset_j = 3 * j;

        if (i < 0 || j < 0) {
            continue;
        }

        scalar_t rij_vec[3];
        scalar_t tmp[3];
        diff_vec3(&coords[offset_i], &coords[offset_j], tmp);
        apply_pbc_triclinic(tmp, box, box_inv, rij_vec);

        scalar_t r = norm_vec3(rij_vec);
        if (r > cutoff) {
            continue;
        }

        scalar_t A_ij = static_cast<scalar_t>(0.0);
        scalar_t B_ij = static_cast<scalar_t>(0.0);
        int64_t type_index = 0;

        if constexpr (USE_TYPE_PAIRS) {
            type_index = atom_types[i] * ntypes + atom_types[j];
            A_ij = A[type_index];
            B_ij = B[type_index];
        } else {
            A_ij = A[index];
            B_ij = B[index];
        }

        scalar_t A_ij_grad = static_cast<scalar_t>(0.0);
        scalar_t B_ij_grad = static_cast<scalar_t>(0.0);
        scalar_t dedr = static_cast<scalar_t>(0.0);

        slater_pairwise_kernel<scalar_t>(
            rij_vec[0],
            rij_vec[1],
            rij_vec[2],
            r,
            A_ij,
            B_ij,
            &ene,
            (coord_grad ? &dedr : nullptr),
            (A_grad ? &A_ij_grad : nullptr),
            (B_grad ? &B_ij_grad : nullptr)
        );

        if constexpr (USE_TYPE_PAIRS) {
            if (A_grad) {
                atomicAdd(&A_grad[type_index], A_ij_grad);
            }
            if (B_grad) {
                atomicAdd(&B_grad[type_index], B_ij_grad);
            }
        } else {
            if (A_grad) {
                A_grad[index] = A_ij_grad;
            }
            if (B_grad) {
                B_grad[index] = B_ij_grad;
            }
        }

        if (coord_grad) {
            scalar_t drx = dedr * rij_vec[0] / r;
            scalar_t dry = dedr * rij_vec[1] / r;
            scalar_t drz = dedr * rij_vec[2] / r;
            atomicAdd(&coord_grad[offset_i], drx);
            atomicAdd(&coord_grad[offset_i + 1], dry);
            atomicAdd(&coord_grad[offset_i + 2], drz);
            atomicAdd(&coord_grad[offset_j], -drx);
            atomicAdd(&coord_grad[offset_j + 1], -dry);
            atomicAdd(&coord_grad[offset_j + 2], -drz);
        }
    }

    if (ene_out) {
        block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
    }
}

class SlaterFunctionCuda : public torch::autograd::Function<SlaterFunctionCuda> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& pairs,
        at::Tensor& box,
        at::Tensor& A,
        at::Tensor& B,
        at::Scalar cutoff,
        c10::optional<at::Tensor> atom_types_optional
    ) {
        int64_t npairs = pairs.size(0);

        at::Tensor atom_types;
        int64_t ntypes = 0;
        bool use_type_pairs = false;
        if (atom_types_optional.has_value()) {
            atom_types = atom_types_optional.value();
            ntypes = A.size(0);
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
        at::Tensor A_grad = at::empty_like(A, opts);
        if (A.requires_grad()) {
            A_grad.zero_();
        }
        at::Tensor B_grad = at::empty_like(B, opts);
        if (B.requires_grad()) {
            B_grad.zero_();
        }

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "slater_cuda_kernel", ([&] {
            DISPATCH_BOOL(use_type_pairs, USE_TYPE_PAIRS, [&] {
                slater_cuda_kernel<scalar_t, BLOCK_SIZE, USE_TYPE_PAIRS><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                    coords.data_ptr<scalar_t>(),
                    pairs.data_ptr<int64_t>(),
                    box.data_ptr<scalar_t>(),
                    cutoff.to<scalar_t>(),
                    npairs,
                    use_type_pairs ? atom_types.data_ptr<int64_t>() : nullptr,
                    A.data_ptr<scalar_t>(),
                    B.data_ptr<scalar_t>(),
                    ntypes,
                    ene.data_ptr<scalar_t>(),
                    (coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr),
                    (A.requires_grad() ? A_grad.data_ptr<scalar_t>() : nullptr),
                    (B.requires_grad() ? B_grad.data_ptr<scalar_t>() : nullptr)
                );
            });
        }));

        ctx->save_for_backward({coord_grad, A_grad, B_grad, A, B});
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
            (saved[3].requires_grad() ? saved[1] * grad_outputs[0] : ignore), // A
            (saved[4].requires_grad() ? saved[2] * grad_outputs[0] : ignore), // B
            ignore,                     // cutoff
            ignore                      // atom_types
        };
    }
};

at::Tensor compute_slater_energy_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& box,
    at::Tensor& A,
    at::Tensor& B,
    at::Scalar cutoff,
    c10::optional<at::Tensor> atom_types_optional
) {
    return SlaterFunctionCuda::apply(coords, pairs, box, A, B, cutoff, atom_types_optional);
}

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl(
        "compute_slater_energy",
        [](at::Tensor coords,
           at::Tensor pairs,
           at::Tensor box,
           at::Tensor A,
           at::Tensor B,
           at::Scalar cutoff,
           c10::optional<at::Tensor> atom_types_optional) {
            return compute_slater_energy_cuda(coords, pairs, box, A, B, cutoff, atom_types_optional);
        }
    );
}
