#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"
#include "common/reduce.cuh"
#include "common/dispatch.cuh"
#include "dispersion/tang_tonnies.cuh"


template <typename scalar_t, int BLOCK_SIZE, bool USE_TYPE_PAIRS>
__global__ void tang_tonnies_dispersion_cuda_kernel(
    scalar_t* coords,
    int64_t* pairs,
    scalar_t* g_box,
    scalar_t cutoff,
    int64_t npairs,
    int64_t* atom_types,
    scalar_t* c6,
    scalar_t* b,
    int64_t ntypes,
    scalar_t* ene_out,
    scalar_t* coord_grad,
    scalar_t* c6_grad,
    scalar_t* b_grad
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

        scalar_t rij_vec[3];
        scalar_t tmp[3];
        diff_vec3(&coords[offset_i], &coords[offset_j], tmp);
        apply_pbc_triclinic(tmp, box, box_inv, rij_vec);

        scalar_t r = norm_vec3(rij_vec);
        if (r > cutoff) {
            continue;
        }

        scalar_t c6_ij = static_cast<scalar_t>(0.0);
        scalar_t b_ij = static_cast<scalar_t>(0.0);
        int64_t type_index = 0;
        if constexpr (USE_TYPE_PAIRS) {
            type_index = atom_types[i] * ntypes + atom_types[j];
            c6_ij = c6[type_index];
            b_ij = b[type_index];
        } else {
            c6_ij = c6[index];
            b_ij = b[index];
        }

        scalar_t pair_ene = static_cast<scalar_t>(0.0);
        scalar_t gx = static_cast<scalar_t>(0.0);
        scalar_t gy = static_cast<scalar_t>(0.0);
        scalar_t gz = static_cast<scalar_t>(0.0);
        scalar_t gc6 = static_cast<scalar_t>(0.0);
        scalar_t gb = static_cast<scalar_t>(0.0);

        tang_tonnies_6_dispersion(
            c6_ij,
            b_ij,
            r,
            rij_vec[0],
            rij_vec[1],
            rij_vec[2],
            &pair_ene,
            &gx,
            &gy,
            &gz,
            (c6_grad ? &gc6 : static_cast<scalar_t*>(nullptr)),
            (b_grad ? &gb : static_cast<scalar_t*>(nullptr)));

        ene += pair_ene;

        if constexpr (USE_TYPE_PAIRS) {
            if (c6_grad) {
                atomicAdd(&c6_grad[type_index], gc6);
            }
            if (b_grad) {
                atomicAdd(&b_grad[type_index], gb);
            }
        } else {
            if (c6_grad) {
                c6_grad[index] = gc6;
            }
            if (b_grad) {
                b_grad[index] = gb;
            }
        }

        if (coord_grad) {
            atomicAdd(&coord_grad[offset_i], gx);
            atomicAdd(&coord_grad[offset_i + 1], gy);
            atomicAdd(&coord_grad[offset_i + 2], gz);
            atomicAdd(&coord_grad[offset_j], -gx);
            atomicAdd(&coord_grad[offset_j + 1], -gy);
            atomicAdd(&coord_grad[offset_j + 2], -gz);
        }
    }

    if (ene_out) {
        block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
    }
}


class TangTonniesDispersionFunctionCuda : public torch::autograd::Function<TangTonniesDispersionFunctionCuda> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& pairs,
        at::Tensor& box,
        at::Tensor& c6,
        at::Tensor& b,
        at::Scalar cutoff,
        c10::optional<at::Tensor> atom_types_optional) {
        int64_t npairs = pairs.size(0);

        at::Tensor atom_types;
        int64_t ntypes = 0;
        bool use_type_pairs = false;
        if (atom_types_optional.has_value()) {
            atom_types = atom_types_optional.value();
            ntypes = c6.size(0);
            use_type_pairs = true;
        }

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        int GRID_SIZE = std::min(
            static_cast<int>((npairs + BLOCK_SIZE - 1) / BLOCK_SIZE),
            props->multiProcessorCount * props->maxBlocksPerMultiProcessor);

        auto opts = coords.options();

        at::Tensor ene = at::empty({}, opts);
        at::Tensor coord_grad = at::zeros_like(coords, opts);
        at::Tensor c6_grad = at::empty_like(c6, opts);
        if (c6.requires_grad()) {
            c6_grad.zero_();
        }
        at::Tensor b_grad = at::empty_like(b, opts);
        if (b.requires_grad()) {
            b_grad.zero_();
        }

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "tang_tonnies_dispersion_cuda_kernel", ([&] {
            DISPATCH_BOOL(use_type_pairs, USE_TYPE_PAIRS, [&] {
                tang_tonnies_dispersion_cuda_kernel<scalar_t, BLOCK_SIZE, USE_TYPE_PAIRS><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                    coords.data_ptr<scalar_t>(),
                    pairs.data_ptr<int64_t>(),
                    box.data_ptr<scalar_t>(),
                    cutoff.to<scalar_t>(),
                    npairs,
                    use_type_pairs ? atom_types.data_ptr<int64_t>() : nullptr,
                    c6.data_ptr<scalar_t>(),
                    b.data_ptr<scalar_t>(),
                    ntypes,
                    ene.data_ptr<scalar_t>(),
                    (coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr),
                    (c6.requires_grad() ? c6_grad.data_ptr<scalar_t>() : nullptr),
                    (b.requires_grad() ? b_grad.data_ptr<scalar_t>() : nullptr));
            });
        }));

        ctx->save_for_backward({coord_grad, c6_grad, b_grad, c6, b});
        return ene;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs) {
        auto saved = ctx->get_saved_variables();
        at::Tensor ignore;
        return {
            saved[0] * grad_outputs[0],
            ignore,
            ignore,
            (saved[3].requires_grad() ? saved[1] * grad_outputs[0] : ignore),
            (saved[4].requires_grad() ? saved[2] * grad_outputs[0] : ignore),
            ignore,
            ignore,
        };
    }
};


at::Tensor compute_tang_tonnies_dispersion_energy_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& box,
    at::Tensor& c6,
    at::Tensor& b,
    at::Scalar cutoff,
    c10::optional<at::Tensor> atom_types_optional) {
    return TangTonniesDispersionFunctionCuda::apply(coords, pairs, box, c6, b, cutoff, atom_types_optional);
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl(
        "compute_tang_tonnies_dispersion_energy",
        [](at::Tensor coords,
           at::Tensor pairs,
           at::Tensor box,
           at::Tensor c6,
           at::Tensor b,
           at::Scalar cutoff,
           c10::optional<at::Tensor> atom_types_optional) {
            return compute_tang_tonnies_dispersion_energy_cuda(coords, pairs, box, c6, b, cutoff, atom_types_optional);
        });
}
