#include <torch/library.h>
#include <torch/autograd.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/reduce.cuh"
#include "common/vec3.cuh"

template <typename scalar_t, int BLOCK_SIZE>
__global__ void morse_bond_cuda_kernel(
    scalar_t* coords,
    int64_t* bonds,
    scalar_t* b0,
    scalar_t* kb,
    scalar_t* d,
    int64_t nbonds,
    scalar_t* ene_out,
    scalar_t* coord_grad,
    scalar_t* b0_grad,
    scalar_t* kb_grad,
    scalar_t* d_grad,
    scalar_t sign
) {
    if (ene_out && threadIdx.x == 0 && blockIdx.x == 0) {
        ene_out[0] = scalar_t(0.0);
    }
    __syncthreads();
    scalar_t ene = scalar_t(0.0);
    for (int index = threadIdx.x + blockIdx.x * BLOCK_SIZE; index < nbonds;
         index += BLOCK_SIZE * gridDim.x) {
        int offset = index * 2;
        int64_t offset_0 = bonds[offset] * 3;
        int64_t offset_1 = bonds[offset + 1] * 3;
        scalar_t* coords_0 = coords + offset_0;
        scalar_t* coords_1 = coords + offset_1;
        scalar_t dx = coords_1[0] - coords_0[0];
        scalar_t dy = coords_1[1] - coords_0[1];
        scalar_t dz = coords_1[2] - coords_0[2];
        scalar_t r = norm3d_(dx, dy, dz);

        scalar_t req = b0[index];
        scalar_t kb_ = kb[index];
        scalar_t d_ = d[index];
        scalar_t beta = sqrt_(kb_ / scalar_t(2.0) / d_);
        scalar_t dr = r - req;
        scalar_t z = scalar_t(1.0) - exp_(-beta * dr);
        scalar_t ene_i = d_ * z * z;
        scalar_t du_coord = scalar_t(2.0) * d_ * beta * z * (scalar_t(1.0) - z);
        scalar_t drx = dx * du_coord / r;
        scalar_t dry = dy * du_coord / r;
        scalar_t drz = dz * du_coord / r;

        if (ene_out) {
            ene += ene_i;
        }

        if (coord_grad) {
            scalar_t gx = drx * sign;
            scalar_t gy = dry * sign;
            scalar_t gz = drz * sign;
            atomicAdd(&coord_grad[offset_0], -gx);
            atomicAdd(&coord_grad[offset_0 + 1], -gy);
            atomicAdd(&coord_grad[offset_0 + 2], -gz);
            atomicAdd(&coord_grad[offset_1], gx);
            atomicAdd(&coord_grad[offset_1 + 1], gy);
            atomicAdd(&coord_grad[offset_1 + 2], gz);
        }
        if (b0_grad) {
            b0_grad[index] = -du_coord;
        }
        if (kb_grad) {
            kb_grad[index] = d_ * z * (scalar_t(1.0) - z) * dr / sqrt_(scalar_t(2.0) * kb_ * d_);
        }
        if (d_grad) {
            d_grad[index] = z * z - z * (scalar_t(1.0) - z) * dr * beta;
        }
    }
    if (ene_out) {
        block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
    }
}

class MorseBondFunctionCuda : public torch::autograd::Function<MorseBondFunctionCuda> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& bonds,
        at::Tensor& b0,
        at::Tensor& kb,
        at::Tensor& d
    ) {
        int nbonds = bonds.size(0);

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        int GRID_SIZE = std::min(
            (nbonds + BLOCK_SIZE - 1) / BLOCK_SIZE,
            props->multiProcessorCount * props->maxBlocksPerMultiProcessor
        );

        at::Tensor ene_out = at::empty({}, coords.options());
        at::Tensor coord_grad = at::zeros_like(coords, coords.options());
        at::Tensor b0_grad = at::empty_like(b0, b0.options());
        at::Tensor kb_grad = at::empty_like(kb, kb.options());
        at::Tensor d_grad = at::empty_like(d, d.options());
        ctx->save_for_backward({coord_grad, b0_grad, kb_grad, d_grad});

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_morse_bond_cuda", ([&] {
            morse_bond_cuda_kernel<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                bonds.data_ptr<int64_t>(),
                b0.data_ptr<scalar_t>(),
                kb.data_ptr<scalar_t>(),
                d.data_ptr<scalar_t>(),
                nbonds,
                ene_out.data_ptr<scalar_t>(),
                (coords.requires_grad()) ? coord_grad.data_ptr<scalar_t>() : nullptr,
                (b0.requires_grad()) ? b0_grad.data_ptr<scalar_t>() : nullptr,
                (kb.requires_grad()) ? kb_grad.data_ptr<scalar_t>() : nullptr,
                (d.requires_grad()) ? d_grad.data_ptr<scalar_t>() : nullptr,
                static_cast<scalar_t>(1.0)
            );
        }));

        return ene_out;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        at::Tensor ignore;
        return {
            saved[0] * grad_outputs[0],
            ignore,
            saved[1] * grad_outputs[0],
            saved[2] * grad_outputs[0],
            saved[3] * grad_outputs[0]
        };
    }
};

at::Tensor compute_morse_bond_energy_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& b0,
    at::Tensor& kb,
    at::Tensor& d
) {
    return MorseBondFunctionCuda::apply(coords, pairs, b0, kb, d);
}

void compute_morse_bond_forces_cuda(
    at::Tensor& coords,
    at::Tensor& bonds,
    at::Tensor& b0,
    at::Tensor& kb,
    at::Tensor& d,
    at::Tensor& forces
) {
    int nbonds = bonds.size(0);
    auto props = at::cuda::getCurrentDeviceProperties();
    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = std::min(
        (nbonds + BLOCK_SIZE - 1) / BLOCK_SIZE,
        props->multiProcessorCount * props->maxBlocksPerMultiProcessor
    );
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_morse_bond_forces_cuda", ([&] {
        morse_bond_cuda_kernel<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            bonds.data_ptr<int64_t>(),
            b0.data_ptr<scalar_t>(),
            kb.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            nbonds,
            nullptr,
            forces.data_ptr<scalar_t>(),
            nullptr,
            nullptr,
            nullptr,
            static_cast<scalar_t>(-1.0)
        );
    }));
}

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_morse_bond_energy", compute_morse_bond_energy_cuda);
    m.impl("compute_morse_bond_forces", compute_morse_bond_forces_cuda);
}
