#include <torch/library.h>
#include <torch/autograd.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

#include "common/vec3.cuh"
#include "common/reduce.cuh"


template <typename scalar_t, int BLOCK_SIZE>
__global__ void harmonic_bond_cuda_kernel(
    scalar_t* coords, 
    int64_t* bonds, 
    scalar_t* b0, 
    scalar_t* k, 
    int64_t nbonds,
    scalar_t* ene_out,
    scalar_t* coord_grad, 
    scalar_t* b0_grad, 
    scalar_t* k_grad,
    scalar_t sign
) {
    if ( ene_out && threadIdx.x == 0 && blockIdx.x == 0 ) {
        ene_out[0] = scalar_t(0.0);
    }
    __syncthreads();
    scalar_t ene = scalar_t(0.0);
    for (int index = threadIdx.x+blockIdx.x*BLOCK_SIZE; index < nbonds; index += BLOCK_SIZE*gridDim.x) {
        int offset = index * 2;
        int64_t offset_0 = bonds[offset] * 3;
        int64_t offset_1 = bonds[offset + 1] * 3;
        scalar_t* coords_0 = coords + offset_0;
        scalar_t* coords_1 = coords + offset_1;
        scalar_t dx = coords_1[0] - coords_0[0];
        scalar_t dy = coords_1[1] - coords_0[1];
        scalar_t dz = coords_1[2] - coords_0[2];
        scalar_t b = norm3d_(dx, dy, dz);
        
        scalar_t k_ = k[index];
        scalar_t db = (b - b0[index]);
    
        if ( ene_out ) {
            ene += k_ * db * db / 2;
        }

        if ( coord_grad ) {
            scalar_t prefix = k_ * db / b; 
            scalar_t gx = dx * prefix * sign;
            scalar_t gy = dy * prefix * sign;
            scalar_t gz = dz * prefix * sign;
            atomicAdd(&coord_grad[offset_0],      -gx);
            atomicAdd(&coord_grad[offset_0 + 1],  -gy);
            atomicAdd(&coord_grad[offset_0 + 2],  -gz);
        
            atomicAdd(&coord_grad[offset_1],      gx);
            atomicAdd(&coord_grad[offset_1 + 1],  gy);
            atomicAdd(&coord_grad[offset_1 + 2],  gz);
        }
        if ( k_grad ) {
            k_grad[index] = db * db / 2;
        }
        if ( b0_grad ) {
            b0_grad[index] = -k_ * db;
        }
    }
    if ( ene_out ) {
        block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
    }
}


class HarmonicBondFunctionCuda: public torch::autograd::Function<HarmonicBondFunctionCuda> {

public: 
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& bonds,
        at::Tensor& b0,
        at::Tensor& k
    )
    {
        int nbonds = bonds.size(0);
        
        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        int GRID_SIZE = std::min(
            (nbonds + BLOCK_SIZE - 1) / BLOCK_SIZE,
            props->multiProcessorCount*props->maxBlocksPerMultiProcessor
        );

        at::Tensor ene_out = at::empty({}, coords.options());
        at::Tensor coord_grad = at::zeros_like(coords, coords.options());
        at::Tensor b0_grad = at::empty_like(b0, b0.options());
        at::Tensor k_grad = at::empty_like(k, k.options());
        ctx->save_for_backward({coord_grad, b0_grad, k_grad});

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_harmonic_bond_cuda", ([&] {
            harmonic_bond_cuda_kernel<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                bonds.data_ptr<int64_t>(),
                b0.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                nbonds,
                ene_out.data_ptr<scalar_t>(),
                (coords.requires_grad()) ? coord_grad.data_ptr<scalar_t>() : nullptr,
                (b0.requires_grad()) ? b0_grad.data_ptr<scalar_t>() : nullptr,
                (k.requires_grad()) ? k_grad.data_ptr<scalar_t>() : nullptr,
                static_cast<scalar_t>(1.0)
            );
        }));
        
        return ene_out;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    )
    {
        auto saved = ctx->get_saved_variables();
        at::Tensor ignore;
        return {saved[0] * grad_outputs[0], ignore, saved[1] * grad_outputs[0], saved[2] * grad_outputs[0]};
    }
};


at::Tensor compute_harmonic_bond_energy_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& b0,
    at::Tensor& k
) {
    return HarmonicBondFunctionCuda::apply(coords, pairs, b0, k);
}


void compute_harmonic_bond_forces_cuda(
    at::Tensor& coords,
    at::Tensor& bonds,
    at::Tensor& b0,
    at::Tensor& k,
    at::Tensor& forces
) {

    int nbonds = bonds.size(0);
    auto props = at::cuda::getCurrentDeviceProperties();
    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = std::min(
        (nbonds + BLOCK_SIZE - 1) / BLOCK_SIZE,
        props->multiProcessorCount*props->maxBlocksPerMultiProcessor
    );
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_harmonic_bond_forces_cuda", ([&] {
        harmonic_bond_cuda_kernel<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            bonds.data_ptr<int64_t>(),
            b0.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            nbonds,
            nullptr,
            forces.data_ptr<scalar_t>(),
            nullptr,
            nullptr,
            static_cast<scalar_t>(-1.0)
        );
    }));
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_harmonic_bond_energy", compute_harmonic_bond_energy_cuda);
    m.impl("compute_harmonic_bond_forces", compute_harmonic_bond_forces_cuda);
}
