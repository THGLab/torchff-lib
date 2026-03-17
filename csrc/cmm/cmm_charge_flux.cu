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

template <typename scalar_t, int BLOCK_SIZE>
__global__ void cmm_charge_flux_bond_forward_kernel(
    scalar_t* coords,
    int64_t* bonds,
    scalar_t* req,
    scalar_t* j_cf,
    scalar_t* j_cf_pauli,
    int64_t nbonds,
    scalar_t* dq_a,
    scalar_t* dq_pauli_a
)
{

    int64_t start = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    for (int64_t index = start; index < nbonds; index += gridDim.x * BLOCK_SIZE) {
        int64_t i = bonds[index * 2];
        int64_t j = bonds[index * 2 + 1];
        scalar_t rx = coords[j*3] - coords[i*3];
        scalar_t ry = coords[j*3+1] - coords[i*3+1];
        scalar_t rz = coords[j*3+2] - coords[i*3+2];
        scalar_t dr = sqrt_(rx*rx+ry*ry+rz*rz) - req[index];
        scalar_t flux;
        
        flux = j_cf[index] * dr;
        atomicAdd(&dq_a[i], -flux);
        atomicAdd(&dq_a[j], flux);

        flux = j_cf_pauli[index] * dr;
        atomicAdd(&dq_pauli_a[i], -flux);
        atomicAdd(&dq_pauli_a[j], flux);
    }
}


template <typename scalar_t, int BLOCK_SIZE>
__global__ void cmm_charge_flux_bond_backward_kernel(
    scalar_t* coords,
    int64_t* bonds,
    scalar_t* req,
    scalar_t* j_cf,
    scalar_t* j_cf_pauli,
    int64_t nbonds,
    scalar_t* dq_a_grad,
    scalar_t* dq_pauli_a_grad,
    scalar_t* coords_grad
)
{
    int64_t start = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    for (int64_t index = start; index < nbonds; index += gridDim.x * BLOCK_SIZE) {
        int64_t i = bonds[index * 2];
        int64_t j = bonds[index * 2 + 1];
        scalar_t rx = coords[j*3] - coords[i*3];
        scalar_t ry = coords[j*3+1] - coords[i*3+1];
        scalar_t rz = coords[j*3+2] - coords[i*3+2];
        scalar_t r = sqrt_(rx*rx+ry*ry+rz*rz);
        scalar_t tmp = (-j_cf[index] * (dq_a_grad[j] - dq_a_grad[i]) - j_cf_pauli[index] * (dq_pauli_a_grad[j] - dq_pauli_a_grad[i])) / r;
        atomicAdd(&coords_grad[i*3],   tmp*rx);
        atomicAdd(&coords_grad[i*3+1], tmp*ry);
        atomicAdd(&coords_grad[i*3+2], tmp*rz);
        atomicAdd(&coords_grad[j*3],  -tmp*rx);
        atomicAdd(&coords_grad[j*3+1],-tmp*ry);
        atomicAdd(&coords_grad[j*3+2],-tmp*rz);
    }
}


class CMMBondChargeFluxCuda: public torch::autograd::Function<CMMBondChargeFluxCuda> {

public: 

static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& coords, at::Tensor& bonds, at::Tensor& req,
    at::Tensor& j_cf, at::Tensor& j_cf_pauli
)
{
    int64_t nbonds = bonds.size(0);
    int64_t natoms = coords.size(0);
    at::Tensor dq_a = at::zeros({natoms}, coords.options());
    at::Tensor dq_a_pauli = at::zeros({natoms}, coords.options());

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int BLOCK_SIZE = 256;
    int64_t grid_dim = std::min(
        static_cast<int64_t>(props->maxBlocksPerMultiProcessor * props->multiProcessorCount),
        (nbonds + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "cmm_charge_flux_bond_forward_kernel", ([&] {
        cmm_charge_flux_bond_forward_kernel<scalar_t, BLOCK_SIZE><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            bonds.data_ptr<int64_t>(),
            req.data_ptr<scalar_t>(),
            j_cf.data_ptr<scalar_t>(),
            j_cf_pauli.data_ptr<scalar_t>(),
            nbonds,
            dq_a.data_ptr<scalar_t>(),
            dq_a_pauli.data_ptr<scalar_t>()
        );
    }));

    ctx->save_for_backward({coords, bonds, req, j_cf, j_cf_pauli});
    ctx->saved_data["nbonds"] = nbonds;
    std::vector<at::Tensor> outs;
    outs.reserve(2);
    outs.push_back(dq_a);
    outs.push_back(dq_a_pauli);
    return outs;
}


static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<at::Tensor> grad_outputs
)
{
    auto saved = ctx->get_saved_variables();
    at::Tensor coords_grad = at::zeros_like(saved[0], saved[0].options());
    int64_t nbonds = static_cast<int64_t>(ctx->saved_data["nbonds"].toInt());

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int BLOCK_SIZE = 256;
    int64_t grid_dim = std::min(
        static_cast<int64_t>(props->maxBlocksPerMultiProcessor * props->multiProcessorCount),
        (nbonds + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(saved[0].scalar_type(), "cmm_charge_flux_bond_backward_kernel", ([&] {
        cmm_charge_flux_bond_backward_kernel<scalar_t, BLOCK_SIZE><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
            saved[0].data_ptr<scalar_t>(),
            saved[1].data_ptr<int64_t>(),
            saved[2].data_ptr<scalar_t>(),
            saved[3].data_ptr<scalar_t>(),
            saved[4].data_ptr<scalar_t>(),
            nbonds,
            grad_outputs[0].contiguous().data_ptr<scalar_t>(),
            grad_outputs[1].contiguous().data_ptr<scalar_t>(),
            coords_grad.data_ptr<scalar_t>()
        );
    }));
    at::Tensor ignore;
    return {coords_grad, ignore, ignore, ignore, ignore};
}

};

std::tuple<at::Tensor, at::Tensor> cmm_bond_charge_flux_cuda(
    at::Tensor& coords, at::Tensor& bonds, at::Tensor& req,
    at::Tensor& j_cf, at::Tensor& j_cf_pauli
) {
    auto outs = CMMBondChargeFluxCuda::apply(coords, bonds, req, j_cf, j_cf_pauli);
    return {outs[0], outs[1]};
}

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("cmm_bond_charge_flux", cmm_bond_charge_flux_cuda);
}
