#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "bond/morse_bond.cuh"
#include "common/reduce.cuh"


template <typename scalar_t, int BLOCK_SIZE>
__global__ void cmm_field_dependent_morse_bond_kernel(
    scalar_t* coords,
    int64_t* bonds,
    scalar_t* req_0,
    scalar_t* kb_0,
    scalar_t* d,
    scalar_t* dipole_deriv_1,
    scalar_t* dipole_deriv_2,
    scalar_t* efield,
    int64_t nbonds,
    scalar_t* ene_out,
    scalar_t* coords_grad,
    scalar_t* efield_grad
)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ene_out[0] = scalar_t(0.0);
    }
    __syncthreads();

    scalar_t ene = scalar_t(0.0);

    int64_t start = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    for (int64_t index = start; index < nbonds; index += gridDim.x * BLOCK_SIZE) {
        int64_t i = bonds[index * 2];
        int64_t j = bonds[index * 2 + 1];
        scalar_t rx = coords[j*3] - coords[i*3];
        scalar_t ry = coords[j*3+1] - coords[i*3+1];
        scalar_t rz = coords[j*3+2] - coords[i*3+2];
        scalar_t r = sqrt_(rx*rx+ry*ry+rz*rz);
        scalar_t e, drx, dry, drz, defx, defy, defz;
        fd_morse_bond_from_kb_d(
            r, rx, ry, rz, req_0[index], kb_0[index], d[index],
            efield[j*3], efield[j*3+1], efield[j*3+2], dipole_deriv_1[index], dipole_deriv_2[index],
            &e, &drx, &dry, &drz, &defx, &defy, &defz
        );
        ene += e;
        atomicAdd(&coords_grad[i*3], -drx);
        atomicAdd(&coords_grad[i*3+1], -dry);
        atomicAdd(&coords_grad[i*3+2], -drz);

        atomicAdd(&coords_grad[j*3], drx);
        atomicAdd(&coords_grad[j*3+1], dry);
        atomicAdd(&coords_grad[j*3+2], drz);

        atomicAdd(&efield_grad[j*3], defx);
        atomicAdd(&efield_grad[j*3+1], defy);
        atomicAdd(&efield_grad[j*3+2], defz);

    }

    block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
}


class CMMFieldDependentMorseBondCuda: public torch::autograd::Function<CMMFieldDependentMorseBondCuda> {

public: 

static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& coords, at::Tensor& bonds,
    at::Tensor& req_0, at::Tensor& kb_0, at::Tensor& D,
    at::Tensor& dipole_deriv_1, at::Tensor& dipole_deriv_2, 
    at::Tensor& efield
)
{
    int64_t nbonds = bonds.size(0);

    auto opts = coords.options();
    at::Tensor coords_grad = at::zeros_like(coords, opts);
    at::Tensor efield_grad = at::zeros_like(efield, opts);
    at::Tensor ene = at::zeros({}, opts);

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int BLOCK_SIZE = 256;
    int64_t grid_dim = std::min(
        static_cast<int64_t>(props->maxBlocksPerMultiProcessor * props->multiProcessorCount),
        (nbonds + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "cmm_field_dependent_morse_bond_kernel", ([&] {
        cmm_field_dependent_morse_bond_kernel<scalar_t, BLOCK_SIZE><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            bonds.data_ptr<int64_t>(),
            req_0.data_ptr<scalar_t>(),
            kb_0.data_ptr<scalar_t>(),
            D.data_ptr<scalar_t>(),
            dipole_deriv_1.data_ptr<scalar_t>(),
            dipole_deriv_2.data_ptr<scalar_t>(),
            efield.data_ptr<scalar_t>(),
            nbonds,
            ene.data_ptr<scalar_t>(),
            coords_grad.data_ptr<scalar_t>(),
            efield_grad.data_ptr<scalar_t>()
        );
    }));
    
    ctx->save_for_backward({coords_grad, efield_grad});
    return ene;
}


static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<at::Tensor> grad_outputs
)
{
    auto saved = ctx->get_saved_variables();
    at::Tensor ignore;
    return {
        saved[0] * grad_outputs[0], 
        ignore, ignore, ignore, ignore, ignore, ignore,
        saved[1] * grad_outputs[0]
    };
}

};

at::Tensor cmm_field_dependent_morse_bond_cuda(
    at::Tensor& coords, at::Tensor& bonds,
    at::Tensor& req_0, at::Tensor& kb_0, at::Tensor& D,
    at::Tensor& dipole_deriv_1, at::Tensor& dipole_deriv_2, 
    at::Tensor& efield
) {
    return CMMFieldDependentMorseBondCuda::apply(
        coords, bonds, req_0, kb_0, D, dipole_deriv_1, dipole_deriv_2,
        efield
    );
}

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("cmm_field_dependent_morse_bond", cmm_field_dependent_morse_bond_cuda);
}
