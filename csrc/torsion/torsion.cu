#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"


template <typename scalar_t>
__global__ void torsion_cuda_kernel(
    const scalar_t* __restrict__ coords,
    const int64_t* __restrict__ torsions,
    const int64_t ntors,
    scalar_t* __restrict__ phi_grad_ptr,
    scalar_t* __restrict__ phi_out,
    scalar_t* __restrict__ coord_grad
) {
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < ntors;
         index += blockDim.x * gridDim.x) {

        const int offset = index * 4;
        const int64_t offset_i = 3 * torsions[offset];
        const int64_t offset_j = 3 * torsions[offset + 1];
        const int64_t offset_k = 3 * torsions[offset + 2];
        const int64_t offset_l = 3 * torsions[offset + 3];

        const scalar_t* coords_i = coords + offset_i;
        const scalar_t* coords_j = coords + offset_j;
        const scalar_t* coords_k = coords + offset_k;
        const scalar_t* coords_l = coords + offset_l;

        scalar_t phi_grad = 1.0;
        if ( phi_grad_ptr ) {
            phi_grad = phi_grad_ptr[index];
        }

        scalar_t b1[3];
        scalar_t b2[3];
        scalar_t b3[3];

        scalar_t n1[3];
        scalar_t n2[3];

        diff_vec3(coords_j, coords_i, b1);
        diff_vec3(coords_k, coords_j, b2);
        diff_vec3(coords_l, coords_k, b3);

        cross_vec3(b1, b2, n1);
        cross_vec3(b2, b3, n2);

        const scalar_t norm_n1 = norm_vec3(n1);
        const scalar_t norm_n2 = norm_vec3(n2);
        const scalar_t norm_b2 = norm_vec3(b2);
        const scalar_t norm_b2_sqr = norm_b2 * norm_b2;

        scalar_t cosval = dot_vec3(n1, n2) / (norm_n1 * norm_n2);
        cosval = clamp_(cosval, scalar_t(-0.999999999), scalar_t(0.99999999));

        const scalar_t sign_phi = dot_vec3(n1, b3) > 0.0 ? scalar_t(1.0) : scalar_t(-1.0);
        scalar_t phi = acos_(cosval) * sign_phi;

        if (phi_out) {
            phi_out[index] = phi;
        }

        if (coord_grad) {
            // Reuse the same geometry derivatives as periodic_torsion, but with prefactor = dphi/dcos
            const scalar_t aux1 = dot_vec3(b1, b2) / norm_b2_sqr;
            const scalar_t aux2 = dot_vec3(b2, b3) / norm_b2_sqr;

            scalar_t cgi, cgj, cgk, cgl;
            #pragma unroll 3
            for (int i = 0; i < 3; i++) {
                cgi = -norm_b2 / (norm_n1 * norm_n1) * n1[i];
                cgl =  norm_b2 / (norm_n2 * norm_n2) * n2[i];
                cgj = (-aux1 - 1) * cgi + aux2 * cgl;
                cgk = -cgi - cgj - cgl;
                atomicAdd(&coord_grad[offset_i + i], cgi * phi_grad);
                atomicAdd(&coord_grad[offset_j + i], cgj * phi_grad);
                atomicAdd(&coord_grad[offset_l + i], cgl * phi_grad);
                atomicAdd(&coord_grad[offset_k + i], cgk * phi_grad);
            }
        }
    }
}


class TorsionFunctionCuda : public torch::autograd::Function<TorsionFunctionCuda> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& torsions
    ) {

        const auto ntors = torsions.size(0);

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        const int GRID_SIZE = std::min(
            static_cast<int>((ntors + BLOCK_SIZE - 1) / BLOCK_SIZE),
            props->multiProcessorCount * props->maxBlocksPerMultiProcessor
        );

        at::Tensor phi = at::empty({ntors}, coords.options());

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_torsion_cuda", ([&] {
            torsion_cuda_kernel<scalar_t><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                torsions.data_ptr<int64_t>(),
                ntors,
                nullptr,
                phi.data_ptr<scalar_t>(),
                nullptr
            );
        }));

        ctx->save_for_backward({coords, torsions});
        return phi;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& coords = saved[0];
        const at::Tensor& torsions = saved[1];
        const at::Tensor& grad_phi = grad_outputs[0];

        const auto ntors = torsions.size(0);

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        const int GRID_SIZE = std::min(
            static_cast<int>((ntors + BLOCK_SIZE - 1) / BLOCK_SIZE),
            props->multiProcessorCount * props->maxBlocksPerMultiProcessor
        );

        at::Tensor grad_coords = at::zeros_like(coords, coords.options());

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_torsion_cuda_backward", ([&] {
            torsion_cuda_kernel<scalar_t><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                torsions.data_ptr<int64_t>(),
                ntors,
                grad_phi.data_ptr<scalar_t>(),
                nullptr,
                grad_coords.data_ptr<scalar_t>()
            );
        }));

        at::Tensor ignore;
        return {grad_coords, ignore};
    }
};


at::Tensor compute_torsion_cuda(
    at::Tensor& coords,
    at::Tensor& torsions
) {
    return TorsionFunctionCuda::apply(coords, torsions);
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_torsion", compute_torsion_cuda);
}

