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
__global__ void angles_cuda_kernel(
    const scalar_t* coords,
    const int64_t* angles,
    int64_t nangles,
    scalar_t* theta_out,
    const scalar_t* grad_theta,
    scalar_t* coord_grad
) {
    for (int index = threadIdx.x + blockIdx.x * blockDim.x; index < nangles;
         index += blockDim.x * gridDim.x) {
        int64_t offset = index * 3;
        int64_t offset_0 = angles[offset] * 3;
        int64_t offset_1 = angles[offset + 1] * 3;
        int64_t offset_2 = angles[offset + 2] * 3;

        const scalar_t* coords_0 = coords + offset_0;
        const scalar_t* coords_1 = coords + offset_1;
        const scalar_t* coords_2 = coords + offset_2;

        scalar_t v1x = coords_0[0] - coords_1[0];
        scalar_t v1y = coords_0[1] - coords_1[1];
        scalar_t v1z = coords_0[2] - coords_1[2];

        scalar_t v2x = coords_2[0] - coords_1[0];
        scalar_t v2y = coords_2[1] - coords_1[1];
        scalar_t v2z = coords_2[2] - coords_1[2];

        scalar_t v1_norm = sqrt_(v1x * v1x + v1y * v1y + v1z * v1z);
        scalar_t v2_norm = sqrt_(v2x * v2x + v2y * v2y + v2z * v2z);

        scalar_t dot_product = v1x * v2x + v1y * v2y + v1z * v2z;
        scalar_t cos_theta = dot_product / (v1_norm * v2_norm);
        cos_theta = clamp_(cos_theta, scalar_t(-1.0), scalar_t(1.0));

        scalar_t theta = acos_(cos_theta);

        if (theta_out) {
            theta_out[index] = theta;
        }

        if (coord_grad && grad_theta) {
            scalar_t gt = grad_theta[index];
            scalar_t sin_sq = scalar_t(1.0) - cos_theta * cos_theta;
            sin_sq = max_(sin_sq, scalar_t(1e-20));
            scalar_t dtheta_dcos = -scalar_t(1.0) / sqrt_(sin_sq);
            scalar_t prefix = gt * dtheta_dcos / (v1_norm * v2_norm);

            scalar_t g1x = prefix * (v2x - cos_theta * v1x / v1_norm * v2_norm);
            scalar_t g1y = prefix * (v2y - cos_theta * v1y / v1_norm * v2_norm);
            scalar_t g1z = prefix * (v2z - cos_theta * v1z / v1_norm * v2_norm);

            scalar_t g3x = prefix * (v1x - cos_theta * v2x / v2_norm * v1_norm);
            scalar_t g3y = prefix * (v1y - cos_theta * v2y / v2_norm * v1_norm);
            scalar_t g3z = prefix * (v1z - cos_theta * v2z / v2_norm * v1_norm);

            atomicAdd(&coord_grad[offset_0], g1x);
            atomicAdd(&coord_grad[offset_0 + 1], g1y);
            atomicAdd(&coord_grad[offset_0 + 2], g1z);

            atomicAdd(&coord_grad[offset_1], -g1x - g3x);
            atomicAdd(&coord_grad[offset_1 + 1], -g1y - g3y);
            atomicAdd(&coord_grad[offset_1 + 2], -g1z - g3z);

            atomicAdd(&coord_grad[offset_2], g3x);
            atomicAdd(&coord_grad[offset_2 + 1], g3y);
            atomicAdd(&coord_grad[offset_2 + 2], g3z);
        }
    }
}

class AnglesFunctionCuda : public torch::autograd::Function<AnglesFunctionCuda> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& angles
    ) {
        const auto nangles = angles.size(0);

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        const int GRID_SIZE = std::min(
            static_cast<int>((nangles + BLOCK_SIZE - 1) / BLOCK_SIZE),
            props->multiProcessorCount * props->maxBlocksPerMultiProcessor
        );

        at::Tensor theta = at::empty({nangles}, coords.options());

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_angles_cuda", ([&] {
            angles_cuda_kernel<scalar_t><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                angles.data_ptr<int64_t>(),
                nangles,
                theta.data_ptr<scalar_t>(),
                nullptr,
                nullptr
            );
        }));

        ctx->save_for_backward({coords, angles});
        return theta;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& coords = saved[0];
        const at::Tensor& angles = saved[1];
        const at::Tensor& grad_theta = grad_outputs[0];

        const auto nangles = angles.size(0);

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        const int GRID_SIZE = std::min(
            static_cast<int>((nangles + BLOCK_SIZE - 1) / BLOCK_SIZE),
            props->multiProcessorCount * props->maxBlocksPerMultiProcessor
        );

        at::Tensor grad_coords = at::zeros_like(coords, coords.options());

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_angles_cuda_backward", ([&] {
            angles_cuda_kernel<scalar_t><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                angles.data_ptr<int64_t>(),
                nangles,
                nullptr,
                grad_theta.data_ptr<scalar_t>(),
                grad_coords.data_ptr<scalar_t>()
            );
        }));

        at::Tensor ignore;
        return {grad_coords, ignore};
    }
};

at::Tensor compute_angles_cuda(at::Tensor& coords, at::Tensor& angles) {
    return AnglesFunctionCuda::apply(coords, angles);
}

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_angles", compute_angles_cuda);
}
