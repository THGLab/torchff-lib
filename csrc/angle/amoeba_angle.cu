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
#include "common/reduce.cuh"

template <typename scalar_t, int BLOCK_SIZE>
__global__ void amoeba_angle_cuda_kernel(
    scalar_t* coords,
    int64_t* angles,
    scalar_t* theta0,
    scalar_t* k,
    scalar_t cubic,
    scalar_t quartic,
    scalar_t pentic,
    scalar_t sextic,
    int64_t nangles,
    scalar_t* ene_out,
    scalar_t* coord_grad,
    scalar_t* theta0_grad,
    scalar_t* k_grad,
    scalar_t sign
) {
    if (ene_out && threadIdx.x == 0 && blockIdx.x == 0) {
        ene_out[0] = scalar_t(0.0);
    }
    __syncthreads();
    scalar_t ene = scalar_t(0.0);
    for (int index = threadIdx.x + blockIdx.x * BLOCK_SIZE; index < nangles; index += BLOCK_SIZE * gridDim.x) {
        int64_t offset = index * 3;
        int64_t offset_0 = angles[offset] * 3;
        int64_t offset_1 = angles[offset + 1] * 3;
        int64_t offset_2 = angles[offset + 2] * 3;

        scalar_t* coords_0 = coords + offset_0;
        scalar_t* coords_1 = coords + offset_1;
        scalar_t* coords_2 = coords + offset_2;

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

        scalar_t k_ = k[index];
        scalar_t dtheta = theta - theta0[index];
        scalar_t dtheta2 = dtheta * dtheta;
        scalar_t dtheta3 = dtheta2 * dtheta;
        scalar_t dtheta4 = dtheta2 * dtheta2;
        scalar_t poly = scalar_t(1.0) + cubic * dtheta + quartic * dtheta2 + pentic * dtheta3 + sextic * dtheta4;

        if (ene_out) {
            ene += k_ * dtheta2 * poly;
        }

        if (coord_grad || theta0_grad) {
            scalar_t dpoly_ddtheta = cubic + scalar_t(2.0) * quartic * dtheta + scalar_t(3.0) * pentic * dtheta2 + scalar_t(4.0) * sextic * dtheta3;
            scalar_t dU_ddtheta = k_ * (scalar_t(2.0) * dtheta * poly + dtheta2 * dpoly_ddtheta);

            if (coord_grad) {
                scalar_t dtheta_dcos = -scalar_t(1.0) / sqrt_(scalar_t(1.0) - cos_theta * cos_theta);
                scalar_t prefix = dU_ddtheta * dtheta_dcos / (v1_norm * v2_norm) * sign;

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
            if (theta0_grad) {
                theta0_grad[index] = -dU_ddtheta;
            }
        }
        if (k_grad) {
            k_grad[index] = dtheta2 * poly;
        }
    }
    if (ene_out) {
        block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
    }
}


class AmoebaAngleFunctionCuda : public torch::autograd::Function<AmoebaAngleFunctionCuda> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& angles,
        at::Tensor& theta0,
        at::Tensor& k,
        at::Scalar cubic,
        at::Scalar quartic,
        at::Scalar pentic,
        at::Scalar sextic
    ) {
        int nangles = angles.size(0);

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();

        constexpr int BLOCK_SIZE = 256;
        int GRID_SIZE = std::min(
            (nangles + BLOCK_SIZE - 1) / BLOCK_SIZE,
            props->multiProcessorCount * props->maxBlocksPerMultiProcessor
        );

        at::Tensor ene_out = at::empty({}, coords.options());
        at::Tensor coord_grad = at::zeros_like(coords, coords.options());
        at::Tensor theta0_grad = at::empty_like(theta0, theta0.options());
        at::Tensor k_grad = at::empty_like(k, k.options());
        ctx->save_for_backward({coord_grad, theta0_grad, k_grad});

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_amoeba_angle_cuda", ([&] {
            scalar_t c = static_cast<scalar_t>(cubic.to<double>());
            scalar_t q = static_cast<scalar_t>(quartic.to<double>());
            scalar_t p = static_cast<scalar_t>(pentic.to<double>());
            scalar_t s = static_cast<scalar_t>(sextic.to<double>());
            amoeba_angle_cuda_kernel<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                angles.data_ptr<int64_t>(),
                theta0.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                c, q, p, s,
                nangles,
                ene_out.data_ptr<scalar_t>(),
                (coords.requires_grad()) ? coord_grad.data_ptr<scalar_t>() : nullptr,
                (theta0.requires_grad()) ? theta0_grad.data_ptr<scalar_t>() : nullptr,
                (k.requires_grad()) ? k_grad.data_ptr<scalar_t>() : nullptr,
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
            saved[0] * grad_outputs[0], ignore, saved[1] * grad_outputs[0], saved[2] * grad_outputs[0],
            ignore, ignore, ignore, ignore
        };
    }
};


at::Tensor compute_amoeba_angle_energy_cuda(
    at::Tensor& coords,
    at::Tensor& angles,
    at::Tensor& theta0,
    at::Tensor& k,
    at::Scalar cubic,
    at::Scalar quartic,
    at::Scalar pentic,
    at::Scalar sextic
) {
    return AmoebaAngleFunctionCuda::apply(coords, angles, theta0, k, cubic, quartic, pentic, sextic);
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_amoeba_angle_energy", compute_amoeba_angle_energy_cuda);
}
