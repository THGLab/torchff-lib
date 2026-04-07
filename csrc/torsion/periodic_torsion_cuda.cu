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
__global__ void periodic_torsion_cuda_kernel(
    scalar_t* coords,
    int64_t* torsions,
    scalar_t* fc,
    int64_t* periodicity,
    scalar_t* phase,
    int64_t ntors,
    scalar_t* ene_out,
    scalar_t* coord_grad,
    scalar_t* fc_grad,
    scalar_t* phase_grad,
    scalar_t sign
) {
    if (ene_out && threadIdx.x == 0 && blockIdx.x == 0) {
        ene_out[0] = scalar_t(0.0);
    }
    __syncthreads();
    scalar_t ene = scalar_t(0.0);
    for (int index = threadIdx.x + blockIdx.x * BLOCK_SIZE; index < ntors; index += BLOCK_SIZE * gridDim.x) {
        int offset = index * 4;
        int64_t offset_i = 3 * torsions[offset];
        int64_t offset_j = 3 * torsions[offset + 1];
        int64_t offset_k = 3 * torsions[offset + 2];
        int64_t offset_l = 3 * torsions[offset + 3];

        scalar_t* coords_i = coords + offset_i;
        scalar_t* coords_j = coords + offset_j;
        scalar_t* coords_k = coords + offset_k;
        scalar_t* coords_l = coords + offset_l;

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

        scalar_t norm_n1 = norm_vec3(n1);
        scalar_t norm_n2 = norm_vec3(n2);
        scalar_t norm_b2 = norm_vec3(b2);
        scalar_t norm_b2_sqr = norm_b2 * norm_b2;

        scalar_t cosval = dot_vec3(n1, n2) / (norm_n1 * norm_n2);
        cosval = clamp_(cosval, scalar_t(-0.999999999), scalar_t(0.99999999));

        scalar_t k = fc[index];
        int64_t n = periodicity[index];
        scalar_t phi = acos_(cosval);
        phi = dot_vec3(n1, b3) > 0.0 ? phi : -phi;
        phi = n * phi - phase[index];
        
        scalar_t tmp1 = 1 + cos_(phi);
        scalar_t tmp2 = k * sin_(phi);

        if (ene_out) {
            ene += k * tmp1;
        }

        if (coord_grad) {
            scalar_t prefactor = tmp2 * n * sign;

            scalar_t aux1 = dot_vec3(b1, b2) / norm_b2_sqr;
            scalar_t aux2 = dot_vec3(b2, b3) / norm_b2_sqr;

            scalar_t cgi, cgj, cgk, cgl;
            #pragma unroll 3
            for (int i = 0; i < 3; i++) {
                cgi = prefactor * norm_b2 / (norm_n1 * norm_n1) * n1[i];
                cgl = -prefactor * norm_b2 / (norm_n2 * norm_n2) * n2[i];
                cgj = (-aux1 - 1) * cgi + aux2 * cgl;
                cgk = -cgi - cgj - cgl;
                atomicAdd(&coord_grad[offset_i + i], cgi);
                atomicAdd(&coord_grad[offset_j + i], cgj);
                atomicAdd(&coord_grad[offset_l + i], cgl);
                atomicAdd(&coord_grad[offset_k + i], cgk);
            }
        }

        if ( fc_grad ) {
            fc_grad[index] = tmp1;
        }
        if ( phase_grad ) {
            phase_grad[index] = tmp2;
        }
    }
    if (ene_out) {
        block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
    }
}


class PeriodicTorsionFunctionCuda: public torch::autograd::Function<PeriodicTorsionFunctionCuda> {

public: 
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& torsions,
        at::Tensor& fc,
        at::Tensor& periodicity,
        at::Tensor& phase
    )
    {
        int ntors = torsions.size(0);

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        int GRID_SIZE = std::min(
            (ntors + BLOCK_SIZE - 1) / BLOCK_SIZE,
            props->multiProcessorCount*props->maxBlocksPerMultiProcessor
        );

        at::Tensor ene_out = at::empty({}, coords.options());
        at::Tensor coord_grad = at::zeros_like(coords, coords.options());
        at::Tensor fc_grad = at::empty_like(fc, fc.options());
        at::Tensor phase_grad = at::empty_like(phase, phase.options());

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_periodic_torsion_cuda", ([&] {
            periodic_torsion_cuda_kernel<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                torsions.data_ptr<int64_t>(),
                fc.data_ptr<scalar_t>(),
                periodicity.data_ptr<int64_t>(),
                phase.data_ptr<scalar_t>(),
                ntors,
                ene_out.data_ptr<scalar_t>(),
                (coords.requires_grad()) ? coord_grad.data_ptr<scalar_t>() : nullptr,
                (fc.requires_grad()) ? fc_grad.data_ptr<scalar_t>() : nullptr,
                (phase.requires_grad()) ? phase_grad.data_ptr<scalar_t>(): nullptr,
                static_cast<scalar_t>(1.0)
            );
        }));
        ctx->save_for_backward({coord_grad, fc_grad, phase_grad});
        return ene_out;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    )
    {
        auto saved = ctx->get_saved_variables();
        at::Tensor ignore;
        return {saved[0] * grad_outputs[0], ignore, saved[1] * grad_outputs[0], ignore, saved[2] * grad_outputs[0]};
    }

};


at::Tensor compute_periodic_torsion_energy_cuda(
    at::Tensor& coords,
    at::Tensor& torsions,
    at::Tensor& fc,
    at::Tensor& periodicity,
    at::Tensor& phase
) {
    return PeriodicTorsionFunctionCuda::apply(coords, torsions, fc, periodicity, phase);
}


void compute_periodic_torsion_forces_cuda(
    at::Tensor& coords,
    at::Tensor& torsions,
    at::Tensor& fc,
    at::Tensor& periodicity,
    at::Tensor& phase,
    at::Tensor& forces
) {

    int ntors = torsions.size(0);
    auto props = at::cuda::getCurrentDeviceProperties();
    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = std::min(
        (ntors + BLOCK_SIZE - 1) / BLOCK_SIZE,
        props->multiProcessorCount*props->maxBlocksPerMultiProcessor
    );
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_periodic_torsion_forces_cuda", ([&] {
        periodic_torsion_cuda_kernel<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            torsions.data_ptr<int64_t>(),
            fc.data_ptr<scalar_t>(),
            periodicity.data_ptr<int64_t>(),
            phase.data_ptr<scalar_t>(),
            ntors,
            nullptr,
            forces.data_ptr<scalar_t>(),
            nullptr,
            nullptr,
            static_cast<scalar_t>(-1.0)
        );
    }));
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_periodic_torsion_energy", compute_periodic_torsion_energy_cuda);
    m.impl("compute_periodic_torsion_forces", compute_periodic_torsion_forces_cuda);
}