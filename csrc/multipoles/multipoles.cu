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
#include "ewald/damps.cuh"
#include "multipoles_new.cuh"


template <typename scalar_t, int BLOCK_SIZE, int RANK, bool DO_EWALD=false>
__global__ void multipolar_interaction_atom_pairs_kernel(
    scalar_t* coords,
    scalar_t* box,
    int64_t* pairs,
    scalar_t cutoff,
    scalar_t ewald_alpha,
    scalar_t prefactor,
    scalar_t* q,
    scalar_t* p,
    scalar_t* t,
    int64_t npairs,
    scalar_t* ene_out,
    scalar_t* coord_grad,
    scalar_t* q_grad,
    scalar_t* p_grad,
    scalar_t* t_grad
) {
    scalar_t ene = static_cast<scalar_t>(0.0);

    __shared__ scalar_t box_inv[9];
    if (threadIdx.x == 0) {
        // Compute reciprocal box matrix once per block
        invert_box_3x3(box, box_inv);
    }
    __syncthreads();

    for (int64_t index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
         index < npairs;
         index += BLOCK_SIZE * gridDim.x) {
        int64_t i = pairs[index * 2];
        int64_t j = pairs[index * 2 + 1];
        if (i < 0 || j < 0) {
            continue;
        }

        scalar_t drvec[3];
        diff_vec3(&coords[j * 3], &coords[i * 3], drvec);
        apply_pbc_triclinic(drvec, box, box_inv, drvec);

        scalar_t dr = norm3d_(drvec[0], drvec[1], drvec[2]);
        if (dr > cutoff) {
            continue;
        }

        MultipoleAccumWithGrad<scalar_t, RANK> mpi;
        MultipoleAccumWithGrad<scalar_t, RANK> mpj;

        mpi.c0 = q[i];
        mpj.c0 = q[j];

        if constexpr (RANK >= 1) {
            mpi.dx = p[i * 3];
            mpi.dy = p[i * 3 + 1];
            mpi.dz = p[i * 3 + 2];
            mpj.dx = p[j * 3];
            mpj.dy = p[j * 3 + 1];
            mpj.dz = p[j * 3 + 2];
        }

        if constexpr (RANK >= 2) {
            mpi.qxx = t[i * 9 + 0] * scalar_t(1/3.0);
            mpi.qxy = (t[i * 9 + 1] + t[i * 9 + 3]) * scalar_t(1/3.0);
            mpi.qxz = (t[i * 9 + 2] + t[i * 9 + 6]) * scalar_t(1/3.0);
            mpi.qyy = t[i * 9 + 4] * scalar_t(1/3.0);
            mpi.qyz = (t[i * 9 + 5] + t[i * 9 + 7]) * scalar_t(1/3.0);
            mpi.qzz = t[i * 9 + 8] * scalar_t(1/3.0);

            mpj.qxx = t[j * 9 + 0] * scalar_t(1/3.0);
            mpj.qxy = (t[j * 9 + 1] + t[j * 9 + 3]) * scalar_t(1/3.0);
            mpj.qxz = (t[j * 9 + 2] + t[j * 9 + 6]) * scalar_t(1/3.0);
            mpj.qyy = t[j * 9 + 4] * scalar_t(1/3.0);
            mpj.qyz = (t[j * 9 + 5] + t[j * 9 + 7]) * scalar_t(1/3.0);
            mpj.qzz = t[j * 9 + 8] * scalar_t(1/3.0);
        }

        scalar_t ene_pair = static_cast<scalar_t>(0.0);
        scalar_t drx_g = static_cast<scalar_t>(0.0);
        scalar_t dry_g = static_cast<scalar_t>(0.0);
        scalar_t drz_g = static_cast<scalar_t>(0.0);

        if constexpr (DO_EWALD) {
            scalar_t damps[RANK * 2 + 2];
            // ORDER 3->2 damps, 7->4 damps, 11->6 damps; kernel needs RANK*2+2
            ewald_erfc_damps<scalar_t, 4 * RANK + 3>(dr, ewald_alpha, damps);
            pairwise_multipole_kernel_with_grad<scalar_t, RANK, true>(
                mpi, mpj,
                drvec[0], drvec[1], drvec[2], dr,
                damps,
                ene_pair, drx_g, dry_g, drz_g
            );
        } else {
            pairwise_multipole_kernel_with_grad<scalar_t, RANK, false>(
                mpi, mpj,
                drvec[0], drvec[1], drvec[2], dr, nullptr,
                ene_pair, drx_g, dry_g, drz_g
            );
        }

        ene += ene_pair;

        if (coord_grad) {
            atomicAdd(&coord_grad[i * 3],     -drx_g * prefactor);
            atomicAdd(&coord_grad[i * 3 + 1], -dry_g * prefactor);
            atomicAdd(&coord_grad[i * 3 + 2], -drz_g * prefactor);
            atomicAdd(&coord_grad[j * 3],      drx_g * prefactor);
            atomicAdd(&coord_grad[j * 3 + 1],  dry_g * prefactor);
            atomicAdd(&coord_grad[j * 3 + 2],  drz_g * prefactor);
        }

        if (q_grad) {
            atomicAdd(&q_grad[i], mpi.ep * prefactor);
            atomicAdd(&q_grad[j], mpj.ep * prefactor);
        }

        if constexpr (RANK >= 1) {
            if (p_grad) {
                atomicAdd(&p_grad[i * 3],     mpi.efx * prefactor);
                atomicAdd(&p_grad[i * 3 + 1], mpi.efy * prefactor);
                atomicAdd(&p_grad[i * 3 + 2], mpi.efz * prefactor);
                atomicAdd(&p_grad[j * 3],     mpj.efx * prefactor);
                atomicAdd(&p_grad[j * 3 + 1], mpj.efy * prefactor);
                atomicAdd(&p_grad[j * 3 + 2], mpj.efz * prefactor);
            }
        }

        if constexpr (RANK >= 2) {
            if (t_grad) {
                atomicAdd(&t_grad[i * 9 + 0], mpi.egxx * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[i * 9 + 1], mpi.egxy * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[i * 9 + 2], mpi.egxz * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[i * 9 + 3], mpi.egxy * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[i * 9 + 4], mpi.egyy * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[i * 9 + 5], mpi.egyz * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[i * 9 + 6], mpi.egxz * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[i * 9 + 7], mpi.egyz * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[i * 9 + 8], mpi.egzz * scalar_t(1/3.0) * prefactor);

                atomicAdd(&t_grad[j * 9 + 0], mpj.egxx * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[j * 9 + 1], mpj.egxy * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[j * 9 + 2], mpj.egxz * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[j * 9 + 3], mpj.egxy * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[j * 9 + 4], mpj.egyy * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[j * 9 + 5], mpj.egyz * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[j * 9 + 6], mpj.egxz * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[j * 9 + 7], mpj.egyz * scalar_t(1/3.0) * prefactor);
                atomicAdd(&t_grad[j * 9 + 8], mpj.egzz * scalar_t(1/3.0) * prefactor);
            }
        }
    }

    ene *= prefactor;
    if (ene_out) {
        block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
    }
}


class MultipolarInteractionAtomPairsFunctionCuda : public torch::autograd::Function<MultipolarInteractionAtomPairsFunctionCuda> {

public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& box,
        at::Tensor& pairs,
        at::Tensor& q,
        c10::optional<at::Tensor> p,
        c10::optional<at::Tensor> t,
        at::Scalar cutoff,
        at::Scalar ewald_alpha,
        at::Scalar prefactor
    ) {
        int64_t npairs = pairs.size(0);
        int64_t rank = 0;
        if (t.has_value()) {
            rank = 2;
        } else if (p.has_value()) {
            rank = 1;
        }

        bool do_ewald = (ewald_alpha.toDouble() >= 0);

        at::Tensor ene = at::zeros({1}, coords.options());
        at::Tensor coord_grad = at::zeros_like(coords, coords.options());
        at::Tensor q_grad = at::zeros_like(q, q.options());
        at::Tensor p_grad = (rank >= 1) ? at::zeros_like(p.value(), p.value().options()) : at::Tensor();
        at::Tensor t_grad = (rank >= 2) ? at::zeros_like(t.value(), t.value().options()) : at::Tensor();

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        int grid_dim = std::min(
            static_cast<int>((npairs + BLOCK_SIZE - 1) / BLOCK_SIZE),
            props->maxBlocksPerMultiProcessor * props->multiProcessorCount
        );

        ctx->saved_data["rank"] = static_cast<int64_t>(rank);

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "multipolar_interaction_atom_pairs_cuda", ([&] {
            const scalar_t cutoff_val = cutoff.to<scalar_t>();
            const scalar_t ewald_alpha_val = ewald_alpha.to<scalar_t>();
            const scalar_t prefactor_val = prefactor.to<scalar_t>();

            scalar_t* p_ptr = (rank >= 1 && p.has_value()) ? p.value().data_ptr<scalar_t>() : nullptr;
            scalar_t* t_ptr = (rank >= 2 && t.has_value()) ? t.value().data_ptr<scalar_t>() : nullptr;
            scalar_t* p_grad_ptr = (rank >= 1 && p.value().requires_grad() && p_grad.defined()) ? p_grad.data_ptr<scalar_t>() : nullptr;
            scalar_t* t_grad_ptr = (rank >= 2 && t.value().requires_grad() && t_grad.defined()) ? t_grad.data_ptr<scalar_t>() : nullptr;

            if (rank == 0) {
                if (do_ewald) {
                    multipolar_interaction_atom_pairs_kernel<scalar_t, BLOCK_SIZE, 0, true><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
                        coords.data_ptr<scalar_t>(),
                        box.data_ptr<scalar_t>(),
                        pairs.data_ptr<int64_t>(),
                        cutoff_val, ewald_alpha_val, prefactor_val,
                        q.data_ptr<scalar_t>(),
                        nullptr, nullptr,
                        npairs,
                        ene.data_ptr<scalar_t>(),
                        coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr,
                        q.requires_grad() ? q_grad.data_ptr<scalar_t>() : nullptr,
                        nullptr, nullptr
                    );
                } else {
                    multipolar_interaction_atom_pairs_kernel<scalar_t, BLOCK_SIZE, 0, false><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
                        coords.data_ptr<scalar_t>(),
                        box.data_ptr<scalar_t>(),
                        pairs.data_ptr<int64_t>(),
                        cutoff_val, ewald_alpha_val, prefactor_val,
                        q.data_ptr<scalar_t>(),
                        nullptr, nullptr,
                        npairs,
                        ene.data_ptr<scalar_t>(),
                        coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr,
                        q.requires_grad() ? q_grad.data_ptr<scalar_t>() : nullptr,
                        nullptr, nullptr
                    );
                }
            } else if (rank == 1) {
                if (do_ewald) {
                    multipolar_interaction_atom_pairs_kernel<scalar_t, BLOCK_SIZE, 1, true><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
                        coords.data_ptr<scalar_t>(),
                        box.data_ptr<scalar_t>(),
                        pairs.data_ptr<int64_t>(),
                        cutoff_val, ewald_alpha_val, prefactor_val,
                        q.data_ptr<scalar_t>(),
                        p_ptr, nullptr,
                        npairs,
                        ene.data_ptr<scalar_t>(),
                        coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr,
                        q.requires_grad() ? q_grad.data_ptr<scalar_t>() : nullptr,
                        p_grad_ptr, nullptr
                    );
                } else {
                    multipolar_interaction_atom_pairs_kernel<scalar_t, BLOCK_SIZE, 1, false><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
                        coords.data_ptr<scalar_t>(),
                        box.data_ptr<scalar_t>(),
                        pairs.data_ptr<int64_t>(),
                        cutoff_val, ewald_alpha_val, prefactor_val,
                        q.data_ptr<scalar_t>(),
                        p_ptr, nullptr,
                        npairs,
                        ene.data_ptr<scalar_t>(),
                        coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr,
                        q.requires_grad() ? q_grad.data_ptr<scalar_t>() : nullptr,
                        p_grad_ptr, nullptr
                    );
                }
            } else {
                if (do_ewald) {
                    multipolar_interaction_atom_pairs_kernel<scalar_t, BLOCK_SIZE, 2, true><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
                        coords.data_ptr<scalar_t>(),
                        box.data_ptr<scalar_t>(),
                        pairs.data_ptr<int64_t>(),
                        cutoff_val, ewald_alpha_val, prefactor_val,
                        q.data_ptr<scalar_t>(),
                        p_ptr, t_ptr,
                        npairs,
                        ene.data_ptr<scalar_t>(),
                        coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr,
                        q.requires_grad() ? q_grad.data_ptr<scalar_t>() : nullptr,
                        p_grad_ptr, t_grad_ptr
                    );
                } else {
                    multipolar_interaction_atom_pairs_kernel<scalar_t, BLOCK_SIZE, 2, false><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
                        coords.data_ptr<scalar_t>(),
                        box.data_ptr<scalar_t>(),
                        pairs.data_ptr<int64_t>(),
                        cutoff_val, ewald_alpha_val, prefactor_val,
                        q.data_ptr<scalar_t>(),
                        p_ptr, t_ptr,
                        npairs,
                        ene.data_ptr<scalar_t>(),
                        coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr,
                        q.requires_grad() ? q_grad.data_ptr<scalar_t>() : nullptr,
                        p_grad_ptr, t_grad_ptr
                    );
                }
            }
        }));

        ctx->save_for_backward({coord_grad, q_grad, p_grad, t_grad});
        return ene;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        int64_t rank = ctx->saved_data["rank"].toInt();

        at::Tensor ignore;
        std::vector<at::Tensor> grads(9);
        grads[0] = saved[0] * grad_outputs[0];   // coords grad
        grads[1] = ignore;                        // box
        grads[2] = ignore;                        // pairs
        grads[3] = saved[1] * grad_outputs[0];   // q grad
        grads[4] = (rank >= 1 && saved[2].defined()) ? saved[2] * grad_outputs[0] : ignore;  // p grad
        grads[5] = (rank >= 2 && saved[3].defined()) ? saved[3] * grad_outputs[0] : ignore;  // t grad
        grads[6] = ignore;                        // cutoff
        grads[7] = ignore;                        // ewald_alpha
        grads[8] = ignore;                        // prefactor
        return grads;
    }
};


at::Tensor compute_multipolar_energy_from_atom_pairs_cuda(
    at::Tensor& coords,
    at::Tensor& box,
    at::Tensor& pairs,
    at::Tensor& q,
    c10::optional<at::Tensor> p,
    c10::optional<at::Tensor> t,
    at::Scalar cutoff,
    at::Scalar ewald_alpha,
    at::Scalar prefactor
) {
    return MultipolarInteractionAtomPairsFunctionCuda::apply(
        coords, box, pairs, q, p, t, cutoff, ewald_alpha, prefactor);
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_multipolar_energy_from_atom_pairs", compute_multipolar_energy_from_atom_pairs_cuda);
}
