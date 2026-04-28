#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <tuple>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"
#include "common/reduce.cuh"
#include "common/dispatch.cuh"
#include "ewald/damps.cuh"
#include "multipoles_forward.cuh"
#include "multipoles_backward.cuh"


constexpr int MULTIPOLES_BLOCK_SIZE = 256;


class MultipolarInteractionAtomPairsFunctionCuda : public torch::autograd::Function<MultipolarInteractionAtomPairsFunctionCuda> {

public:

static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& coords,
    at::Tensor& box,
    at::Tensor& pairs,
    c10::optional<at::Tensor> pairs_excl,
    at::Tensor& q,
    c10::optional<at::Tensor> p,
    c10::optional<at::Tensor> t,
    at::Scalar cutoff,
    at::Scalar ewald_alpha,
    at::Scalar prefactor
) {
    int64_t rank = 0;
    if (t.has_value()) {
        rank = 2;
    } else if (p.has_value()) {
        rank = 1;
    }

    int64_t npairs = pairs.size(0);
    int64_t npairs_excl = (pairs_excl.has_value() && pairs_excl.value().defined())
        ? pairs_excl.value().size(0) : 0;
    const int64_t* pairs_excl_ptr = (pairs_excl.has_value() && pairs_excl.value().defined() && npairs_excl > 0)
        ? pairs_excl.value().data_ptr<int64_t>() : nullptr;

    bool do_ewald = (ewald_alpha.toDouble() >= 0);

    bool need_coord_grad = coords.requires_grad();
    bool need_mpole_grad = q.requires_grad()
        || (p.has_value() && p.value().requires_grad())
        || (t.has_value() && t.value().requires_grad());

    at::Tensor ene = at::zeros({}, coords.options());
    at::Tensor coord_grad = need_coord_grad ? at::zeros_like(coords) : at::Tensor();
    at::Tensor q_grad = need_mpole_grad ? at::zeros_like(q) : at::Tensor();
    at::Tensor p_grad = (need_mpole_grad && rank >= 1) ? at::zeros_like(p.value()) : at::Tensor();
    at::Tensor t_grad = (need_mpole_grad && rank >= 2) ? at::zeros_like(t.value()) : at::Tensor();

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t max_pairs = npairs_excl > npairs ? npairs_excl : npairs;
    int grid_dim = std::min(
        static_cast<int>((max_pairs + MULTIPOLES_BLOCK_SIZE - 1) / MULTIPOLES_BLOCK_SIZE),
        props->maxBlocksPerMultiProcessor * props->multiProcessorCount
    );

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "multipolar_interaction_atom_pairs_cuda", ([&] {
        const scalar_t cutoff_val = cutoff.to<scalar_t>();
        const scalar_t ewald_alpha_val = ewald_alpha.to<scalar_t>();
        const scalar_t prefactor_val = prefactor.to<scalar_t>();

        scalar_t* p_ptr = (rank >= 1 && p.has_value()) ? p.value().data_ptr<scalar_t>() : nullptr;
        scalar_t* t_ptr = (rank >= 2 && t.has_value()) ? t.value().data_ptr<scalar_t>() : nullptr;
        scalar_t* coord_grad_ptr = need_coord_grad ? coord_grad.data_ptr<scalar_t>() : nullptr;
        scalar_t* q_grad_ptr = need_mpole_grad ? q_grad.data_ptr<scalar_t>() : nullptr;
        scalar_t* p_grad_ptr = (need_mpole_grad && rank >= 1 && p_grad.defined()) ? p_grad.data_ptr<scalar_t>() : nullptr;
        scalar_t* t_grad_ptr = (need_mpole_grad && rank >= 2 && t_grad.defined()) ? t_grad.data_ptr<scalar_t>() : nullptr;

        DISPATCH_RANK(rank, RANK, [&] {
            DISPATCH_BOOL(do_ewald, DO_EWALD, [&] {
                DISPATCH_BOOL(need_coord_grad, DO_COORD_GRAD, [&] {
                    DISPATCH_BOOL(need_mpole_grad, DO_MPOLE_GRAD, [&] {
                        auto kernel = multipolar_interaction_atom_pairs_kernel<
                            scalar_t, MULTIPOLES_BLOCK_SIZE, RANK, DO_EWALD, true, DO_COORD_GRAD, DO_MPOLE_GRAD>;
                        kernel<<<grid_dim, MULTIPOLES_BLOCK_SIZE, 0, stream>>>(
                            coords.data_ptr<scalar_t>(), box.data_ptr<scalar_t>(), pairs.data_ptr<int64_t>(),
                            pairs_excl_ptr, npairs, npairs_excl,
                            cutoff_val, ewald_alpha_val, prefactor_val,
                            q.data_ptr<scalar_t>(), p_ptr, t_ptr,
                            ene.data_ptr<scalar_t>(), coord_grad_ptr, q_grad_ptr, p_grad_ptr, t_grad_ptr);
                    });
                });
            });
        });
    }));

    ctx->saved_data["rank"] = rank;
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
    std::vector<at::Tensor> grads(10);
    grads[0] = saved[0].defined() ? saved[0] * grad_outputs[0] : ignore;   // coords grad
    grads[1] = ignore;                        // box
    grads[2] = ignore;                        // pairs
    grads[3] = ignore;                        // pairs_excl
    grads[4] = saved[1].defined() ? saved[1] * grad_outputs[0] : ignore;   // q grad
    grads[5] = (rank >= 1 && saved[2].defined()) ? saved[2] * grad_outputs[0] : ignore;  // p grad
    grads[6] = (rank >= 2 && saved[3].defined()) ? saved[3] * grad_outputs[0] : ignore;  // t grad
    grads[7] = ignore;                        // cutoff
    grads[8] = ignore;                        // ewald_alpha
    grads[9] = ignore;                        // prefactor
    return grads;
}

};

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_multipolar_energy_from_atom_pairs",
        [](const at::Tensor& coords,
           const at::Tensor& box,
           const at::Tensor& pairs,
           c10::optional<at::Tensor> pairs_excl,
           const at::Tensor& q,
           c10::optional<at::Tensor> p,
           c10::optional<at::Tensor> t,
           at::Scalar cutoff,
           at::Scalar ewald_alpha,
           at::Scalar prefactor) {
            return MultipolarInteractionAtomPairsFunctionCuda::apply(
                const_cast<at::Tensor&>(coords), const_cast<at::Tensor&>(box),
                const_cast<at::Tensor&>(pairs), pairs_excl, const_cast<at::Tensor&>(q),
                p, t, cutoff, ewald_alpha, prefactor);
        });
}
