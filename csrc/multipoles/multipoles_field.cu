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


class MultipolarInteractionWithFieldsAtomPairsFunctionCuda : public torch::autograd::Function<MultipolarInteractionWithFieldsAtomPairsFunctionCuda> {

public:

static std::vector<at::Tensor> forward(
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

    at::Tensor ene = at::zeros({1}, coords.options());
    at::Tensor q_grad = at::zeros_like(q);
    at::Tensor p_grad = (rank >= 1) ? at::zeros_like(p.value()) : at::Tensor();
    at::Tensor t_grad = (rank >= 2) ? at::zeros_like(t.value()) : at::Tensor();

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

        scalar_t* p_ptr = (rank >= 1) ? p.value().data_ptr<scalar_t>() : nullptr;
        scalar_t* t_ptr = (rank >= 2) ? t.value().data_ptr<scalar_t>() : nullptr;
        scalar_t* q_grad_ptr = q_grad.data_ptr<scalar_t>();
        scalar_t* p_grad_ptr = (rank >= 1) ? p_grad.data_ptr<scalar_t>() : nullptr;
        scalar_t* t_grad_ptr = (rank >= 2) ? t_grad.data_ptr<scalar_t>() : nullptr;

        DISPATCH_RANK(rank, RANK, [&] {
            DISPATCH_BOOL(do_ewald, DO_EWALD, [&] {
                auto kernel = multipolar_interaction_atom_pairs_kernel<
                    scalar_t, MULTIPOLES_BLOCK_SIZE, RANK, DO_EWALD, true, false, true>;
                kernel<<<grid_dim, MULTIPOLES_BLOCK_SIZE, 0, stream>>>(
                    coords.data_ptr<scalar_t>(), box.data_ptr<scalar_t>(), pairs.data_ptr<int64_t>(),
                    pairs_excl_ptr, npairs, npairs_excl,
                    cutoff_val, ewald_alpha_val, prefactor_val,
                    q.data_ptr<scalar_t>(), p_ptr, t_ptr,
                    ene.data_ptr<scalar_t>(), nullptr, q_grad_ptr, p_grad_ptr, t_grad_ptr);
            });
        });
    }));

    bool need_coord_grad = coords.requires_grad();
    bool need_q_grad = q.requires_grad();
    bool need_p_grad = p.has_value() && p.value().requires_grad();
    bool need_t_grad = t.has_value() && t.value().requires_grad();
    ctx->saved_data["need_coord_grad"] = need_coord_grad;
    ctx->saved_data["need_q_grad"] = need_q_grad;
    ctx->saved_data["need_p_grad"] = need_p_grad;
    ctx->saved_data["need_t_grad"] = need_t_grad;

    ctx->saved_data["rank"] = rank;
    ctx->saved_data["cutoff"] = cutoff.to<double>();
    ctx->saved_data["ewald_alpha"] = ewald_alpha.to<double>();
    ctx->saved_data["prefactor"] = prefactor.to<double>();
    at::Tensor pairs_excl_t = (pairs_excl.has_value() && pairs_excl.value().defined())
        ? pairs_excl.value() : at::empty({0, 2}, coords.options().dtype(at::kLong));
    ctx->save_for_backward({
        coords, box, pairs, pairs_excl_t, 
        q, 
        (rank >= 1) ? p.value() : at::Tensor(),
        (rank >= 2) ? t.value() : at::Tensor()
    });

    auto efield = (rank >= 1) ? -p_grad : at::zeros({coords.size(0), 3}, coords.options());
    return {ene, q_grad, efield};
}

static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<at::Tensor> grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    at::Tensor coords = saved[0];
    at::Tensor box = saved[1];
    at::Tensor pairs = saved[2];
    at::Tensor pairs_excl_saved = saved[3];
    
    int rank = ctx->saved_data["rank"].toInt();
    at::Tensor q = saved[4];
    at::Tensor p = ( rank >= 1 ) ? saved[5] : at::zeros({coords.size(0), 3}, coords.options());
    at::Tensor t = ( rank >= 2 ) ? saved[6] : at::zeros({coords.size(0), 3, 3}, coords.options());

    double cutoff_val = ctx->saved_data["cutoff"].toDouble();
    double ewald_alpha_val = ctx->saved_data["ewald_alpha"].toDouble();
    double prefactor_val = ctx->saved_data["prefactor"].toDouble();

    int64_t npairs = pairs.size(0);
    int64_t npairs_excl = (pairs_excl_saved.defined() && pairs_excl_saved.numel() > 0)
        ? pairs_excl_saved.size(0) : 0;
    const int64_t* pairs_excl_ptr = (pairs_excl_saved.defined() && npairs_excl > 0)
        ? pairs_excl_saved.data_ptr<int64_t>() : nullptr;

    bool do_ewald = (ewald_alpha_val >= 0);
    bool need_coord_grad = ctx->saved_data["need_coord_grad"].toBool();
    bool need_q_grad = ctx->saved_data["need_q_grad"].toBool();
    bool need_p_grad = ctx->saved_data["need_p_grad"].toBool();
    bool need_t_grad = ctx->saved_data["need_t_grad"].toBool();
    bool need_mpole_grad = need_q_grad || need_p_grad || need_t_grad;

    at::Tensor coord_grad = need_coord_grad ? at::zeros_like(coords) : at::Tensor();
    at::Tensor q_grad = need_mpole_grad ? at::zeros_like(q) : at::Tensor();
    at::Tensor p_grad = need_mpole_grad ? at::zeros_like(p) : at::Tensor();
    at::Tensor t_grad = need_mpole_grad ? at::zeros_like(t) : at::Tensor();

    at::Tensor d_energy = grad_outputs[0].defined() && grad_outputs[0].numel() > 0
        ? grad_outputs[0].contiguous()
        : at::zeros({1}, coords.options());
    at::Tensor d_epot = grad_outputs[1].defined()
        ? grad_outputs[1].contiguous()
        : at::zeros_like(q);
    at::Tensor d_efield_tensor = grad_outputs[2].defined()
        ? (grad_outputs[2]).contiguous()
        : at::zeros_like(p);

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t max_pairs = npairs_excl > npairs ? npairs_excl : npairs;
    int grid_dim = std::min(
        static_cast<int>((max_pairs + MULTIPOLES_BLOCK_SIZE - 1) / MULTIPOLES_BLOCK_SIZE),
        props->maxBlocksPerMultiProcessor * props->multiProcessorCount
    );

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "multipolar_with_fields_backward_cuda", ([&] {
        const scalar_t cutoff = static_cast<scalar_t>(cutoff_val);
        const scalar_t ewald_alpha = static_cast<scalar_t>(ewald_alpha_val);
        const scalar_t prefactor = static_cast<scalar_t>(prefactor_val);

        scalar_t* coord_grad_ptr = need_coord_grad ? coord_grad.data_ptr<scalar_t>() : nullptr;
        scalar_t* q_grad_ptr = need_mpole_grad ? q_grad.data_ptr<scalar_t>() : nullptr;
        scalar_t* p_grad_ptr = need_mpole_grad ? p_grad.data_ptr<scalar_t>() : nullptr;
        scalar_t* t_grad_ptr = need_mpole_grad ? t_grad.data_ptr<scalar_t>() : nullptr;

        DISPATCH_BOOL(do_ewald, DO_EWALD, [&] {
            DISPATCH_BOOL(need_coord_grad, DO_COORD_GRAD, [&] {
                DISPATCH_BOOL(need_mpole_grad, DO_MPOLE_GRAD, [&] {
                    pairwise_multipole_with_fields_backward_kernel<
                        scalar_t, MULTIPOLES_BLOCK_SIZE, 2, DO_EWALD, true, DO_COORD_GRAD, DO_MPOLE_GRAD>
                    <<<grid_dim, MULTIPOLES_BLOCK_SIZE, 0, stream>>>(
                        coords.data_ptr<scalar_t>(),
                        box.data_ptr<scalar_t>(),
                        pairs.data_ptr<int64_t>(),
                        pairs_excl_ptr,
                        npairs,
                        npairs_excl,
                        cutoff,
                        ewald_alpha,
                        prefactor,
                        q.data_ptr<scalar_t>(),
                        p.data_ptr<scalar_t>(),
                        t.data_ptr<scalar_t>(),
                        d_energy.data_ptr<scalar_t>(),
                        d_epot.data_ptr<scalar_t>(),
                        d_efield_tensor.data_ptr<scalar_t>(),
                        coord_grad_ptr,
                        q_grad_ptr,
                        p_grad_ptr,
                        t_grad_ptr
                    );
                });
            });
        });
    }));

    at::Tensor ignore;
    std::vector<at::Tensor> grads(10);
    grads[0] = (ctx->saved_data["need_coord_grad"].toBool()) ? coord_grad : ignore;
    grads[1] = ignore;
    grads[2] = ignore;
    grads[3] = ignore;  // pairs_excl
    grads[4] = (ctx->saved_data["need_q_grad"].toBool()) ? q_grad : ignore;
    grads[5] = (ctx->saved_data["need_p_grad"].toBool()) ? p_grad : ignore;
    grads[6] = (ctx->saved_data["need_t_grad"].toBool()) ? t_grad : ignore;
    grads[7] = ignore;
    grads[8] = ignore;
    grads[9] = ignore;
    return grads;
}

};


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_multipolar_energy_and_fields_from_atom_pairs",
        [](const at::Tensor& coords,
            const at::Tensor& box,
            const at::Tensor& pairs,
            c10::optional<at::Tensor> pairs_excl,
            const at::Tensor& q,
            c10::optional<at::Tensor> p,
            c10::optional<at::Tensor> t,
            at::Scalar cutoff,
            at::Scalar ewald_alpha,
            at::Scalar prefactor) -> std::tuple<at::Tensor, at::Tensor, at::Tensor> {
            auto outs = MultipolarInteractionWithFieldsAtomPairsFunctionCuda::apply(
                const_cast<at::Tensor&>(coords), const_cast<at::Tensor&>(box),
                const_cast<at::Tensor&>(pairs), pairs_excl, const_cast<at::Tensor&>(q),
                p, t, cutoff, ewald_alpha, prefactor);
            return std::make_tuple(outs[0], outs[1], outs[2]);
        });
}