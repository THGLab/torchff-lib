#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <iomanip>
#include <ATen/cuda/CUDAContext.h>
#include "kernels.cuh"
#include "kernels_with_field.cuh"
#include "ewald/self_contribution.cuh"


class PMEEnergyFunction : public torch::autograd::Function<PMEEnergyFunction> {

public:

static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& coords,
    const at::Tensor& box,
    const at::Tensor& q,
    c10::optional<at::Tensor> p,
    c10::optional<at::Tensor> t,
    at::Scalar K_t,
    at::Scalar alpha_t,
    const at::Tensor& xmoduli, const at::Tensor& ymoduli, const at::Tensor& zmoduli
){

    // Determine rank from optional tensors
    int64_t rank = 0;
    if (t.has_value()) {
        rank = 2;
    } else if (p.has_value()) {
        rank = 1;
    }
    const int64_t K = K_t.toLong();
    const int K1 = K, K2 = K, K3 = K;

    // Resolve optional tensors into (possibly undefined) Tensors
    at::Tensor p_used = p.has_value() ? p.value() : at::Tensor();
    at::Tensor t_used = t.has_value() ? t.value() : at::Tensor();

    // --- 1. Setup Grid & Geometry ---
    at::Tensor box_contig = box.contiguous().to(coords.dtype());
    int N = coords.size(0);
    auto options = coords.options();

    // // --- 3. Allocate Outputs ---
    at::Tensor epot = torch::zeros({N}, options);
    // Only allocate internal E/EG buffers when the rank requires them.
    at::Tensor efield;
    at::Tensor efield_grad; // stores full 3x3 per atom as flat N*9
    if (rank >= 1) {
        efield = torch::zeros({N, 3}, options);
    }
    if (rank >= 2) {
        efield_grad = torch::zeros({N, 3, 3}, options);
    }
    at::Tensor forces = torch::zeros({N, 3}, options);

    at::Tensor energy = at::zeros({}, options);

    // --- 4. Run CUDA Pipeline (inlined from compute_pme_cuda_pipeline) ---
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "pme_long_range_forward", ([&] {
        scalar_t alpha_val = static_cast<scalar_t>(alpha_t.toDouble());

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        scalar_t* coords_ptr = coords.data_ptr<scalar_t>();
        const scalar_t* q_ptr = q.data_ptr<scalar_t>();
        scalar_t* energy_ptr = energy.data_ptr<scalar_t>();

        scalar_t* box_ptr = box_contig.data_ptr<scalar_t>();
        scalar_t* xmoduli_ptr = xmoduli.data_ptr<scalar_t>();
        scalar_t* ymoduli_ptr = ymoduli.data_ptr<scalar_t>();
        scalar_t* zmoduli_ptr = zmoduli.data_ptr<scalar_t>();
        // scalar_t* phi_ptr = phi.data_ptr<scalar_t>();
        // scalar_t* force_ptr = forces.data_ptr<scalar_t>();

        int K3_complex = K3 / 2 + 1;

        // 1. Allocate 3D grid
        auto grid_Q = torch::zeros({K1, K2, K3}, coords.options());
        scalar_t* grid_Q_ptr = grid_Q.data_ptr<scalar_t>();
        constexpr int BLOCK_SIZE = 256;
        int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // 2. Set rank-dependent pointers
        const scalar_t* p_ptr = nullptr;
        const scalar_t* Q_ptr = nullptr;

        if (rank >= 1 && p_used.defined()) {
            p_ptr = p_used.data_ptr<scalar_t>();
        }
        if (rank >= 2 && t_used.defined()) {
            Q_ptr = t_used.data_ptr<scalar_t>();
        }

        // 3. Spread charges / multipoles
        if (rank == 0) {
            spread_q_kernel<scalar_t, 0><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords_ptr, q_ptr, p_ptr, Q_ptr, box_ptr,
                grid_Q_ptr, N, K1, K2, K3);
        } else if (rank == 1) {
            spread_q_kernel<scalar_t, 1><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords_ptr, q_ptr, p_ptr, Q_ptr, box_ptr,
                grid_Q_ptr, N, K1, K2, K3);
        } else {
            spread_q_kernel<scalar_t, 2><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords_ptr, q_ptr, p_ptr, Q_ptr, box_ptr,
                grid_Q_ptr, N, K1, K2, K3);
        }
        // std::cout << std::fixed << std::setprecision(8) << "Grid Q: " << grid_Q.cpu() << std::endl;

        // 4. Forward FFT (real to complex)
        auto grid_Q_fft = torch::fft::rfftn(grid_Q).contiguous();
        c10::complex<scalar_t>* grid_Q_fft_ptr = grid_Q_fft.data_ptr<c10::complex<scalar_t>>();

        dim3 dimBlock(8, 8, 8);
        dim3 dimGrid((K1 + 7) / 8, (K2 + 7) / 8, (K3_complex + 7) / 8);
        pme_convolution_fused_kernel<scalar_t><<<dimGrid, dimBlock, 0, stream>>>(
            grid_Q_fft_ptr, box_ptr, xmoduli_ptr, ymoduli_ptr, zmoduli_ptr,
            K1, K2, K3, alpha_val);

        // 5. Inverse FFT (complex to real)
        auto grid_Phi = torch::fft::irfftn(grid_Q_fft, {K1, K2, K3}, c10::nullopt, "forward");

        // 6. Compute Energy
        at::sum_out(energy, grid_Phi * grid_Q);
        energy *= 0.5;
        // std::cout << std::fixed << std::setprecision(8) << "Energy: " << energy.cpu().item() << std::endl;

        // 7. Self-correction
        energy_ptr = energy.data_ptr<scalar_t>();
        if (rank == 0) {
            compute_self_contribution_kernel_rank_0<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                q_ptr, N, alpha_val,
                energy_ptr, epot.data_ptr<scalar_t>()
            );
        }
        else if (rank == 1) {
            compute_self_contribution_kernel_rank_1<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                q_ptr, p_ptr, N, alpha_val,
                energy_ptr, epot.data_ptr<scalar_t>(), efield.data_ptr<scalar_t>()
            );
        }
        else {
            compute_self_contribution_kernel_rank_2<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                q_ptr, p_ptr, Q_ptr, N, alpha_val,
                energy_ptr, epot.data_ptr<scalar_t>(), efield.data_ptr<scalar_t>(), efield_grad.data_ptr<scalar_t>()
            );
        }

        // Save for backward pass
        ctx->saved_data["rank"] = rank;
        ctx->saved_data["alpha"] = alpha_t;
        ctx->saved_data["N"] = N;
        ctx->saved_data["K1"] = K1;
        ctx->saved_data["K2"] = K2;
        ctx->saved_data["K3"] = K3;
    
        ctx->save_for_backward({coords, box, q, p_used, t_used, epot, efield, efield_grad, grid_Phi});
    }));
    return energy;

}

static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs) {
    // Only the gradient w.r.t. the energy output is used
    const at::Tensor& g_energy = grad_outputs[0];

    auto saved = ctx->get_saved_variables();
    at::Tensor& coords = saved[0];
    at::Tensor& box = saved[1];
    at::Tensor& q = saved[2];
    at::Tensor& p = saved[3];
    at::Tensor& t = saved[4];
    at::Tensor& epot = saved[5];
    at::Tensor& efield = saved[6];
    at::Tensor& efield_grad = saved[7];
    at::Tensor& grid_Phi = saved[8];

    int64_t rank = ctx->saved_data["rank"].toInt();
    int64_t K1 = ctx->saved_data["K1"].toInt();
    int64_t K2 = ctx->saved_data["K2"].toInt();
    int64_t K3 = ctx->saved_data["K3"].toInt();
    int64_t N = ctx->saved_data["N"].toInt();

    auto opts = saved[0].options();
    at::Tensor dcoords = torch::zeros_like(saved[0], opts);

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "pme_long_range_backward", ([&] {
        scalar_t alpha_val = static_cast<scalar_t>(ctx->saved_data["alpha"].toDouble());

        constexpr int BLOCK_SIZE = 256;
        int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        auto stream = at::cuda::getCurrentCUDAStream();

        scalar_t* coords_ptr = coords.data_ptr<scalar_t>();
        scalar_t* box_ptr = box.data_ptr<scalar_t>();
        scalar_t* q_ptr = q.data_ptr<scalar_t>();
        scalar_t* p_ptr = rank >= 1 ? p.data_ptr<scalar_t>() : nullptr;
        scalar_t* Q_ptr = rank >= 2 ? t.data_ptr<scalar_t>() : nullptr;

        scalar_t* grid_Phi_ptr = grid_Phi.data_ptr<scalar_t>();
        scalar_t* epot_ptr = epot.data_ptr<scalar_t>();
        scalar_t* efield_ptr = (rank >= 1 && efield.defined()) ? efield.data_ptr<scalar_t>() : nullptr;
        scalar_t* efield_grad_ptr = (rank >= 2 && efield_grad.defined()) ? efield_grad.data_ptr<scalar_t>() : nullptr;
        scalar_t* dcoords_ptr = dcoords.data_ptr<scalar_t>();

        if (rank == 0) {
            interpolate_kernel<scalar_t, 0><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                grid_Phi_ptr, coords_ptr, box_ptr, q_ptr, p_ptr, Q_ptr,
                epot_ptr, efield_ptr, efield_grad_ptr, dcoords_ptr, alpha_val, N, K1, K2, K3);
        } else if (rank == 1) {
            interpolate_kernel<scalar_t, 1><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                grid_Phi_ptr, coords_ptr, box_ptr, q_ptr, p_ptr, Q_ptr,
                epot_ptr, efield_ptr, efield_grad_ptr, dcoords_ptr, alpha_val, N, K1, K2, K3);
        } else {
            interpolate_kernel<scalar_t, 2><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                grid_Phi_ptr, coords_ptr, box_ptr, q_ptr, p_ptr, Q_ptr,
                epot_ptr, efield_ptr, efield_grad_ptr, dcoords_ptr, alpha_val, N, K1, K2, K3);
        }
    }));

    dcoords.mul_(-g_energy);
    epot.mul_(g_energy);
    if (rank >= 1 && efield.defined()) efield.mul_(-g_energy);
    if (rank >= 2 && efield_grad.defined()) efield_grad.mul_(-g_energy/3.0);

    at::Tensor ignore;
    return {
        dcoords,        // coords
        ignore,   // box
        epot,   // q (monopoles)
        rank >= 1 ? efield : ignore,            // p (Dipoles)
        rank >= 2 ? efield_grad : ignore,            // t (Quadrupoles)
        ignore,   // K
        ignore,   // alpha
        ignore,   // xmoduli
        ignore,   // ymoduli
        ignore    // zmoduli
    };
}

};


class PMEAllFunction : public torch::autograd::Function<PMEAllFunction> {

public:

static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& coords,
    const at::Tensor& box,
    const at::Tensor& q,
    c10::optional<at::Tensor> p,
    c10::optional<at::Tensor> t,
    at::Scalar K_t,
    at::Scalar alpha_t,
    const at::Tensor& xmoduli, const at::Tensor& ymoduli, const at::Tensor& zmoduli
){

    // Determine rank from optional tensors
    int64_t rank = 0;
    if (t.has_value()) {
        rank = 2;
    } else if (p.has_value()) {
        rank = 1;
    }
    const int64_t K = K_t.toLong();
    const int K1 = K, K2 = K, K3 = K;

    // Resolve optional tensors into (possibly undefined) Tensors
    at::Tensor p_used = p.has_value() ? p.value() : at::Tensor();
    at::Tensor t_used = t.has_value() ? t.value() : at::Tensor();

    // --- 1. Setup Grid & Geometry ---
    at::Tensor box_contig = box.contiguous().to(coords.dtype());
    int N = coords.size(0);
    auto options = coords.options();

    // // --- 3. Allocate Outputs ---
    at::Tensor epot = torch::zeros({N}, options);
    at::Tensor efield = torch::zeros({N, 3}, options);
    at::Tensor efield_grad = torch::zeros({N, 3, 3}, options);
    at::Tensor forces = torch::zeros({N, 3}, options);

    at::Tensor energy = at::zeros({}, options);

    // --- 4. Run CUDA Pipeline (inlined from compute_pme_cuda_pipeline) ---
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "pme_long_range_forward", ([&] {
        scalar_t alpha_val = static_cast<scalar_t>(alpha_t.toDouble());

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        scalar_t* coords_ptr = coords.data_ptr<scalar_t>();
        const scalar_t* q_ptr = q.data_ptr<scalar_t>();
        scalar_t* energy_ptr = energy.data_ptr<scalar_t>();

        scalar_t* box_ptr = box_contig.data_ptr<scalar_t>();
        scalar_t* xmoduli_ptr = xmoduli.data_ptr<scalar_t>();
        scalar_t* ymoduli_ptr = ymoduli.data_ptr<scalar_t>();
        scalar_t* zmoduli_ptr = zmoduli.data_ptr<scalar_t>();

        // Gradients
        scalar_t* forces_ptr = forces.data_ptr<scalar_t>();
        scalar_t* epot_ptr = epot.data_ptr<scalar_t>();
        scalar_t* efield_ptr = (rank >= 1) ? efield.data_ptr<scalar_t>() : nullptr;
        scalar_t* efield_grad_ptr = (rank >= 2) ? efield_grad.data_ptr<scalar_t>() : nullptr;

        int K3_complex = K3 / 2 + 1;

        // 1. Allocate 3D grid
        auto grid_Q = torch::zeros({K1, K2, K3}, coords.options());
        scalar_t* grid_Q_ptr = grid_Q.data_ptr<scalar_t>();
        constexpr int BLOCK_SIZE = 256;
        int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // 2. Set rank-dependent pointers
        const scalar_t* p_ptr = nullptr;
        const scalar_t* Q_ptr = nullptr;

        if (rank >= 1 && p_used.defined()) {
            p_ptr = p_used.data_ptr<scalar_t>();
        }
        if (rank >= 2 && t_used.defined()) {
            Q_ptr = t_used.data_ptr<scalar_t>();
        }

        // 3. Spread charges / multipoles
        if (rank == 0) {
            spread_q_kernel<scalar_t, 0><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords_ptr, q_ptr, p_ptr, Q_ptr, box_ptr,
                grid_Q_ptr, N, K1, K2, K3);
        } else if (rank == 1) {
            spread_q_kernel<scalar_t, 1><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords_ptr, q_ptr, p_ptr, Q_ptr, box_ptr,
                grid_Q_ptr, N, K1, K2, K3);
        } else {
            spread_q_kernel<scalar_t, 2><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords_ptr, q_ptr, p_ptr, Q_ptr, box_ptr,
                grid_Q_ptr, N, K1, K2, K3);
        }
        // std::cout << std::fixed << std::setprecision(8) << "Grid Q: " << grid_Q.cpu() << std::endl;

        // 4. Forward FFT (real to complex)
        auto grid_Q_fft = torch::fft::rfftn(grid_Q).contiguous();
        c10::complex<scalar_t>* grid_Q_fft_ptr = grid_Q_fft.data_ptr<c10::complex<scalar_t>>();

        dim3 dimBlock(8, 8, 8);
        dim3 dimGrid((K1 + 7) / 8, (K2 + 7) / 8, (K3_complex + 7) / 8);
        pme_convolution_fused_kernel<scalar_t><<<dimGrid, dimBlock, 0, stream>>>(
            grid_Q_fft_ptr, box_ptr, xmoduli_ptr, ymoduli_ptr, zmoduli_ptr,
            K1, K2, K3, alpha_val);

        // 5. Inverse FFT (complex to real)
        auto grid_Phi = torch::fft::irfftn(grid_Q_fft, {K1, K2, K3}, c10::nullopt, "forward");

        // 6. Compute Energy
        at::sum_out(energy, grid_Phi * grid_Q);
        energy *= 0.5;
        // std::cout << std::fixed << std::setprecision(8) << "Energy: " << energy.cpu().item() << std::endl;

        // 7. Self-correction
        energy_ptr = energy.data_ptr<scalar_t>();
        if (rank == 0) {
            compute_self_contribution_kernel_rank_0<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                q_ptr, N, alpha_val,
                energy_ptr, epot_ptr
            );
        }
        else if (rank == 1) {
            compute_self_contribution_kernel_rank_1<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                q_ptr, p_ptr, N, alpha_val,
                energy_ptr, epot_ptr, efield_ptr
            );
        }
        else {
            compute_self_contribution_kernel_rank_2<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                q_ptr, p_ptr, Q_ptr, N, alpha_val,
                energy_ptr, epot_ptr, efield_ptr, efield_grad_ptr
            );
        }

        scalar_t* grid_Phi_ptr = grid_Phi.data_ptr<scalar_t>();
        if (rank == 0) {
            interpolate_kernel<scalar_t, 0><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                grid_Phi_ptr, coords_ptr, box_ptr, q_ptr, p_ptr, Q_ptr,
                epot_ptr, efield_ptr, efield_grad_ptr, forces_ptr, alpha_val, N, K1, K2, K3);
        } else if (rank == 1) {
            interpolate_kernel<scalar_t, 1><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                grid_Phi_ptr, coords_ptr, box_ptr, q_ptr, p_ptr, Q_ptr,
                epot_ptr, efield_ptr, efield_grad_ptr, forces_ptr, alpha_val, N, K1, K2, K3);
        } else {
            interpolate_kernel<scalar_t, 2><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                grid_Phi_ptr, coords_ptr, box_ptr, q_ptr, p_ptr, Q_ptr,
                epot_ptr, efield_ptr, efield_grad_ptr, forces_ptr, alpha_val, N, K1, K2, K3);
        }

        // Save for backward pass
        ctx->saved_data["rank"] = rank;
        ctx->saved_data["alpha"] = alpha_t;
        ctx->saved_data["N"] = N;
        ctx->saved_data["K1"] = K1;
        ctx->saved_data["K2"] = K2;
        ctx->saved_data["K3"] = K3;
    
        ctx->save_for_backward({coords, box, q, p_used, t_used, xmoduli, ymoduli, zmoduli, grid_Phi});
    }));
    return {energy, epot, efield};

}

static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs) {
    // forward returns a single energy scalar, so grad_outputs has one element
    const at::Tensor& g_energy = grad_outputs[0];
   

    auto saved = ctx->get_saved_variables();
    auto opts = saved[0].options();
    at::Tensor& coords = saved[0];
    at::Tensor& box = saved[1];
    at::Tensor& q = saved[2];
    at::Tensor& p = saved[3];
    at::Tensor& t = saved[4];
    at::Tensor& xmoduli = saved[5];
    at::Tensor& ymoduli = saved[6];
    at::Tensor& zmoduli = saved[7];

    // expand "multipoles"
    int64_t rank = ctx->saved_data["rank"].toInt();
    at::Tensor q_expand = q * g_energy + grad_outputs[1];
    at::Tensor p_expand;
    at::Tensor t_expand;
    at::Tensor q_grad = at::zeros_like(q, opts);
    at::Tensor p_grad;
    at::Tensor t_grad;
    if ( rank >= 1 ) {
        p_expand = p * g_energy - grad_outputs[2];
        p_grad = at::zeros_like(p, opts);
    }
    if ( rank >= 2 ) {
        t_expand = t * g_energy;
        t_grad = at::zeros_like(t, opts);
    }

    int64_t K1 = ctx->saved_data["K1"].toInt();
    int64_t K2 = ctx->saved_data["K2"].toInt();
    int64_t K3 = ctx->saved_data["K3"].toInt();
    int64_t N = ctx->saved_data["N"].toInt();

    at::Tensor coords_grad = torch::zeros_like(saved[0], opts);

    // Here the energy is just a placeholder
    at::Tensor energy = at::zeros({}, opts);

    // --- 4. Run CUDA Pipeline (inlined from compute_pme_cuda_pipeline) ---
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "pme_long_range_forward", ([&] {
        scalar_t alpha_val = static_cast<scalar_t>(ctx->saved_data["alpha"].toDouble());

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        scalar_t* coords_ptr = coords.data_ptr<scalar_t>();
        scalar_t* energy_ptr = energy.data_ptr<scalar_t>();

        scalar_t* box_ptr = box.data_ptr<scalar_t>();
        scalar_t* xmoduli_ptr = xmoduli.data_ptr<scalar_t>();
        scalar_t* ymoduli_ptr = ymoduli.data_ptr<scalar_t>();
        scalar_t* zmoduli_ptr = zmoduli.data_ptr<scalar_t>();

        // Gradients
        scalar_t* coords_grad_ptr = coords_grad.data_ptr<scalar_t>();
        scalar_t* q_grad_ptr = q_grad.data_ptr<scalar_t>();
        scalar_t* p_grad_ptr = ( rank >= 1 ) ? p_grad.data_ptr<scalar_t>() : nullptr;
        scalar_t* t_grad_ptr = ( rank >= 2 ) ? t_grad.data_ptr<scalar_t>() : nullptr;

        int K3_complex = K3 / 2 + 1;

        // 1. Allocate 3D grid
        auto grid_Q = torch::zeros({K1, K2, K3}, coords.options());
        scalar_t* grid_Q_ptr = grid_Q.data_ptr<scalar_t>();
        constexpr int BLOCK_SIZE = 256;
        int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // 2. Set rank-dependent pointers
        const scalar_t* q_ptr = q_expand.data_ptr<scalar_t>();
        const scalar_t* p_ptr = ( rank >= 1 ) ? p_expand.data_ptr<scalar_t>() : nullptr;
        const scalar_t* t_ptr = ( rank >= 2 ) ? t_expand.data_ptr<scalar_t>() : nullptr;

        // 3. Spread charges / multipoles
        if (rank == 0) {
            spread_q_kernel<scalar_t, 0><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords_ptr, q_ptr, p_ptr, t_ptr, box_ptr,
                grid_Q_ptr, N, K1, K2, K3);
        } else if (rank == 1) {
            spread_q_kernel<scalar_t, 1><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords_ptr, q_ptr, p_ptr, t_ptr, box_ptr,
                grid_Q_ptr, N, K1, K2, K3);
        } else {
            spread_q_kernel<scalar_t, 2><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords_ptr, q_ptr, p_ptr, t_ptr, box_ptr,
                grid_Q_ptr, N, K1, K2, K3);
        }

        // 4. Forward FFT (real to complex)
        auto grid_Q_fft = torch::fft::rfftn(grid_Q).contiguous();
        c10::complex<scalar_t>* grid_Q_fft_ptr = grid_Q_fft.data_ptr<c10::complex<scalar_t>>();

        dim3 dimBlock(8, 8, 8);
        dim3 dimGrid((K1 + 7) / 8, (K2 + 7) / 8, (K3_complex + 7) / 8);
        pme_convolution_fused_kernel<scalar_t><<<dimGrid, dimBlock, 0, stream>>>(
            grid_Q_fft_ptr, box_ptr, xmoduli_ptr, ymoduli_ptr, zmoduli_ptr,
            K1, K2, K3, alpha_val);


        auto grid_Phi = torch::fft::irfftn(grid_Q_fft, {K1, K2, K3}, c10::nullopt, "forward");

        if (rank == 0) {
            compute_self_contribution_kernel_rank_0<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                q_ptr, N, alpha_val,
                energy_ptr, q_grad_ptr
            );
        }
        else if (rank == 1) {
            compute_self_contribution_kernel_rank_1<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                q_ptr, p_ptr, N, alpha_val,
                energy_ptr, q_grad_ptr, p_grad_ptr
            );
        }
        else {
            compute_self_contribution_kernel_rank_2<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                q_ptr, p_ptr, t_ptr, N, alpha_val,
                energy_ptr, q_grad_ptr, p_grad_ptr, t_grad_ptr
            );
        }

        scalar_t* grid_Phi_ptr_backward = grid_Phi.data_ptr<scalar_t>();
        scalar_t* grid_Phi_ptr_forward = saved[8].data_ptr<scalar_t>();
        if (rank == 0) {
            interpolate_kernel_with_field<scalar_t, 0><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                grid_Phi_ptr_backward, grid_Phi_ptr_forward, coords_ptr, box_ptr, q_ptr, p_ptr, t_ptr,
                grad_outputs[1].data_ptr<scalar_t>(), grad_outputs[2].data_ptr<scalar_t>(),
                q_grad_ptr, p_grad_ptr, t_grad_ptr, coords_grad_ptr, alpha_val, N, K1, K2, K3);
        } else if (rank == 1) {
            interpolate_kernel_with_field<scalar_t, 1><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                grid_Phi_ptr_backward, grid_Phi_ptr_forward, coords_ptr, box_ptr, q_ptr, p_ptr, t_ptr,
                grad_outputs[1].data_ptr<scalar_t>(), grad_outputs[2].data_ptr<scalar_t>(),
                q_grad_ptr, p_grad_ptr, t_grad_ptr, coords_grad_ptr, alpha_val, N, K1, K2, K3);
        } else {
            interpolate_kernel_with_field<scalar_t, 2><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                grid_Phi_ptr_backward, grid_Phi_ptr_forward, coords_ptr, box_ptr, q_ptr, p_ptr, t_ptr,
                grad_outputs[1].data_ptr<scalar_t>(), grad_outputs[2].data_ptr<scalar_t>(),
                q_grad_ptr, p_grad_ptr, t_grad_ptr, coords_grad_ptr, alpha_val, N, K1, K2, K3);
        }
    
    }));

    at::Tensor ignore;
    return {
        -coords_grad,                  // coords
        ignore,                        // box
        q_grad,                        // q (monopoles)
        rank >= 1 ? -p_grad : ignore,   // p (Dipoles)
        rank >= 2 ? -t_grad * (1.0/3.0) : ignore,   // t (Quadrupoles)
        ignore,                        // K
        ignore,                        // alpha
        ignore,                        // xmoduli
        ignore,                        // ymoduli
        ignore                         // zmoduli
    };
}
};



TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("pme_long_range",
        [](const at::Tensor& coords,
           const at::Tensor& box,
           const at::Tensor& q,
           c10::optional<at::Tensor> p,
           c10::optional<at::Tensor> t,
           at::Scalar K,
           at::Scalar alpha,
           const at::Tensor& xmoduli,
           const at::Tensor& ymoduli,
           const at::Tensor& zmoduli) {
           return PMEEnergyFunction::apply(coords, box, q, p, t, K, alpha, xmoduli, ymoduli, zmoduli);
        });
    m.impl("pme_long_range_all",
        [](const at::Tensor& coords,
           const at::Tensor& box,
           const at::Tensor& q,
           c10::optional<at::Tensor> p,
           c10::optional<at::Tensor> t,
           at::Scalar K,
           at::Scalar alpha,
           const at::Tensor& xmoduli,
           const at::Tensor& ymoduli,
           const at::Tensor& zmoduli) -> std::tuple<at::Tensor, at::Tensor, at::Tensor> {
           auto outs = PMEAllFunction::apply(coords, box, q, p, t, K, alpha, xmoduli, ymoduli, zmoduli);
           return std::make_tuple(outs[0], outs[1], outs[2]);
        });
}
