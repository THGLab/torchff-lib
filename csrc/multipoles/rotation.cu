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


enum AxisType {
    ZThenX            = 0,
    Bisector          = 1,
    ZBisect           = 2,
    ThreeFold         = 3,
    ZOnly             = 4,
    NoAxisType        = 5,
    LastAxisTypeIndex = 6,
};


template <typename scalar_t>
__global__ void compute_rotation_matrices_forward_kernel(
    scalar_t* coords,
    int64_t* zatoms, int64_t* xatoms, int64_t* yatoms,
    int64_t* axistypes,
    int64_t natoms,
    scalar_t* rot_matrices
)
{
    constexpr scalar_t ONE = scalar_t(1.0);
    constexpr scalar_t ZERO = scalar_t(0.0);
    constexpr scalar_t THRESH = scalar_t(0.866);

    int64_t start = threadIdx.x + blockDim.x * blockIdx.x;
    for (int64_t index = start; index < natoms; index += gridDim.x * blockDim.x) {
        int64_t type = axistypes[index];

        scalar_t xvec[3] = {};
        scalar_t zvec[3] = {};
        scalar_t yvec[3] = {};
        if ( type == NoAxisType ) {
            xvec[0] = ONE; yvec[1] = ONE; zvec[2] = ONE;
        }
        else {
            scalar_t ri[3];
            ri[0] = coords[index*3]; ri[1] = coords[index*3+1]; ri[2] = coords[index*3+2];

            /* zatom */
            int64_t z = zatoms[index];
            if ( z >= 0 ) {
                diff_vec3(&coords[z*3], ri, zvec);
                normalize_vec3(zvec, zvec);
            }
            /* xatom */
            int64_t x = xatoms[index];
            if ( x >= 0 ) {
                diff_vec3(&coords[x*3], ri, xvec);
            }
            /* yatom */
            int64_t y = yatoms[index]; 
            if ( y >= 0 ) {
                diff_vec3(&coords[y*3], ri, yvec);
            }

            if ( type == Bisector ) {
                normalize_vec3(xvec, xvec);
                add_vec3(zvec, xvec, zvec);
                normalize_vec3(zvec, zvec);
            }
            else if ( type == ZBisect ) {
                normalize_vec3(xvec, xvec);
                add_vec3(xvec, yvec, xvec);
            }
            else if ( type == ThreeFold ) {
                normalize_vec3(xvec, xvec);
                normalize_vec3(yvec, yvec);
                zvec[0] = zvec[0] + xvec[0] + yvec[0];
                zvec[1] = zvec[1] + xvec[1] + yvec[1];
                zvec[2] = zvec[2] + xvec[2] + yvec[2];
                normalize_vec3(zvec, zvec);
            }
            else if ( type == ZOnly ) {
                if ( abs_(zvec[0]) < THRESH ) {
                    xvec[0] = ONE;
                }
                else {
                    xvec[1] = ONE;
                }
            }

            scalar_t dot = dot_vec3(xvec, zvec);
            xvec[0] -= dot * zvec[0];
            xvec[1] -= dot * zvec[1];
            xvec[2] -= dot * zvec[2];
            normalize_vec3(xvec, xvec);

            if ( type == ZThenX && y >= 0 ) {
                scalar_t tmp[3] = {};
                cross_vec3(zvec, xvec, tmp);
                if ( dot_vec3(tmp, yvec) > 0 ) {
                    yvec[0] = tmp[0]; yvec[1] = tmp[1]; yvec[2] = tmp[2];
                }
                else{
                    yvec[0] = -tmp[0]; yvec[1] = -tmp[1]; yvec[2] = -tmp[2];
                }
            }
            else {
                cross_vec3(zvec, xvec, yvec);
            }
        }
        rot_matrices[index*9]   = xvec[0]; rot_matrices[index*9+1] = xvec[1]; rot_matrices[index*9+2] = xvec[2]; 
        rot_matrices[index*9+3] = yvec[0]; rot_matrices[index*9+4] = yvec[1]; rot_matrices[index*9+5] = yvec[2]; 
        rot_matrices[index*9+6] = zvec[0]; rot_matrices[index*9+7] = zvec[1]; rot_matrices[index*9+8] = zvec[2]; 
    }
}


template <typename scalar_t>
__global__ void compute_rotation_matrices_backward_kernel(
    scalar_t* coords,
    int64_t* zatoms, int64_t* xatoms, int64_t* yatoms,
    int64_t* axistypes,
    int64_t natoms,
    scalar_t* rot_matrices_grad,
    scalar_t* coords_grad
)
{
    constexpr scalar_t ONE = scalar_t(1.0);
    constexpr scalar_t MINUS_ONE = scalar_t(-1.0);
    constexpr scalar_t ZERO = scalar_t(0.0);
    constexpr scalar_t THRESH = scalar_t(0.866);

    int64_t start = threadIdx.x + blockDim.x * blockIdx.x;
    for (int64_t index = start; index < natoms; index += gridDim.x * blockDim.x) {

        int64_t type = axistypes[index];
        if ( type == NoAxisType ) {
            continue;
        }

        /* central atom */
        scalar_t ri[3];
        ri[0] = coords[index*3]; ri[1] = coords[index*3+1]; ri[2] = coords[index*3+2];
        /* z atom */
        int64_t z = zatoms[index];
        scalar_t rzi_norm_inv = ZERO;
        scalar_t rzi[3] = {};
        if ( z >= 0 ) {
            diff_vec3(&coords[z*3], ri, rzi);
            rzi_norm_inv = rnorm3d_(rzi[0], rzi[1], rzi[2]);
        }
        /* x atom */
        int64_t x = xatoms[index];
        scalar_t rxi_norm_inv = ZERO;
        scalar_t rxi[3] = {};
        if ( x >= 0 ) {
            diff_vec3(&coords[x*3], ri, rxi);
            rxi_norm_inv = rnorm3d_(rxi[0], rxi[1], rxi[2]);
        }
        /* y atom */
        int64_t y = yatoms[index];
        scalar_t ryi_norm_inv = ZERO;
        scalar_t ryi[3] = {};
        if ( y >= 0 ) {
            diff_vec3(&coords[y*3], ri, ryi);
            ryi_norm_inv = rnorm3d_(ryi[0], ryi[1], ryi[2]);
        }


        scalar_t u[3] = {};
        scalar_t w[3] = {};
        if ( type == ZThenX ) {
            /* u = rzi; w = rxi */
            u[0] = rzi[0]; u[1] = rzi[1]; u[2] = rzi[2];
            w[0] = rxi[0]; w[1] = rxi[1]; w[2] = rxi[2];
        }
        else if ( type == Bisector ) {
            /* u = norm(rzi)+norm(rxi); w = rxi */
            w[0] = rxi[0]; w[1] = rxi[1]; w[2] = rxi[2];
            u[0] = rzi[0]*rzi_norm_inv + rxi[0]*rxi_norm_inv;
            u[1] = rzi[1]*rzi_norm_inv + rxi[1]*rxi_norm_inv; 
            u[2] = rzi[2]*rzi_norm_inv + rxi[2]*rxi_norm_inv;
        }
        else if ( type == ZBisect ) {
            /* u = rzi; w = norm(rxi)+norm(ryi) */
            u[0] = rzi[0]; u[1] = rzi[1]; u[2] = rzi[2];
            w[0] = rxi[0]*rxi_norm_inv + ryi[0]*ryi_norm_inv; 
            w[1] = rxi[1]*rxi_norm_inv + ryi[1]*ryi_norm_inv; 
            w[2] = rxi[2]*rxi_norm_inv + ryi[2]*ryi_norm_inv;
        }
        else if ( type == ThreeFold ) {
            /* u = norm(rxi)+norm(ryi)+norm(rzi); w = rxi; */
            w[0] = rxi[0]; w[1] = rxi[1]; w[2] = rxi[2];
            u[0] = rxi[0]*rxi_norm_inv + ryi[0]*ryi_norm_inv + rzi[0]*rzi_norm_inv;
            u[1] = rxi[1]*rxi_norm_inv + ryi[1]*ryi_norm_inv + rzi[1]*rzi_norm_inv;
            u[2] = rxi[2]*rxi_norm_inv + ryi[2]*ryi_norm_inv + rzi[2]*rzi_norm_inv;
        }
        else if ( type == ZOnly ) {
            /* u = rzi; w = random */
            u[0] = rzi[0]; u[1] = rzi[1]; u[2] = rzi[2];
            if (abs_(u[0]) * rzi_norm_inv < THRESH) {
                w[0] = ONE; w[1] = ZERO; w[2] = ZERO;
            }
            else {
                w[0] = ZERO; w[1] = ONE; w[2] = ZERO;
            }
        }

        /* z = norm(u) */
        scalar_t zvec[3];
        scalar_t u_norm_inv = rnorm3d_(u[0], u[1], u[2]);
        zvec[0] = u[0]*u_norm_inv; zvec[1] = u[1]*u_norm_inv; zvec[2] = u[2]*u_norm_inv;

        /* v = w - (w.z)z */
        scalar_t v[3];
        scalar_t dot = w[0]*zvec[0] + w[1]*zvec[1] + w[2]*zvec[2];
        v[0] = w[0] - dot * zvec[0];
        v[1] = w[1] - dot * zvec[1];
        v[2] = w[2] - dot * zvec[2];
        scalar_t v_norm_inv = rnorm3d_(v[0], v[1], v[2]);

        /* x = norm(v) */
        scalar_t xvec[3];
        xvec[0] = v[0]*v_norm_inv; xvec[1] = v[1]*v_norm_inv; xvec[2] = v[2]*v_norm_inv;

        /* check chirality - revese y */
        scalar_t reverse = ONE;
        scalar_t* tmp = v;
        if ( type == ZThenX && y >= 0 ) {
            cross_vec3(zvec, xvec, tmp);
            if ( dot_vec3(tmp, ryi) < 0 ) {
                reverse = MINUS_ONE;
            }
        }

        /* Backward */
        scalar_t dzvec[3];
        scalar_t dxvec[3];
        
        // back prop of computing y - cross product
        cross_vec3(xvec, &rot_matrices_grad[index*9+3], tmp);
        dzvec[0] = rot_matrices_grad[index*9+6] + tmp[0] * reverse;
        dzvec[1] = rot_matrices_grad[index*9+7] + tmp[1] * reverse;
        dzvec[2] = rot_matrices_grad[index*9+8] + tmp[2] * reverse;

        cross_vec3(&rot_matrices_grad[index*9+3], zvec, tmp);
        dxvec[0] = rot_matrices_grad[index*9]   + tmp[0] * reverse;
        dxvec[1] = rot_matrices_grad[index*9+1] + tmp[1] * reverse;
        dxvec[2] = rot_matrices_grad[index*9+2] + tmp[2] * reverse;

        // back prop of norm x = v / |v|
        scalar_t dv[3];
        dv[0] = v_norm_inv * ( dxvec[0] - xvec[0]*xvec[0]*dxvec[0] - xvec[0]*xvec[1]*dxvec[1] - xvec[0]*xvec[2]*dxvec[2]);
        dv[1] = v_norm_inv * ( dxvec[1] - xvec[1]*xvec[0]*dxvec[0] - xvec[1]*xvec[1]*dxvec[1] - xvec[1]*xvec[2]*dxvec[2]); 
        dv[2] = v_norm_inv * ( dxvec[2] - xvec[2]*xvec[0]*dxvec[0] - xvec[2]*xvec[1]*dxvec[1] - xvec[2]*xvec[2]*dxvec[2]);

        // back prop of v = w - (w.z)z  w.r.t. z
        scalar_t dot_dv_zvec = dot_vec3(dv, zvec);
        dzvec[0] += -dv[0]*dot - w[0]*dot_dv_zvec;
        dzvec[1] += -dv[1]*dot - w[1]*dot_dv_zvec;
        dzvec[2] += -dv[2]*dot - w[2]*dot_dv_zvec;

        // back prop of norm z = u / |u|
        scalar_t du[3];
        du[0] = u_norm_inv * ( dzvec[0] - zvec[0]*zvec[0]*dzvec[0] - zvec[0]*zvec[1]*dzvec[1] - zvec[0]*zvec[2]*dzvec[2]);
        du[1] = u_norm_inv * ( dzvec[1] - zvec[1]*zvec[0]*dzvec[0] - zvec[1]*zvec[1]*dzvec[1] - zvec[1]*zvec[2]*dzvec[2]); 
        du[2] = u_norm_inv * ( dzvec[2] - zvec[2]*zvec[0]*dzvec[0] - zvec[2]*zvec[1]*dzvec[1] - zvec[2]*zvec[2]*dzvec[2]);

        // back prop of v = w - (w.z)z  w.r.t. w
        scalar_t dw[3];
        dw[0] = dv[0] - zvec[0]*zvec[0]*dv[0] - zvec[0]*zvec[1]*dv[1] - zvec[0]*zvec[2]*dv[2];
        dw[1] = dv[1] - zvec[1]*zvec[0]*dv[0] - zvec[1]*zvec[1]*dv[1] - zvec[1]*zvec[2]*dv[2];
        dw[2] = dv[2] - zvec[2]*zvec[0]*dv[0] - zvec[2]*zvec[1]*dv[1] - zvec[2]*zvec[2]*dv[2];

        scalar_t drx[3] = {};
        scalar_t dry[3] = {};
        scalar_t drz[3] = {};
        if ( type == ZThenX ) {
            /* u = rzi; w = rxi */
            drz[0] = du[0]; drz[1] = du[1]; drz[2] = du[2]; 
            drx[0] = dw[0]; drx[1] = dw[1]; drx[2] = dw[2]; 
        }
        else if ( type == Bisector ) {
            /* u = norm(rzi)+norm(rxi); w = rxi */
            drx[0] = dw[0] + rxi_norm_inv*du[0] - rxi_norm_inv*rxi_norm_inv*rxi_norm_inv * (rxi[0]*rxi[0]*du[0] + rxi[0]*rxi[1]*du[1] + rxi[0]*rxi[2]*du[2]);
            drx[1] = dw[1] + rxi_norm_inv*du[1] - rxi_norm_inv*rxi_norm_inv*rxi_norm_inv * (rxi[1]*rxi[0]*du[0] + rxi[1]*rxi[1]*du[1] + rxi[1]*rxi[2]*du[2]);
            drx[2] = dw[2] + rxi_norm_inv*du[2] - rxi_norm_inv*rxi_norm_inv*rxi_norm_inv * (rxi[2]*rxi[0]*du[0] + rxi[2]*rxi[1]*du[1] + rxi[2]*rxi[2]*du[2]);

            drz[0] = rzi_norm_inv*du[0] - rzi_norm_inv*rzi_norm_inv*rzi_norm_inv * (rzi[0]*rzi[0]*du[0] + rzi[0]*rzi[1]*du[1] + rzi[0]*rzi[2]*du[2]);
            drz[1] = rzi_norm_inv*du[1] - rzi_norm_inv*rzi_norm_inv*rzi_norm_inv * (rzi[1]*rzi[0]*du[0] + rzi[1]*rzi[1]*du[1] + rzi[1]*rzi[2]*du[2]);
            drz[2] = rzi_norm_inv*du[2] - rzi_norm_inv*rzi_norm_inv*rzi_norm_inv * (rzi[2]*rzi[0]*du[0] + rzi[2]*rzi[1]*du[1] + rzi[2]*rzi[2]*du[2]);
        }
        else if ( type == ZBisect ) {
            /* u = rzi; w = norm(rxi)+norm(ryi) */
            drz[0] = du[0]; drz[1] = du[1]; drz[2] = du[2]; 

            drx[0] = rxi_norm_inv*dw[0] - rxi_norm_inv*rxi_norm_inv*rxi_norm_inv * (rxi[0]*rxi[0]*dw[0] + rxi[0]*rxi[1]*dw[1] + rxi[0]*rxi[2]*dw[2]);
            drx[1] = rxi_norm_inv*dw[1] - rxi_norm_inv*rxi_norm_inv*rxi_norm_inv * (rxi[1]*rxi[0]*dw[0] + rxi[1]*rxi[1]*dw[1] + rxi[1]*rxi[2]*dw[2]);
            drx[2] = rxi_norm_inv*dw[2] - rxi_norm_inv*rxi_norm_inv*rxi_norm_inv * (rxi[2]*rxi[0]*dw[0] + rxi[2]*rxi[1]*dw[1] + rxi[2]*rxi[2]*dw[2]);

            dry[0] = ryi_norm_inv*dw[0] - ryi_norm_inv*ryi_norm_inv*ryi_norm_inv * (ryi[0]*ryi[0]*dw[0] + ryi[0]*ryi[1]*dw[1] + ryi[0]*ryi[2]*dw[2]);
            dry[1] = ryi_norm_inv*dw[1] - ryi_norm_inv*ryi_norm_inv*ryi_norm_inv * (ryi[1]*ryi[0]*dw[0] + ryi[1]*ryi[1]*dw[1] + ryi[1]*ryi[2]*dw[2]);
            dry[2] = ryi_norm_inv*dw[2] - ryi_norm_inv*ryi_norm_inv*ryi_norm_inv * (ryi[2]*ryi[0]*dw[0] + ryi[2]*ryi[1]*dw[1] + ryi[2]*ryi[2]*dw[2]);
        }
        else if ( type == ThreeFold ) {
            /* u = norm(rxi)+norm(ryi)+norm(rzi); w = rxi; */
            drx[0] = dw[0] + rxi_norm_inv*du[0] - rxi_norm_inv*rxi_norm_inv*rxi_norm_inv * (rxi[0]*rxi[0]*du[0] + rxi[0]*rxi[1]*du[1] + rxi[0]*rxi[2]*du[2]);
            drx[1] = dw[1] + rxi_norm_inv*du[1] - rxi_norm_inv*rxi_norm_inv*rxi_norm_inv * (rxi[1]*rxi[0]*du[0] + rxi[1]*rxi[1]*du[1] + rxi[1]*rxi[2]*du[2]);
            drx[2] = dw[2] + rxi_norm_inv*du[2] - rxi_norm_inv*rxi_norm_inv*rxi_norm_inv * (rxi[2]*rxi[0]*du[0] + rxi[2]*rxi[1]*du[1] + rxi[2]*rxi[2]*du[2]);

            drz[0] = rzi_norm_inv*du[0] - rzi_norm_inv*rzi_norm_inv*rzi_norm_inv * (rzi[0]*rzi[0]*du[0] + rzi[0]*rzi[1]*du[1] + rzi[0]*rzi[2]*du[2]);
            drz[1] = rzi_norm_inv*du[1] - rzi_norm_inv*rzi_norm_inv*rzi_norm_inv * (rzi[1]*rzi[0]*du[0] + rzi[1]*rzi[1]*du[1] + rzi[1]*rzi[2]*du[2]);
            drz[2] = rzi_norm_inv*du[2] - rzi_norm_inv*rzi_norm_inv*rzi_norm_inv * (rzi[2]*rzi[0]*du[0] + rzi[2]*rzi[1]*du[1] + rzi[2]*rzi[2]*du[2]);

            dry[0] = ryi_norm_inv*du[0] - ryi_norm_inv*ryi_norm_inv*ryi_norm_inv * (ryi[0]*ryi[0]*du[0] + ryi[0]*ryi[1]*du[1] + ryi[0]*ryi[2]*du[2]);
            dry[1] = ryi_norm_inv*du[1] - ryi_norm_inv*ryi_norm_inv*ryi_norm_inv * (ryi[1]*ryi[0]*du[0] + ryi[1]*ryi[1]*du[1] + ryi[1]*ryi[2]*du[2]);
            dry[2] = ryi_norm_inv*du[2] - ryi_norm_inv*ryi_norm_inv*ryi_norm_inv * (ryi[2]*ryi[0]*du[0] + ryi[2]*ryi[1]*du[1] + ryi[2]*ryi[2]*du[2]);
        }
        else if ( type == ZOnly ) {
            /* u = rzi; w = random */
            drz[0] = du[0]; drz[1] = du[1]; drz[2] = du[2];
        }

        /* write back */
        if ( z >= 0 ) {
            atomicAdd(&coords_grad[z*3], drz[0]);
            atomicAdd(&coords_grad[z*3+1], drz[1]);
            atomicAdd(&coords_grad[z*3+2], drz[2]);
        }
        if ( x >= 0 ) {
            atomicAdd(&coords_grad[x*3], drx[0]);
            atomicAdd(&coords_grad[x*3+1], drx[1]);
            atomicAdd(&coords_grad[x*3+2], drx[2]);
        } 
        if ( y >= 0 ) {
            atomicAdd(&coords_grad[y*3], dry[0]);
            atomicAdd(&coords_grad[y*3+1], dry[1]);
            atomicAdd(&coords_grad[y*3+2], dry[2]);
        } 
        atomicAdd(&coords_grad[index*3], -drx[0]-dry[0]-drz[0]);
        atomicAdd(&coords_grad[index*3+1], -drx[1]-dry[1]-drz[1]);
        atomicAdd(&coords_grad[index*3+2], -drx[2]-dry[2]-drz[2]);

    }
}



class RotationMatricesFunctionCuda: public torch::autograd::Function<RotationMatricesFunctionCuda> {

public: 

static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& coords,
    at::Tensor& zatoms, at::Tensor& xatoms, at::Tensor& yatoms, 
    at::Tensor& axistypes
)
{
    int64_t natoms = coords.size(0);
    
    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    
    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = std::min(
        static_cast<int>((natoms + BLOCK_SIZE - 1) / BLOCK_SIZE),
        props->multiProcessorCount*props->maxBlocksPerMultiProcessor
    );

    at::Tensor rot_matrices = at::zeros({natoms, 3, 3}, coords.options());

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_rotation_matrices_forward_kernel", ([&] {
        compute_rotation_matrices_forward_kernel<scalar_t><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            zatoms.data_ptr<int64_t>(), xatoms.data_ptr<int64_t>(), yatoms.data_ptr<int64_t>(),
            axistypes.data_ptr<int64_t>(),
            natoms,
            rot_matrices.data_ptr<scalar_t>()
        );
    }));
    ctx->save_for_backward({coords, zatoms, xatoms, yatoms, axistypes});
    ctx->saved_data["natoms"] = natoms;
    ctx->saved_data["block_dim"] = BLOCK_SIZE;
    ctx->saved_data["grid_dim"] = GRID_SIZE;
    return rot_matrices;
}

static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<at::Tensor> grad_outputs
)
{
    auto saved = ctx->get_saved_variables();
    at::Tensor coords_grad = at::zeros_like(saved[0], saved[0].options());
    int64_t natoms = static_cast<int64_t>(ctx->saved_data["natoms"].toInt());
    int64_t block_dim = static_cast<int64_t>(ctx->saved_data["block_dim"].toInt());
    int64_t grid_dim = static_cast<int64_t>(ctx->saved_data["grid_dim"].toInt());
    auto stream = at::cuda::getCurrentCUDAStream();
    

    AT_DISPATCH_FLOATING_TYPES(saved[0].scalar_type(), "compute_rotation_matrices_backward_kernel", ([&] {
        compute_rotation_matrices_backward_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            saved[0].data_ptr<scalar_t>(),
            saved[1].data_ptr<int64_t>(), saved[2].data_ptr<int64_t>(), saved[3].data_ptr<int64_t>(),
            saved[4].data_ptr<int64_t>(),
            natoms,
            grad_outputs[0].contiguous().data_ptr<scalar_t>(),
            coords_grad.data_ptr<scalar_t>()
        );
    }));

    at::Tensor ignore;
    return {coords_grad, ignore, ignore, ignore, ignore};
}

};


at::Tensor compute_rotation_matrices_cuda(
    at::Tensor& coords,
    at::Tensor& zatoms, at::Tensor& xatoms, at::Tensor& yatoms, 
    at::Tensor& axistypes
) {
    return RotationMatricesFunctionCuda::apply(coords, zatoms, xatoms, yatoms, axistypes);
}



TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_rotation_matrices", compute_rotation_matrices_cuda);
}
