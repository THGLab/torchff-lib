import torch 
import torch.nn as nn

import torchff
import torchff_ewald


@torch._dynamo.disable
def ewald_long_range(
    coords: torch.Tensor, box: torch.Tensor, 
    q: torch.Tensor, p: torch.Tensor | None, t: torch.Tensor | None, 
    kmax: int, alpha: float
):
    return torch.ops.torchff.ewald_long_range(coords, box, q, p, t, kmax, alpha)


@torch._dynamo.disable
def ewald_long_range_with_fields(
    coords: torch.Tensor, box: torch.Tensor, 
    q: torch.Tensor, p: torch.Tensor | None, t: torch.Tensor | None, 
    kmax: int, alpha: float
):
    return torch.ops.torchff.ewald_long_range_all(coords, box, q, p, t, kmax, alpha)


class Ewald(nn.Module):
    def __init__(self, alpha: float, kmax: int, rank: int, use_customized_ops: bool = True, return_fields: bool = True):
        super().__init__()
        self.kmax = kmax
        self.alpha = alpha
        self.rank = rank
        self.return_fields = return_fields
        self.use_customized_ops = use_customized_ops

        if not self.use_customized_ops:
            kvecs = []
            for kx in range(0, kmax+1):
                for ky in range(-kmax, kmax+1):
                    for kz in range(-kmax, kmax+1):
                        if kx > 0:
                            kvecs.append([kx, ky, kz])
                        elif (kx == 0) and (ky > 0):
                            kvecs.append([kx, ky, kz])
                        elif (kx == 0) and (ky == 0) and (kz > 0):
                            kvecs.append([kx, ky, kz])
                            
            self.register_buffer('kvecs', torch.tensor(kvecs, dtype=torch.get_default_dtype()))
            self.alpha2 = alpha * alpha
            self.alpha_over_root_pi = self.alpha / torch.sqrt(torch.tensor(torch.pi))
    
    def forward(self, coords, box, q, p=None, t=None):
        if self.use_customized_ops:
            if not self.return_fields:
                return ewald_long_range(coords, box, q, p, t, self.kmax, self.alpha)
            else:
                return ewald_long_range_with_fields(coords, box, q, p, t, self.kmax, self.alpha)
        else:
            return self._forward_python(coords, box, q, p, t)

    def _forward_python(self, coords, box, q, p=None, t=None):
        # box_inv = torch.inverse(box)
        box_inv, _ = torch.linalg.inv_ex(box)
        V = torch.det(box)

        # Convert h,k,l indices to reciprocal space vectors
        kvectors = torch.matmul(self.kvecs, box_inv)   # (M, 3)
    
        # Apply spherical cutoff based on k^2
        k_squared = torch.einsum('ij,ij->i', kvectors, kvectors) # (M,)
        gaussian_factors = torch.exp(-torch.pi * torch.pi * k_squared / self.alpha2) / k_squared # (M,)

        # Calculating all structure factors
        k_dot_r = torch.matmul(kvectors, coords.T) # (M, N)
        cos_k_dot_r = torch.cos(2 * torch.pi * k_dot_r) # (M, N)
        sin_k_dot_r = torch.sin(2 * torch.pi * k_dot_r) # (M, N)

        if self.rank == 2:
            # eqn (2.7) (M, N)
            L_real = q.expand(kvectors.size(0), -1) - torch.einsum('kj,nij,ki->kn', kvectors, t, kvectors) * (2 * torch.pi) * (2 * torch.pi) / 3
            L_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
        elif self.rank == 1:
            L_real = q.expand(kvectors.size(0), -1)
            L_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
        else:
            L_real = q.expand(kvectors.size(0), -1)
            L_imag = torch.zeros(kvectors.size(0), q.size(0), device=q.device, dtype=q.dtype)
        
        # Structure factors, eqn 2.6
        S_real = torch.sum(cos_k_dot_r * L_real - sin_k_dot_r * L_imag, dim=1) # (M,)
        S_imag = torch.sum(cos_k_dot_r * L_imag + sin_k_dot_r * L_real, dim=1) # (M,)
        
        energy = torch.sum(gaussian_factors * (S_real ** 2 + S_imag ** 2)) / torch.pi / V
        energy -= self.alpha_over_root_pi * torch.sum(q*q)
        if self.rank >= 1:
            energy -= self.alpha_over_root_pi * (2*self.alpha2/3) * torch.sum(p*p)
        if self.rank == 2:
            energy -= self.alpha_over_root_pi * (8*self.alpha2*self.alpha2/45) * torch.sum(t*t)
        if not self.return_fields:
            return energy

        S_real_expanded = gaussian_factors * S_real
        S_imag_expanded = gaussian_factors * S_imag

        # S * exp(-i * 2 * pi * k * r)
        K_real = S_real_expanded.unsqueeze(1) * cos_k_dot_r + S_imag_expanded.unsqueeze(1) * sin_k_dot_r # (M,N)
        
        # eqn 2.8
        potential = (2.0 / torch.pi / V) * torch.sum(K_real, dim=0)
        potential = potential - 2 * self.alpha_over_root_pi * q  # self contributions
        if self.rank == 0:
            return energy, potential, torch.zeros((q.size(0), 3), device=q.device, dtype=q.dtype)
        
        # equation 2.10
        K_imag = S_real_expanded.unsqueeze(1) * sin_k_dot_r - S_imag_expanded.unsqueeze(1) * cos_k_dot_r # (M,N)
        field = 4.0 / V * (torch.matmul(K_imag.T, kvectors))
        field = field + self.alpha_over_root_pi * (4 * self.alpha2 / 3) * p

        if self.rank == 1:
            return energy, potential, field
        
        # k_outer = torch.einsum('bi,bj->bij', kvectors, kvectors) # (M, 3, 3)
        # field_grad = 8.0 * torch.pi / V * (torch.matmul(K_real.T, k_outer.reshape(-1, 9))).reshape(-1, 3, 3)
        field_grad = 8.0 * torch.pi / V * torch.einsum('mj,mx,my->jxy', K_real, kvectors, kvectors)
        field_grad = field_grad + self.alpha_over_root_pi * (16 * self.alpha2 * self.alpha2 / 5) * t / 3
        if self.rank == 2:
            return energy, potential, field
        
