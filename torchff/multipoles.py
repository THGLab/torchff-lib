from typing import Optional
import math

import torch
import torch.nn as nn
import torchff_multipoles

from torchff.pbc import PBC


def computeInteractionTensor(drVec: torch.Tensor, dampFactors: Optional[torch.Tensor] = None, drInv: Optional[torch.Tensor] = None, rank: int = 2):
    if drInv is None:
        drInv = 1 / torch.norm(drVec, dim=1)
    
    if rank == 0:
        # For rank-0, dampFactors (if present) is a per-pair vector (erfc(b r)).
        # We should apply it elementwise, not index it as if it were a stacked tensor.
        return drInv if dampFactors is None else drInv * dampFactors
    
    # calculate inversions
    drInv2 = drInv * drInv
    drInv3 = drInv2 * drInv
    drInv5 = drInv3 * drInv2

    drVec2 = drVec * drVec
    x, y, z = drVec[:, 0], drVec[:, 1], drVec[:, 2]
    x2, y2, z2 = drVec2[:, 0], drVec2[:, 1], drVec2[:, 2]
    xy, xz, yz = x * y, x * z, y * z

    drInv7 = drInv5 * drInv2
    drInv9 = drInv7 * drInv2

    if dampFactors is not None:
        drInv = drInv * dampFactors[0]
        if rank > 0:
            drInv3 = drInv3 * dampFactors[1]
            drInv5 = drInv5 * dampFactors[2]
        if rank > 1:
            drInv7 = drInv7 * dampFactors[3]
            drInv9 = drInv9 * dampFactors[4]

    tx, ty, tz = -x * drInv3, -y * drInv3, -z * drInv3
    
    txx = 3 * x2 * drInv5 - drInv3
    txy = 3 * xy * drInv5
    txz = 3 * xz * drInv5
    tyy = 3 * y2 * drInv5 - drInv3
    tyz = 3 * yz * drInv5
    tzz = 3 * z2 * drInv5 - drInv3     

    if rank == 1:
        return torch.vstack((
            drInv, -tx,   -ty,   -tz,   
            tx,    -txx,  -txy,  -txz,  
            ty,    -txy,  -tyy,  -tyz,  
            tz,    -txz,  -tyz,  -tzz,  
        )).T.reshape(-1, 4, 4)
    
    txxx = -15 * x2 * x * drInv7 + 9 * x * drInv5
    txxy = -15 * x2 * y * drInv7 + 3 * y * drInv5
    txxz = -15 * x2 * z * drInv7 + 3 * z * drInv5
    tyyy = -15 * y2 * y * drInv7 + 9 * y * drInv5
    tyyx = -15 * y2 * x * drInv7 + 3 * x * drInv5
    tyyz = -15 * y2 * z * drInv7 + 3 * z * drInv5
    tzzz = -15 * z2 * z * drInv7 + 9 * z * drInv5
    tzzx = -15 * z2 * x * drInv7 + 3 * x * drInv5
    tzzy = -15 * z2 * y * drInv7 + 3 * y * drInv5
    txyz = -15 * x * y * z * drInv7

    txxxx = 105 * x2 * x2 * drInv9 - 90 * x2 * drInv7 + 9 * drInv5
    txxxy = 105 * x2 * xy * drInv9 - 45 * xy * drInv7
    txxxz = 105 * x2 * xz * drInv9 - 45 * xz * drInv7
    txxyy = 105 * x2 * y2 * drInv9 - 15 * (x2 + y2) * drInv7 + 3 * drInv5
    txxzz = 105 * x2 * z2 * drInv9 - 15 * (x2 + z2) * drInv7 + 3 * drInv5
    txxyz = 105 * x2 * yz * drInv9 - 15 * yz * drInv7

    tyyyy = 105 * y2 * y2 * drInv9 - 90 * y2 * drInv7 + 9 * drInv5
    tyyyx = 105 * y2 * xy * drInv9 - 45 * xy * drInv7
    tyyyz = 105 * y2 * yz * drInv9 - 45 * yz * drInv7
    tyyzz = 105 * y2 * z2 * drInv9 - 15 * (y2 + z2) * drInv7 + 3 * drInv5
    tyyxz = 105 * y2 * xz * drInv9 - 15 * xz * drInv7

    tzzzz = 105 * z2 * z2 * drInv9 - 90 * z2 * drInv7 + 9 * drInv5
    tzzzx = 105 * z2 * xz * drInv9 - 45 * xz * drInv7
    tzzzy = 105 * z2 * yz * drInv9 - 45 * yz * drInv7                
    tzzxy = 105 * z2 * xy * drInv9 - 15 * xy * drInv7

    return torch.vstack((
        drInv, -tx,   -ty,   -tz,   txx,   txy,   txz,   tyy,   tyz,   tzz,
        tx,    -txx,  -txy,  -txz,  txxx,  txxy,  txxz,  tyyx,  txyz,  tzzx,
        ty,    -txy,  -tyy,  -tyz,  txxy,  tyyx,  txyz,  tyyy,  tyyz,  tzzy,
        tz,    -txz,  -tyz,  -tzz,  txxz,  txyz,  tzzx,  tyyz,  tzzy,  tzzz,
        txx,   -txxx, -txxy, -txxz, txxxx, txxxy, txxxz, txxyy, txxyz, txxzz,
        txy,   -txxy, -tyyx, -txyz, txxxy, txxyy, txxyz, tyyyx, tyyxz, tzzxy,
        txz,   -txxz, -txyz, -tzzx, txxxz, txxyz, txxzz, tyyxz, tzzxy, tzzzx,
        tyy,   -tyyx, -tyyy, -tyyz, txxyy, tyyyx, tyyxz, tyyyy, tyyyz, tyyzz,
        tyz,   -txyz, -tyyz, -tzzy, txxyz, tyyxz, tzzxy, tyyyz, tyyzz, tzzzy,
        tzz,   -tzzx, -tzzy, -tzzz, txxzz, tzzxy, tzzzx, tyyzz, tzzzy, tzzzz
    )).T.reshape(-1, 10, 10)


def computeDampFactorsErfc(dr: torch.Tensor, b: float, rank: int):
    u = b * dr
    erfc_u = torch.erfc(u)
    if rank == 0:
        return erfc_u

    exp2_u = torch.exp(-u * u)

    u2 = u * u
    u3 = u2 * u
    u5 = u3 * u2
    u7 = u5 * u2
    
    p1 = 0.0
    p3  = u 
    p5  = (3*u + 2*u3) / 3
    p7  = (15*u + 10*u3 + 4*u5) / 15
    p9  = (8*u7 + 28*u5 + 70*u3 + 105*u) / 105
    prefactor = 2 / math.sqrt(math.pi)

    return torch.stack([erfc_u + prefactor * p * exp2_u for p in [p1, p3, p5, p7, p9]], dim=0)


@torch._dynamo.disable
def _compute_multipolar_energy_and_fields_from_atom_pairs(
    coords: torch.Tensor,
    box: torch.Tensor,
    pairs: torch.Tensor,
    pairs_excl: torch.Tensor | None,
    q: torch.Tensor,
    p: torch.Tensor | None,
    t: torch.Tensor | None,
    cutoff: float,
    ewald_alpha: float,
    prefactor: float,
):
    return torch.ops.torchff.compute_multipolar_energy_and_fields_from_atom_pairs(
        coords, box, pairs, pairs_excl, q, p, t,
        cutoff, ewald_alpha, prefactor,
    )


@torch._dynamo.disable
def _compute_multipolar_energy_from_atom_pairs(
    coords: torch.Tensor,
    box: torch.Tensor,
    pairs: torch.Tensor,
    pairs_excl: torch.Tensor | None,
    q: torch.Tensor,
    p: torch.Tensor | None,
    t: torch.Tensor | None,
    cutoff: float,
    ewald_alpha: float,
    prefactor: float,
):
    return torch.ops.torchff.compute_multipolar_energy_from_atom_pairs(
        coords, box, pairs, pairs_excl, q, p, t,
        cutoff, ewald_alpha, prefactor,
    )


class MultipolePacker(nn.Module):
    """
    Convert monopole, dipole, and quadrupole to (N, 10) polytensor with
    quadrupole entries scaled so symmetry-equivalent operations are avoided.
    """

    def __init__(self, rank: int = 2):
        super().__init__()
        self.rank = rank
        
    def forward(
        self,
        mono: torch.Tensor,
        dipo: torch.Tensor | None = None,
        quad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mono : torch.Tensor
            Monopoles, shape (N,).
        dipo : torch.Tensor
            Dipoles, shape (N, 3).
        quad : torch.Tensor
            Quadrupoles, shape (N, 3, 3).

        Returns
        -------
        torch.Tensor
            Polytensor [q, ux, uy, uz, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz], shape (N, 10).
        """
        if self.rank == 0:
            return mono
        elif self.rank == 1:
            return torch.hstack((mono.unsqueeze(1), dipo))
        else:
            return torch.hstack((
                mono.unsqueeze(1),
                dipo,
                quad[:, 0, 0].unsqueeze(1) / 3,
                (quad[:, 0, 1] + quad[:, 1, 0]).unsqueeze(1) / 3,
                (quad[:, 0, 2] + quad[:, 2, 0]).unsqueeze(1) / 3,
                quad[:, 1, 1].unsqueeze(1) / 3,
                (quad[:, 1, 2] + quad[:, 2, 1]).unsqueeze(1) / 3,
                quad[:, 2, 2].unsqueeze(1) / 3,
            ))


class MultipolarInteraction(nn.Module):
    def __init__(
        self,
        rank: int,
        cutoff: float,
        ewald_alpha: float = -1.0,
        prefactor: float = 1.0,
        return_fields: bool = False,
        use_customized_ops: bool = True,
        cuda_graph_compat: bool = True,
    ):
        super().__init__()
        self.rank = rank
        self.cutoff = cutoff
        self.ewald_alpha = ewald_alpha
        self.prefactor = prefactor
        self.return_fields = return_fields
        self.use_customized_ops = use_customized_ops
        if not use_customized_ops:
            self.pbc = PBC()
            self.packer = MultipolePacker(rank=rank)
            self.cuda_graph_compat = cuda_graph_compat

    def forward(self, coords, box, pairs, q, p=None, t=None, pairs_excl=None):
        if self.use_customized_ops:
            return self._forward_cpp(coords, box, pairs, q, p, t, pairs_excl)
        else:
            return self._forward_python(coords, box, pairs, q, p, t, pairs_excl)

    def _forward_python_from_packed_multipoles(self, coords, box, box_inv, multipoles, pairs, is_excl=False):
        dr_vecs = self.pbc(coords[pairs[:, 1]] - coords[pairs[:, 0]], box, box_inv)
        dr = torch.norm(dr_vecs, dim=1, keepdim=False)

        mask = dr <= self.cutoff
        if not self.cuda_graph_compat:
            pairs = pairs[mask]
            dr_vecs = dr_vecs[mask]
            dr = dr[mask]

        if self.ewald_alpha >= 0:
            damps = computeDampFactorsErfc(dr, self.ewald_alpha, rank=self.rank)
            if is_excl:
                damps = damps - 1.0
            i_tensor = computeInteractionTensor(dr_vecs, damps, 1.0/dr, rank=self.rank)
        else:
            i_tensor = computeInteractionTensor(dr_vecs, None, 1.0/dr, rank=self.rank)
        
        if ( not self.return_fields ):
            if self.rank == 0:
                # For monopoles, i_tensor already includes any Ewald damping (when ewald_alpha >= 0),
                # so we should not multiply by damps again here.
                ene_pairs = multipoles[pairs[:, 0]] * multipoles[pairs[:, 1]] * i_tensor
            else:
                m_j = multipoles[pairs[:, 1]]
                m_i = multipoles[pairs[:, 0]]
                ene_pairs = torch.bmm(m_j.unsqueeze(1), torch.bmm(i_tensor, m_i.unsqueeze(2))).squeeze(-1).squeeze(-1)
            
            if self.cuda_graph_compat:
                return self.prefactor * torch.sum(ene_pairs * mask)
            else:
                return self.prefactor * torch.sum(ene_pairs)
        else:
            N = coords.shape[0]
            device = coords.device
            dtype = coords.dtype
            if self.rank == 0:
                epot = torch.zeros(N, device=device, dtype=dtype)
                if self.cuda_graph_compat:
                    epot.scatter_add_(0, pairs[:, 0], (multipoles[pairs[:, 1]] * i_tensor) * mask)
                    epot.scatter_add_(0, pairs[:, 1], (multipoles[pairs[:, 0]] * i_tensor) * mask)
                else:
                    epot.scatter_add_(0, pairs[:, 0], multipoles[pairs[:, 1]] * i_tensor)
                    epot.scatter_add_(0, pairs[:, 1], multipoles[pairs[:, 0]] * i_tensor)
                epot *= self.prefactor
                return torch.sum(epot * multipoles) / 2, epot, None
            else:
                m_j = multipoles[pairs[:, 1]]
                m_i = multipoles[pairs[:, 0]]
                n_edata = 4 if self.rank == 1 else 10
                edata_ij = torch.bmm(i_tensor, m_i.unsqueeze(2)).squeeze(2)
                i_tensor_ji = i_tensor.permute(0, 2, 1)
                edata_ji = torch.bmm(i_tensor_ji, m_j.unsqueeze(2)).squeeze(2)
                edata = torch.zeros(N, n_edata, device=device, dtype=dtype)
                if self.cuda_graph_compat:
                    # Scatter masked contributions so invalid pairs add zero
                    mask_expand = mask.unsqueeze(1).expand(-1, n_edata)
                    edata.scatter_add_(0, pairs[:, 1].unsqueeze(1).expand(-1, n_edata), edata_ij * mask_expand)
                    edata.scatter_add_(0, pairs[:, 0].unsqueeze(1).expand(-1, n_edata), edata_ji * mask_expand)
                else:
                    edata.scatter_add_(0, pairs[:, 1].unsqueeze(1).expand(-1, n_edata), edata_ij)
                    edata.scatter_add_(0, pairs[:, 0].unsqueeze(1).expand(-1, n_edata), edata_ji)
                edata *= self.prefactor
                epot = edata[:, 0]
                efield = -edata[:, 1:4]
                energy = torch.sum(edata * multipoles) / 2
                return energy, epot, efield

    def _forward_python(self, coords, box, pairs, q, p=None, t=None, pairs_excl=None):
        box_inv, _ = torch.linalg.inv_ex(box)
        multipoles = self.packer(q, p, t)
        if pairs_excl is None or self.ewald_alpha <= 0:
            return self._forward_python_from_packed_multipoles(coords, box, box_inv, multipoles, pairs)
        else:
            # With exclusions and Ewald (\( \alpha > 0 \)), combine the regular and
            # exclusion contributions. The return type depends on whether fields are
            # requested:
            # - return_fields == False: scalar energy
            # - return_fields == True: (energy, epot, efield)
            ret = self._forward_python_from_packed_multipoles(
                coords, box, box_inv, multipoles, pairs
            )
            ret_excl = self._forward_python_from_packed_multipoles(
                coords, box, box_inv, multipoles, pairs_excl, True
            )

            if not self.return_fields:
                # Both calls returned scalar energies.
                return ret + ret_excl

            # Both calls returned (energy, epot, efield).
            energy = ret[0] + ret_excl[0]
            epot = ret[1] + ret_excl[1]
            if self.rank == 0:
                efield = None
            else:
                efield = ret[2] + ret_excl[2]
            return energy, epot, efield

    def _forward_cpp(self, coords, box, pairs, q, p=None, t=None, pairs_excl=None):
        # pairs_excl is only effective when ewald_alpha > 0; when None or ewald_alpha <= 0
        # the kernel receives nullptr and npairs_excl=0 (handled in C++/CUDA).
        if self.return_fields:
            energy, epot, efield = _compute_multipolar_energy_and_fields_from_atom_pairs(
                coords, box, pairs, pairs_excl, q, p, t,
                self.cutoff, self.ewald_alpha, self.prefactor,
            )
            return energy, epot, efield
        return _compute_multipolar_energy_from_atom_pairs(
            coords, box, pairs, pairs_excl, q, p, t,
            self.cutoff, self.ewald_alpha, self.prefactor,
        )
