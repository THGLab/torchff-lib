"""
AMOEBA water box: TorchFF Amoeba bond, Urey–Bradley, Amoeba angle, buffered 14–7 vdW,
multipole real-space + reciprocal PME, and direct polarization.

Initialization uses a plain configuration object (see ``examples/amoeba/md.py`` for
:class:`AmoebaTorchFFConfig` and construction from OpenMM). Place buffers and submodules on the
inference device and dtype with ``model.to(device, dtype)`` before calling :meth:`TorchFFAmoeba.forward`.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

import torchff_amoeba  # noqa: F401 — ``compute_amoeba_induced_field_from_atom_pairs``

from torchff.angle import AmoebaAngle
from torchff.bond import AmoebaBond, HarmonicBond
from torchff.multipoles import MultipolarInteraction, computeDampFactorsErfc, computeInteractionTensor
from torchff.nblist import NeighborList
from torchff.pme import PME
from torchff.ewald import Ewald
from torchff.multipoles import MultipolarRotation
from torchff.vdw import Vdw

COULOMB_PREFACTOR_KJMOL = 138.935456

# ``amoeba2018.xml`` water ``AmoebaBond``: ``k*(d^2 + c3*d^3 + c4*d^4); d=r-r0`` (nm).
_AMOEBA2018_BOND_CUBIC = -25.5
_AMOEBA2018_BOND_QUARTIC = 379.3125

# ``amoeba2018.xml`` water ``AmoebaAngle``: degree-polynomial terms mapped to radian displacement.
_AMOEBA2018_ANGLE_CUBIC = -0.8021409131831525
_AMOEBA2018_ANGLE_QUARTIC = 0.18383715560065766
_AMOEBA2018_ANGLE_PENTIC = -0.13166366417009362
_AMOEBA2018_ANGLE_SEXTIC = 0.23708998569690343


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
) -> torch.Tensor:
    return torch.ops.torchff.compute_multipolar_energy_from_atom_pairs(
        coords, box, pairs, pairs_excl, q, p, t, cutoff, ewald_alpha, prefactor
    )


def _compute_amoeba_induced_field_from_atom_pairs(
    coords: torch.Tensor,
    box: torch.Tensor,
    pairs: torch.Tensor,
    pairs_excl: torch.Tensor | None,
    q: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor | None,
    polarity: torch.Tensor,
    thole: torch.Tensor | float,
    cutoff: float,
    ewald_alpha: float,
    prefactor: float,
) -> torch.Tensor:
    return torch.ops.torchff.compute_amoeba_induced_field_from_atom_pairs(
        coords,
        box,
        pairs,
        pairs_excl,
        q,
        p,
        t,
        polarity,
        thole,
        cutoff,
        ewald_alpha,
        prefactor,
    )


def computeDampFactorsThole(
    dr: torch.Tensor,
    thole: float | torch.Tensor,
    factor: torch.Tensor
) -> torch.Tensor:
    u = dr * factor
    x = thole * (u**3)
    exp_x = torch.exp(-x)
    x2 = x * x
    p1 = torch.zeros_like(dr)
    p3 = -exp_x
    p5 = -(1.0 + x) * exp_x
    p7 = -(1.0 + x + 3.0 / 5.0 * x2) * exp_x
    p9 = -(1.0 + x + 18.0 / 35.0 * x2 + 9.0 / 35.0 * x2*x) * exp_x
    return torch.stack((p1, p3, p5, p7, p9), dim=0)


class MultipolarAmoeba(MultipolarInteraction):
    def __init__(
        self,
        rank: int,
        cutoff: float,
        ewald_alpha: float = -1.0,
        use_customized_ops: bool = True,
        cuda_graph_compat: bool = True,
        thole: torch.Tensor | float = 0.39,
    ):
        super().__init__(rank, cutoff, ewald_alpha, 1.0, True, use_customized_ops, cuda_graph_compat)
        self.thole = thole

    def forward(self, coords, box, pairs, q, p, t, polarity, pairs_excl=None):
        if self.use_customized_ops:
            return self._forward_cpp(coords, box, pairs, q, p, t, polarity, pairs_excl)
        else:
            return self._forward_python(coords, box, pairs, q, p, t, polarity, pairs_excl)

    def _forward_python_from_packed_multipoles(self, coords, box, box_inv, multipoles, polarity, pairs):
        dr_vecs = self.pbc(coords[pairs[:, 1]] - coords[pairs[:, 0]], box, box_inv)
        dr = torch.norm(dr_vecs, dim=1, keepdim=False)

        mask = dr <= self.cutoff
        if not self.cuda_graph_compat:
            pairs = pairs[mask]
            dr_vecs = dr_vecs[mask]
            dr = dr[mask]

        drInv = 1 / dr
        factor = torch.pow(polarity[pairs[:, 0]] * polarity[pairs[:, 1]], -1/6)
        if self.ewald_alpha >= 0:
            damps_perm = computeDampFactorsErfc(dr, self.ewald_alpha, rank=self.rank)
            damps_pol = damps_perm + computeDampFactorsThole(dr, self.thole, factor)
        else:
            damps_pol = 1.0 + computeDampFactorsThole(dr, self.thole, factor)
            
        i_tensor_perm = computeInteractionTensor(dr_vecs, damps_perm, drInv, rank=self.rank)
        m_j = multipoles[pairs[:, 1]]
        m_i = multipoles[pairs[:, 0]]
        
        ene_pairs = torch.bmm(m_j.unsqueeze(1), torch.bmm(i_tensor_perm, m_i.unsqueeze(2))).squeeze(-1).squeeze(-1)
        ene_perm = self.prefactor * torch.sum(ene_pairs * mask) if self.cuda_graph_compat else self.prefactor * torch.sum(ene_pairs)

        N = coords.shape[0]
        i_tensor_pol_ij = computeInteractionTensor(dr_vecs, damps_pol, drInv, rank=self.rank)
        i_tensor_pol_ji = i_tensor_pol_ij.permute(0, 2, 1)
        i_tensor_pol_ij = i_tensor_pol_ij[:, 1:4, :]
        i_tensor_pol_ji = i_tensor_pol_ji[:, 1:4, :]

        n_edata = 3
        edata_ij = torch.bmm(i_tensor_pol_ij, m_i.unsqueeze(2)).squeeze(2)
        edata_ji = torch.bmm(i_tensor_pol_ji, m_j.unsqueeze(2)).squeeze(2)
        edata = torch.zeros(N, n_edata, device=coords.device, dtype=coords.dtype)
        if self.cuda_graph_compat:
            # Scatter masked contributions so invalid pairs add zero
            mask_expand = mask.unsqueeze(1).expand(-1, n_edata)
            edata.scatter_add_(0, pairs[:, 1].unsqueeze(1).expand(-1, n_edata), edata_ij * mask_expand)
            edata.scatter_add_(0, pairs[:, 0].unsqueeze(1).expand(-1, n_edata), edata_ji * mask_expand)
        else:
            edata.scatter_add_(0, pairs[:, 1].unsqueeze(1).expand(-1, n_edata), edata_ij)
            edata.scatter_add_(0, pairs[:, 0].unsqueeze(1).expand(-1, n_edata), edata_ji)
        edata *= self.prefactor
        efield = -edata
        return ene_perm, efield
        
    def _forward_python(self, coords, box, pairs, q, p, t, polarity, pairs_excl=None):
        box_inv, _ = torch.linalg.inv_ex(box)
        multipoles = self.packer(q, p, t)
        if pairs_excl is None or self.ewald_alpha <= 0:
            return self._forward_python_from_packed_multipoles(coords, box, box_inv, multipoles, polarity, pairs)
        else:
            ene_perm, efield = self._forward_python_from_packed_multipoles(
                coords, box, box_inv, multipoles, polarity, pairs
            )
            ene_perm_excl, _, efield_excl = super()._forward_python_from_packed_multipoles(
                coords, box, box_inv, multipoles, pairs_excl, True
            )
            return ene_perm+ene_perm_excl, efield+efield_excl

    def _forward_cpp(self, coords, box, pairs, q, p, t, polarity, pairs_excl=None):
        # pairs_excl is only effective when ewald_alpha > 0; when None or ewald_alpha <= 0
        # the kernel receives nullptr and npairs_excl=0 (handled in C++/CUDA).
        perm_elec_energy = _compute_multipolar_energy_from_atom_pairs(
            coords, box, pairs, pairs_excl, q, p, t,
            self.cutoff, self.ewald_alpha, self.prefactor,
        )
        efield = _compute_amoeba_induced_field_from_atom_pairs(
            coords, box, pairs, pairs_excl, q, p, t,
            polarity, self.thole, self.cutoff, self.ewald_alpha, self.prefactor,
        )
        return perm_elec_energy, efield


class TorchFFAmoeba(nn.Module):
    """
    Full AMOEBA water energy: Amoeba bond, Urey–Bradley (harmonic bond), Amoeba angle,
    buffered 14–7 vdW, multipole real-space + PME reciprocal, and direct polarization.

    ``forward`` expects positions and box in **nanometers**, on the same device/dtype as buffers.
    """

    def __init__(self, config: Any, *, use_customized_ops: bool = True):
        """
        Parameters
        ----------
        config
            Namespace (e.g. :class:`AmoebaTorchFFConfig` in ``md.py``) with attributes:

            ``natoms``, ``cutoff_nm``, ``ewald_alpha_inv_nm``, ``max_hkl``,
            ``initial_positions_nm``, ``initial_box_nm``,
            ``bonds_amoeba``, ``b0_amoeba``, ``kb_amoeba``, ``bonds_ub``, ``b0_ub``, ``kb_ub``,
            ``angles``, ``th0``, ``k_angle``, ``sigma_table``, ``epsilon_table``,
            ``atom_types``, ``vdw_parent``, ``vdw_reduction``,
            ``q``, ``p_local``, ``t_local``, ``z_atoms``, ``x_atoms``, ``y_atoms``, ``axis_types``,
            ``polarity`` (isotropic polarizability volume in nm^3, OpenMM),
            ``thole``, ``excluded_pairs``, ``intra_pairs``.
        use_customized_ops
            Passed through to TorchFF modules.
        """
        super().__init__()
        c = config
        n_atoms = int(c.natoms)
        cutoff_nm = float(c.cutoff_nm)
        max_hkl = int(c.max_hkl)
        self.ewald_alpha = float(c.ewald_alpha_inv_nm)

        self.natoms = n_atoms
        self.n_waters = n_atoms // 3
        self.cutoff_nm = cutoff_nm
        self.use_customized_ops = use_customized_ops

        self.register_buffer("initial_positions_nm", c.initial_positions_nm.detach().clone())
        self.register_buffer("initial_box_nm", c.initial_box_nm.detach().clone())
        self.register_buffer("bonds_amoeba", c.bonds_amoeba.detach().clone())
        self.register_buffer("b0_amoeba", c.b0_amoeba.detach().clone())
        self.register_buffer("kb_amoeba", c.kb_amoeba.detach().clone())
        self.register_buffer("bonds_ub", c.bonds_ub.detach().clone())
        self.register_buffer("b0_ub", c.b0_ub.detach().clone())
        self.register_buffer("kb_ub", c.kb_ub.detach().clone())
        self.register_buffer("angles", c.angles.detach().clone())
        self.register_buffer("th0", c.th0.detach().clone())
        self.register_buffer("k_angle", c.k_angle.detach().clone())
        self.register_buffer("sigma_table", c.sigma_table.detach().clone())
        self.register_buffer("epsilon_table", c.epsilon_table.detach().clone())
        self.register_buffer("atom_types", c.atom_types.detach().clone())
        self.register_buffer("vdw_parent", c.vdw_parent.detach().clone())
        self.register_buffer("vdw_reduction", c.vdw_reduction.detach().clone())
        self.register_buffer("q", c.q.detach().clone())
        self.register_buffer("p_local", c.p_local.detach().clone())
        self.register_buffer("t_local", c.t_local.detach().clone())
        self.register_buffer("z_atoms", c.z_atoms.detach().clone())
        self.register_buffer("x_atoms", c.x_atoms.detach().clone())
        self.register_buffer("y_atoms", c.y_atoms.detach().clone())
        self.register_buffer("axis_types", c.axis_types.detach().clone())
        self.register_buffer("polarity", c.polarity.detach().clone())
        self.register_buffer("excluded_pairs", c.excluded_pairs.detach().clone())
        self.register_buffer("intra_pairs", c.intra_pairs.detach().clone())
        
        if self.use_customized_ops:
            self.register_buffer("thole", c.thole.detach().clone())
        else:
            self.thole = 0.39

        self.max_hkl = max_hkl

        u = use_customized_ops
        self.amoeba_bond = torch.compile(AmoebaBond(use_customized_ops=u))
        self.ub_bond = torch.compile(HarmonicBond(use_customized_ops=u))
        self.amoeba_angle = torch.compile(AmoebaAngle(use_customized_ops=u))
        self.vdw = torch.compile(Vdw(
            "AmoebaVdw147",
            cutoff=cutoff_nm,
            use_customized_ops=u,
            use_type_pairs=True,
        ))
        self.multipole = torch.compile(MultipolarAmoeba(
            rank=2,
            cutoff=self.cutoff_nm,
            ewald_alpha=self.ewald_alpha,
            use_customized_ops=u,
            thole=self.thole,
        ))
        self.pme = PME(
            alpha=self.ewald_alpha,
            max_hkl=max_hkl,
            rank=2,
            use_customized_ops=u,
            return_fields=True,
        )
        self.max_npairs = int(n_atoms * 4.0 / 3.0 * math.pi * cutoff_nm**3 * 100.0 / 2.0 * 1.2)
        self.nblist = NeighborList(
            n_atoms,
            exclusions=self.excluded_pairs,
            use_customized_ops=u,
            algorithm="nsquared",
        )
        self.rotation = torch.compile(MultipolarRotation(use_customized_ops=u))

    def energy_components(self, coords_nm: torch.Tensor, box_nm: torch.Tensor) -> dict[str, torch.Tensor]:
        """Energy terms: valence/vdW in kJ/mol; electrostatic pieces in Hartree (``*_hartree``) and kJ/mol (``*_kjmol``)."""
        pairs = self.nblist(coords_nm, box_nm, self.cutoff_nm, self.max_npairs, padding=True)
        pairs = pairs[0] if self.use_customized_ops else pairs
        e_amoeba_bond = self.amoeba_bond(
            coords_nm,
            self.bonds_amoeba,
            self.b0_amoeba,
            self.kb_amoeba,
            cubic=_AMOEBA2018_BOND_CUBIC,
            quartic=_AMOEBA2018_BOND_QUARTIC,
        )
        e_ub = self.ub_bond(coords_nm, self.bonds_ub, self.b0_ub, self.kb_ub)
        e_angle = self.amoeba_angle(
            coords_nm,
            self.angles,
            self.th0,
            self.k_angle,
            cubic=_AMOEBA2018_ANGLE_CUBIC,
            quartic=_AMOEBA2018_ANGLE_QUARTIC,
            pentic=_AMOEBA2018_ANGLE_PENTIC,
            sextic=_AMOEBA2018_ANGLE_SEXTIC,
        )
        f = self.vdw_reduction.unsqueeze(-1)
        parent_pos = coords_nm[self.vdw_parent]
        coords_vdw = f * coords_nm + (1.0 - f) * parent_pos

        e_vdw = self.vdw(
            coords_vdw,
            pairs,
            box_nm,
            self.sigma_table,
            self.epsilon_table,
            self.cutoff_nm,
            atom_types=self.atom_types,
        )

        p_b, t_b = self.rotation(
            coords_nm, self.z_atoms, self.x_atoms, self.y_atoms, self.axis_types,
            self.p_local, self.t_local
        )

        ene_real, field_real = self.multipole(
            coords_nm, box_nm, pairs, self.q, p_b, t_b, self.polarity, pairs_excl=self.intra_pairs
        )
        ene_recip, _, field_recip = self.pme(coords_nm, box_nm, self.q, p_b, t_b)
        efield = field_real + field_recip
        e_pol = -torch.sum(self.polarity.reshape(-1, 1) * efield * efield) / 2.0

        return {
            "AmoebaBond": e_amoeba_bond,
            "HarmonicBond": e_ub,
            "AmoebaAngle": e_angle,
            "AmoebaVdw": e_vdw,
            "AmoebaPermElec": COULOMB_PREFACTOR_KJMOL * (ene_real + ene_recip),
            "AmoebaPolarization": COULOMB_PREFACTOR_KJMOL * e_pol,
        }

    def forward(self, coords_nm: torch.Tensor, box_nm: torch.Tensor) -> torch.Tensor:
        c = self.energy_components(coords_nm, box_nm)
        return (
            c["AmoebaBond"]
            + c["HarmonicBond"]
            + c["AmoebaAngle"]
            + c["AmoebaVdw"]
            + c["AmoebaPermElec"]
            + c["AmoebaPolarization"]
        )
