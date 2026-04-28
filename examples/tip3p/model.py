"""
TIP3P water box: TorchFF bond, angle, Lennard-Jones, screened real-space Coulomb, and reciprocal PME.

Initialization uses a plain configuration object (see ``examples/tip3p/md.py`` for
:class:`Tip3pTorchFFConfig` and construction from OpenMM). Place buffers and submodules on the
inference device and dtype with ``model.to(device, dtype)`` before calling :meth:`Tip3pTorchFF.forward`.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from torchff.angle import HarmonicAngle
from torchff.bond import HarmonicBond
from torchff.multipoles import MultipolarInteraction
from torchff.nblist import NeighborList
from torchff.pme import PME
from torchff.vdw import Vdw

# kJ mol^-1 nm e^-2 (ONE_4PI_EPS0), same as tests/test_nonbonded.py and OpenMM fixed charges.
COULOMB_PREFACTOR_KJMOL = 138.935456


class Tip3pTorchFF(nn.Module):
    """TIP3P water box: bond + angle + LJ + screened real-space Coulomb + reciprocal PME."""

    def __init__(self, config: Any, *, use_customized_ops: bool = False):
        """
        Parameters
        ----------
        config
            Namespace (e.g. :class:`Tip3pTorchFFConfig` in ``md.py``) with attributes:

            ``natoms``, ``cutoff_nm``, ``ewald_alpha``, ``max_hkl``,
            ``initial_positions_nm``, ``initial_box_nm``,
            ``bonds``, ``b0``, ``kb``, ``angles``, ``th0``, ``kth``, ``charges``,
            ``sigma``, ``epsilon`` (type-pair LJ tables),
            ``atom_types``, ``excluded_pairs``, ``coulomb_excl_pairs``.
        use_customized_ops
            Passed through to TorchFF modules.
        """
        super().__init__()
        c = config
        n_atoms = int(c.natoms)
        cutoff_nm = float(c.cutoff_nm)
        ewald_alpha = float(c.ewald_alpha)
        max_hkl = int(c.max_hkl)

        self.natoms = n_atoms
        self.cutoff_nm = cutoff_nm

        self.register_buffer("initial_positions_nm", c.initial_positions_nm.detach().clone())
        self.register_buffer("initial_box_nm", c.initial_box_nm.detach().clone())
        self.register_buffer("bonds", c.bonds.detach().clone())
        self.register_buffer("b0", c.b0.detach().clone())
        self.register_buffer("kb", c.kb.detach().clone())
        self.register_buffer("angles", c.angles.detach().clone())
        self.register_buffer("th0", c.th0.detach().clone())
        self.register_buffer("kth", c.kth.detach().clone())
        self.register_buffer("charges", c.charges.detach().clone())
        self.register_buffer("sigma", c.sigma.detach().clone())
        self.register_buffer("epsilon", c.epsilon.detach().clone())
        self.register_buffer("atom_types", c.atom_types.detach().clone())
        self.register_buffer("excluded_pairs", c.excluded_pairs.detach().clone())
        self.register_buffer("coulomb_excl_pairs", c.coulomb_excl_pairs.detach().clone())

        self.bond = HarmonicBond(use_customized_ops=use_customized_ops)
        self.angle = HarmonicAngle(use_customized_ops=use_customized_ops)
        self.lj = Vdw(
            "LennardJones",
            cutoff=cutoff_nm,
            use_customized_ops=use_customized_ops,
            use_type_pairs=True,
            sum_output=True,
        )
        self.coul = MultipolarInteraction(
            rank=0,
            cutoff=cutoff_nm,
            ewald_alpha=ewald_alpha,
            prefactor=COULOMB_PREFACTOR_KJMOL,
            use_customized_ops=use_customized_ops,
        )
        self.pme = PME(
            alpha=ewald_alpha,
            max_hkl=max_hkl,
            rank=0,
            use_customized_ops=use_customized_ops,
        )
        self.nblist = NeighborList(
            n_atoms,
            exclusions=self.excluded_pairs,
            use_customized_ops=use_customized_ops,
            algorithm="nsquared",
        )
        self.max_npairs = int(n_atoms * 4 / 3 * 3.1416 * cutoff_nm**3 * 100 / 2 * 1.2)
        self.use_customized_ops = use_customized_ops

    def energy_components(self, coords: torch.Tensor, box: torch.Tensor) -> dict[str, torch.Tensor]:
        """TorchFF energy terms (scalar tensors), same grouping as OpenMM valence/nonbonded names."""
        pairs = self.nblist(coords, box, self.cutoff_nm, self.max_npairs, padding=True)
        pairs = pairs[0] if self.use_customized_ops else pairs
        
        e_bond = self.bond(coords, self.bonds, self.b0, self.kb)
        e_ang = self.angle(coords, self.angles, self.th0, self.kth)
        e_lj = self.lj(
            coords, pairs, box, self.sigma, self.epsilon, self.cutoff_nm, atom_types=self.atom_types
        )
        e_coul = self.coul(
            coords, box, pairs, self.charges, pairs_excl=self.coulomb_excl_pairs
        )
        e_pme_raw = self.pme(coords, box, self.charges)
        e_pme = e_pme_raw * COULOMB_PREFACTOR_KJMOL
        return {
            "HarmonicBondForce": e_bond,
            "HarmonicAngleForce": e_ang,
            "LennardJones": e_lj,
            "Coulomb_real": e_coul,
            "Coulomb_reciprocal": e_pme,
            "Coulomb_reciprocal_raw": e_pme_raw,
            "Coulomb": e_coul + e_pme,
        }

    def forward(self, coords: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        c = self.energy_components(coords, box)
        return (
            c["HarmonicBondForce"]
            + c["HarmonicAngleForce"]
            + c["LennardJones"]
            + c["Coulomb_real"]
            + c["Coulomb_reciprocal"]
        )
