import torch
import torch.nn as nn

import torchff_nb  # noqa: F401 - ensure CUDA extension is loaded

from .coulomb import compute_coulomb_energy_ref
from .vdw import compute_lennard_jones_energy_ref
from .pbc import PBC


@torch._dynamo.disable
def compute_nonbonded_energy_from_atom_pairs(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    charges: torch.Tensor,
    coul_constant,
    cutoff: float,
    do_shift: bool = True,
) -> torch.Tensor:
    """Compute fused Coulomb + Lennard-Jones nonbonded energies using custom CUDA/C++ ops."""
    # CUDA kernel expects int64 index pairs
    if pairs.dtype != torch.int64:
        pairs = pairs.to(torch.int64)
    return torch.ops.torchff.compute_nonbonded_energy_from_atom_pairs(
        coords,
        pairs,
        box,
        sigma,
        epsilon,
        charges,
        coul_constant,
        cutoff,
        do_shift,
    )


def compute_nonbonded_energy_from_atom_pairs_ref(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    charges: torch.Tensor,
    coul_constant: float,
    cutoff: float,
    do_shift: bool = True,
) -> torch.Tensor:
    """Reference fused Coulomb + Lennard-Jones implementation using native PyTorch ops."""
    ene_coul = compute_coulomb_energy_ref(
        coords,
        pairs,
        box,
        charges,
        coul_constant,
        cutoff,
        do_shift,
    )
    ene_lj = compute_lennard_jones_energy_ref(
        coords,
        pairs,
        box,
        sigma,
        epsilon,
        cutoff,
    )
    return ene_coul + ene_lj


def compute_nonbonded_forces_from_atom_pairs(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    charges: torch.Tensor,
    coul_constant: float,
    cutoff: float,
    forces: torch.Tensor,
) -> torch.Tensor:
    """Compute fused Coulomb + Lennard-Jones nonbonded forces in-place using custom CUDA/C++ ops."""
    if pairs.dtype != torch.int64:
        pairs = pairs.to(torch.int64)
    return torch.ops.torchff.compute_nonbonded_forces_from_atom_pairs(
        coords,
        pairs,
        box,
        sigma,
        epsilon,
        charges,
        coul_constant,
        cutoff,
        forces,
    )


class Nonbonded(nn.Module):
    """Fused fixed-charge nonbonded (Coulomb + Lennard-Jones) interaction."""

    def __init__(self, use_customized_ops: bool = False):
        super().__init__()
        self.use_customized_ops = use_customized_ops

    def forward(
        self,
        coords: torch.Tensor,
        pairs: torch.Tensor,
        box: torch.Tensor,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        charges: torch.Tensor,
        coul_constant: float,
        cutoff: float,
        do_shift: bool = True,
    ) -> torch.Tensor:
        if self.use_customized_ops:
            return compute_nonbonded_energy_from_atom_pairs(
                coords,
                pairs,
                box,
                sigma,
                epsilon,
                charges,
                coul_constant,
                cutoff,
                do_shift,
            )
        else:
            return compute_nonbonded_energy_from_atom_pairs_ref(
                coords,
                pairs,
                box,
                sigma,
                epsilon,
                charges,
                coul_constant,
                cutoff,
                do_shift,
            )

