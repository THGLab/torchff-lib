"""Tang–Tonnies damped C6 dispersion: U = -C6 f_6(br) / r^6."""

from typing import Optional

import torch
import torch.nn as nn
import torchff_dispersion

from .pbc import PBC


def _tang_tonnies_f6(u: torch.Tensor) -> torch.Tensor:
    """Order-6 Tang–Tonnies damping f_6(u) = 1 - e^{-u} sum_{k=0}^6 u^k/k!."""
    # Horner form matching tang_tonnies.cuh
    return 1.0 - torch.exp(-u) * (
        1.0
        + u * (1.0 + u / 2.0 * (1.0 + u / 3.0 * (1.0 + u / 4.0 * (1.0 + u / 5.0 * (1.0 + u / 6.0)))))
    )


@torch._dynamo.disable
def compute_tang_tonnies_dispersion_energy(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    c6: torch.Tensor,
    b: torch.Tensor,
    cutoff: float,
    atom_types: torch.Tensor | None = None,
) -> torch.Tensor:
    """Total Tang–Tonnies C6 dispersion energy (custom CUDA)."""
    return torch.ops.torchff.compute_tang_tonnies_dispersion_energy(
        coords, pairs, box, c6, b, cutoff, atom_types
    )


def compute_tang_tonnies_dispersion_energy_ref(
    r_ij: torch.Tensor,
    c6_ij: torch.Tensor,
    b_ij: torch.Tensor,
    sum: bool = True,
) -> torch.Tensor:
    """
    Reference: U_ij = -C6_ij f_6(b_ij r_ij) / r_ij^6.

    Parameters
    ----------
    r_ij : torch.Tensor
        Pair distances, shape (P,) or broadcastable.
    c6_ij, b_ij : torch.Tensor
        Per-pair C6 and Tang–Tonnies inverse length b, same shape as ``r_ij`` after broadcast.
    sum : bool
        If True, return scalar sum over pairs; else per-pair energies (P,).
    """
    u = b_ij * r_ij
    f6 = _tang_tonnies_f6(u)
    ene_ij = -(c6_ij * f6) / (r_ij**6)
    return torch.sum(ene_ij) if sum else ene_ij


class Dispersion(nn.Module):
    """
    Tang–Tonnies C6 dispersion pair energy.

    Dispatches to :func:`compute_tang_tonnies_dispersion_energy` when
    :attr:`use_customized_ops` is True; otherwise PBC minimum-image distances
    and :func:`compute_tang_tonnies_dispersion_energy_ref`.
    """

    def __init__(
        self,
        cutoff: Optional[float] = None,
        use_customized_ops: bool = False,
        use_type_pairs: bool = False,
        sum_output: bool = False,
        cuda_graph_compat: bool = True,
    ) -> None:
        super().__init__()
        self.use_customized_ops = use_customized_ops
        self.use_type_pairs = use_type_pairs
        self.sum_output = sum_output
        self.cuda_graph_compat = cuda_graph_compat
        self.pbc = PBC()
        self.cutoff = cutoff
        if self.sum_output:
            assert not self.use_customized_ops

    def expand_type_pairs(
        self,
        c6: torch.Tensor,
        b: torch.Tensor,
        pairs: torch.Tensor,
        atom_types: torch.Tensor | None,
    ):
        if self.use_type_pairs:
            assert atom_types is not None
            ti, tj = atom_types[pairs[:, 0]], atom_types[pairs[:, 1]]
            c6_ij = c6[ti, tj]
            b_ij = b[ti, tj]
            return c6_ij, b_ij
        return c6, b

    def forward(
        self,
        coords: torch.Tensor,
        pairs: torch.Tensor,
        box: torch.Tensor,
        c6: torch.Tensor,
        b: torch.Tensor,
        cutoff: float,
        atom_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_customized_ops:
            return compute_tang_tonnies_dispersion_energy(
                coords, pairs, box, c6, b, cutoff, atom_types
            )
        dr_vecs = self.pbc(coords[pairs[:, 1]] - coords[pairs[:, 0]], box)
        c6_ij, b_ij = self.expand_type_pairs(c6, b, pairs, atom_types)
        dr = torch.norm(dr_vecs, dim=1)
        if not self.cuda_graph_compat:
            m = dr <= cutoff
            ene_pairs = compute_tang_tonnies_dispersion_energy_ref(
                dr[m], c6_ij[m], b_ij[m], sum=False
            )
        else:
            ene_pairs = compute_tang_tonnies_dispersion_energy_ref(
                dr, c6_ij, b_ij, sum=False
            )
            ene_pairs = torch.where(dr <= cutoff, ene_pairs, 0.0)
        if self.sum_output:
            return torch.sum(ene_pairs)
        return ene_pairs
