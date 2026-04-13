"""Slater-type pair energies: A * P(B r) * exp(-B r) with P = (B r)^2 / 3 + B r + 1."""

from typing import Optional

import torch
import torch.nn as nn
import torchff_slater

from .pbc import PBC


@torch._dynamo.disable
def compute_slater_energy(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    cutoff: float,
    atom_types: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Slater pair energies via custom CUDA/C++ ops.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    pairs : torch.Tensor
        Shape (P, 2), integer indices (i, j) of interacting pairs.
    box : torch.Tensor
        Shape (3, 3) or broadcastable, periodic box (same convention as :mod:`torchff.pbc`).
    A : torch.Tensor
        Per-pair or type-pair amplitude :math:`A_{ij}` (see ``atom_types``).
    B : torch.Tensor
        Per-pair or type-pair inverse length :math:`B_{ij}` (see ``atom_types``).
    cutoff : float
        Distance cutoff; interactions beyond cutoff are excluded by the kernel.
    atom_types : torch.Tensor, optional
        If provided, used by the backend for type-based indexing together with ``A`` and ``B``.

    Returns
    -------
    torch.Tensor
        Scalar total Slater energy.
    """
    return torch.ops.torchff.compute_slater_energy(
        coords, pairs, box, A, B, cutoff, atom_types
    )


def compute_slater_energy_ref(
    r_ij: torch.Tensor, A_ij: torch.Tensor, B_ij: torch.Tensor, sum: bool = True
):
    """
    Reference Slater pair energy in PyTorch.

    With :math:`x = B_{ij} r_{ij}`, :math:`P = x^2/3 + x + 1`,
    :math:`E_{ij} = A_{ij} P \\exp(-x)`.

    Parameters
    ----------
    r_ij : torch.Tensor
        Pair distances, shape (P,) or broadcastable.
    A_ij : torch.Tensor
        :math:`A_{ij}` for each pair, same shape as ``r_ij`` (after broadcast).
    B_ij : torch.Tensor
        :math:`B_{ij}` for each pair, same shape as ``r_ij`` (after broadcast).
    sum : bool, optional
        If True (default), return the sum over pairs; otherwise return per-pair energies.

    Returns
    -------
    torch.Tensor
        Scalar total energy if ``sum`` is True, else shape (P,) per-pair energies.
    """
    x = B_ij * r_ij
    P = x * x / 3.0 + x + 1.0
    ene_ij = A_ij * P * torch.exp(-x)
    return torch.sum(ene_ij) if sum else ene_ij


class Slater(nn.Module):
    """
    Slater pair energy module.

    Dispatches to :func:`compute_slater_energy` when :attr:`use_customized_ops` is True;
    otherwise uses minimum-image displacements via :class:`torchff.pbc.PBC` and
    :func:`compute_slater_energy_ref`.
    """

    def __init__(
        self,
        cutoff: Optional[float] = None,
        use_customized_ops: bool = False,
        use_type_pairs: bool = False,
        sum_output: bool = False,
        cuda_graph_compat: bool = True,
    ):
        """
        Parameters
        ----------
        cutoff : float, optional
            Stored on the module; the active cutoff is the ``cutoff`` argument to :meth:`forward`.
        use_customized_ops : bool, optional
            If True, use custom CUDA/C++ kernels; otherwise use the PyTorch reference path.
        use_type_pairs : bool, optional
            If True, ``A`` and ``B`` are indexed by ``atom_types`` for each pair
            (shape ``(n_types, n_types)``).
        sum_output : bool, optional
            If True, return a scalar sum over pairs. Requires ``use_customized_ops`` False.
        cuda_graph_compat : bool, optional
            If True (default), apply the cutoff with :func:`torch.where` so tensor shapes are
            stable; if False, distances are filtered with boolean indexing before the energy expression.
        """
        super().__init__()
        self.use_customized_ops = use_customized_ops
        self.use_type_pairs = use_type_pairs
        self.sum_output = sum_output
        self.cuda_graph_compat = cuda_graph_compat
        self.pbc = PBC()
        self.cutoff = cutoff
        if self.sum_output:
            assert self.use_customized_ops is False

    def expand_type_pairs(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        pairs: torch.Tensor,
        atom_types: torch.Tensor,
    ):
        if self.use_type_pairs:
            atypes_i, atypes_j = atom_types[pairs[:, 0]], atom_types[pairs[:, 1]]
            A_ij = A[atypes_i, atypes_j]
            B_ij = B[atypes_i, atypes_j]
            return A_ij, B_ij
        return A, B

    def forward(
        self,
        coords: torch.Tensor,
        pairs: torch.Tensor,
        box: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        cutoff: float,
        atom_types: torch.Tensor | None = None,
    ):
        """
        Compute Slater energy.

        Parameters
        ----------
        coords : torch.Tensor
            Shape (N, 3), atom coordinates.
        pairs : torch.Tensor
            Shape (P, 2), pair indices (i, j).
        box : torch.Tensor
            Periodic box, same convention as :class:`torchff.pbc.PBC`.
        A : torch.Tensor
            Per-pair ``(P,)`` or type table ``(T, T)`` when :attr:`use_type_pairs` is True.
        B : torch.Tensor
            Same layout as ``A``.
        cutoff : float
            Pair distance cutoff.
        atom_types : torch.Tensor, optional
            Shape (N,), integer atom types; required when :attr:`use_type_pairs` is True.

        Returns
        -------
        torch.Tensor
            If :attr:`use_customized_ops` is True, scalar total energy from the custom op.
            Otherwise per-pair energies of shape (P,), or a scalar if :attr:`sum_output` is True.
        """
        if self.use_customized_ops:
            return compute_slater_energy(
                coords, pairs, box, A, B, cutoff, atom_types
            )
        dr_vecs = self.pbc(coords[pairs[:, 1]] - coords[pairs[:, 0]], box)
        A_ij, B_ij = self.expand_type_pairs(A, B, pairs, atom_types)
        dr = torch.norm(dr_vecs, dim=1)
        if not self.cuda_graph_compat:
            dr = dr[dr <= cutoff]
        ene_pairs = compute_slater_energy_ref(dr, A_ij, B_ij, sum=False)
        if self.cuda_graph_compat:
            ene_pairs = torch.where(dr <= cutoff, ene_pairs, 0.0)
        if self.sum_output:
            return torch.sum(ene_pairs)
        return ene_pairs
