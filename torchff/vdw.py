"""Van der Waals (vdW) pair energies: Lennard-Jones 12-6 and AMOEBA buffered 14-7."""

from typing import Literal, Optional
import torch
import torch.nn as nn
import torchff_vdw
from .pbc import PBC


@torch._dynamo.disable
def compute_vdw_14_7_energy(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    cutoff: float,
    atom_types: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute AMOEBA buffered 14-7 vdW pair energies via custom CUDA/C++ ops.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    pairs : torch.Tensor
        Shape (P, 2), integer indices (i, j) of interacting pairs.
    box : torch.Tensor
        Shape (3, 3) or broadcastable, periodic box (same convention as :mod:`torchff.pbc`).
    sigma : torch.Tensor
        Per-pair or type-pair :math:`\\sigma` (see ``atom_types``).
    epsilon : torch.Tensor
        Per-pair or type-pair :math:`\\epsilon` (see ``atom_types``).
    cutoff : float
        Distance cutoff; interactions beyond cutoff are excluded by the kernel.
    atom_types : torch.Tensor, optional
        If provided, used by the backend for type-based indexing together with ``sigma`` and ``epsilon``.

    Returns
    -------
    torch.Tensor
        Scalar total vdW energy for the buffered 14-7 potential.
    """
    return torch.ops.torchff.compute_vdw_14_7_energy(coords, pairs, box, sigma, epsilon, cutoff, atom_types)


@torch._dynamo.disable
def compute_lennard_jones_energy(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    cutoff: float,
    atom_types: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Lennard-Jones 12-6 vdW pair energies via custom CUDA/C++ ops.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    pairs : torch.Tensor
        Shape (P, 2), integer indices (i, j) of interacting pairs.
    box : torch.Tensor
        Shape (3, 3) or broadcastable, periodic box (same convention as :mod:`torchff.pbc`).
    sigma : torch.Tensor
        Per-pair or type-pair :math:`\\sigma` (see ``atom_types``).
    epsilon : torch.Tensor
        Per-pair or type-pair :math:`\\epsilon` (see ``atom_types``).
    cutoff : float
        Distance cutoff; interactions beyond cutoff are excluded by the kernel.
    atom_types : torch.Tensor, optional
        If provided, used by the backend for type-based indexing together with ``sigma`` and ``epsilon``.

    Returns
    -------
    torch.Tensor
        Scalar total Lennard-Jones energy.
    """
    return torch.ops.torchff.compute_lennard_jones_energy(coords, pairs, box, sigma, epsilon, cutoff, atom_types)


def compute_lennard_jones_energy_ref(r_ij, sigma_ij, epsilon_ij, sum=True):
    """
    Reference Lennard-Jones 12-6 pair energy in PyTorch.

    Per pair:
    :math:`E_{ij} = 4 \\epsilon_{ij} \\left[ (\\sigma_{ij}/r_{ij})^{12} - (\\sigma_{ij}/r_{ij})^6 \\right]`.

    Parameters
    ----------
    r_ij : torch.Tensor
        Pair distances, shape (P,) or broadcastable.
    sigma_ij : torch.Tensor
        :math:`\\sigma` for each pair, same shape as ``r_ij`` (after broadcast).
    epsilon_ij : torch.Tensor
        :math:`\\epsilon` for each pair, same shape as ``r_ij`` (after broadcast).
    sum : bool, optional
        If True (default), return the sum over pairs; otherwise return per-pair energies.

    Returns
    -------
    torch.Tensor
        Scalar total energy if ``sum`` is True, else shape (P,) per-pair energies.
    """
    tmp = (sigma_ij / r_ij) ** 6
    ene_ij = 4 * epsilon_ij * tmp * (tmp - 1)
    return torch.sum(ene_ij) if sum else ene_ij


def compute_vdw_14_7_energy_ref(r_ij, sigma_ij, epsilon_ij, sum=True):
    """
    Reference AMOEBA buffered 14-7 vdW pair energy in PyTorch.

    With :math:`\\rho = r_{ij} / \\sigma_{ij}`,
    :math:`E_{ij} = \\epsilon_{ij} \\left( \\frac{1.07}{\\rho + 0.07} \\right)^7 \\left( \\frac{1.12}{\\rho^7 + 0.12} - 2 \\right)`.

    Parameters
    ----------
    r_ij : torch.Tensor
        Pair distances, shape (P,) or broadcastable.
    sigma_ij : torch.Tensor
        :math:`\\sigma` for each pair, same shape as ``r_ij`` (after broadcast).
    epsilon_ij : torch.Tensor
        :math:`\\epsilon` for each pair, same shape as ``r_ij`` (after broadcast).
    sum : bool, optional
        If True (default), return the sum over pairs; otherwise return per-pair energies.

    Returns
    -------
    torch.Tensor
        Scalar total energy if ``sum`` is True, else shape (P,) per-pair energies.
    """
    rho = r_ij / sigma_ij
    ene_ij = epsilon_ij * (1.07 / (rho + 0.07)) ** 7 * (1.12 / (rho**7 + 0.12) - 2.0)
    return torch.sum(ene_ij) if sum else ene_ij


class Vdw(nn.Module):
    """
    Van der Waals pair energy module (Lennard-Jones 12-6 or AMOEBA buffered 14-7).

    Dispatches to :func:`compute_lennard_jones_energy` / :func:`compute_vdw_14_7_energy`
    when :attr:`use_customized_ops` is True; otherwise uses minimum-image displacements
    via :class:`torchff.pbc.PBC` and the reference formulas
    :func:`compute_lennard_jones_energy_ref` / :func:`compute_vdw_14_7_energy_ref`.
    """

    def __init__(
        self,
        function: Literal['LennardJones', 'AmoebaVdw147'] = 'LennardJones',
        cutoff: Optional[float] = None,
        use_customized_ops: bool = False,
        use_type_pairs: bool = False,
        sum_output: bool = True,
        cuda_graph_compat: bool = True,
    ):
        """
        Parameters
        ----------
        function : {'LennardJones', 'AmoebaVdw147'}, optional
            Potential form: standard LJ 12-6 or AMOEBA buffered 14-7.
        cutoff : float, optional
            Stored on the module; the active cutoff is the ``cutoff`` argument to :meth:`forward`.
        use_customized_ops : bool, optional
            If True, use custom CUDA/C++ kernels; otherwise use the PyTorch reference path.
        use_type_pairs : bool, optional
            If True, ``sigma`` and ``epsilon`` are indexed by ``atom_types`` for each pair
            (shape ``(n_types, n_types)``).
        sum_output : bool, optional
            If True (default), return a scalar sum over pairs. Must be True when
            ``use_customized_ops`` is True because the custom kernels only return total energy.
            When ``use_customized_ops`` is False, if False return per-pair energies of shape ``(P,)``.
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
        if self.use_customized_ops and not self.sum_output:
            raise ValueError(
                "sum_output must be True when use_customized_ops is True "
                "(custom vdW kernels only compute total energy, not per-pair terms)."
            )
        self.function = function
        assert self.function in ('LennardJones', 'AmoebaVdw147'), f'Invalid vdw function: {function}'
    
    def expand_type_pairs(self, sigma, epsilon, pairs, atom_types):
        if self.use_type_pairs:
            atypes_i, atypes_j = atom_types[pairs[:, 0]], atom_types[pairs[:, 1]]
            sigma_ij = sigma[atypes_i, atypes_j]
            epsilon_ij = epsilon[atypes_i, atypes_j]
            return sigma_ij, epsilon_ij
        else:
            return sigma, epsilon

    def forward(
        self,
        coords: torch.Tensor,
        pairs: torch.Tensor,
        box: torch.Tensor,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        cutoff: float,
        atom_types: torch.Tensor | None = None,
    ):
        """
        Compute vdW energy for the configured potential.

        Parameters
        ----------
        coords : torch.Tensor
            Shape (N, 3), atom coordinates.
        pairs : torch.Tensor
            Shape (P, 2), pair indices (i, j).
        box : torch.Tensor
            Periodic box, same convention as :class:`torchff.pbc.PBC`.
        sigma : torch.Tensor
            Per-pair ``(P,)`` or type table ``(T, T)`` when :attr:`use_type_pairs` is True.
        epsilon : torch.Tensor
            Same layout as ``sigma``.
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
            if self.function == 'LennardJones':
                return compute_lennard_jones_energy(coords, pairs, box, sigma, epsilon, cutoff, atom_types)
            else:
                return compute_vdw_14_7_energy(coords, pairs, box, sigma, epsilon, cutoff, atom_types)
        else:
            drVecs = self.pbc(coords[pairs[:, 1]] - coords[pairs[:, 0]], box)
            sigma_ij, epsilon_ij = self.expand_type_pairs(sigma, epsilon, pairs, atom_types)
            dr = torch.norm(drVecs, dim=1)
            if not self.cuda_graph_compat:
                dr = dr[dr <= cutoff]
            if self.function == 'LennardJones':
                ene_pairs = compute_lennard_jones_energy_ref(dr, sigma_ij, epsilon_ij, sum=False)
            else:
                ene_pairs = compute_vdw_14_7_energy_ref(dr, sigma_ij, epsilon_ij, sum=False)
            if self.cuda_graph_compat:
                ene_pairs = torch.where(dr <= cutoff, ene_pairs, 0.0)
            if self.sum_output:
                return torch.sum(ene_pairs)
            else:
                return ene_pairs
