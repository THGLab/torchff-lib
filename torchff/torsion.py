"""Torsion (dihedral) angles, harmonic torsion energy, and periodic torsion energies."""

import torch
import torch.nn as nn
import torchff_torsion


def compute_torsion_ref(coords: torch.Tensor, torsions: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of torsion (dihedral) angles using PyTorch ops.

    For each torsion (i, j, k, l), computes the dihedral angle between the
    planes formed by (i, j, k) and (j, k, l). The convention matches the CUDA
    implementation used by the custom operator.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    torsions : torch.Tensor
        Shape (M, 4), integer indices; each row is (i, j, k, l).

    Returns
    -------
    torch.Tensor
        Shape (M,), torsion angles in radians.
    """
    i = torsions[:, 0]
    j = torsions[:, 1]
    k = torsions[:, 2]
    l = torsions[:, 3]

    r_i = coords[i]
    r_j = coords[j]
    r_k = coords[k]
    r_l = coords[l]

    b1 = r_j - r_i
    b2 = r_k - r_j
    b3 = r_l - r_k

    n1 = torch.cross(b1, b2, dim=1)
    n2 = torch.cross(b2, b3, dim=1)

    n1_norm = torch.norm(n1, dim=1)
    n2_norm = torch.norm(n2, dim=1)
    b2_norm = torch.norm(b2, dim=1)

    cosval = torch.sum(n1 * n2, dim=1) / (n1_norm * n2_norm)
    cosval = torch.clamp(cosval, -0.999999999, 0.99999999)

    phi = torch.acos(cosval)
    sign = torch.sign(torch.sum(n1 * b3, dim=1))
    phi = torch.where(sign > 0, phi, -phi)
    return phi


@torch._dynamo.disable
def compute_torsion(coords: torch.Tensor, torsions: torch.Tensor) -> torch.Tensor:
    """
    Compute torsion (dihedral) angles via custom CUDA/CPU ops.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    torsions : torch.Tensor
        Shape (M, 4), integer indices; each row is (i, j, k, l).

    Returns
    -------
    torch.Tensor
        Shape (M,), torsion angles in radians.
    """
    return torch.ops.torchff.compute_torsion(coords, torsions)


class Torsion(nn.Module):
    """
    Torsion (dihedral) angle module returning raw angle values in radians.

    Dispatches to custom ops or a PyTorch reference implementation based on
    :attr:`use_customized_ops`.
    """

    def __init__(self, use_customized_ops: bool = False):
        """
        Parameters
        ----------
        use_customized_ops : bool, optional
            If True, use custom CUDA/CPU kernels; otherwise use reference implementation.
        """
        super().__init__()
        self.use_customized_ops = use_customized_ops

    def forward(self, coords: torch.Tensor, torsions: torch.Tensor) -> torch.Tensor:
        """
        Compute torsion (dihedral) angles.

        Parameters
        ----------
        coords : torch.Tensor
            Shape (N, 3), atom coordinates.
        torsions : torch.Tensor
            Shape (M, 4), torsion indices (i, j, k, l).

        Returns
        -------
        torch.Tensor
            Shape (M,), torsion angles in radians.
        """
        if self.use_customized_ops:
            return compute_torsion(coords, torsions)
        else:
            return compute_torsion_ref(coords, torsions)


@torch._dynamo.disable
def compute_periodic_torsion_energy(
    coords: torch.Tensor, 
    torsions: torch.Tensor, 
    fc: torch.Tensor, 
    periodicity: torch.Tensor,
    phase: torch.Tensor
) -> torch.Tensor :
    """Compute periodic torsion energies"""
    return torch.ops.torchff.compute_periodic_torsion_energy(coords, torsions, fc, periodicity, phase)


def compute_periodic_torsion_energy_ref(coords, torsions, fc, periodicity, phase):
    phi = compute_torsion_ref(coords, torsions)
    ene = fc * (1 + torch.cos(periodicity * phi - phase))
    return torch.sum(ene)


class PeriodicTorsion(nn.Module):
    def __init__(self, use_customized_ops: bool = False):
        super().__init__()
        self.use_customized_ops = use_customized_ops

    def forward(self, coords, torsions, fc, periodicity, phase):
        if self.use_customized_ops:
            return compute_periodic_torsion_energy(coords, torsions, fc, periodicity, phase)
        else:
            return compute_periodic_torsion_energy_ref(coords, torsions, fc, periodicity, phase)


def compute_harmonic_torsion_energy_ref(
    coords: torch.Tensor,
    torsions: torch.Tensor,
    k: torch.Tensor,
    torsion_ref: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reference harmonic torsion energy using PyTorch dihedral angles.

    Total energy is ``sum_i (k_i / 2) * (phi_i - phi0_i)^2`` when ``torsion_ref``
    (phi0) is given; if ``torsion_ref`` is None, uses ``sum_i (k_i / 2) * phi_i^2``.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    torsions : torch.Tensor
        Shape (M, 4), integer indices (i, j, k, l) per torsion.
    k : torch.Tensor
        Shape (M,), force constants.
    torsion_ref : torch.Tensor, optional
        Shape (M,), equilibrium torsion angles in radians. If None, the
        equilibrium angle is taken as zero for every term.

    Returns
    -------
    torch.Tensor
        Scalar total energy.
    """
    phi = compute_torsion_ref(coords, torsions)
    if torsion_ref is None:
        return torch.sum(0.5 * k * phi**2)
    return torch.sum(0.5 * k * (phi - torsion_ref) ** 2)


class HarmonicTorsion(nn.Module):
    """
    Harmonic torsion energy: sum_i (k_i/2) (phi_i - phi0_i)^2.

    Dihedral angles phi_i come from :class:`Torsion` (custom or reference ops).
    The energy reduction is pure PyTorch; there is no separate fused CUDA kernel.
    """

    def __init__(self, use_customized_ops: bool = False):
        """
        Parameters
        ----------
        use_customized_ops : bool, optional
            If True, dihedral angles use :func:`compute_torsion`; otherwise
            :func:`compute_torsion_ref`.
        """
        super().__init__()
        self.use_customized_ops = use_customized_ops

    def forward(
        self,
        coords: torch.Tensor,
        torsions: torch.Tensor,
        k: torch.Tensor,
        torsion_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute total harmonic torsion energy.

        Parameters
        ----------
        coords : torch.Tensor
            Shape (N, 3), atom coordinates.
        torsions : torch.Tensor
            Shape (M, 4), torsion indices (i, j, k, l).
        k : torch.Tensor
            Shape (M,), force constants.
        torsion_ref : torch.Tensor, optional
            Shape (M,), equilibrium angles in radians. If None, uses
            ``sum (k/2) phi^2``.

        Returns
        -------
        torch.Tensor
            Scalar total energy.
        """
        if self.use_customized_ops:
            phi = compute_torsion(coords, torsions)
        else:
            phi = compute_torsion_ref(coords, torsions)
        if torsion_ref is None:
            return torch.sum(0.5 * k * phi**2)
        return torch.sum(0.5 * k * (phi - torsion_ref) ** 2)


def compute_periodic_torsion_forces(
    coords: torch.Tensor, 
    torsions: torch.Tensor, 
    fc: torch.Tensor, 
    periodicity: torch.Tensor,
    phase: torch.Tensor,
    forces: torch.Tensor
) -> None:
    """
    Compute periodic torsion forces in-place, backward calculation does not supported, used for fast MD
    """
    return torch.ops.torchff.compute_periodic_torsion_forces(coords, torsions, fc, periodicity, phase, forces)