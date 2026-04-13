"""Bond energy and force terms (harmonic and Amoeba-style) for force field calculations."""

import torch
import torch.nn as nn
import torchff_bond


@torch._dynamo.disable
def compute_harmonic_bond_energy(coords: torch.Tensor, bonds: torch.Tensor, b0: torch.Tensor, k: torch.Tensor) -> torch.Tensor :
    """
    Compute harmonic bond energies via custom CUDA/CPU ops.

    Energy per bond: :math:`E = (1/2) k (r - r_0)^2` with :math:`r` the bond length.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    bonds : torch.Tensor
        Shape (M, 2), integer indices; each row is (i, j).
    b0 : torch.Tensor
        Shape (M,), equilibrium bond lengths.
    k : torch.Tensor
        Shape (M,), force constants.

    Returns
    -------
    torch.Tensor
        Scalar total harmonic bond energy.
    """
    return torch.ops.torchff.compute_harmonic_bond_energy(coords, bonds, b0, k)


def compute_harmonic_bond_energy_ref(coords, bonds, b0, k):
    """
    Reference implementation of harmonic bond energy (PyTorch only).

    Same formula as :func:`compute_harmonic_bond_energy`; used when custom ops are disabled.
    """
    r = torch.norm(coords[bonds[:, 0]] - coords[bonds[:, 1]], dim=1)
    ene = (r - b0) ** 2 * k / 2
    return torch.sum(ene)


class HarmonicBond(nn.Module):
    """
    Harmonic bond energy module: :math:`E = (1/2) k (r - r_0)^2`.

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

    def forward(self, coords, bonds, b0, k):
        """
        Compute total harmonic bond energy.

        Parameters
        ----------
        coords : torch.Tensor
            Shape (N, 3), atom coordinates.
        bonds : torch.Tensor
            Shape (M, 2), bond indices.
        b0 : torch.Tensor
            Shape (M,), equilibrium lengths.
        k : torch.Tensor
            Shape (M,), force constants.

        Returns
        -------
        torch.Tensor
            Scalar total energy.
        """
        if self.use_customized_ops:
            return compute_harmonic_bond_energy(coords, bonds, b0, k)
        else:
            return compute_harmonic_bond_energy_ref(coords, bonds, b0, k)


def compute_harmonic_bond_forces(
    coords: torch.Tensor, bonds: torch.Tensor, b0: torch.Tensor, k: torch.Tensor,
    forces: torch.Tensor
):
    """
    Compute harmonic bond forces in-place (no autograd).

    Adds bond force contributions into :attr:`forces`. Intended for fast MD;
    backward is not supported.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    bonds : torch.Tensor
        Shape (M, 2), bond indices.
    b0 : torch.Tensor
        Shape (M,), equilibrium lengths.
    k : torch.Tensor
        Shape (M,), force constants.
    forces : torch.Tensor
        Shape (N, 3), modified in-place with bond forces.
    """
    return torch.ops.torchff.compute_harmonic_bond_forces(coords, bonds, b0, k, forces)


@torch._dynamo.disable
def compute_amoeba_bond_energy(
    coords: torch.Tensor,
    bonds: torch.Tensor,
    b0: torch.Tensor,
    k: torch.Tensor,
    cubic: float = -2.55,
    quartic: float = 3.793125,
) -> torch.Tensor:
    """
    Compute Amoeba-style bond energies via custom ops.

    :math:`E = k (b - b_0)^2 [1 + c_3 (b - b_0) + c_4 (b - b_0)^2]` with
    :math:`b` the bond length and :math:`c_3, c_4` the cubic and quartic coefficients.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    bonds : torch.Tensor
        Shape (M, 2), bond indices.
    b0 : torch.Tensor
        Shape (M,), equilibrium lengths.
    k : torch.Tensor
        Shape (M,), force constants.
    cubic : float, optional
        Cubic coefficient in the polynomial (default -2.55).
    quartic : float, optional
        Quartic coefficient (default 3.793125).

    Returns
    -------
    torch.Tensor
        Scalar total Amoeba bond energy.
    """
    return torch.ops.torchff.compute_amoeba_bond_energy(coords, bonds, b0, k, cubic, quartic)


def compute_amoeba_bond_energy_ref(
    coords, bonds, b0, k, cubic: float = -2.55, quartic: float = 3.793125
):
    """
    Reference implementation of Amoeba bond energy (PyTorch only).

    Same formula as :func:`compute_amoeba_bond_energy`; used when custom ops are disabled.
    """
    r = torch.norm(coords[bonds[:, 0]] - coords[bonds[:, 1]], dim=1)
    db = r - b0
    poly = 1.0 + cubic * db + quartic * db * db
    ene = k * db * db * poly
    return torch.sum(ene)


class AmoebaBond(nn.Module):
    """
    Amoeba-style bond energy with cubic and quartic correction terms.

    Dispatches to custom ops or reference based on :attr:`use_customized_ops`.
    See :class:`HarmonicBond` for constructor and dispatch behavior.
    """

    def __init__(self, use_customized_ops: bool = False):
        """Same as :class:`HarmonicBond.__init__`."""
        super().__init__()
        self.use_customized_ops = use_customized_ops

    def forward(self, coords, bonds, b0, k, cubic: float = -2.55, quartic: float = 3.793125):
        """
        Compute total Amoeba bond energy.

        Parameters
        ----------
        coords, bonds, b0, k
            Same as :meth:`HarmonicBond.forward`.
        cubic : float, optional
            Cubic coefficient (default -2.55).
        quartic : float, optional
            Quartic coefficient (default 3.793125).

        Returns
        -------
        torch.Tensor
            Scalar total energy.
        """
        if self.use_customized_ops:
            return compute_amoeba_bond_energy(coords, bonds, b0, k, cubic, quartic)
        else:
            return compute_amoeba_bond_energy_ref(coords, bonds, b0, k, cubic, quartic)


@torch._dynamo.disable
def compute_morse_bond_energy(
    coords: torch.Tensor,
    bonds: torch.Tensor,
    b0: torch.Tensor,
    k: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Morse bond energies via custom CUDA/CPU ops.

    Energy per bond: :math:`E = D [1 - e^{-\\beta (r - r_0)}]^2` with
    :math:`\\beta = \\sqrt{k/(2D)}`, :math:`r` the bond length, :math:`r_0` the
    equilibrium length, :math:`k` the stiffness, and :math:`D` the well depth.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    bonds : torch.Tensor
        Shape (M, 2), integer indices; each row is (i, j).
    b0 : torch.Tensor
        Shape (M,), equilibrium bond lengths :math:`r_0`.
    k : torch.Tensor
        Shape (M,), stiffness parameters :math:`k`.
    d : torch.Tensor
        Shape (M,), well depth :math:`D` (must be positive).

    Returns
    -------
    torch.Tensor
        Scalar total Morse bond energy.
    """
    return torch.ops.torchff.compute_morse_bond_energy(coords, bonds, b0, k, d)


def compute_morse_bond_energy_ref(coords, bonds, b0, k, d):
    """
    Reference implementation of Morse bond energy (PyTorch only).

    Same formula as :func:`compute_morse_bond_energy`; used when custom ops are disabled.
    """
    r = torch.norm(coords[bonds[:, 0]] - coords[bonds[:, 1]], dim=1)
    beta = torch.sqrt(k / (2 * d))
    z = 1.0 - torch.exp(-beta * (r - b0))
    ene = d * z * z
    return torch.sum(ene)


class MorseBond(nn.Module):
    """
    Morse bond energy module.

    Dispatches to custom ops or a PyTorch reference implementation based on
    :attr:`use_customized_ops`.
    """

    def __init__(self, use_customized_ops: bool = False):
        super().__init__()
        self.use_customized_ops = use_customized_ops

    def forward(self, coords, bonds, b0, k, d):
        """
        Compute total Morse bond energy.

        Parameters
        ----------
        coords : torch.Tensor
            Shape (N, 3), atom coordinates.
        bonds : torch.Tensor
            Shape (M, 2), bond indices.
        b0 : torch.Tensor
            Shape (M,), equilibrium lengths.
        k : torch.Tensor
            Shape (M,), stiffness parameters.
        d : torch.Tensor
            Shape (M,), well depths.

        Returns
        -------
        torch.Tensor
            Scalar total energy.
        """
        if self.use_customized_ops:
            return compute_morse_bond_energy(coords, bonds, b0, k, d)
        return compute_morse_bond_energy_ref(coords, bonds, b0, k, d)


def compute_morse_bond_forces(
    coords: torch.Tensor,
    bonds: torch.Tensor,
    b0: torch.Tensor,
    k: torch.Tensor,
    d: torch.Tensor,
    forces: torch.Tensor,
):
    """
    Compute Morse bond forces in-place (no autograd).

    Adds bond force contributions into ``forces``.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    bonds : torch.Tensor
        Shape (M, 2), bond indices.
    b0 : torch.Tensor
        Shape (M,), equilibrium lengths.
    k : torch.Tensor
        Shape (M,), stiffness parameters.
    d : torch.Tensor
        Shape (M,), well depths.
    forces : torch.Tensor
        Shape (N, 3), modified in-place with bond forces.
    """
    return torch.ops.torchff.compute_morse_bond_forces(coords, bonds, b0, k, d, forces)