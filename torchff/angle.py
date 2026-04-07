"""Angle energy and force terms (harmonic and Amoeba-style) for force field calculations."""

import torch
import torch.nn as nn
import torchff_angle


def compute_angles(coords: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Compute angle (theta) at the central atom for each angle triple.

    For each angle (i, j, k), computes the angle at j between vectors
    (coords[i] - coords[j]) and (coords[k] - coords[j]).

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    angles : torch.Tensor
        Shape (M, 3), integer indices; each row is (i, j, k).

    Returns
    -------
    torch.Tensor
        Shape (M,), angle in radians for each triple.
    """
    v1 = coords[angles[:, 0]] - coords[angles[:, 1]]
    v2 = coords[angles[:, 2]] - coords[angles[:, 1]]
    dot_product = torch.sum(v1 * v2, dim=1)
    mag_v1 = torch.norm(v1, dim=1)
    mag_v2 = torch.norm(v2, dim=1)
    cos_theta = dot_product / (mag_v1 * mag_v2)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    return theta


@torch._dynamo.disable
def compute_harmonic_angle_energy(coords: torch.Tensor, angles: torch.Tensor, theta0: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Compute harmonic angle energies via custom CUDA/CPU ops.

    Energy per angle: :math:`E = (1/2) k (\\theta - \\theta_0)^2` with :math:`\\theta` in radians.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    angles : torch.Tensor
        Shape (M, 3), integer indices; each row is (i, j, k), j the central atom.
    theta0 : torch.Tensor
        Shape (M,), equilibrium angles in radians.
    k : torch.Tensor
        Shape (M,), force constants.

    Returns
    -------
    torch.Tensor
        Scalar total harmonic angle energy.
    """
    return torch.ops.torchff.compute_harmonic_angle_energy(coords, angles, theta0, k)


def compute_harmonic_angle_energy_ref(coords, angles, theta0, k):
    """
    Reference implementation of harmonic angle energy (PyTorch only).

    Uses :func:`compute_angles` then same formula as :func:`compute_harmonic_angle_energy`;
    used when custom ops are disabled.
    """
    theta = compute_angles(coords, angles)
    ene = (theta - theta0) ** 2 * k / 2
    return torch.sum(ene)


class HarmonicAngle(nn.Module):
    """
    Harmonic angle energy module: :math:`E = (1/2) k (\\theta - \\theta_0)^2`.

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

    def forward(self, coords, angles, theta0, k):
        """
        Compute total harmonic angle energy.

        Parameters
        ----------
        coords : torch.Tensor
            Shape (N, 3), atom coordinates.
        angles : torch.Tensor
            Shape (M, 3), angle indices (i, j, k).
        theta0 : torch.Tensor
            Shape (M,), equilibrium angles in radians.
        k : torch.Tensor
            Shape (M,), force constants.

        Returns
        -------
        torch.Tensor
            Scalar total energy.
        """
        if self.use_customized_ops:
            return compute_harmonic_angle_energy(coords, angles, theta0, k)
        else:
            return compute_harmonic_angle_energy_ref(coords, angles, theta0, k)


def compute_harmonic_angle_forces(
    coords: torch.Tensor,
    angles: torch.Tensor,
    theta0: torch.Tensor,
    k: torch.Tensor,
    forces: torch.Tensor
) -> torch.Tensor:
    """
    Compute harmonic angle forces in-place (no autograd).

    Adds angle force contributions into :attr:`forces`. Intended for fast MD;
    backward is not supported.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    angles : torch.Tensor
        Shape (M, 3), angle indices.
    theta0 : torch.Tensor
        Shape (M,), equilibrium angles in radians.
    k : torch.Tensor
        Shape (M,), force constants.
    forces : torch.Tensor
        Shape (N, 3), modified in-place with angle forces.

    Returns
    -------
    torch.Tensor
        Reference to :attr:`forces`.
    """
    return torch.ops.torchff.compute_harmonic_angle_forces(coords, angles, theta0, k, forces)


@torch._dynamo.disable
def compute_amoeba_angle_energy(
    coords: torch.Tensor,
    angles: torch.Tensor,
    theta0: torch.Tensor,
    k: torch.Tensor,
    cubic: float = -0.014,
    quartic: float = 5.6e-5,
    pentic: float = -7.0e-7,
    sextic: float = 2.2e-8,
) -> torch.Tensor:
    """
    Compute Amoeba-style angle energies via custom ops.

    :math:`E = k (\\theta - \\theta_0)^2 [1 + c_3 \\Delta\\theta + c_4 \\Delta\\theta^2 + c_5 \\Delta\\theta^3 + c_6 \\Delta\\theta^4]`
    with :math:`\\Delta\\theta = \\theta - \\theta_0` in radians.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    angles : torch.Tensor
        Shape (M, 3), angle indices (i, j, k).
    theta0 : torch.Tensor
        Shape (M,), equilibrium angles in radians.
    k : torch.Tensor
        Shape (M,), force constants.
    cubic, quartic, pentic, sextic : float, optional
        Polynomial coefficients (defaults: -0.014, 5.6e-5, -7.0e-7, 2.2e-8).

    Returns
    -------
    torch.Tensor
        Scalar total Amoeba angle energy.
    """
    return torch.ops.torchff.compute_amoeba_angle_energy(coords, angles, theta0, k, cubic, quartic, pentic, sextic)


def compute_amoeba_angle_energy_ref(
    coords, angles, theta0, k,
    cubic: float = -0.014,
    quartic: float = 5.6e-5,
    pentic: float = -7.0e-7,
    sextic: float = 2.2e-8,
):
    """
    Reference implementation of Amoeba angle energy (PyTorch only).

    Uses :func:`compute_angles`; same formula as :func:`compute_amoeba_angle_energy`;
    used when custom ops are disabled.
    """
    theta = compute_angles(coords, angles)
    dtheta = theta - theta0
    poly = 1.0 + cubic * dtheta + quartic * dtheta**2 + pentic * dtheta**3 + sextic * dtheta**4
    ene = k * dtheta**2 * poly
    return torch.sum(ene)


class AmoebaAngle(nn.Module):
    """
    Amoeba-style angle energy with cubic through sextic correction terms.

    Dispatches to custom ops or reference based on :attr:`use_customized_ops`.
    See :class:`HarmonicAngle` for constructor and dispatch behavior.
    """

    def __init__(self, use_customized_ops: bool = False):
        """Same as :class:`HarmonicAngle.__init__`."""
        super().__init__()
        self.use_customized_ops = use_customized_ops

    def forward(
        self, coords, angles, theta0, k,
        cubic: float = -0.014,
        quartic: float = 5.6e-5,
        pentic: float = -7.0e-7,
        sextic: float = 2.2e-8,
    ):
        """
        Compute total Amoeba angle energy.

        Parameters
        ----------
        coords, angles, theta0, k
            Same as :meth:`HarmonicAngle.forward`.
        cubic, quartic, pentic, sextic : float, optional
            Polynomial coefficients (defaults: -0.014, 5.6e-5, -7.0e-7, 2.2e-8).

        Returns
        -------
        torch.Tensor
            Scalar total energy.
        """
        if self.use_customized_ops:
            return compute_amoeba_angle_energy(coords, angles, theta0, k, cubic, quartic, pentic, sextic)
        else:
            return compute_amoeba_angle_energy_ref(coords, angles, theta0, k, cubic, quartic, pentic, sextic)