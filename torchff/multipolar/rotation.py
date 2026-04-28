from __future__ import annotations

import math
from enum import IntEnum
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import torchff_multipoles  # noqa: F401 — registers ``torch.ops.torchff.compute_rotation_matrices``


class AxisTypes(IntEnum):
    ZThenX = 0
    Bisector = 1
    ZBisect = 2
    ThreeFold = 3
    ZOnly = 4
    NoAxisType = 5
    LastAxisTypeIndex = 6


def normVec(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.norm(vec, dim=1, keepdim=True)


def _compute_rotation_matrices_python(
    positions: torch.Tensor,
    z_atoms: torch.Tensor,
    x_atoms: torch.Tensor,
    y_atoms: torch.Tensor,
    axis_types: torch.Tensor,
) -> torch.Tensor:
    """Python reference for local-to-global rotation matrices (see :class:`MultipolarRotation`)."""
    z_vec = normVec(positions[z_atoms] - positions)
    x_vec = torch.zeros_like(z_vec)
    y_vec = torch.zeros_like(z_vec)

    filter_z_only = torch.logical_or(
        axis_types == AxisTypes.ZOnly.value,
        axis_types == AxisTypes.NoAxisType.value,
    )
    x_vec_not_z_only = positions[x_atoms][~filter_z_only] - positions[~filter_z_only]
    x_vec_new = x_vec.clone()
    x_vec_new[~filter_z_only] = x_vec[~filter_z_only] + normVec(x_vec_not_z_only)
    x_vec_new[filter_z_only, 0] = x_vec[filter_z_only, 0] + (1.0 - z_vec[filter_z_only, 0])
    x_vec = x_vec_new
    x_vec[filter_z_only, 1] = x_vec[filter_z_only, 1] + z_vec[filter_z_only, 0]

    filter_bisector = axis_types == AxisTypes.Bisector.value
    if torch.any(filter_bisector):
        z_vec = z_vec.clone()
        z_vec[filter_bisector] = z_vec[filter_bisector] + x_vec[filter_bisector]
        z_vec = normVec(z_vec)

    filter_z_bisect = axis_types == AxisTypes.ZBisect.value
    if torch.any(filter_z_bisect):
        y_vec_zb = positions[y_atoms][filter_z_bisect] - positions[filter_z_bisect]
        y_vec_zb = normVec(y_vec_zb)
        x_vec_zb = normVec(x_vec[filter_z_bisect] + y_vec_zb)
        x_vec = x_vec.clone()
        x_vec[filter_z_bisect] = x_vec_zb

    filter_three_fold = axis_types == AxisTypes.ThreeFold.value
    if torch.any(filter_three_fold):
        y_vec_tf = positions[y_atoms][filter_three_fold] - positions[filter_three_fold]
        y_vec_tf = normVec(y_vec_tf)
        x_vec_tf = x_vec[filter_three_fold]
        z_vec_tf = z_vec[filter_three_fold]
        z_vec = z_vec.clone()
        z_vec[filter_three_fold] = normVec(z_vec_tf + x_vec_tf + y_vec_tf)

    x_vec = normVec(x_vec - z_vec * torch.sum(z_vec * x_vec, dim=1, keepdim=True))
    y_vec = torch.linalg.cross(z_vec, x_vec)

    filter_no_axis = axis_types == AxisTypes.NoAxisType.value
    if torch.any(filter_no_axis):
        fa = filter_no_axis.view(-1, 1)
        eye_z = torch.tensor(
            [0.0, 0.0, 1.0], dtype=z_vec.dtype, device=z_vec.device
        )
        eye_x = torch.tensor(
            [1.0, 0.0, 0.0], dtype=x_vec.dtype, device=x_vec.device
        )
        eye_y = torch.tensor(
            [0.0, 1.0, 0.0], dtype=y_vec.dtype, device=y_vec.device
        )
        z_vec = torch.where(fa, eye_z, z_vec)
        x_vec = torch.where(fa, eye_x, x_vec)
        y_vec = torch.where(fa, eye_y, y_vec)

    rot_matrix = torch.hstack((x_vec, y_vec, z_vec)).reshape(-1, 3, 3)
    return rot_matrix


@torch._dynamo.disable
def _compute_rotation_matrices_torchff(
    coords: torch.Tensor,
    z_atoms: torch.Tensor,
    x_atoms: torch.Tensor,
    y_atoms: torch.Tensor,
    axis_types: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA implementation of local-to-global rotation matrices (custom op).

    Wrapped with :func:`torch._dynamo.disable` so ``torch.compile`` on calling modules does
    not trace through the dispatcher.
    """
    return torch.ops.torchff.compute_rotation_matrices(
        coords, z_atoms, x_atoms, y_atoms, axis_types
    )


class MultipolarRotation(nn.Module):
    """
    Multipole local frame: build rotation matrices and rotate dipole / quadrupole tensors.

    Vectorized local-to-global rotation matrices for multipole sites use the same batched
    construction as the historical TorchFF Python path: differences
    ``positions[neighbor] - positions[site]`` with **no** periodic minimum-image
    wrapping. Intramolecular (or otherwise local) geometry should already lie in one
    periodic image so that these raw vectors match the intended local frame.

    Parameters
    ----------
    use_customized_ops : bool
        If ``True``, use :func:`_compute_rotation_matrices_torchff` (CUDA custom op); if ``False``,
        use :func:`_compute_rotation_matrices_python` (Python reference). Dipole and
        quadrupole rotation use :func:`rotateDipoles` and :func:`rotateQuadrupoles` in
        either case.

    Notes
    -----
    Rotation matrices have shape ``(N, 3, 3)``; rows are local X, Y, Z in global
    coordinates (``hstack`` of ``xVec``, ``yVec``, ``zVec``). Inputs ``z_atoms``,
    ``x_atoms``, ``y_atoms``, and ``axis_types`` have shape ``(N,)`` (see :class:`AxisTypes`).
    """

    def __init__(self, use_customized_ops: bool = False) -> None:
        super().__init__()
        self.use_customized_ops = use_customized_ops

    @classmethod
    def compute_matrices(
        cls,
        coords: torch.Tensor,
        z_atoms: torch.Tensor,
        x_atoms: torch.Tensor,
        y_atoms: torch.Tensor,
        axis_types: torch.Tensor,
        *,
        use_customized_ops: bool = False,
    ) -> torch.Tensor:
        """Shape ``(N, 3, 3)`` rotation matrices (rows = local X, Y, Z in global coordinates)."""
        if use_customized_ops:
            return _compute_rotation_matrices_torchff(
                coords, z_atoms, x_atoms, y_atoms, axis_types
            )
        return _compute_rotation_matrices_python(
            coords, z_atoms, x_atoms, y_atoms, axis_types
        )

    @classmethod
    def rotate_dipoles(
        cls, matrices: torch.Tensor, dipoles: torch.Tensor
    ) -> torch.Tensor:
        """Rotate dipoles of shape ``(N, 3)``; returns ``(N, 1, 3)`` (same as :func:`rotateDipoles`)."""
        return rotateDipoles(dipoles, matrices)

    @classmethod
    def rotate_quadrupoles(
        cls, matrices: torch.Tensor, quadrupoles: torch.Tensor
    ) -> torch.Tensor:
        """Rotate quadrupoles of shape ``(N, 3, 3)``."""
        return rotateQuadrupoles(quadrupoles, matrices)

    def forward(
        self,
        coords: torch.Tensor,
        z_atoms: torch.Tensor,
        x_atoms: torch.Tensor,
        y_atoms: torch.Tensor,
        axis_types: torch.Tensor,
        dipoles: torch.Tensor,
        quadrupoles: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply local-to-global rotation to dipoles and optionally quadrupoles.

        Returns
        -------
        torch.Tensor or tuple
            Rotated dipoles ``(N, 3)``. If ``quadrupoles`` is given, returns
            ``(dipoles_rot, quadrupoles_rot)``.
        """
        rot = self.compute_matrices(
            coords,
            z_atoms,
            x_atoms,
            y_atoms,
            axis_types,
            use_customized_ops=self.use_customized_ops,
        )
        d = self.rotate_dipoles(rot, dipoles).squeeze(1)
        if quadrupoles is None:
            return d
        q = self.rotate_quadrupoles(rot, quadrupoles)
        return d, q


@torch.compile
def scaleMultipoles(
    mPoles: torch.Tensor,
    monoScales: torch.Tensor,
    dipoScales: torch.Tensor,
    quadScales: torch.Tensor,
):
    # The monopoles are set directly from the parameter list while the
    # multipoles are directly scaled versions of the electric multipoles.
    mPolesScaled = torch.zeros_like(mPoles)
    mPolesScaled[:, 0] += monoScales
    mPolesScaled[:, 1:4] += mPoles[:, 1:4] * dipoScales.unsqueeze(1)
    mPolesScaled[:, 4:] += mPoles[:, 4:] * quadScales.unsqueeze(1)
    return mPolesScaled


def rotateDipoles(dipo: torch.Tensor, rotMatrix: torch.Tensor):
    return torch.bmm(dipo.unsqueeze(1), rotMatrix)


def rotateQuadrupoles(quad: torch.Tensor, rotMatrix: torch.Tensor):
    return torch.bmm(torch.bmm(rotMatrix.permute(0, 2, 1), quad), rotMatrix)


def rotateMultipoles(mono: torch.Tensor, dipo: torch.Tensor, quad: torch.Tensor, rotMatrix: torch.Tensor):
    """
    Rotate multipoles

    Parameters
    ----------
    mono: torch.Tensor
        Monopoles, shape (N,)
    dipo: torch.Tensor
        Dipoles, shape (N, 3)
    quad: torch.Tensor
        Quadrupoles, shape (N, 3, 3)

    Returns
    -------
    mPoles: torch.Tensor
        Multipoles [q, ux, uy, uz, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz], shape (N, 10)
    """
    mono = mono.unsqueeze(1)
    dipo = rotateDipoles(dipo, rotMatrix).squeeze(1)
    quad = rotateQuadrupoles(quad, rotMatrix)[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    return torch.hstack((mono, dipo, quad))


@torch.compile
def convertMultipolesToPolytensor(mono: torch.Tensor, dipo: torch.Tensor, quad: torch.Tensor):
    """
    Takes already-rotated multipoles and flattens to (N, 10) polytensor with quadrupole
    entries appropriately scaled so that symmetry-equivalent operations are avoided.

    Parameters
    ----------
    mono: torch.Tensor
        Monopoles, shape (N,)
    dipo: torch.Tensor
        Dipoles, shape (N, 3)
    quad: torch.Tensor
        Quadrupoles, shape (N, 3, 3)

    Returns
    -------
    mPoles: torch.Tensor
        Multipoles [q, ux, uy, uz, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz], shape (N, 10)
    """
    scales = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1 / 3, 2 / 3, 2 / 3, 1 / 3, 2 / 3, 1 / 3],
        device=mono.device,
        dtype=mono.dtype,
    )
    return (
        torch.hstack(
            (mono.unsqueeze(1), dipo, quad[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]])
        )
        * scales
    )


def computeCartesianQuadrupoles(quad_s: torch.Tensor):
    """
    Compute cartesian quadrupoles from spheric-harmonics quadrupoles

    Parameters
    ----------
    quad_s: torch.Tensor
        Quadrupoles in spherical harmonics form (Q20, Q21c, Q21s, Q22c, Q22s), shape (N, 5).

    Returns
    -------
    quad: torch.Tensor
        Quadrupoles in cartesian form, shape N x 3 x 3
    """
    HALF_SQRT3 = math.sqrt(3) / 2
    qxx = quad_s[:, 3] * HALF_SQRT3 - quad_s[:, 0] / 2
    qxy = quad_s[:, 4] * HALF_SQRT3
    qxz = quad_s[:, 1] * HALF_SQRT3
    qyy = -quad_s[:, 3] * HALF_SQRT3 - quad_s[:, 0] / 2
    qyz = quad_s[:, 2] * HALF_SQRT3
    qzz = quad_s[:, 0]
    quad = torch.vstack((qxx, qxy, qxz, qxy, qyy, qyz, qxz, qyz, qzz)).T.reshape(-1, 3, 3)
    return quad


def computeSphericalQuadrupoles(quad_c: torch.Tensor):
    """
    Compute cartesian quadrupoles from spheric-harmonics quadrupoles

    Parameters
    ----------
    quad_c: torch.Tensor
        Quadrupoles in cartesian form (Qxx, Qxy, Qxz, Qyy, Qyz, Qzz), shape (N, 6).

    Returns
    -------
    quad_s: torch.Tensor
        Quadrupoles in spherical harmonics form, shape (N, 5)
    """
    HALF_SQRT3 = math.sqrt(3) / 2
    q20 = quad_c[:, 5]
    q21c = quad_c[:, 2] / HALF_SQRT3
    q21s = quad_c[:, 4] / HALF_SQRT3
    q22c = (quad_c[:, 0] - quad_c[:, 3]) / HALF_SQRT3 / 2
    q22s = quad_c[:, 1] / HALF_SQRT3
    quad_s = torch.vstack((q20, q21c, q21s, q22c, q22s)).T
    return quad_s
