import math
from typing import List, Optional, Union
import torch
from enum import IntEnum
from .pbc import applyPBC


class AxisTypes(IntEnum):
    ZThenX            = 0
    Bisector          = 1
    ZBisect           = 2
    ThreeFold         = 3
    ZOnly             = 4
    NoAxisType        = 5
    LastAxisTypeIndex = 6


def normVec(vec):
    return vec / torch.norm(vec, dim=1, keepdim=True)


@torch.compile
def computeLocal2GlobalRotationMatrices(
    positions: torch.Tensor, 
    zAtoms: torch.Tensor, 
    xAtoms: torch.Tensor, 
    yAtoms: torch.Tensor, 
    axisTypes: torch.Tensor, 
    box: Optional[torch.Tensor] = None, 
    boxInv: Optional[torch.Tensor] = None
):
    """
    Compute local to global rotation matrix for a set of atoms

    Parameters
    ----------
    positions: torch.Tensor
        Atom positions, shape (N, 3)
    zAtoms: torch.Tensor[int]
        Atomic indices specifying Z-axis, shape (N,)
    xAtoms: torch.Tensor[int]
        Atomic indices specifying X-axis, shape (N,)
    yAtoms: torch.Tensor[int]
        Atomic indices specifying Y-axis, shape (N,)
    axisTypes: torch.Tensor[int]
        Integers specifying local axis types, shape (N,)
    box: torch.Tensor
        Peroidic box, shape (3, 3), optional
    """
    zVec = applyPBC(positions[zAtoms] - positions, box, boxInv)
    zVec = normVec(zVec)
    xVec = torch.zeros_like(zVec)
    yVec = torch.zeros_like(zVec)

    # Z-Only
    filterZOnly = torch.logical_or(axisTypes == AxisTypes.ZOnly.value, axisTypes == AxisTypes.NoAxisType.value)
    xVecNotZOnly = applyPBC(positions[xAtoms][~filterZOnly] - positions[~filterZOnly], box, boxInv)
    #xVec[~filterZOnly] += normVec(xVecNotZOnly)
    #xVec[filterZOnly, 0] += 1 - zVec[filterZOnly, 0]
    xVec_new = xVec.clone()
    xVec_new[~filterZOnly] = xVec[~filterZOnly] + normVec(xVecNotZOnly)
    xVec_new[filterZOnly, 0] = xVec[filterZOnly, 0] + (1 - zVec[filterZOnly, 0])
    xVec = xVec_new
    xVec[filterZOnly, 1] += zVec[filterZOnly, 0]

    # Bisector
    filterBisector = (axisTypes == AxisTypes.Bisector.value)
    if torch.any(filterBisector):
        zVec[filterBisector] += xVec[filterBisector]
        zVec = normVec(zVec)
    
    # Z-Bisect
    filterZBisect = (axisTypes == AxisTypes.ZBisect.value)
    if torch.any(filterZBisect):
        yVecZBisect = applyPBC(positions[yAtoms][filterZBisect] - positions[filterZBisect], box, boxInv)
        yVecZBisect = normVec(yVecZBisect)
        xVecZBisect = normVec(xVec[filterZBisect] + yVecZBisect)
        xVec[filterZBisect] = xVecZBisect
    
    # Threefold
    filterThreeFold = (axisTypes == AxisTypes.ThreeFold.value)
    if torch.any(filterThreeFold):
        yVecThreeFold = applyPBC(positions[yAtoms][filterThreeFold] - positions[filterThreeFold], box, boxInv)
        yVecThreeFold = normVec(yVecThreeFold)
        xVecThreeFold = xVec[filterThreeFold]
        zVecThreeFold = zVec[filterThreeFold]
        zVec[filterThreeFold] = normVec(zVecThreeFold + xVecThreeFold + yVecThreeFold)

    xVec = normVec(xVec - zVec * torch.sum(zVec * xVec, dim=1, keepdim=True))
    yVec = torch.linalg.cross(zVec, xVec)

    # No axis
    filterNoAxis = (axisTypes == AxisTypes.NoAxisType.value)
    if torch.any(filterNoAxis):
        filterNoAxis = filterNoAxis.view(-1, 1)
        zVec = torch.where(filterNoAxis, torch.tensor([0.0, 0.0, 1.0], dtype=zVec.dtype, device=zVec.device), zVec)
        xVec = torch.where(filterNoAxis, torch.tensor([1.0, 0.0, 0.0], dtype=xVec.dtype, device=xVec.device), xVec)
        yVec = torch.where(filterNoAxis, torch.tensor([0.0, 1.0, 0.0], dtype=yVec.dtype, device=yVec.device), yVec)

    rotMatrix = torch.hstack((xVec, yVec, zVec)).reshape(-1, 3, 3)
    return rotMatrix


@torch.compile
def scaleMultipoles(
    mPoles: torch.Tensor, 
    monoScales: torch.Tensor, dipoScales: torch.Tensor, quadScales: torch.Tensor,
):
    # The monopoles are set directly from the parameter list while the
    # multipoles are directly scaled versions of the electric multipoles.
    mPolesScaled = torch.zeros_like(mPoles)
    mPolesScaled[:, 0]   += monoScales
    mPolesScaled[:, 1:4] += mPoles[:, 1:4] * dipoScales.unsqueeze(1)
    mPolesScaled[:, 4:]  += mPoles[:, 4:] * quadScales.unsqueeze(1)
    return mPolesScaled


@torch.compile
def rotateDipoles(dipo: torch.Tensor, rotMatrix: torch.Tensor):
    return torch.bmm(dipo.unsqueeze(1), rotMatrix)


@torch.compile
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
    return torch.hstack((mono.unsqueeze(1), dipo, quad[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]])) * torch.tensor([1., 1., 1., 1., 1/3, 2/3, 2/3, 1/3, 2/3, 1/3], device=mono.device)


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
    q20  = quad_c[:, 5] 
    q21c = quad_c[:, 2] / HALF_SQRT3
    q21s = quad_c[:, 4] / HALF_SQRT3
    q22c = (quad_c[:, 0] - quad_c[:, 3]) / HALF_SQRT3 / 2
    q22s = quad_c[:, 1] / HALF_SQRT3
    quad_s = torch.vstack((q20, q21c, q21s, q22c, q22s)).T
    return quad_s
