from typing import Optional
import math

import torch
import torch.nn as nn
import torchff_multipoles


def computeInteractionTensor(drVec: torch.Tensor, dampFactors: Optional[torch.Tensor] = None, drInv: Optional[torch.Tensor] = None, rank: int = 2):
    if drInv is None:
        drInv = 1 / torch.norm(drVec, dim=1)
    
    if rank == 0:
        # For rank-0, dampFactors (if present) is a per-pair vector (erfc(b r)).
        # We should apply it elementwise, not index it as if it were a stacked tensor.
        return drInv if dampFactors is None else drInv * dampFactors
    
    # calculate inversions
    drInv2 = drInv * drInv
    drInv3 = drInv2 * drInv
    drInv5 = drInv3 * drInv2

    drVec2 = drVec * drVec
    x, y, z = drVec[:, 0], drVec[:, 1], drVec[:, 2]
    x2, y2, z2 = drVec2[:, 0], drVec2[:, 1], drVec2[:, 2]
    xy, xz, yz = x * y, x * z, y * z

    drInv7 = drInv5 * drInv2
    drInv9 = drInv7 * drInv2

    if dampFactors is not None:
        drInv = drInv * dampFactors[0]
        if rank > 0:
            drInv3 = drInv3 * dampFactors[1]
            drInv5 = drInv5 * dampFactors[2]
        if rank > 1:
            drInv7 = drInv7 * dampFactors[3]
            drInv9 = drInv9 * dampFactors[4]

    tx, ty, tz = -x * drInv3, -y * drInv3, -z * drInv3
    
    txx = 3 * x2 * drInv5 - drInv3
    txy = 3 * xy * drInv5
    txz = 3 * xz * drInv5
    tyy = 3 * y2 * drInv5 - drInv3
    tyz = 3 * yz * drInv5
    tzz = 3 * z2 * drInv5 - drInv3     

    if rank == 1:
        return torch.vstack((
            drInv, -tx,   -ty,   -tz,   
            tx,    -txx,  -txy,  -txz,  
            ty,    -txy,  -tyy,  -tyz,  
            tz,    -txz,  -tyz,  -tzz,  
        )).T.reshape(-1, 4, 4)
    
    txxx = -15 * x2 * x * drInv7 + 9 * x * drInv5
    txxy = -15 * x2 * y * drInv7 + 3 * y * drInv5
    txxz = -15 * x2 * z * drInv7 + 3 * z * drInv5
    tyyy = -15 * y2 * y * drInv7 + 9 * y * drInv5
    tyyx = -15 * y2 * x * drInv7 + 3 * x * drInv5
    tyyz = -15 * y2 * z * drInv7 + 3 * z * drInv5
    tzzz = -15 * z2 * z * drInv7 + 9 * z * drInv5
    tzzx = -15 * z2 * x * drInv7 + 3 * x * drInv5
    tzzy = -15 * z2 * y * drInv7 + 3 * y * drInv5
    txyz = -15 * x * y * z * drInv7

    txxxx = 105 * x2 * x2 * drInv9 - 90 * x2 * drInv7 + 9 * drInv5
    txxxy = 105 * x2 * xy * drInv9 - 45 * xy * drInv7
    txxxz = 105 * x2 * xz * drInv9 - 45 * xz * drInv7
    txxyy = 105 * x2 * y2 * drInv9 - 15 * (x2 + y2) * drInv7 + 3 * drInv5
    txxzz = 105 * x2 * z2 * drInv9 - 15 * (x2 + z2) * drInv7 + 3 * drInv5
    txxyz = 105 * x2 * yz * drInv9 - 15 * yz * drInv7

    tyyyy = 105 * y2 * y2 * drInv9 - 90 * y2 * drInv7 + 9 * drInv5
    tyyyx = 105 * y2 * xy * drInv9 - 45 * xy * drInv7
    tyyyz = 105 * y2 * yz * drInv9 - 45 * yz * drInv7
    tyyzz = 105 * y2 * z2 * drInv9 - 15 * (y2 + z2) * drInv7 + 3 * drInv5
    tyyxz = 105 * y2 * xz * drInv9 - 15 * xz * drInv7

    tzzzz = 105 * z2 * z2 * drInv9 - 90 * z2 * drInv7 + 9 * drInv5
    tzzzx = 105 * z2 * xz * drInv9 - 45 * xz * drInv7
    tzzzy = 105 * z2 * yz * drInv9 - 45 * yz * drInv7                
    tzzxy = 105 * z2 * xy * drInv9 - 15 * xy * drInv7

    return torch.vstack((
        drInv, -tx,   -ty,   -tz,   txx,   txy,   txz,   tyy,   tyz,   tzz,
        tx,    -txx,  -txy,  -txz,  txxx,  txxy,  txxz,  tyyx,  txyz,  tzzx,
        ty,    -txy,  -tyy,  -tyz,  txxy,  tyyx,  txyz,  tyyy,  tyyz,  tzzy,
        tz,    -txz,  -tyz,  -tzz,  txxz,  txyz,  tzzx,  tyyz,  tzzy,  tzzz,
        txx,   -txxx, -txxy, -txxz, txxxx, txxxy, txxxz, txxyy, txxyz, txxzz,
        txy,   -txxy, -tyyx, -txyz, txxxy, txxyy, txxyz, tyyyx, tyyxz, tzzxy,
        txz,   -txxz, -txyz, -tzzx, txxxz, txxyz, txxzz, tyyxz, tzzxy, tzzzx,
        tyy,   -tyyx, -tyyy, -tyyz, txxyy, tyyyx, tyyxz, tyyyy, tyyyz, tyyzz,
        tyz,   -txyz, -tyyz, -tzzy, txxyz, tyyxz, tzzxy, tyyyz, tyyzz, tzzzy,
        tzz,   -tzzx, -tzzy, -tzzz, txxzz, tzzxy, tzzzx, tyyzz, tzzzy, tzzzz
    )).T.reshape(-1, 10, 10)