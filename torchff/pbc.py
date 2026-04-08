import torch
import torch.nn as nn
from typing import Optional


class PBC(nn.Module):
    """
    Apply periodic boundary conditions to vectors.

    Supports both fixed box (provided at construction) and variable box
    (passed to :meth:`forward`). If neither box nor box_inv is available,
    returns the input vectors unchanged.
    """

    def __init__(
        self,
        box: Optional[torch.Tensor] = None,
        box_inv: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if box is not None and box_inv is not None:
            self.register_buffer("box", box)
            self.register_buffer("box_inv", box_inv)
        else:
            self.box = None
            self.box_inv = None

    def forward(
        self,
        dr_vecs: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        box_inv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply PBC to displacement vectors.

        Parameters
        ----------
        dr_vecs : torch.Tensor
            Real space vectors in Cartesian, shape (N, 3).
        box : torch.Tensor, optional
            Simulation box with axes in rows, shape (3, 3).
            Overrides the buffer if provided.
        box_inv : torch.Tensor, optional
            Inverse of the box matrix, shape (3, 3).
            Overrides the buffer if provided.

        Returns
        -------
        torch.Tensor
            PBC-folded vectors in Cartesian, shape (N, 3).
        """
        box = box if box is not None else getattr(self, "box", None)
        box_inv = box_inv if box_inv is not None else getattr(self, "box_inv", None)
        if box is not None and box_inv is None:
            box_inv, _ = torch.linalg.inv_ex(box)
        if box is None or box_inv is None:
            return dr_vecs
        ds_vecs = torch.matmul(dr_vecs, box_inv)
        ds_vecs_pbc = ds_vecs - torch.floor(ds_vecs + 0.5)
        dr_vecs_pbc = torch.matmul(ds_vecs_pbc, box)
        return dr_vecs_pbc


def applyPBC(
    dr_vecs: torch.Tensor,
    box: Optional[torch.Tensor] = None,
    box_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply periodic boundary conditions to a set of vectors.

    Functional interface to :class:`PBC`. Prefer using :class:`PBC` when
    integrating with :class:`nn.Module`-based models.

    Parameters
    ----------
    dr_vecs : torch.Tensor
        Real space vectors in Cartesian, shape (N, 3).
    box : torch.Tensor, optional
        Simulation box with axes in rows, shape (3, 3).
    box_inv : torch.Tensor, optional
        Inverse of the box matrix, shape (3, 3).

    Returns
    -------
    torch.Tensor
        PBC-folded vectors in Cartesian, shape (N, 3).
    """
    if box is not None and box_inv is None:
        box_inv, _ = torch.linalg.inv_ex(box)
    if box is None or box_inv is None:
        return dr_vecs
    ds_vecs = torch.matmul(dr_vecs, box_inv)
    ds_vecs_pbc = ds_vecs - torch.floor(ds_vecs + 0.5)
    dr_vecs_pbc = torch.matmul(ds_vecs_pbc, box)
    return dr_vecs_pbc
