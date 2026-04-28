"""Benchmark torch-pme Ewald using water data and TorchFF neighbor lists."""

import math
import os
import sys
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
import torch.nn as nn
import torchff  # noqa: E402
from torchff.test_utils import perf_op  # noqa: E402
from torchpme import EwaldCalculator, PMECalculator  # noqa: E402
from torchpme.potentials import CoulombPotential  # noqa: E402
from torchpme.tuning import tune_ewald  # noqa: E402
from torchpme.lib import generate_kvectors_for_ewald  # noqa: E402


def _pbc_pair_distances(
    coords: torch.Tensor,
    box: torch.Tensor,
    pairs: torch.Tensor,
) -> torch.Tensor:
    """Compute minimum-image pairwise distances for periodic boundary conditions.

    Differentiable with respect to ``coords``. Uses the same PBC convention as
    :func:`torchff.coulomb.compute_coulomb_energy_ref` (fractional wrap then
    Cartesian distance).

    Parameters
    ----------
    coords : torch.Tensor
        Atomic coordinates, shape ``(N, 3)``.
    box : torch.Tensor
        Periodic box matrix, shape ``(3, 3)``, row vectors.
    pairs : torch.Tensor
        Pair indices, shape ``(P, 2)``, dtype integer.

    Returns
    -------
    torch.Tensor
        Distances for each pair, shape ``(P,)``.
    """
    dr_vecs = coords[pairs[:, 0]] - coords[pairs[:, 1]]
    box_inv = torch.linalg.inv(box)
    ds_vecs = torch.matmul(dr_vecs, box_inv)
    ds_vecs_pbc = ds_vecs - torch.floor(ds_vecs + 0.5)
    dr_vecs_pbc = torch.matmul(ds_vecs_pbc, box)
    return torch.linalg.norm(dr_vecs_pbc, dim=1)


def _build_torchff_neighbor_list(
    coords: torch.Tensor,
    box: torch.Tensor,
    cutoff: float,
    exclude_same_molecule: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build neighbor list and distances using TorchFF and PBC distances.

    Uses :func:`torchff.build_neighbor_list_nsquared` then computes
    minimum-image distances. Optionally excludes pairs within the same
    molecule (e.g. water: ``exclude_same_molecule=3``).

    Parameters
    ----------
    coords : torch.Tensor
        Atomic coordinates, shape ``(N, 3)``.
    box : torch.Tensor
        Periodic box, shape ``(3, 3)``.
    cutoff : float
        Distance cutoff for the neighbor list.
    exclude_same_molecule : int, optional
        If set (e.g. 3 for water), drop pairs where
        ``i // exclude_same_molecule == j // exclude_same_molecule``.

    Returns
    -------
    neighbor_indices : torch.Tensor
        Pair indices, shape ``(P, 2)``, dtype int64.
    neighbor_distances : torch.Tensor
        PBC distances for each pair, shape ``(P,)``.
    """
    nblist_trial, _ = torchff.build_neighbor_list_nsquared(
        coords, box, cutoff, -1, False
    )
    pairs = nblist_trial
    if exclude_same_molecule is not None:
        mask = torch.floor_divide(pairs[:, 0], exclude_same_molecule) != torch.floor_divide(
            pairs[:, 1], exclude_same_molecule
        )
        pairs = pairs[mask]
    neighbor_indices = pairs.to(dtype=torch.int64)
    neighbor_distances = _pbc_pair_distances(coords, box, neighbor_indices)
    return neighbor_indices, neighbor_distances


def _estimate_ewald_params(box: torch.Tensor) -> Tuple[float, int]:
    """Estimate (alpha, kmax) for Ewald given a periodic box.

    Uses the same heuristic as in :func:`examples.fixed_charge_benchmark._estimate_ewald_params`.
    """
    alpha = math.sqrt(-math.log10(2.0 * 1e-6)) / 0.8
    box_len = float(torch.mean(torch.diag(box)).item())
    kmax = 50
    for i in range(2, 50):
        error_estimate = (
            i * math.sqrt(box_len * alpha) / 20.0
        ) * math.exp(
            -math.pi * math.pi * i * i / (box_len * alpha * box_len * alpha)
        )
        if error_estimate < 1e-6:
            kmax = i
            break
    return alpha, kmax


class EwaldTorchPME(nn.Module):
    def __init__(self, alpha, kmax):
        super().__init__()
 
        self.calculator = EwaldCalculator(
            potential=CoulombPotential(smearing=alpha),
            lr_wavelength=1.0, # assign a random value, this will be overwritten by kmax
        )
        # The definition of kmax in torch PME is different
        self.kmax = kmax * 2 + 1
    
    def forward(self, coords, box, charges):
        kvecs = generate_kvectors_for_ewald(cell=box, ns=torch.tensor([self.kmax, self.kmax, self.kmax], device=coords.device))
        potentials = self.calculator._compute_kspace(charges, box, coords, kvectors=kvecs)
        energy = torch.sum(charges * potentials)
        return energy


class PMETorchPME(nn.Module):
    def __init__(self, alpha, mesh):
        super().__init__()
 
        self.calculator = PMECalculator(
            potential=CoulombPotential(smearing=alpha),
            mesh_spacing=mesh,
            interpolation_nodes=6
        )
    
    def forward(self, coords, box, charges):
        potentials = self.calculator._compute_kspace(charges, box, coords)
        energy = torch.sum(charges * potentials)
        return energy



def main() -> None:
    """Run Ewald benchmark on water systems using perf_op (see tests/test_ewald.py)."""
    from tests.get_reference import get_water_data  # noqa: E402

    device = "cuda"
    dtype = torch.float64
    cutoff_nm = 0.8
    num_waters = 3000
    warmup = 10
    repeat = 1000

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark.")

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    wd = get_water_data(
        n=num_waters,
        cutoff=cutoff_nm,
        dtype=dtype,
        device=device,
        coord_grad=True,
        box_grad=False,
        param_grad=False,
    )
    coords = wd.coords
    box = wd.box
    charges = wd.charges
    if charges.ndim == 1:
        charges = charges.unsqueeze(1)

    alpha, kmax = _estimate_ewald_params(box)
    model = torch.compile(EwaldTorchPME(alpha=alpha, kmax=kmax).to(device=device, dtype=dtype))

    perf_op(
        model,
        coords,
        box,
        charges,
        desc=f"EwaldTorchPME (N={num_waters * 3}, alpha={alpha:.6f}, kmax={kmax})",
        warmup=warmup,
        repeat=repeat,
        run_backward=True,
        # use_cuda_graph=True,
    )


if __name__ == "__main__":
    main()
