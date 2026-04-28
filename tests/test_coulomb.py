import pytest
import torch

from torchff.test_utils import check_op, perf_op
from .get_reference import get_water_data
from torchff.coulomb import Coulomb


def _build_coulomb_pairs(num_atoms: int, device: str) -> torch.Tensor:
    """
    Very simple neighbor list: all unique atom pairs (i < j).
    This is O(N^2) and meant only for test sizes.
    """
    idx = torch.arange(num_atoms, device=device, dtype=torch.int64)
    pairs = torch.combinations(idx, r=2)
    return pairs


def _create_coulomb_test_data(
    num_waters: int,
    cutoff: float,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
):
    """
    Create Coulomb test data from the water reference system.
    Uses OpenMM-derived coordinates, box, and charges from TIP3P.
    """
    wd = get_water_data(
        n=num_waters,
        cutoff=cutoff,
        dtype=dtype,
        device=device,
        coord_grad=True,
        box_grad=True,
        param_grad=True,
    )

    # coords: (Natoms, 3) in nm, box: (3, 3) in nm
    coords = wd.coords
    box = wd.box
    charges = wd.charges

    pairs = _build_coulomb_pairs(coords.shape[0], device=device)

    return coords, box, charges, pairs


@pytest.mark.dependency()
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
def test_coulomb_energy(device, dtype):
    """
    Compare custom CUDA Coulomb kernel against Python reference implementation.
    """
    cutoff = 0.4  # in nm, consistent with water reference setup
    num_waters = 100

    coords, box, charges, pairs = _create_coulomb_test_data(
        num_waters=num_waters,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
    )

    # Coulomb prefactor in kJ/mol·nm·e^-2 (can be any scalar since both implementations share it)
    coulomb_constant = 138.935456

    func = Coulomb(use_customized_ops=True).to(device=device, dtype=dtype)
    func_ref = Coulomb(use_customized_ops=False).to(device=device, dtype=dtype)

    check_op(
        func,
        func_ref,
        {
            "coords": coords,
            "pairs": pairs,
            "box": box,
            "charges": charges,
            "coulomb_constant": coulomb_constant,
            "cutoff": cutoff,
        },
        check_grad=True,
        atol=1e-6 if dtype is torch.float64 else 1e-4,
        rtol=1e-5,
    )


@pytest.mark.performance
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
def test_perf_coulomb(device, dtype):
    """
    Performance comparison between Python and custom CUDA Coulomb implementations.
    """
    cutoff = 0.4  # in nm
    num_waters = 300  # modest size for perf test

    coords, box, charges, pairs = _create_coulomb_test_data(
        num_waters=num_waters,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
    )

    coulomb_constant = 138.935456

    func_ref = torch.compile(
        Coulomb(use_customized_ops=False)
    ).to(device=device, dtype=dtype)
    func = Coulomb(use_customized_ops=True).to(device=device, dtype=dtype)

    perf_op(
        func_ref,
        coords,
        pairs,
        box,
        charges,
        coulomb_constant,
        cutoff,
        desc=f"coulomb_ref (N={coords.shape[0]})",
        repeat=100,
        run_backward=True,
    )
    perf_op(
        func,
        coords,
        pairs,
        box,
        charges,
        coulomb_constant,
        cutoff,
        desc=f"coulomb_torchff (N={coords.shape[0]})",
        repeat=100,
        run_backward=True,
    )
