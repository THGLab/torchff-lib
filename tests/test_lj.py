import pytest
import torch

from torchff.test_utils import check_op, perf_op
from .get_reference import get_water_data
from torchff.vdw import LennardJones


def _build_lj_pairs(num_atoms: int, device: str) -> torch.Tensor:
    """
    Very simple neighbor list: all unique atom pairs (i < j).
    This is O(N^2) and meant only for test sizes.
    """
    # Use torch.combinations to generate all unordered pairs.
    idx = torch.arange(num_atoms, device=device, dtype=torch.int64)
    pairs = torch.combinations(idx, r=2)
    return pairs


def _create_lj_test_data(
    num_waters: int,
    cutoff: float,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
):
    """
    Create Lennard-Jones test data from the water reference system.
    Uses OpenMM-derived coordinates, box, sigma, epsilon from TIP3P.
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
    sigma = wd.sigma
    epsilon = wd.epsilon

    pairs = _build_lj_pairs(coords.shape[0], device=device)

    return coords, box, sigma, epsilon, pairs


@pytest.mark.dependency()
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
def test_lennard_jones_energy(device, dtype):
    """
    Compare custom CUDA Lennard-Jones kernel against Python reference implementation.
    """
    cutoff = 0.4  # in nm, consistent with water reference setup
    num_waters = 100

    coords, box, sigma, epsilon, pairs = _create_lj_test_data(
        num_waters=num_waters,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
    )

    func = LennardJones(use_customized_ops=True).to(device=device, dtype=dtype)
    func_ref = LennardJones(use_customized_ops=False).to(device=device, dtype=dtype)

    check_op(
        func,
        func_ref,
        {
            "coords": coords,
            "pairs": pairs,
            "box": box,
            "sigma": sigma,
            "epsilon": epsilon,
            "cutoff": cutoff,
        },
        check_grad=True,
        atol=1e-6 if dtype is torch.float64 else 1e-4,
        rtol=1e-5,
    )


@pytest.mark.performance
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
def test_perf_lennard_jones(device, dtype):
    """
    Performance comparison between Python and custom CUDA Lennard-Jones implementations.
    """
    cutoff = 0.4  # in nm
    num_waters = 300  # modest size for perf test

    coords, box, sigma, epsilon, pairs = _create_lj_test_data(
        num_waters=num_waters,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
    )

    func_ref = torch.compile(
        LennardJones(use_customized_ops=False)
    ).to(device=device, dtype=dtype)
    func = LennardJones(use_customized_ops=True).to(device=device, dtype=dtype)

    perf_op(
        func_ref,
        coords,
        pairs,
        box,
        sigma,
        epsilon,
        cutoff,
        desc=f"lj_ref (N={coords.shape[0]})",
        repeat=100,
        run_backward=True,
    )
    perf_op(
        func,
        coords,
        pairs,
        box,
        sigma,
        epsilon,
        cutoff,
        desc=f"lj_torchff (N={coords.shape[0]})",
        repeat=100,
        run_backward=True,
    )

