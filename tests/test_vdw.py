"""Tests for :mod:`torchff.vdw` custom ops vs PyTorch reference paths."""

import pytest
import torch
torch.set_printoptions(precision=8)

from torchff.test_utils import check_op
from torchff.vdw import Vdw


def test_vdw_custom_requires_sum_output() -> None:
    """Custom vdW ops cannot return per-pair energies; sum_output must stay True."""
    with pytest.raises(ValueError, match="sum_output must be True"):
        Vdw(use_customized_ops=True, sum_output=False)


def _build_all_pairs(num_atoms: int, device: torch.device) -> torch.Tensor:
    """All unique pairs (i < j), shape (P, 2) with P = n*(n-1)/2."""
    idx = torch.arange(num_atoms, device=device, dtype=torch.int64)
    return torch.combinations(idx, r=2)


def _make_synthetic_vdw_system(
    num_atoms: int,
    device: torch.device,
    dtype: torch.dtype,
    box_edge_nm: float = 5.0,
):
    """
    Periodic cubic box of edge ``box_edge_nm`` with ``num_atoms`` coordinates drawn
    uniformly in ``[0, box_edge_nm)`` per axis, per-pair sigma/epsilon, and all
    unordered pairs.
    """
    assert num_atoms == 100
    g = torch.Generator(device=device)
    g.manual_seed(42)
    # U(0, 1) * L places atoms in [0, L) along each Cartesian axis.
    coords = (
        torch.rand(num_atoms, 3, generator=g, device=device, dtype=dtype) * box_edge_nm
    )
    coords.requires_grad_(True)

    box = torch.diag(
        torch.tensor([box_edge_nm, box_edge_nm, box_edge_nm], device=device, dtype=dtype)
    )
    box.requires_grad_(True)

    pairs = _build_all_pairs(num_atoms, device)
    p = pairs.shape[0]
    assert p == num_atoms * (num_atoms - 1) // 2

    sigma = torch.full((p,), 0.32, device=device, dtype=dtype, requires_grad=True)
    epsilon = torch.full((p,), 0.85, device=device, dtype=dtype, requires_grad=True)

    # Min-image distance for a cubic box is at most (sqrt(3)/2) * L; stay above that.
    cutoff = 0.5 * box_edge_nm * (3.0**0.5) + 1e-3

    return coords, pairs, box, sigma, epsilon, cutoff


@pytest.mark.parametrize("vdw_function", ["LennardJones", "AmoebaVdw147"])
@pytest.mark.parametrize("dtype", [torch.float64])
def test_vdw_custom_matches_reference(vdw_function: str, dtype: torch.dtype) -> None:
    """Custom CUDA vdW total energy and gradients match the PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for customized vdW ops.")

    device = torch.device("cuda")
    n = 100
    coords, pairs, box, sigma, epsilon, cutoff = _make_synthetic_vdw_system(
        n, device, dtype
    )

    kwargs = {
        "coords": coords,
        "pairs": pairs,
        "box": box,
        "sigma": sigma,
        "epsilon": epsilon,
        "cutoff": cutoff,
    }

    func = Vdw(
        function=vdw_function,
        use_customized_ops=True,
    ).to(device=device, dtype=dtype)

    func_ref = Vdw(
        function=vdw_function,
        use_customized_ops=False,
        cuda_graph_compat=True,
    ).to(device=device, dtype=dtype)

    # Forward matches tightly; coord gradients can differ slightly (PBC + custom vs autograd).
    check_op(
        func,
        func_ref,
        kwargs,
        check_grad=True,
        atol=1e-6,
        rtol=0.0,
        verbose=True
    )
