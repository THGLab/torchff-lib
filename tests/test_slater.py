"""Tests for :mod:`torchff.slater` custom ops vs PyTorch reference paths."""

import pytest
import torch

torch.set_printoptions(precision=8)

from torchff.test_utils import check_op
from torchff.slater import Slater


def _build_all_pairs(num_atoms: int, device: torch.device) -> torch.Tensor:
    """All unique pairs (i < j), shape (P, 2) with P = n*(n-1)/2."""
    idx = torch.arange(num_atoms, device=device, dtype=torch.int64)
    return torch.combinations(idx, r=2)


def _make_synthetic_slater_system(
    num_atoms: int,
    device: torch.device,
    dtype: torch.dtype,
    box_edge_nm: float = 5.0,
):
    """
    Periodic cubic box of edge ``box_edge_nm`` with ``num_atoms`` coordinates drawn
    uniformly in ``[0, box_edge_nm)`` per axis, per-pair A/B, and all unordered pairs.
    """
    assert num_atoms == 100
    g = torch.Generator(device=device)
    g.manual_seed(42)
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

    A = torch.full((p,), 1.2, device=device, dtype=dtype, requires_grad=True)
    B = torch.full((p,), 2.5, device=device, dtype=dtype, requires_grad=True)

    cutoff = 0.5 * box_edge_nm * (3.0**0.5) + 1e-3

    return coords, pairs, box, A, B, cutoff


@pytest.mark.parametrize("dtype", [torch.float64])
def test_slater_custom_matches_reference(dtype: torch.dtype) -> None:
    """Custom CUDA Slater total energy and gradients match the PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for customized Slater ops.")

    device = torch.device("cuda")
    n = 100
    coords, pairs, box, A, B, cutoff = _make_synthetic_slater_system(n, device, dtype)

    kwargs = {
        "coords": coords,
        "pairs": pairs,
        "box": box,
        "A": A,
        "B": B,
        "cutoff": cutoff,
    }

    func = Slater(use_customized_ops=True).to(device=device, dtype=dtype)

    func_ref = Slater(use_customized_ops=False, sum_output=True, cuda_graph_compat=True).to(
        device=device, dtype=dtype
    )

    check_op(
        func,
        func_ref,
        kwargs,
        check_grad=True,
        atol=1e-6,
        rtol=0.0,
        verbose=True,
    )


@pytest.mark.parametrize("dtype", [torch.float64])
def test_slater_type_pairs_custom_matches_reference(dtype: torch.dtype) -> None:
    """Type-pair tables: custom op matches reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for customized Slater ops.")

    device = torch.device("cuda")
    n = 100
    n_types = 8
    g = torch.Generator(device=device)
    g.manual_seed(43)

    coords = torch.rand(n, 3, generator=g, device=device, dtype=dtype) * 5.0
    coords.requires_grad_(True)
    box = torch.diag(torch.tensor([5.0, 5.0, 5.0], device=device, dtype=dtype))
    box.requires_grad_(True)

    atom_types = torch.randint(0, n_types, (n,), device=device, dtype=torch.int64)
    pairs = _build_all_pairs(n, device)
    p = pairs.shape[0]

    # Leaf tensors (required for check_op / .grad); avoid `rand * a + b` which is non-leaf.
    A_tab = torch.rand(n_types, n_types, device=device, dtype=dtype, generator=g)
    A_tab.mul_(0.5).add_(0.5)
    A_tab.requires_grad_(True)
    B_tab = torch.rand(n_types, n_types, device=device, dtype=dtype, generator=g)
    B_tab.mul_(2.0).add_(0.5)
    B_tab.requires_grad_(True)

    cutoff = 0.5 * 5.0 * (3.0**0.5) + 1e-3

    kwargs = {
        "coords": coords,
        "pairs": pairs,
        "box": box,
        "A": A_tab,
        "B": B_tab,
        "cutoff": cutoff,
        "atom_types": atom_types,
    }

    func = Slater(use_customized_ops=True, use_type_pairs=True).to(device=device, dtype=dtype)
    func_ref = Slater(
        use_customized_ops=False,
        use_type_pairs=True,
        sum_output=True,
        cuda_graph_compat=True,
    ).to(device=device, dtype=dtype)

    check_op(
        func,
        func_ref,
        kwargs,
        check_grad=True,
        atol=1e-6,
        rtol=0.0,
        verbose=False,
    )
