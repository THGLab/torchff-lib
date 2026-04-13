"""Tests for :mod:`torchff.dispersion` custom ops vs PyTorch reference paths."""

import pytest
import torch

torch.set_printoptions(precision=8)

from torchff.dispersion import Dispersion
from torchff.test_utils import check_op


def _build_all_pairs(num_atoms: int, device: torch.device) -> torch.Tensor:
    """All unique pairs (i < j), shape (P, 2) with P = n*(n-1)/2."""
    idx = torch.arange(num_atoms, device=device, dtype=torch.int64)
    return torch.combinations(idx, r=2)


def _make_synthetic_dispersion_system(
    num_atoms: int,
    device: torch.device,
    dtype: torch.dtype,
    box_edge_nm: float = 5.0,
):
    """Periodic cubic box with per-pair c6, b and all unordered pairs."""
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

    c6 = torch.full((p,), 1.2, device=device, dtype=dtype, requires_grad=True)
    b = torch.full((p,), 2.5, device=device, dtype=dtype, requires_grad=True)

    cutoff = 0.5 * box_edge_nm * (3.0**0.5) + 1e-3

    return coords, pairs, box, c6, b, cutoff


@pytest.mark.parametrize("dtype", [torch.float64])
def test_dispersion_custom_matches_reference(dtype: torch.dtype) -> None:
    """Custom CUDA Tang–Tonnies energy and gradients match the PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for customized dispersion ops.")

    device = torch.device("cuda")
    n = 100
    coords, pairs, box, c6, b, cutoff = _make_synthetic_dispersion_system(
        n, device, dtype
    )

    kwargs = {
        "coords": coords,
        "pairs": pairs,
        "box": box,
        "c6": c6,
        "b": b,
        "cutoff": cutoff,
    }

    func = Dispersion(use_customized_ops=True).to(device=device, dtype=dtype)
    func_ref = Dispersion(
        use_customized_ops=False,
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
        verbose=True,
    )


@pytest.mark.parametrize("dtype", [torch.float64])
def test_dispersion_type_pairs(dtype: torch.dtype) -> None:
    """Type-pair (T,T) tables match reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for customized dispersion ops.")

    device = torch.device("cuda")
    n = 32
    g = torch.Generator(device=device)
    g.manual_seed(123)
    coords = torch.rand(n, 3, generator=g, device=device, dtype=dtype) * 4.0
    coords.requires_grad_(True)
    box = torch.diag(
        torch.tensor([4.0, 4.0, 4.0], device=device, dtype=dtype)
    )
    box.requires_grad_(True)

    n_types = 5
    atom_types = torch.randint(0, n_types, (n,), device=device, dtype=torch.int64)
    pairs = _build_all_pairs(n, device)
    p = pairs.shape[0]

    c6 = torch.rand(n_types, n_types, device=device, dtype=dtype, requires_grad=True)
    b = torch.rand(n_types, n_types, device=device, dtype=dtype, requires_grad=True)
    cutoff = 2.0

    kwargs = {
        "coords": coords,
        "pairs": pairs,
        "box": box,
        "c6": c6,
        "b": b,
        "cutoff": cutoff,
        "atom_types": atom_types,
    }

    func = Dispersion(use_customized_ops=True, use_type_pairs=True).to(
        device=device, dtype=dtype
    )
    func_ref = Dispersion(
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
