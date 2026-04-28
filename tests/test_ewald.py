import math
import numpy as np
import pytest
import torch

from torchff.test_utils import check_op, perf_op
from torchff.ewald import Ewald


torch.set_printoptions(precision=8)


def create_test_data(num: int, rank: int = 2, device: str = "cuda", dtype: torch.dtype = torch.float64):
    """Create random test data for Ewald tests."""
    # Set a physically reasonable box length scaling with number of atoms
    boxLen = float((num * 10.0) ** (1.0 / 3.0))

    # Random coordinates in [0, boxLen)
    coords_np = np.random.rand(num, 3) * boxLen

    # Random charges, shifted so that the total charge is zero
    q_np = np.random.randn(num)
    q_np -= q_np.mean()

    # Random dipoles
    d_np = np.random.randn(num, 3)

    # Random symmetric, traceless quadrupoles per atom
    t_np = np.empty((num, 3, 3), dtype=float)
    for i in range(num):
        A = np.random.randn(3, 3)
        sym = 0.5 * (A + A.T)
        trace = np.trace(sym) / 3.0
        sym -= np.eye(3) * trace  # make traceless
        t_np[i] = sym

    # Cubic box
    box_np = np.eye(3) * boxLen

    # Convert to torch tensors with the requested device and dtype
    coords = torch.tensor(coords_np, device=device, dtype=dtype, requires_grad=True)
    box = torch.tensor(box_np, device=device, dtype=dtype)
    q = torch.tensor(q_np, device=device, dtype=dtype, requires_grad=True)
    p = (
        torch.tensor(d_np, device=device, dtype=dtype, requires_grad=True)
        if rank >= 1
        else None
    )
    t = (
        torch.tensor(t_np, device=device, dtype=dtype, requires_grad=True)
        if rank >= 2
        else None
    )

    # Find appropriate Ewald parameters.
    alpha_ewald = math.sqrt(-math.log10(2 * 1e-6)) / 9.0
    k_max = 50
    for i in range(2, 50):
        error_estimate = (i * math.sqrt(boxLen * alpha_ewald) / 20.0) * math.exp(
            -torch.pi
            * torch.pi
            * i
            * i
            / (boxLen * alpha_ewald * boxLen * alpha_ewald)
        )
        if error_estimate < 1e-6:
            k_max = i
            break
    print("Number of kmax:", k_max)

    return coords, box, q, p, t, alpha_ewald, k_max



@pytest.mark.dependency()
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_ewald_energy(device, dtype, rank):
    """Compare custom CUDA Ewald kernel against Python reference implementation."""
    N = 1000
    coords, box, q, p, t, alpha, kmax = create_test_data(N, rank, device=device, dtype=dtype)

    func = Ewald(alpha, kmax, rank, use_customized_ops=True, return_fields=False).to(
        device=device, dtype=dtype
    )
    func_ref = Ewald(
        alpha, kmax, rank, use_customized_ops=False, return_fields=False
    ).to(device=device, dtype=dtype)

    check_op(
        func,
        func_ref,
        {"coords": coords, "box": box, "q": q, "p": p, "t": t},
        check_grad=True,
        atol=1e-6 if dtype is torch.float64 else 1e-4,
        rtol=1e-5,
    )


@pytest.mark.performance
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
@pytest.mark.parametrize("rank", [0])
def test_perf_ewald(device, dtype, rank):
    """Performance comparison between Python and custom CUDA Ewald implementations."""
    N = 3000
    coords, box, q, p, t, alpha, kmax = create_test_data(N, rank, device=device, dtype=dtype)

    func_ref = torch.compile(
        Ewald(alpha, kmax, rank, use_customized_ops=False, return_fields=False)
    ).to(device=device, dtype=dtype)
    func = Ewald(alpha, kmax, rank, use_customized_ops=True, return_fields=False).to(
        device=device, dtype=dtype
    )

    perf_op(
        func_ref,
        coords,
        box,
        q,
        p,
        t,
        desc=f"ewald_ref (N={N}, rank={rank})",
        repeat=1000,
        run_backward=True,
    )
    perf_op(
        func,
        coords,
        box,
        q,
        p,
        t,
        desc=f"ewald_torchff (N={N}, rank={rank})",
        repeat=1000,
        run_backward=True,
    )
