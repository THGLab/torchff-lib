import os
import numpy as np
import pytest
import torch

from torchff.test_utils import check_op, perf_op
from torchff.pme import PME


torch.set_printoptions(precision=8)
torch.set_default_dtype(torch.float64)


def create_test_data(num: int, rank: int = 2, device: str = "cuda", dtype: torch.dtype = torch.float64):
    """Load test data from npz file and convert to torch tensors."""
    # Get the directory where the water test data is stored
    test_dir = os.path.dirname(os.path.abspath(__file__))
    water_dir = os.path.join(test_dir, "water")
    npz_path = os.path.join(water_dir, f"random_water_{num}.npz")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Test data file not found: {npz_path}")
    
    # Load data from npz file
    data = np.load(npz_path)
    coords_np = data["coords"]
    box_np = data["box"]
    q_np = data["q"]
    p_np = data["p"]
    t_np = data["t"]
    alpha_pme = float(data["alpha"])
    max_hkl = int(data["max_hkl"])
    loaded_rank = int(data["rank"])
    
    # Use the rank from the file if not specified, otherwise use the requested rank
    if rank is None:
        rank = loaded_rank

    # Convert to torch tensors with the requested device and dtype
    coords = torch.tensor(coords_np, device=device, dtype=dtype, requires_grad=True)
    box = torch.tensor(box_np, device=device, dtype=dtype)
    q = torch.tensor(q_np, device=device, dtype=dtype, requires_grad=True)
    p = (
        torch.tensor(p_np, device=device, dtype=dtype, requires_grad=True)
        if rank >= 1
        else None
    )
    t = (
        torch.tensor(t_np, device=device, dtype=dtype, requires_grad=True)
        if rank >= 2
        else None
    )

    return coords, box, q, p, t, alpha_pme, max_hkl


@pytest.mark.parametrize("device, dtype", [("cuda", torch.float64)])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_pme_execution(device, dtype, rank):
    """Compare custom CUDA PME kernel against Python reference implementation."""
    N = 300
    coords, box, q, p, t, alpha, max_hkl = create_test_data(N, rank, device=device, dtype=dtype)

    func = PME(alpha, max_hkl, rank, use_customized_ops=True).to(
        device=device, dtype=dtype
    )
    func_ref = PME(alpha, max_hkl, rank, use_customized_ops=False).to(
        device=device, dtype=dtype
    )

    # PME returns a tuple; compare energies (index 3) and their gradients.
    def func_ref_wrapper(coords, box, q, p, t):
        result = func_ref(coords, box, q, p, t)
        return result[3] if isinstance(result, tuple) else result

    def func_wrapper(coords, box, q, p, t):
        result = func(coords, box, q, p, t)
        return result[3] if isinstance(result, tuple) else result

    check_op(
        func_wrapper,
        func_ref_wrapper,
        {"coords": coords, "box": box, "q": q, "p": p, "t": t},
        check_grad=True,
        atol=1e-5 if dtype is torch.float64 else 1e-4,
        rtol=0.0,
        verbose=True
    )


@pytest.mark.parametrize("device, dtype", [("cuda", torch.float64)])
@pytest.mark.parametrize("rank", [0, 1])
def test_pme_with_field_simple(device, dtype, rank):
    """Verify that E = (q*pot - p*field - t*field_grad/3) / 2 holds."""
    N = 300
    coords, box, q, p, t, alpha, max_hkl = create_test_data(N, rank, device=device, dtype=dtype)
    coords.requires_grad_(True)

    pme_ref = PME(alpha, max_hkl, rank, use_customized_ops=True, return_fields=True).to(
        device=device, dtype=dtype
    )
    pme = PME(alpha, max_hkl, rank, use_customized_ops=True, return_fields=True).to(
        device=device, dtype=dtype
    )

    def ref_func(coords, box, q, p, t):
        energy, pot, field = pme_ref(coords, box, q, p, t)
        if rank == 0:
            return torch.sum(q * pot) / 2
        else:
            return (torch.sum(q * pot) - torch.sum(p * field)) / 2

    def func(coords, box, q, p, t):
        energy, pot, field = pme(coords, box, q, p, t)
        if rank == 0:
            return torch.sum(q * pot) / 2
        else:
            return (torch.sum(q * pot) - torch.sum(p * field)) / 2

    check_op(
        func,
        ref_func,
        {"coords": coords, "box": box, "q": q, "p": p, "t": t},
        check_grad=True,
        verbose=True,
    )


@pytest.mark.performance
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float64)])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_perf_pme(device, dtype, rank):
    """Performance comparison between Python and custom CUDA PME implementations."""
    N = 300
    coords, box, q, p, t, alpha, max_hkl = create_test_data(N, 2, device=device, dtype=dtype)
    print(alpha, max_hkl)

    func_ref = PME(alpha, max_hkl, rank, use_customized_ops=False).to(device=device, dtype=dtype)
    func = PME(alpha, max_hkl, rank, use_customized_ops=True).to(
        device=device, dtype=dtype
    )

    # PME returns a tuple, so we need to extract the energy for backward pass
    def func_ref_wrapper(*args):
        result = func_ref(*args)
        return result[3] if isinstance(result, tuple) else result
    
    def func_wrapper(*args):
        result = func(*args)
        return result[3] if isinstance(result, tuple) else result

    perf_op(
        func_ref_wrapper,
        coords,
        box,
        q,
        p,
        t,
        desc=f"pme_ref (N={N}, rank={rank})",
        repeat=100,
        run_backward=True,
    )
    perf_op(
        func_wrapper,
        coords,
        box,
        q,
        p,
        t,
        desc=f"pme_torchff (N={N}, rank={rank})",
        repeat=1000,
        run_backward=True,
        use_cuda_graph=True
    )
