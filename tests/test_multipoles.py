import numpy as np
import pytest
import torch

from torchff.test_utils import check_op, perf_op
from torchff.multipoles import MultipolarInteraction


torch.set_printoptions(precision=8)


def create_test_data(
    num: int,
    rank: int = 2,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
    cutoff: float = 9.0,
    ewald_alpha: float = -1.0,
    random_state: int | None = 42,
):
    """Create random test data for multipolar interaction tests."""
    box_len = float((num * 10.0) ** (1.0 / 3.0))

    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    coords_np = rng.rand(num, 3) * box_len
    q_np = rng.randn(num) * 0.1
    d_np = rng.randn(num, 3) * 0.1

    t_np = np.empty((num, 3, 3), dtype=float)
    for i in range(num):
        A = rng.randn(3, 3)
        sym = 0.5 * (A + A.T)
        trace = np.trace(sym) / 3.0
        sym -= np.eye(3) * trace
        t_np[i] = sym

    box_np = np.eye(3) * box_len

    # PBC minimum-image distance to build pairs
    dr = coords_np[:, None, :] - coords_np[None, :, :]
    box_inv_np = np.linalg.inv(box_np)
    ds = dr @ box_inv_np
    ds = ds - np.floor(ds + 0.5)
    dr_pbc = ds @ box_np
    dist = np.linalg.norm(dr_pbc, axis=2)
    ii, jj = np.triu_indices(num, k=1)
    mask = dist[ii, jj] < cutoff
    pairs_np = np.column_stack([ii[mask], jj[mask]])

    if pairs_np.size == 0:
        pairs_np = np.array([[0, 1]], dtype=np.int64)

    coords = torch.tensor(coords_np, device=device, dtype=dtype, requires_grad=True)
    box = torch.tensor(box_np, device=device, dtype=dtype)
    pairs = torch.tensor(pairs_np, device=device, dtype=torch.int64)
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

    prefactor = 1.0
    return coords, box, pairs, q, p, t, cutoff, ewald_alpha, prefactor


@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("ewald, excl", [(-1.0, False), (0.4, True), (0.4, False)])
@pytest.mark.parametrize("cuda_graph", [True, False])
@pytest.mark.parametrize("return_fields", [True, False])
def test_multipolar(device, dtype, rank, ewald, excl, cuda_graph, return_fields):
    """Compare custom CUDA multipolar kernel against Python reference implementation."""
    N = 200
    coords, box, pairs, q, p, t, cutoff, ewald_alpha, prefactor = create_test_data(
        N, rank, device=device, dtype=dtype, ewald_alpha=ewald
    )
    if excl:
        pairs_excl = torch.tensor([[i % N, (i + 1) % N] for i in range(N)], device=device, dtype=pairs.dtype)
    else:
        pairs_excl = None

    func = MultipolarInteraction(
        rank, cutoff, ewald_alpha, prefactor, use_customized_ops=True, return_fields=return_fields
    ).to(device=device, dtype=dtype)
    func_ref = MultipolarInteraction(
        rank, cutoff, ewald_alpha, prefactor, use_customized_ops=False, cuda_graph_compat=cuda_graph,
        return_fields=return_fields
    ).to(device=device, dtype=dtype)

    def _ref(coords, box, pairs, q, p, t, pairs_excl):
        ret = func_ref(coords, box, pairs, q, p, t, pairs_excl)
        if isinstance(ret, tuple):
            total = ret[0] + torch.sum(ret[1] ** 2)
            if rank >= 1:
                total += torch.sum(ret[2] ** 2)
            return total
        else:
            return ret
    
    def _prb(coords, box, pairs, q, p, t, pairs_excl):
        ret = func(coords, box, pairs, q, p, t, pairs_excl)
        if isinstance(ret, tuple):
            total = ret[0] + torch.sum(ret[1] ** 2)
            if rank >= 1:
                total += torch.sum(ret[2] ** 2)
            return total
        else:
            return ret

    check_op(
        _ref,
        _prb,
        {"coords": coords, "box": box, "pairs": pairs, "q": q, "p": p, "t": t, "pairs_excl": pairs_excl},
        check_func=True if dtype is torch.float64 else False,
        check_grad=True if dtype is torch.float64 else False,
        atol=1e-6,
        rtol=1e-6,
        verbose=False
    )


@pytest.mark.performance
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_perf_multipolar(device, dtype, rank):
    """Performance comparison between Python and custom CUDA multipolar implementations."""
    N = 500
    coords, box, pairs, q, p, t, cutoff, ewald_alpha, prefactor = create_test_data(
        N, rank, device=device, dtype=dtype
    )

    func_ref = torch.compile(
        MultipolarInteraction(
            rank, cutoff, ewald_alpha, prefactor,
            use_customized_ops=False, cuda_graph_compat=False,
        )
    ).to(device=device, dtype=dtype)
    func = MultipolarInteraction(
        rank, cutoff, ewald_alpha, prefactor, use_customized_ops=True
    ).to(device=device, dtype=dtype)

    perf_op(
        func_ref,
        coords,
        box,
        pairs,
        q,
        p,
        t,
        desc=f"multipolar_ref (N={N}, rank={rank})",
        repeat=1000,
        run_backward=True,
    )
    perf_op(
        func,
        coords,
        box,
        pairs,
        q,
        p,
        t,
        desc=f"multipolar_torchff (N={N}, rank={rank})",
        repeat=1000,
        run_backward=True,
    )

