"""Benchmark ALCHEMI (nvalchemi-toolkit-ops) neighbor-list construction."""
from typing import Any, Dict, List

import torch

from benchmark_utils import (
    create_benchmark_data,
    get_common_parser,
    make_failed_result,
    make_result,
    parse_dtype,
    perf_op,
    setup_seeds,
    write_results_to_csv,
)


def _max_neighbors_estimate(num_atoms: int, density: float, cutoff: float) -> int:
    """Conservative per-atom max neighbors for cubic box."""
    return min(num_atoms - 1, max(64, int((cutoff ** 3) * density * 2)))


def run_alchemi(
    num_atoms: int,
    density: float,
    cutoff: float,
    device: str,
    dtype: torch.dtype,
    repeat: int,
    method: str = "cell_list",
) -> Dict[str, Any]:
    """Run a single ALCHEMI neighbor-list benchmark and return timing results."""
    from nvalchemiops.neighborlist import cell_list, naive_neighbor_list

    coords, box, cutoff, _ = create_benchmark_data(
        num_atoms, density, cutoff, device, dtype
    )
    cell = box.unsqueeze(0)
    pbc = torch.tensor([[True, True, True]], device=device, dtype=torch.bool)

    if method == "cell_list":
        engine = "alchemi_cl"

        def step():
            return cell_list(coords, cutoff, cell, pbc)
    else:
        engine = "alchemi_naive"
        max_neighbors = _max_neighbors_estimate(num_atoms, density, cutoff)

        def step():
            return naive_neighbor_list(
                coords, cutoff, cell=cell, pbc=pbc, max_neighbors=max_neighbors
            )

    desc = f"nblist_{engine} (N={num_atoms})"
    try:
        perf = perf_op(
            step,
            desc=desc, warmup=10, repeat=repeat,
            run_backward=False, use_cuda_graph=False, explicit_sync=True,
        )
    except Exception as e:
        print(f"  [{engine}] N={num_atoms}: FAILED ({e})")
        return make_failed_result(engine, num_atoms)

    result = make_result(engine, num_atoms, perf)
    print(f"  [{engine}] N={num_atoms}: "
          f"{result['mean_ms_per_step']:.4f} +/- {result['std_ms_per_step']:.4f} ms/step")
    return result


def main() -> None:
    parser = get_common_parser(
        "Benchmark ALCHEMI (nvalchemi-toolkit-ops) neighbor-list construction."
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when using a CUDA Torch device.")

    dtype = parse_dtype(args.dtype)
    setup_seeds()

    all_results: List[Dict[str, Any]] = []

    for n_atoms in args.atoms:
        print(f"\n=== N = {n_atoms} atoms ===")
        for method in ("naive", "cell_list"):
            all_results.append(run_alchemi(
                num_atoms=n_atoms, density=args.density, cutoff=args.cutoff,
                device=args.device, dtype=dtype, repeat=args.repeat,
                method=method,
            ))

    write_results_to_csv(all_results, args.output)


if __name__ == "__main__":
    main()
