"""Benchmark Vesin neighbor-list construction (brute_force, cell_list)."""
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


def run_vesin(
    num_atoms: int,
    density: float,
    cutoff: float,
    device: str,
    dtype: torch.dtype,
    repeat: int,
    algorithm: str = "cell_list",
) -> Dict[str, Any]:
    """Run a single Vesin neighbor-list benchmark.

    Parameters
    ----------
    algorithm : str
        One of ``"brute_force"`` or ``"cell_list"`` (Vesin API).
    """
    from vesin.torch import NeighborList as VesinNeighborList

    coords, box, cutoff, _ = create_benchmark_data(
        num_atoms, density, cutoff, device, dtype
    )
    calculator = VesinNeighborList(
        cutoff=cutoff, full_list=False, algorithm=algorithm
    )
    engine = "vesin_bf" if algorithm == "brute_force" else "vesin_cl"
    desc = f"nblist_{engine} (N={num_atoms})"

    try:
        perf = perf_op(
            calculator.compute, coords, box, True, "ij",
            desc=desc, warmup=10, repeat=repeat,
            run_backward=False, use_cuda_graph=False, explicit_sync=True,
        )
    except Exception as e:
        print(f"  [{engine}] N={num_atoms}: FAILED ({e})")
        return make_failed_result(engine, num_atoms)

    result = make_result(engine, num_atoms, perf)
    print(
        f"  [{engine}] N={num_atoms}: "
        f"{result['mean_ms_per_step']:.4f} +/- {result['std_ms_per_step']:.4f} ms/step"
    )
    return result


def main() -> None:
    parser = get_common_parser(
        "Benchmark Vesin neighbor-list construction (brute_force, cell_list)."
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when using a CUDA Torch device.")

    dtype = parse_dtype(args.dtype)
    setup_seeds()

    all_results: List[Dict[str, Any]] = []

    for n_atoms in args.atoms:
        print(f"\n=== N = {n_atoms} atoms ===")
        for algorithm in ("brute_force", "cell_list"):
        # for algorithm in ("cell_list",):
            all_results.append(run_vesin(
                num_atoms=n_atoms, density=args.density, cutoff=args.cutoff,
                device=args.device, dtype=dtype, repeat=args.repeat,
                algorithm=algorithm,
            ))

    write_results_to_csv(all_results, args.output)


if __name__ == "__main__":
    main()
