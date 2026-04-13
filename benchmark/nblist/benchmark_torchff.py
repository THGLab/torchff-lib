"""Benchmark TorchFF neighbor-list construction (nsquared, cell_list, python)."""
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
from torchff.nblist import NeighborList


def run_torchff(
    num_atoms: int,
    density: float,
    cutoff: float,
    device: str,
    dtype: torch.dtype,
    use_customized_ops: bool,
    repeat: int,
    algorithm: str = "nsquared",
) -> Dict[str, Any]:
    """Run a single TorchFF neighbor-list benchmark and return timing results."""
    coords, box, cutoff, max_npairs = create_benchmark_data(
        num_atoms, density, cutoff, device, dtype
    )

    if not use_customized_ops:
        engine = "python"
    elif algorithm == "cell_list":
        engine = "cell_list"
    else:
        engine = "nsquared"

    try:
        nblist = NeighborList(
            num_atoms, use_customized_ops=use_customized_ops,
            algorithm=algorithm,
        ).to(device)
    except Exception as e:
        print(f"  [{engine}] N={num_atoms}: SKIPPED (init failed: {e})")
        return make_failed_result(engine, num_atoms)

    desc = f"nblist_{engine} (N={num_atoms})"
    use_cuda_graph = (engine != "python")

    try:
        perf = perf_op(
            nblist, coords, box, cutoff, max_npairs, True,
            desc=desc, warmup=10, repeat=repeat,
            run_backward=False, use_cuda_graph=False,
            explicit_sync=True,
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
        "Benchmark TorchFF neighbor-list construction (nsquared, cell_list, python)."
    )
    parser.add_argument(
        "--max-python-atoms", type=int, default=30000,
        help="Skip the pure-Python path for systems larger than this.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when using a CUDA Torch device.")

    dtype = parse_dtype(args.dtype)
    setup_seeds()

    all_results: List[Dict[str, Any]] = []

    for n_atoms in args.atoms:
        print(f"\n=== N = {n_atoms} atoms ===")

        all_results.append(run_torchff(
            n_atoms, args.density, args.cutoff, args.device, dtype,
            use_customized_ops=True, repeat=args.repeat, algorithm="nsquared",
        ))
        all_results.append(run_torchff(
            n_atoms, args.density, args.cutoff, args.device, dtype,
            use_customized_ops=True, repeat=args.repeat, algorithm="cell_list",
        ))

        if n_atoms <= args.max_python_atoms:
            all_results.append(run_torchff(
                n_atoms, args.density, args.cutoff, args.device, dtype,
                use_customized_ops=False, repeat=args.repeat,
            ))
        else:
            print(
                f"  [python] N={n_atoms}: SKIPPED "
                f"(exceeds --max-python-atoms={args.max_python_atoms})"
            )

    write_results_to_csv(all_results, args.output)


if __name__ == "__main__":
    main()
