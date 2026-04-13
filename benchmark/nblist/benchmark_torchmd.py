"""Benchmark TorchMD-Net OptimizedDistance neighbor-list construction."""
from typing import Any, Dict, List

import numpy as np
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


def run_torchmd(
    num_atoms: int,
    density: float,
    cutoff: float,
    device: str,
    dtype: torch.dtype,
    repeat: int,
    strategy: str,
) -> Dict[str, Any]:
    """Run a single TorchMD-Net OptimizedDistance benchmark."""
    from torchmdnet.models.utils import OptimizedDistance

    coords, box, cutoff_val, max_npairs = create_benchmark_data(
        num_atoms, density, cutoff, device, dtype
    )

    engine = f"torchmd_{strategy}"
    batch = torch.zeros(num_atoms, dtype=torch.int64, device=device)

    try:
        nl = OptimizedDistance(
            cutoff_upper=cutoff_val,
            max_num_pairs=max_npairs,
            strategy=strategy,
            box=box,
            loop=False,
            include_transpose=False,
            resize_to_fit=False,
        )
    except Exception as e:
        print(f"  [{engine}] N={num_atoms}: SKIPPED (init failed: {e})")
        return make_failed_result(engine, num_atoms)

    desc = f"nblist_{engine} (N={num_atoms})"

    try:
        perf = perf_op(
            nl, coords, batch,
            desc=desc, warmup=10, repeat=repeat,
            run_backward=False, use_cuda_graph=False, explicit_sync=False,
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
        "Benchmark TorchMD-Net OptimizedDistance neighbor-list (brute + cell)."
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when using a CUDA Torch device.")

    dtype = parse_dtype(args.dtype)
    setup_seeds()

    all_results: List[Dict[str, Any]] = []

    for n_atoms in args.atoms:
        print(f"\n=== N = {n_atoms} atoms ===")
        for strategy in ("brute", "cell"):
            all_results.append(run_torchmd(
                num_atoms=n_atoms, density=args.density, cutoff=args.cutoff,
                device=args.device, dtype=dtype, repeat=args.repeat,
                strategy=strategy,
            ))

    write_results_to_csv(all_results, args.output)


if __name__ == "__main__":
    main()
