"""Benchmark ALCHEMI (nvalchemi-toolkit-ops) neighbor-list construction.

Uses the ``nvalchemiops.torch.neighbors`` high-level API with pre-allocated
buffers to avoid per-call memory allocation overhead during timing.
"""
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

from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.cell_list import estimate_cell_list_sizes
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    compute_naive_num_shifts,
    estimate_max_neighbors,
)


def _prepare_inputs(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    method: str,
    device: str,
    density: float = 100.0,
) -> Dict[str, Any]:
    """Build input dict with pre-allocated buffers for ``neighbor_list()``.

    Mirrors the allocation strategy from the ALCHEMI developer benchmarks:
    output buffers (neighbor_matrix, shifts, counts) and method-specific
    internal structures (cell-list cache or naive shift tables) are allocated
    once and reused across timed iterations.
    """
    total_atoms = positions.shape[0]

    max_neighbors = estimate_max_neighbors(
        cutoff, atomic_density=density, safety_factor=1.2
    )
    neighbor_matrix = torch.full(
        (total_atoms, max_neighbors),
        total_atoms,
        dtype=torch.int32,
        device=device,
    )
    neighbor_matrix_shifts = torch.zeros(
        (total_atoms, max_neighbors, 3),
        dtype=torch.int32,
        device=device,
    )
    num_neighbors = torch.zeros(
        total_atoms, dtype=torch.int32, device=device
    )

    inputs: Dict[str, Any] = {
        "positions": positions,
        "cutoff": cutoff,
        "cell": cell,
        "pbc": pbc,
        "method": method,
        "neighbor_matrix": neighbor_matrix,
        "neighbor_matrix_shifts": neighbor_matrix_shifts,
        "num_neighbors": num_neighbors,
    }

    if "naive" in method:
        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell, cutoff, pbc)
        )
        inputs["shift_range_per_dimension"] = shift_range_per_dimension
        inputs["shift_offset"] = shift_offset
        inputs["total_shifts"] = total_shifts
    elif "cell_list" in method:
        max_total_cells, neighbor_search_radius = (
            estimate_cell_list_sizes(cell, pbc, cutoff)
        )
        cell_list_cache = allocate_cell_list(
            total_atoms, max_total_cells, neighbor_search_radius, device,
        )
        inputs["cells_per_dimension"] = cell_list_cache[0]
        inputs["neighbor_search_radius"] = cell_list_cache[1]
        inputs["atom_periodic_shifts"] = cell_list_cache[2]
        inputs["atom_to_cell_mapping"] = cell_list_cache[3]
        inputs["atoms_per_cell_count"] = cell_list_cache[4]
        inputs["cell_atom_start_indices"] = cell_list_cache[5]
        inputs["cell_atom_list"] = cell_list_cache[6]

    return inputs


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
    coords, box, cutoff, _ = create_benchmark_data(
        num_atoms, density, cutoff, device, dtype
    )
    cell = box.unsqueeze(0)
    pbc = torch.tensor([[True, True, True]], device=device, dtype=torch.bool)

    engine = "alchemi_cl" if method == "cell_list" else "alchemi_naive"

    try:
        inputs = _prepare_inputs(
            coords, cell, pbc, cutoff, method, device, density=density,
        )
    except Exception as e:
        print(f"  [{engine}] N={num_atoms}: SKIPPED (pre-alloc failed: {e})")
        return make_failed_result(engine, num_atoms)

    def step():
        return neighbor_list(**inputs)

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
