import csv
import os
from typing import List, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, '..'))
sys.path.insert(0, _script_dir)
from torchff.test_utils import perf_op
from tests.test_ewald import create_test_data
from torchff.ewald import Ewald


def run_single_benchmark(
    num_atoms: int,
    rank: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    repeat: int = 200,
) -> List[Dict[str, Any]]:
    """Run performance benchmark for a given system size and multipole rank.

    This benchmarks both the reference (no customized ops) and the customized
    CUDA Ewald implementation and returns a list of results dictionaries.
    """
    results: List[Dict[str, Any]] = []

    # Create test data (may raise OOM)
    coords, box, q, p, t, alpha, kmax = create_test_data(
        num_atoms, rank, device=device, dtype=dtype
    )

    # Reference implementation: Python / non-custom ops, compiled
    torch.cuda.empty_cache()
    try:
        func_ref = torch.compile(
            Ewald(alpha, kmax, rank, use_customized_ops=False, return_fields=False).to(device=device, dtype=dtype),
            mode='max-autotune-no-cudagraphs'
        )

        perf_ref = perf_op(
            func_ref,
            coords,
            box,
            q,
            p,
            t,
            desc=f"ewald_ref (N={num_atoms}, rank={rank})",
            repeat=repeat,
            run_backward=True,
            use_cuda_graph=True
        )
        results.append(
            {
                "rank": rank,
                "num_atoms": num_atoms,
                "impl": "ref",
                "mean_ms": float(np.mean(perf_ref)),
                "std_ms": float(np.std(perf_ref)),
                "success": True,
            }
        )
    except RuntimeError as e:
        # Catch CUDA OOM or similar runtime errors
        if "out of memory" in str(e).lower():
            print(
                f"Encountered OOM for ref implementation at "
                f"N={num_atoms}, rank={rank}: {e}"
            )
        else:
            print(
                f"Encountered RuntimeError for ref implementation at "
                f"N={num_atoms}, rank={rank}: {e}"
            )
        torch.cuda.empty_cache()
        results.append(
            {
                "rank": rank,
                "num_atoms": num_atoms,
                "impl": "ref",
                "mean_ms": float("nan"),
                "std_ms": float("nan"),
                "success": False,
            }
        )

    # Customized CUDA ops implementation
    torch.cuda.empty_cache()
    try:
        func_custom = torch.compile(Ewald(
            alpha, kmax, rank, use_customized_ops=True, return_fields=False
        ).to(device=device, dtype=dtype), mode='max-autotune-no-cudagraphs')

        perf_custom = perf_op(
            func_custom,
            coords,
            box,
            q,
            p,
            t,
            desc=f"ewald_torchff (N={num_atoms}, rank={rank})",
            repeat=repeat,
            run_backward=True,
            use_cuda_graph=True
        )
        results.append(
            {
                "rank": rank,
                "num_atoms": num_atoms,
                "impl": "custom",
                "mean_ms": float(np.mean(perf_custom)),
                "std_ms": float(np.std(perf_custom)),
                "success": True,
            }
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(
                f"Encountered OOM for customized implementation at "
                f"N={num_atoms}, rank={rank}: {e}"
            )
        else:
            print(
                f"Encountered RuntimeError for customized implementation at "
                f"N={num_atoms}, rank={rank}: {e}"
            )
        torch.cuda.empty_cache()
        results.append(
            {
                "rank": rank,
                "num_atoms": num_atoms,
                "impl": "custom",
                "mean_ms": float("nan"),
                "std_ms": float("nan"),
                "success": False,
            }
        )

    # TorchPME implementation (charge-only, rank=0)
    torch.cuda.empty_cache()
    if rank == 0:
        try:
            from torchpme_impl import EwaldTorchPME

            charges = q.unsqueeze(1) if q.ndim == 1 else q
            func_torchpme = torch.compile(
                EwaldTorchPME(alpha=alpha, kmax=kmax).to(device=device, dtype=dtype),
                mode='max-autotune'
            )
            perf_torchpme = perf_op(
                func_torchpme,
                coords,
                box,
                charges,
                desc=f"ewald_torchpme (N={num_atoms}, rank={rank})",
                repeat=repeat,
                run_backward=True,
                use_cuda_graph=False,
            )
            results.append(
                {
                    "rank": rank,
                    "num_atoms": num_atoms,
                    "impl": "torchpme",
                    "mean_ms": float(np.mean(perf_torchpme)),
                    "std_ms": float(np.std(perf_torchpme)),
                    "success": True,
                }
            )
        except ImportError as e:
            print(
                f"Skipping TorchPME benchmark (not installed or import failed): {e}"
            )
            results.append(
                {
                    "rank": rank,
                    "num_atoms": num_atoms,
                    "impl": "torchpme",
                    "mean_ms": float("nan"),
                    "std_ms": float("nan"),
                    "success": False,
                }
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(
                    f"Encountered OOM for TorchPME at "
                    f"N={num_atoms}, rank={rank}: {e}"
                )
            else:
                print(
                    f"Encountered RuntimeError for TorchPME at "
                    f"N={num_atoms}, rank={rank}: {e}"
                )
            torch.cuda.empty_cache()
            results.append(
                {
                    "rank": rank,
                    "num_atoms": num_atoms,
                    "impl": "torchpme",
                    "mean_ms": float("nan"),
                    "std_ms": float("nan"),
                    "success": False,
                }
            )

    return results


def write_results_to_csv(results: List[Dict[str, Any]], csv_path: str) -> None:
    """Write benchmark results to CSV."""
    fieldnames = ["rank", "num_atoms", "impl", "mean_ms", "std_ms", "success"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Wrote benchmark results to {csv_path}")


def plot_results(results: List[Dict[str, Any]], pdf_path: str) -> None:
    """Create a 3-panel PDF plot (one per rank) comparing implementations.

    Each subplot uses log-log scales with system size on the x-axis and time
    in milliseconds on the y-axis. Two lines are drawn per subplot:
    one for the reference implementation and one for the customized ops.
    """
    ranks = [0, 1, 2]
    impl_labels = {"ref": "TorchFF (Python)", "custom": "TorchFF (CUDA)", "torchpme": "TorchPME"}
    impl_styles = {"ref": "o-", "custom": "s-", "torchpme": "^-"}

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

    for ax, rank in zip(axes, ranks):
        # Filter results for current rank and successful runs
        rank_results = [r for r in results if r["rank"] == rank and r["success"]]

        impls_to_plot = ("ref", "custom", "torchpme") if rank == 0 else ("ref", "custom")
        for impl in impls_to_plot:
            impl_results = [r for r in rank_results if r["impl"] == impl]
            if not impl_results:
                continue

            # Sort by system size for consistent plotting
            impl_results.sort(key=lambda r: r["num_atoms"])
            sizes = [r["num_atoms"] for r in impl_results]
            mean_ms = [r["mean_ms"] for r in impl_results]

            ax.plot(
                sizes,
                mean_ms,
                impl_styles[impl],
                label=impl_labels[impl],
                linewidth=1.5,
                markersize=5,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("System size (number of atoms)")
        ax.set_title(f"Rank {rank}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        if rank == 0:
            ax.set_ylabel("Time (ms)")
        ax.legend()

    plt.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved benchmark plot to {pdf_path}")


def main() -> None:
    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = "cuda"
    dtype = torch.float64

    # Choose a range of system sizes up to 10,000 atoms.
    system_sizes = [100, 300, 1000, 3000, 10000, 20000]
    ranks = [0, 1, 2]

    all_results: List[Dict[str, Any]] = []

    for rank in ranks:
        for num_atoms in system_sizes:
            print(f"Running benchmark for N={num_atoms}, rank={rank}")
            try:
                bench_results = run_single_benchmark(
                    num_atoms=num_atoms,
                    rank=rank,
                    device=device,
                    dtype=dtype,
                    repeat=200,
                )
                all_results.extend(bench_results)
            except RuntimeError as e:
                # Handle possible OOM during data creation or other runtime errors
                if "out of memory" in str(e).lower():
                    print(
                        f"Encountered OOM while setting up benchmark for "
                        f"N={num_atoms}, rank={rank}: {e}"
                    )
                else:
                    print(
                        f"Encountered RuntimeError while setting up benchmark for "
                        f"N={num_atoms}, rank={rank}: {e}"
                    )
                torch.cuda.empty_cache()
                # Record failed entries for implementations
                impls = ("ref", "custom", "torchpme") if rank == 0 else ("ref", "custom")
                for impl in impls:
                    all_results.append(
                        {
                            "rank": rank,
                            "num_atoms": num_atoms,
                            "impl": impl,
                            "mean_ms": float("nan"),
                            "std_ms": float("nan"),
                            "success": False,
                        }
                    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "benchmark_ewald.csv")
    pdf_path = os.path.join(script_dir, "benchmark_ewald.pdf")

    write_results_to_csv(all_results, csv_path)
    plot_results(all_results, pdf_path)


if __name__ == "__main__":
    main()

